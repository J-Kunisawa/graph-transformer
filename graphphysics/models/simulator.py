import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data

from graphphysics.models.layers import Normalizer
from graphphysics.utils.nodetype import NodeType


class Simulator(nn.Module):
    """
    Neural simulator wrapper.

    - 正規化器（Normalizer）を管理し、入力をモデルに渡すための前処理を行う。
    - トレーニング時は正規化された target-delta を返し、評価時は最終出力も復元して返す。
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        feature_index_start: int,
        feature_index_end: int,
        output_index_start: int,
        output_index_end: int,
        node_type_index: int,
        model: nn.Module,
        device: torch.device,
        model_dir: str = "checkpoint/simulator.pth",
    ) -> None:
        super().__init__()

        self.node_input_size = int(node_input_size)
        self.edge_input_size = int(edge_input_size) if edge_input_size and edge_input_size > 0 else None
        self.output_size = int(output_size)

        self.feature_index_start = int(feature_index_start)
        self.feature_index_end = int(feature_index_end)
        self.node_type_index = int(node_type_index)

        self.output_index_start = int(output_index_start)
        self.output_index_end = int(output_index_end)

        self.model_dir = str(model_dir)
        self.model = model.to(device)

        # Normalizers: バッファとして統計を保持
        self._output_normalizer = Normalizer(size=self.output_size, name="output_normalizer", device=device)
        self._node_normalizer = Normalizer(size=self.node_input_size, name="node_normalizer", device=device)
        self._edge_normalizer = (
            Normalizer(size=self.edge_input_size, name="edge_normalizer", device=device)
            if self.edge_input_size is not None
            else None
        )

        self.device = device

    def _get_pre_target(self, inputs: Data) -> torch.Tensor:
        """inputs.x から pre-target を安全に取り出す。"""
        if not hasattr(inputs, "x"):
            raise AttributeError("inputs must have attribute 'x'")
        return inputs.x[:, self.output_index_start : self.output_index_end]

    def _get_target_normalized(self, inputs: Data, is_training: bool = True) -> Optional[torch.Tensor]:
        """
        target delta を計算して正規化する。
        Normalizer は None を受け取った場合 None を返す実装を想定。
        """
        target = getattr(inputs, "y", None)
        if target is None:
            return None
        pre_target = self._get_pre_target(inputs)
        target_delta = target - pre_target
        # Normalizer は Optional を扱えるため直接呼び出し
        return self._output_normalizer(target_delta, is_training)

    def _get_one_hot_type(self, inputs: Data) -> torch.Tensor:
        """node type を one-hot に変換して返す。"""
        if not hasattr(inputs, "x"):
            raise AttributeError("inputs must have attribute 'x'")
        # node_type は x の列の一つとして格納されている想定（float であっても cast する）
        node_type_col = inputs.x[:, self.node_type_index].long().squeeze()
        # 保険: 範囲外の場合はゼロにする
        if node_type_col.numel() == 0:
            return torch.zeros((inputs.x.size(0), NodeType.SIZE), device=self.device, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(node_type_col, NodeType.SIZE).to(dtype=torch.float32, device=self.device)
        return one_hot

    def _build_node_features(self, inputs: Data, one_hot_type: torch.Tensor) -> torch.Tensor:
        """feature_index_start:end を抽出し one_hot と連結する。"""
        if not hasattr(inputs, "x"):
            raise AttributeError("inputs must have attribute 'x'")
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]
        # features が空次元の可能性に対応
        if features.numel() == 0:
            features = torch.zeros((inputs.x.size(0), 0), device=self.device, dtype=torch.float32)
        # one_hot は float32 にしておく
        return torch.cat([features.to(device=self.device, dtype=torch.float32), one_hot_type.to(dtype=torch.float32)], dim=1)

    def _build_input_graph(self, inputs: Data, is_training: bool) -> Tuple[Data, Optional[torch.Tensor]]:
        """
        モデルへ渡す入力グラフを構築する：
        - node features を正規化
        - edge_attr を正規化（存在する場合）
        - target delta の正規化を返す
        """
        target_delta_normalized = self._get_target_normalized(inputs, is_training)
        one_hot_type = self._get_one_hot_type(inputs)
        node_features = self._build_node_features(inputs, one_hot_type)

        node_features_normalized = self._node_normalizer(node_features, is_training)

        # edge_attr が None の場合はそのまま None を伝播
        edge_attr = getattr(inputs, "edge_attr", None)
        if self._edge_normalizer is not None:
            edge_attr = self._edge_normalizer(edge_attr, is_training)

        graph = Data(
            x=node_features_normalized,
            pos=getattr(inputs, "pos", None),
            edge_attr=edge_attr,
            edge_index=getattr(inputs, "edge_index", None),
        )

        return graph, target_delta_normalized

    def build_outputs(self, inputs: Data, network_output: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        network_output を逆正規化して pre_target に加え、最終出力を復元する。
        network_output が None の場合は None を返す。
        """
        if network_output is None:
            return None
        pre_target = self._get_pre_target(inputs)
        update = self._output_normalizer.inverse(network_output)
        if update is None:
            return pre_target
        return pre_target + update

    def forward(self, inputs: Data) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward:
         - training: (network_output, target_delta_normalized, None)
         - eval:     (network_output, target_delta_normalized, outputs)
        """
        graph, target_delta_normalized = self._build_input_graph(inputs=inputs, is_training=self.training)
        network_output = self.model(graph)

        if self.training:
            return network_output, target_delta_normalized, None

        outputs = self.build_outputs(inputs=inputs, network_output=network_output)
        return network_output, target_delta_normalized, outputs

    def freeze_all(self) -> None:
        """モデルパラメータを全て凍結する。"""
        for param in self.model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, ckpdir: Optional[str] = None) -> None:
        """チェックポイントから model / normalizer 状態を復元する。"""
        if ckpdir is None:
            ckpdir = self.model_dir
        if not os.path.exists(ckpdir):
            logger.error(f"Checkpoint not found: {ckpdir}")
            return

        checkpoint = torch.load(ckpdir, map_location=self.device)
        model_state = checkpoint.get("model", checkpoint.get("model_state_dict", None))
        if model_state is None:
            logger.error("No model state found in checkpoint")
        else:
            try:
                self.load_state_dict(model_state)
            except Exception:
                # 古いキー名差分等の保険として部分読み込み
                self.model.load_state_dict(model_state, strict=False)

        # normalizer の復元（辞書を期待）
        for key in ["_output_normalizer", "_node_normalizer", "_edge_normalizer"]:
            state = checkpoint.get(key, None)
            normalizer = getattr(self, key, None)
            if normalizer is not None and isinstance(state, dict):
                # state は get_variable() で保存した辞書想定 -> バッファに戻す
                for attr_name, value in state.items():
                    try:
                        # 既存バッファ／属性があれば上書き
                        if hasattr(normalizer, attr_name):
                            setattr(normalizer, attr_name, value)
                    except Exception:
                        pass

        logger.success(f"Simulator model loaded checkpoint {ckpdir}")

    def save_checkpoint(self, savedir: Optional[str] = None) -> None:
        """モデルと normalizer 状態をチェックポイントに保存する。"""
        if savedir is None:
            savedir = self.model_dir
        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        model_state = self.state_dict()
        out_norm = self._output_normalizer.get_variable() if hasattr(self._output_normalizer, "get_variable") else None
        node_norm = self._node_normalizer.get_variable() if hasattr(self._node_normalizer, "get_variable") else None
        edge_norm = self._edge_normalizer.get_variable() if (self._edge_normalizer and hasattr(self._edge_normalizer, "get_variable")) else None

        to_save = {
            "model": model_state,
            "_output_normalizer": out_norm,
            "_node_normalizer": node_norm,
            "_edge_normalizer": edge_norm,
        }

        torch.save(to_save, savedir)
        logger.success(f"Simulator model saved checkpoint {savedir}")