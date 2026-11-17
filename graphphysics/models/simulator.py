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
    シミュレータ・ラッパークラス。
    コアとなるAIモデル（MeshGraphNetやTransformer等）を包み込み、
    入力物理量・エッジ・予測ターゲットの正規化（標準化）処理、特徴量整形、
    および学習状態（チェックポイント）のライフサイクルを一括管理する。
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
    ):
        """
        パラメータ:
            node_input_size (int): ネットワークに入力するノード特徴量の最終次元数。
            edge_input_size (int): ネットワークに入力するエッジ特徴量の最終次元数。
            output_size (int): モデルが予測するターゲット（変化量）の次元数。
            feature_index_start (int): 生データ内における、入力特徴量（速度等）の開始インデックス。
            feature_index_end (int): 生データ内における、入力特徴量の終了インデックス。
            output_index_start (int): 生データ内における、ターゲットに対応する現時刻フィールドの開始インデックス。
            output_index_end (int): 生データ内における、ターゲットに対応する現時刻フィールドの終了インデックス。
            node_type_index (int): 生データ内における、ノード属性（壁や流体など）の格納インデックス。
            model (nn.Module): 演算コアとなるAIモデル（Processorを内包する実体）。
            device (torch.device): 演算を実行するデバイス（CUDA/CPU）。
            model_dir (str): チェックポイントファイルの保存・読込先パス。
        """
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size if edge_input_size > 0 else None
        self.output_size = output_size

        self.feature_index_start = feature_index_start
        self.feature_index_end = feature_index_end
        self.node_type_index = node_type_index

        self.output_index_start = output_index_start
        self.output_index_end = output_index_end

        self.model_dir = model_dir
        self.model = model.to(device)
        
        # --- 各種オンラインノーマライザーの初期化 ---
        # 予測出力（物理量の変化量Δx）のスケールを標準化するためのノーマライザー
        self._output_normalizer = Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        # ネットワークに入力するノード全体特徴量を標準化するためのノーマライザー
        self._node_normalizer = Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        # ネットワークに入力するエッジ幾何特徴量を標準化するためのノーマライザー
        self._edge_normalizer = (
            Normalizer(size=edge_input_size, name="edge_normalizer", device=device)
            if self.edge_input_size is not None
            else None
        )

        self.device = device

    def _get_pre_target(self, inputs: Data) -> torch.Tensor:
        """
        現時刻のグラフ（inputs）から、予測対象の基準となる現在の物理量（pre-target: 例、現時刻の速度）を抽出する。
        """
        return inputs.x[:, self.output_index_start : self.output_index_end]

    def _get_target_normalized(
        self, inputs: Data, is_training: bool = True
    ) -> torch.Tensor:
        """
        正解の物理変化量（残差: Δx = y - x_t）を計算し、さらに出力を標準化（正規化）して教師ラベルを構築する。
        """
        target = inputs.y           # 次の時刻 t+dt の正解物理量
        pre_target = self._get_pre_target(inputs) # 現在の時刻 t の物理量
        
        # 物理量の時間変化量（差分）を計算（残差学習のコア）
        target_delta = target - pre_target
        
        # 変化量をノーマライザーに通してスケーリング（訓練時は統計情報も更新する）
        target_delta_normalized = self._output_normalizer(target_delta, is_training)

        return target_delta_normalized

    def _get_one_hot_type(self, inputs: Data) -> torch.Tensor:
        """
        ノードの属性フィールド（インデックス番号）を、NodeType.SIZEの長さのワンホットベクトルに変換する。
        例: 壁(1) ➔ [0, 1, 0, 0]
        """
        node_type = inputs.x[:, self.node_type_index]
        return torch.nn.functional.one_hot(
            torch.squeeze(node_type.long()), NodeType.SIZE
        )

    def _build_node_features(
        self, inputs: Data, one_hot_type: torch.Tensor
    ) -> torch.Tensor:
        """
        抽出した純粋な物理特徴量（速度等）の後ろに、ノード属性のワンホットベクトルを結合して
        モデルに入力する一連の複合ノード特徴量マトリクスを形成する。
        """
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]
        node_features = torch.cat([features, one_hot_type], dim=1)

        return node_features

    def _build_input_graph(
        self, inputs: Data, is_training: bool
    ) -> Tuple[Data, torch.Tensor]:
        """
        生データをモデルがそのまま安全に食べられる「正規化済みのバッチグラフ」へと変換加工する。
        """
        # 1. 教師ラベルとなる、正規化された物理変化量（Δx）を取得
        target_delta_normalized = self._get_target_normalized(inputs, is_training)
        
        # 2. ノードのワンホット属性を構築し、特徴量を結合
        one_hot_type = self._get_one_hot_type(inputs)
        node_features = self._build_node_features(inputs, one_hot_type)

        # 3. 結合した複合ノード特徴量を標準化
        node_features_normalized = self._node_normalizer(node_features, is_training)

        # 4. エッジ属性（距離ベクトル等）が存在する場合はエッジも標準化
        if self._edge_normalizer is not None:
            edge_attr = self._edge_normalizer(inputs.edge_attr, is_training)
        else:
            edge_attr = inputs.edge_attr

        # 5. モデルに入力するための綺麗な正規化済みPyG Dataオブジェクトを新しくパッキング
        graph = Data(
            x=node_features_normalized,
            pos=inputs.pos,
            edge_attr=edge_attr,
            edge_index=inputs.edge_index,
            node_type=inputs.x[:, self.node_type_index].long(),
        )
        # 動的な隣接行列拡張が異なるサンプル間を接続しないよう、PyG Batch情報を保持する。
        input_batch = getattr(inputs, "batch", None)
        if input_batch is not None:
            graph.batch = input_batch

        return graph, target_delta_normalized

    def build_outputs(self, inputs: Data, network_output: torch.Tensor) -> torch.Tensor:
        """
        ネットワークの予測出力（標準化された変化量）から、
        正規化の逆変換を行い、現時刻の値を加算して最終的な「次ステップの実物理量」を復元・再構築する。
        """
        pre_target = self._get_pre_target(inputs) # 現時刻の実物理量（x_t）
        update = self._output_normalizer.inverse(network_output) # AIの出力を物理スケール（m/s等）に復元
        
        # 次の時刻の絶対物理量を確定（ x_t+1 = x_t + Δx ）
        return pre_target + update

    def forward(
        self, inputs: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        シミュレータの順伝播エントリポイント。
        
        戻り値（タプル）:
            - network_output: モデルが直接予測した標準化デルタ値（Lossの計算用）
            - target_delta_normalized: 正規化された正解デルタ値（Lossの計算用）
            - outputs: 実スケールに復元された次ステップの予測物理場（検証・推論時のみ出力）
        """
        # 1. 入力データの正規化と整形を実行
        graph, target_delta_normalized = self._build_input_graph(
            inputs=inputs, is_training=self.training
        )
        
        # 2. 演算コアモデル（Processorスタック）にグラフを投入して順伝播計算
        network_output = self.model(graph)

        # 3. 訓練モードか評価・推論モードかで出力を出し分ける
        if self.training:
            # 訓練時は逆伝播用の標準化出力テンソルのみを返し、計算コストとメモリをカット
            return network_output, target_delta_normalized, None
        else:
            # 検証・推論時は、自律回帰ループ(Rollout)を回すために、実物理スケールに復元した値も計算して返す
            outputs = self.build_outputs(inputs=inputs, network_output=network_output)
            return network_output, target_delta_normalized, outputs

    def freeze_all(self) -> None:
        """モデル内の全パラメータの勾配計算をストップ（フリーズ）させる。再学習の固定用。"""
        for param in self.model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, ckpdir: Optional[str] = None) -> None:
        """
        チェックポイントバイナリ（.pth）から、モデルの重みだけでなく、
        各ノーマライザーが記憶していた走行統計情報（平均・標準偏差バッファ）も完全にロードして復元する。
        """
        if ckpdir is None:
            ckpdir = self.model_dir
            
        checkpoint = torch.load(ckpdir, map_location=self.device)
        
        # コアモデルの重みのロード
        self.load_state_dict(checkpoint["model"])

        # 各ノーマライザーの内部バッファ値（平均、平方和、蓄積ステップ等）を復旧
        normalizer_keys = ["_output_normalizer", "_node_normalizer", "_edge_normalizer"]
        for key in normalizer_keys:
            normalizer_state = checkpoint.get(key, {})
            normalizer = getattr(self, key, None)
            if normalizer and normalizer_state:
                for attr_name, value in normalizer_state.items():
                    setattr(normalizer, attr_name, value)

        logger.success(f"Simulatorモデルがチェックポイントのロードに成功しました: {ckpdir}")

    def save_checkpoint(self, savedir: Optional[str] = None) -> None:
        """
        現在のモデルの重みマトリクス、および各ノーマライザーの走行統計情報を
        1つのチェックポイント辞書ファイルとしてまとめて安全に保存する。
        """
        if savedir is None:
            savedir = self.model_dir

        # 保存先の親ディレクトリが存在しない場合は自動作成
        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        # 保存するステートディクショナリのパッキング
        model_state = self.state_dict()
        output_normalizer_state = self._output_normalizer.get_variable()
        node_normalizer_state = self._node_normalizer.get_variable()
        edge_normalizer_state = (
            self._edge_normalizer.get_variable() if self._edge_normalizer else None
        )

        to_save = {
            "model": model_state,
            "_output_normalizer": output_normalizer_state,
            "_node_normalizer": node_normalizer_state,
            "_edge_normalizer": edge_normalizer_state,
        }

        torch.save(to_save, savedir)
        logger.success(f"Simulatorモデルがチェックポイントの保存に成功しました: {savedir}")
