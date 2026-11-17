import csv
import json
import os
import sys
from time import perf_counter
from typing import Dict, List, Optional

import lightning as L
import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.models.spatial_mtp_1hop import SpatialMTP1Hop
from graphphysics.training.parse_parameters import (
    get_gradient_method,
    get_loss,
    get_model,
    get_simulator,
)
from graphphysics.utils.loss import L2Loss, MultiLoss
from graphphysics.utils.meshio_mesh import (
    append_mesh_to_xdmf,
    convert_to_meshio_vtu,
    meshes_to_xdmf,
)
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.scheduler import CosineWarmupScheduler


def build_mask(param: dict, graph: Batch):
    """
    メッシュ内のノード属性から、値を強制上書き（リセット）すべき境界条件ノード（壁面、流入、障害物など）
    の領域を特定するためのバイナリマスクを作成する。
    通常の流体ノード（NORMAL）と流出ノード（OUTFLOW）以外（固定すべき条件領域）が True になる。
    """
    if len(graph.x.shape) > 2:
        # 3次元バッチテンソル形状 [Batch, NumNodes, FeatureDim] の場合
        node_type = graph.x[:, 0, param["index"]["node_type_index"]]
    else:
        # 通常のフラットグラフ形状 [NumNodes, FeatureDim] の場合
        node_type = graph.x[:, param["index"]["node_type_index"]]
        
    # NORMAL または OUTFLOW である領域を特定し、それを論理反転（not）させる
    mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
    mask = torch.logical_not(mask)

    return mask


class LightningModule(L.LightningModule):
    """
    PyTorch Lightning 実行管理モジュール。
    物理AIモデルの訓練ステップ、時系列自律回帰を用いた検証ループ、推論、
    および空間自己教師あり学習（Spatial MTP）の統合パイプラインを司る。
    """
    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
        warmup_ratio: Optional[float] = None,
        trajectory_length: int = 599,
        timestep: float = 1.0,
        only_processor: bool = False,
        masks: list[NodeType] = [NodeType.NORMAL, NodeType.OUTFLOW],
        use_previous_data: bool = False,
        previous_data_start: int = None,
        previous_data_end: int = None,
        prediction_save_path: str = "predictions",
        enable_vram_optimizations: bool = False,
    ):
        super().__init__()
        # ハイパーパラメータ（引数）を checkpoint に自動保存
        self.save_hyperparameters()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.param = parameters
        self.wandb_run_id = None # Weights & Biases の実行ID保持用

        # 1. 設定JSONからコア演算プロセッサ（GNN / Transformer等）を取得・初期化
        processor = get_model(param=parameters, only_processor=only_processor)
        print(processor)

        # 2. プロセッサをシミュレータラッパー（正規化・残差処理担当）に注入
        self.model = get_simulator(param=parameters, model=processor, device=device)

        # 3. 損失関数の初期化（物理制約をブレンドした MultiLoss にも対応）
        self.loss, self.loss_name = get_loss(param=parameters)
        logger.info(f"Using loss {self.loss_name}")
        self.is_multiloss = False
        if isinstance(self.loss, MultiLoss):
            self.is_multiloss = True

        self.loss_masks = masks
        self.val_loss = L2Loss() # 検証用は一律L2（RMSEベース）で評価
        self.gradient_method = get_gradient_method(param=parameters)  # 有限差分法(finite_diff) または 最小二乗法(least_squares)

        # 最適化・学習率スケジューラ用設定
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup_ratio = warmup_ratio
        if warmup_ratio is not None and warmup_ratio >= 0:
            resolved_warmup = int(self.num_steps * warmup_ratio)
            if warmup_ratio > 0 and self.num_steps > 0:
                resolved_warmup = max(1, resolved_warmup)
            self.warmup = min(self.num_steps, resolved_warmup)
            logger.info(
                f"Using warmup_ratio={warmup_ratio} -> warmup={self.warmup} steps"
            )
        else:
            self.warmup = warmup

        # 各種状態カウンタおよびバッファの初期化
        self.step_counter = 0
        self.first_step_losses: List[torch.Tensor] = []
        self.val_step_outputs = []
        self.val_step_targets = []
        self.trajectory_length = trajectory_length
        self.timestep = timestep
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None

        # 過去履歴フィードバック用のインデックス範囲設定
        self.use_previous_data = use_previous_data
        self.previous_data_start = previous_data_start
        self.previous_data_end = previous_data_end

        # 推論（Prediction）モード用の時系列管理プロパティ
        self.prediction_save_path: str = prediction_save_path
        self.current_pred_trajectory = 0
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None

        # 推論時の誤差統計用バッファ
        self.predict_step_metrics: List[Dict[str, float]] = []
        self.predict_step_timings: List[Dict[str, float]] = []
        self.predict_global_squared_error = 0.0
        self.predict_global_absolute_error = 0.0
        self.predict_global_target_squared = 0.0
        self.predict_global_numel = 0
        self.predict_component_squared_error = None
        self.predict_component_absolute_error = None
        self.predict_component_numel = None
        self.predict_trajectory_step = 0
        self.predict_timing_step = 0
        self.predict_field_slices = self._resolve_predict_field_slices()
        self.predict_case_error_statistics: Dict[int, Dict[str, float]] = {}

        # 推論結果をディスクに書き出す際の圧縮フラグ
        self.compress_predictions = (
            parameters["compress_predictions"]
            if hasattr(self.parameters, "compress_predictions")
            else False
        )

        # --- 空間自己教師あり学習（Spatial Masked Target Prediction）のパラメータ定義 ---
        training_params: Dict = parameters.setdefault("training", {})
        self.enable_vram_optimizations: bool = bool(
            enable_vram_optimizations or training_params.get("enable_vram_optimizations", False)
        )
        training_params["enable_vram_optimizations"] = self.enable_vram_optimizations

        self.multi_step_enabled = bool(training_params.get("multi_step_loss", False))
        self.multi_step_steps = int(training_params.get("multi_step_loss_steps", 5))
        self.multi_step_weight = float(training_params.get("multi_step_loss_weight", 0.5))
        sampling = training_params.get("scheduled_sampling", {})
        self.scheduled_sampling_enabled = bool(sampling.get("enabled", False))
        self.scheduled_sampling_start = float(sampling.get("start_probability", 0.0))
        self.scheduled_sampling_end = float(sampling.get("end_probability", 0.5))
        self.scheduled_sampling_start_epoch = int(sampling.get("start_epoch", 0))
        self.scheduled_sampling_end_epoch = int(sampling.get("end_epoch", 20))
        if self.multi_step_enabled and self.multi_step_steps < 2:
            raise ValueError("multi_step_loss_stepsは2以上である必要があります。")
        if self.multi_step_weight < 0.0:
            raise ValueError("multi_step_loss_weightは0以上である必要があります。")
        if not 0.0 <= self.scheduled_sampling_start <= 1.0 or not 0.0 <= self.scheduled_sampling_end <= 1.0:
            raise ValueError("scheduled samplingの確率は0から1の範囲で指定してください。")
        if self.scheduled_sampling_end_epoch < self.scheduled_sampling_start_epoch:
            raise ValueError("scheduled samplingのend_epochはstart_epoch以上にしてください。")
        
        self.use_spatial_mtp: bool = training_params.get("use_spatial_mtp", False)
        self.spatial_mtp_alpha: float = training_params.get("spatial_mtp_alpha", 0.20) # 補助損失のブレンド重み
        self.spatial_mtp_centers_per_step: int = training_params.get("spatial_mtp_centers_per_step", 256)
        self.spatial_mtp_num_heads: int = training_params.get("spatial_mtp_num_heads", 4)
        self.spatial_mtp_num_layers: int = training_params.get("spatial_mtp_num_layers", 1)
        self.spatial_mtp_assume_undirected: bool = training_params.get("spatial_mtp_assume_undirected", True)
        self.spatial_mtp_max_neighbors: Optional[int] = training_params.get("spatial_mtp_max_neighbors")

        # 内部特徴量フック用のプレースホルダー
        self.output_head: Optional[nn.Module] = None
        self.node_encoder: Optional[nn.Module] = None
        self.spatial_mtp: Optional[SpatialMTP1Hop] = None
        self._head_hook = None
        self._nodeenc_hook = None
        self._penultimate_hidden = None
        self._H_nodeenc = None

        # 空間MTPが有効な場合、専用フック回路を組み立てる
        if self.use_spatial_mtp:
            self._setup_spatial_mtp(processor, device)

    def forward(self, graph: Batch):
        return self.model(graph)

    def _setup_spatial_mtp(self, processor: nn.Module, device: str) -> None:
        """
        プロセッサの内部構造を解析し、前方向計算（Forward）の途中の潜在特徴量を
        自動的に傍受（フック）して空間ターゲット予測レイヤーと紐付ける。
        """
        # 標準的なモデル構造（EncodeProcessDecode）からコンポーネントを検索
        out_head = getattr(processor, "decode_module", None)
        node_encoder = getattr(processor, "nodes_encoder", None)

        # 見つからない場合、Transolverアーキテクチャの内部モジュール名で再検索（フォールバック）
        if not isinstance(out_head, nn.Module) or not isinstance(node_encoder, nn.Module):
            transolver_model = getattr(processor, "model", None)
            if isinstance(transolver_model, nn.Module):
                maybe_encoder = getattr(transolver_model, "preprocess", None)
                maybe_head = getattr(transolver_model, "output_proj", None)
                if isinstance(maybe_encoder, nn.Module) and isinstance(maybe_head, nn.Module):
                    node_encoder = maybe_encoder
                    out_head = maybe_head

        if not isinstance(out_head, nn.Module):
            raise ValueError("空間MTPを有効にするには、プロセッサに出力ヘッドモジュール（'decode_module' または 'model.output_proj'）が必要です。")
        if not isinstance(node_encoder, nn.Module):
            raise ValueError("空間MTPを有効にするには、プロセッサにエンコーダモジュール（'nodes_encoder' または 'model.preprocess'）が必要です。")

        # 出力デコーダ内の最初の線形層を探索して隠れ層の次元数（d_model）を自動検出
        first_linear = next(
            (module for module in out_head.modules() if isinstance(module, nn.Linear)), None,
        )
        if first_linear is None:
            raise ValueError("デコーダモジュール内にLinear層が見つからないため、空間MTP用の次元数を自動推転できません。")

        d_model = first_linear.in_features
        torch_device = torch.device(device)

        self.output_head = out_head
        self.node_encoder = node_encoder
        self.spatial_mtp = SpatialMTP1Hop(
            d_model=d_model, num_heads=self.spatial_mtp_num_heads, num_layers=self.spatial_mtp_num_layers,
            assume_undirected=self.spatial_mtp_assume_undirected, max_neighbors=self.spatial_mtp_max_neighbors,
        ).to(torch_device)

        # --- フック関数の定義（順伝播の最中に特徴量をバッファへコピーする） ---
        def _capture_head_input(module, inputs):
            hidden = inputs[0]
            if isinstance(hidden, torch.Tensor) and hidden.dim() > 2:
                hidden = hidden.reshape(-1, hidden.size(-1))
            self._penultimate_hidden = hidden # デコーダ直前の潜在空間特徴量をキャプチャ

        def _capture_nodeenc_output(module, inputs, outputs):
            features = outputs
            if isinstance(features, torch.Tensor) and features.dim() > 2:
                features = features.reshape(-1, features.size(-1))
            self._H_nodeenc = features # エンコーダ直後の初期グラフ特徴量をキャプチャ

        # モデルへフックを登録
        self._head_hook = out_head.register_forward_pre_hook(_capture_head_input)
        self._nodeenc_hook = node_encoder.register_forward_hook(_capture_nodeenc_output)

    def _remove_spatial_mtp_hooks(self) -> None:
        """メモリリークや意図しない動作を防ぐため、登録されたフックを安全に解除する"""
        if self._head_hook is not None:
            self._head_hook.remove()
            self._head_hook = None
        if self._nodeenc_hook is not None:
            self._nodeenc_hook.remove()
            self._nodeenc_hook = None

    def _compute_spatial_mtp_loss(
        self, batch: Batch, target: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """フックしてプールしておいた特徴量を基に、空間的な穴埋め自己教師ありタスクのLossを計算する。"""
        if (not self.use_spatial_mtp) or (self.spatial_mtp is None):
            return None, {}

        H = self._penultimate_hidden
        edge_index = getattr(batch, "edge_index", None)
        if H is None or edge_index is None or target is None:
            return None, {}

        N = H.size(0)
        if N == 0:
            return None, {}

        B = min(N, self.spatial_mtp_centers_per_step)
        if B == 0:
            return None, {}

        # グラフ全体から評価対象の中心ノードをランダムサンプリングしてタスクを実行
        centers = torch.randperm(N, device=H.device)[:B]
        aux_loss, stats = self.spatial_mtp(
            H=H, edge_index=edge_index, centers=centers,
            out_head=self.output_head, target=target, H_neigh=self._H_nodeenc,
        )
        return aux_loss, stats

    def _training_step_single(self, batch: Batch):
        """1ミニバッチ（グラフ）に対する訓練演算ステップ（主損失＋補助物理損失の計算とログ）。"""
        batch = batch.to(self.device, non_blocking=True)
        
        # 空間MTPのバッファを各ステップの開始時にクリア
        if self.use_spatial_mtp:
            self._penultimate_hidden = None
            self._H_nodeenc = None
            
        node_type = batch.x[:, self.model.node_type_index]
        
        # シミュレータの順伝播を実行
        network_output, target_delta_normalized, _ = self.model(batch)

        # 損失（Loss）の計算分岐
        if self.is_multiloss:
            # 物理制約ブレンドロスの世界線：物理場での勾配や発散を解くために、実物理スケールに出力を復元
            network_output_physical = self.model.build_outputs(batch, network_output)
            target_physical = self.model.build_outputs(batch, target_delta_normalized)
            
            loss, train_losses = self.loss(
                graph=batch, target=target_delta_normalized, network_output=network_output,
                node_type=node_type, masks=self.loss_masks,
                network_output_physical=network_output_physical, target_physical=target_physical,
                gradient_method=self.gradient_method, return_all_losses=True,
            )
            # 各物理ロス（質量保存誤差、滑らかさ誤差など）を個別にログ記録
            for train_loss, loss_name in zip(train_losses, self.loss_name):
                self.log(f"train_{loss_name}", train_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_multiloss", loss, on_step=True, on_epoch=True, prog_bar=True)

        else:
            # 標準的な単一データのL2誤差の世界線
            loss = self.loss(
                graph=batch, target=target_delta_normalized, network_output=network_output,
                node_type=node_type, masks=self.loss_masks, gradient_method=self.gradient_method,
            )
            self.log(f"train_{self.loss_name}", loss, on_step=True, on_epoch=True, prog_bar=True)
            
        # 空間MTP自己教師あり補助損失の計算とメインLossへのブレンド
        if self.use_spatial_mtp and self.training:
            aux_loss, stats = self._compute_spatial_mtp_loss(batch, target_delta_normalized)
            if aux_loss is not None:
                loss = loss + self.spatial_mtp_alpha * aux_loss # 補助ロスを加算結合
                log_stats = {"sp_mtp/aux_loss": aux_loss.detach()}
                for key, value in stats.items():
                    log_stats[key] = value.detach()
                self.log_dict(log_stats, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # 各ステップ終了時にフックバッファをクリアしてメモリを解放
        if self.use_spatial_mtp:
            self._penultimate_hidden = None
            self._H_nodeenc = None

        predicted_outputs = self.model.build_outputs(batch, network_output)
        return loss, predicted_outputs

    def _scheduled_sampling_probability(self) -> float:
        if not self.scheduled_sampling_enabled:
            return 0.0
        start = self.scheduled_sampling_start_epoch
        end = self.scheduled_sampling_end_epoch
        if self.current_epoch <= start:
            return self.scheduled_sampling_start
        if self.current_epoch >= end or end == start:
            return self.scheduled_sampling_end
        progress = (self.current_epoch - start) / (end - start)
        return self.scheduled_sampling_start + progress * (self.scheduled_sampling_end - self.scheduled_sampling_start)

    def training_step(self, batch):
        if not self.multi_step_enabled:
            loss, _ = self._training_step_single(batch)
            return loss
        if not isinstance(batch, (list, tuple)) or len(batch) != self.multi_step_steps:
            raise ValueError(
                f"multi-step学習は{self.multi_step_steps}個の連続Batchを必要とします。"
            )

        sampling_probability = self._scheduled_sampling_probability()
        previous_prediction = None
        step_losses = []
        for step, step_batch in enumerate(batch):
            step_batch = step_batch.to(self.device, non_blocking=True)
            use_prediction = (
                step > 0
                and previous_prediction is not None
                and torch.rand((), device=self.device) < sampling_probability
            )
            if use_prediction:
                step_batch = step_batch.clone()
                start = self.model.output_index_start
                end = self.model.output_index_end
                step_batch.x[:, start:end] = previous_prediction

            step_loss, predicted_outputs = self._training_step_single(step_batch)
            boundary_mask = build_mask(self.param, step_batch).unsqueeze(-1)
            previous_prediction = torch.where(
                boundary_mask, step_batch.y, predicted_outputs
            )
            step_losses.append(step_loss)

        if len(step_losses) == 1:
            loss = step_losses[0]
        else:
            loss = step_losses[0] + self.multi_step_weight * torch.stack(step_losses[1:]).mean()
        self.log("train_multi_step_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("scheduled_sampling_probability", sampling_probability, on_step=False, on_epoch=True)
        return loss

    def teardown(self, stage: Optional[str] = None) -> None:
        """モジュールの実行終了時に呼び出され、フック回路を完全に破壊する"""
        self._remove_spatial_mtp_hooks()
        super().teardown(stage)

    def _save_batch_to_xdmf(
        self, batch: Batch, save_dir: str, archive_filename: str, timestep: float = 1
    ):
        """現在のグラフ（バッチ状態）を、ParaView対応の時系列 XDMF/H5 アーカイブファイルとして保存・追記する。"""
        os.makedirs(save_dir, exist_ok=True)
        archive_path = os.path.join(save_dir, archive_filename)

        # meshio形式のVTU格子オブジェクトへ変換
        mesh = convert_to_meshio_vtu(batch, add_all_data=True) 
        
        # ファイルが未存在なら新規作成、存在していれば後ろの時間フレームとして追記（Append）する
        if not os.path.exists(f"{archive_path}.h5") or not os.path.exists(f"{archive_path}.xdmf"):
            meshes_to_xdmf(filename=archive_path, meshes=[mesh], timestep=timestep)
        else:
            append_mesh_to_xdmf(
                filename=archive_path, mesh=mesh, timestep=timestep, compress=self.compress_predictions,
            )

    def _batch_with_prediction(
        self, batch: Batch, predicted_outputs: torch.Tensor
    ) -> Batch:
        """XDMF出力用に、入力場を今回の予測結果へ更新したグラフを返す。"""
        output_batch = batch.clone()
        start = self.model.output_index_start
        end = self.model.output_index_end
        if predicted_outputs.shape != output_batch.x[:, start:end].shape:
            raise ValueError(
                "保存する推論値と出力フィールドの形状が一致しません: "
                f"prediction={tuple(predicted_outputs.shape)}, "
                f"field={tuple(output_batch.x[:, start:end].shape)}"
            )
        output_batch.x[:, start:end] = predicted_outputs.detach()
        return output_batch

    def _reset_validation_trajectory(self):
        """新しいデータ系列（Trajectory）に移行した際、時間軸や前のステップの予測キャッシュを初期化する"""
        self.current_val_trajectory += 1
        self.last_val_prediction = None
        self.last_previous_data_prediction = None

    def _make_prediction(self, batch, last_prediction, last_previous_data_prediction):
        """時系列を自律的に進めるための予測一括オペレーション（自律回帰ロールアウトのコア）。"""
        batch = batch.clone()
        
        # 2ステップ目以降：前ステップでAI自身が弾き出した予測値を次の入力場へ上書きマッピング
        if last_prediction is not None:
            batch.x[:, self.model.output_index_start : self.model.output_index_end] = (
                last_prediction.detach()
            )
            # 過去データ履歴のフィードバックが有効な場合、差分（微分値）を特定領域へフィードバック上書き
            if self.use_previous_data:
                batch.x[:, self.previous_data_start : self.previous_data_end] = (
                    last_previous_data_prediction.detach()
                )
                
        # 境界条件ノード（壁面、流入等）の位置を指し示すマスクをビルド
        mask = build_mask(self.param, batch)
        target = batch.y # 次の時刻の正解データ
        current_output = batch.x[:, self.model.output_index_start : self.model.output_index_end]

        # 評価モードで1ステップ先をシミュレーション推論
        with torch.no_grad():
            _, _, predicted_outputs = self.model(inputs=batch)

        # 誤差累積の防波堤：AIモデルが境界領域に対して出力した予測値を、正解の固定条件値（流速0等）で強制上書きリセット
        predicted_outputs[mask] = target[mask]
        
        # キャッシュの更新
        last_prediction = predicted_outputs
        if self.use_previous_data:
            last_previous_data_prediction = predicted_outputs - current_output

        return (
            batch, predicted_outputs, target,
            last_prediction, last_previous_data_prediction,
        )

    def validation_step(self, batch: Batch, batch_idx: int):
        """時系列を最後まで自律回帰（Rollout）させて長期的な物理場の予測RMSE誤差を測る検証ステップ。"""
        batch = batch.to(self.device, non_blocking=True)
        
        # 現在処理しているサンプルが、新しいTrajectory（ケース）に突入したかを検知してリセット
        if batch.traj_index > self.current_val_trajectory:
            self._reset_validation_trajectory()
            self.step_counter = 0

        # 1ステップ先の予測と自律回帰キャッシュの更新
        (
            batch, predicted_outputs, target,
            self.last_val_prediction, self.last_previous_data_prediction,
        ) = self._make_prediction(
            batch, self.last_val_prediction, self.last_previous_data_prediction
        )

        node_type = batch.x[:, self.model.node_type_index]

        # エポック終了時の全累積RMSE算出のために予測とターゲットをプール
        self.val_step_outputs.append(predicted_outputs.cpu())
        self.val_step_targets.append(target.cpu())
        
        # 現在のステップ単体でのValidation Lossの計算とログ
        val_loss = self.val_loss(target, predicted_outputs, node_type, masks=self.loss_masks)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

        # シミュレーションの最初のフレーム（t=0➔t=1）単体での精度（1step_rmse）を追跡記録
        if self.step_counter == 0:
            squared_diff = (predicted_outputs - target) ** 2
            rmse = torch.sqrt(squared_diff.mean()).detach().cpu()
            self.first_step_losses.append(rmse)
        self.step_counter += 1

    def _reset_validation_epoch_end(self):
        """検証エポックの終了時に、次のエポックのために全ての累積バッファを完全に初期化する"""
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None
        self.step_counter = 0
        self.first_step_losses = []

    def on_validation_epoch_end(self):
        """検証エポックの終了時に、プールされた時系列全体（全ロールアウト）に対する統合累積RMSEを算出して記録する。"""
        predicteds = torch.cat(self.val_step_outputs, dim=0)
        targets = torch.cat(self.val_step_targets, dim=0)

        # 全シミュレーションの全ステップを通した累積の平均平方二乗誤差（RMSE）
        squared_diff = (predicteds - targets) ** 2
        all_rollout_rmse = torch.sqrt(squared_diff.mean()).item()

        self.log("val_all_rollout_rmse", all_rollout_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # 最初の1ステップ目だけの平均RMSE（初期微分の予測精度）をログ
        if self.first_step_losses:
            mean_first_step_loss = torch.stack(self.first_step_losses).mean().item()
            self.log("val_1step_rmse", mean_first_step_loss, on_epoch=True, prog_bar=True)

        self._reset_validation_epoch_end()

    def configure_optimizers(self):
        """最適化アルゴリズム（AdamW）および線形ウォームアップ付きコサイン学習率スケジューラの設定。"""
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95),
        )
        # 指定された総ステップ数（num_steps）とウォームアップ長に沿って動的に学習率を上下させる
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.num_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch, "monitor": "train_loss", "interval": "step", "frequency": 1,
            },
        }

    def _reset_prediction_trajectory(self):
        self.current_pred_trajectory += 1
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None
        self.predict_trajectory_step = 0
        self.predict_timing_step = 0

    def _resolve_predict_field_slices(self) -> List[tuple[str, int, int]]:
        """設定されたターゲット名とmeta情報から、出力テンソル上の範囲を求める。"""
        dataset_params = self.param.get("dataset", {})
        targets = dataset_params.get("targets", [])
        meta_path = dataset_params.get("meta_path")
        if not targets or not meta_path:
            return []

        try:
            with open(meta_path, "r", encoding="utf-8") as meta_file:
                features = json.load(meta_file)["features"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(f"物理量別RMSE用のmeta情報を読み込めませんでした: {exc}")
            return []

        field_slices = []
        start = 0
        for target_name in targets:
            try:
                width = int(features[target_name]["shape"][-1])
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                logger.warning(f"物理量 '{target_name}' の次元を解決できませんでした: {exc}")
                return []
            field_slices.append((str(target_name), start, start + width))
            start += width
        return field_slices

    def _update_predict_error_statistics(
        self, predicted_outputs: torch.Tensor, target: torch.Tensor
    ) -> None:
        """推論1ステップ分の誤差を蓄積する。"""
        prediction = predicted_outputs.detach().float()
        target = target.detach().float()

        if prediction.shape != target.shape:
            raise ValueError(
                f"推論値と正解値の形状が一致しません: "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
            )

        error = prediction - target
        squared_error = error.square()
        absolute_error = error.abs()

        rmse = torch.sqrt(squared_error.mean())
        mae = absolute_error.mean()
        relative_l2 = torch.linalg.vector_norm(error) / torch.clamp(
            torch.linalg.vector_norm(target), min=1.0e-8
        )

        record = {
            "trajectory": int(self.current_pred_trajectory),
            "step": int(self.predict_trajectory_step),
            "time": float(self.predict_trajectory_step * self.timestep),
            "rmse": float(rmse.cpu()),
            "mae": float(mae.cpu()),
            "relative_l2": float(relative_l2.cpu()),
            "max_abs_error": float(absolute_error.max().cpu()),
        }

        resolved_fields = self.predict_field_slices
        if resolved_fields and resolved_fields[-1][2] != error.shape[-1]:
            logger.warning("metaの物理量総次元と推論出力次元が不一致のため、成分名を使用します。")
            resolved_fields = []
            self.predict_field_slices = []
        if not resolved_fields:
            resolved_fields = [(f"component_{idx}", idx, idx + 1) for idx in range(error.shape[-1])]

        trajectory = int(self.current_pred_trajectory)
        case_stats = self.predict_case_error_statistics.setdefault(
            trajectory, {"steps": 0, "squared_error": 0.0, "numel": 0}
        )
        case_stats["steps"] += 1
        case_stats["squared_error"] += float(squared_error.sum().cpu())
        case_stats["numel"] += error.numel()
        for field_name, start, end in resolved_fields:
            field_squared = squared_error[..., start:end]
            record[f"{field_name}_rmse"] = float(torch.sqrt(field_squared.mean()).cpu())
            squared_key = f"{field_name}_squared_error"
            numel_key = f"{field_name}_numel"
            case_stats[squared_key] = case_stats.get(squared_key, 0.0) + float(field_squared.sum().cpu())
            case_stats[numel_key] = case_stats.get(numel_key, 0) + field_squared.numel()

            # ベクトル物理量は軸ごとのRMSEも名前付きで集計する。
            axis_names = ("x", "y", "z")
            if end - start > 1:
                for axis_idx in range(min(end - start, len(axis_names))):
                    component_name = f"{field_name}_{axis_names[axis_idx]}"
                    component_squared = squared_error[..., start + axis_idx]
                    record[f"{component_name}_rmse"] = float(
                        torch.sqrt(component_squared.mean()).cpu()
                    )
                    squared_key = f"{component_name}_squared_error"
                    numel_key = f"{component_name}_numel"
                    case_stats[squared_key] = case_stats.get(
                        squared_key, 0.0
                    ) + float(component_squared.sum().cpu())
                    case_stats[numel_key] = (
                        case_stats.get(numel_key, 0) + component_squared.numel()
                    )

        for component_idx in range(error.shape[-1]):
            component_squared = squared_error[:, component_idx]
            component_absolute = absolute_error[:, component_idx]
            record[f"component_{component_idx}_rmse"] = float(
                torch.sqrt(component_squared.mean()).cpu()
            )
            record[f"component_{component_idx}_mae"] = float(
                component_absolute.mean().cpu()
            )

        self.predict_step_metrics.append(record)
        self.predict_global_squared_error += float(squared_error.sum().cpu())
        self.predict_global_absolute_error += float(absolute_error.sum().cpu())
        self.predict_global_target_squared += float(target.square().sum().cpu())
        self.predict_global_numel += error.numel()

        component_dims = tuple(range(error.ndim - 1))
        component_squared_sum = squared_error.sum(dim=component_dims).cpu()
        component_absolute_sum = absolute_error.sum(dim=component_dims).cpu()
        component_numel = torch.full(
            (error.shape[-1],),
            error.numel() // error.shape[-1],
            dtype=torch.long,
        )

        if self.predict_component_squared_error is None:
            self.predict_component_squared_error = component_squared_sum
            self.predict_component_absolute_error = component_absolute_sum
            self.predict_component_numel = component_numel
        else:
            self.predict_component_squared_error += component_squared_sum
            self.predict_component_absolute_error += component_absolute_sum
            self.predict_component_numel += component_numel

        self.predict_trajectory_step += 1

    def _save_predict_error_statistics(self) -> None:
        """推論誤差をCSVへ保存し、集計値をログとWandBへ出力する。"""
        if not self.predict_step_metrics or self.predict_global_numel == 0:
            logger.warning("推論誤差統計が空のため、集計をスキップします。")
            return

        os.makedirs(self.prediction_save_path, exist_ok=True)
        csv_path = os.path.join(
            self.prediction_save_path, "inference_error_statistics.csv"
        )
        fieldnames = list(self.predict_step_metrics[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.predict_step_metrics)

        case_records = []
        for trajectory, case_stats in sorted(self.predict_case_error_statistics.items()):
            case_record = {
                "trajectory": trajectory,
                "steps": int(case_stats["steps"]),
                "overall_rmse": (case_stats["squared_error"] / case_stats["numel"]) ** 0.5,
            }
            field_slices = self.predict_field_slices or [
                (f"component_{idx}", idx, idx + 1)
                for idx in range(len(self.predict_component_squared_error))
            ]
            for field_name, _, _ in field_slices:
                count = int(case_stats[f"{field_name}_numel"])
                case_record[f"{field_name}_rmse"] = (
                    case_stats[f"{field_name}_squared_error"] / count
                ) ** 0.5
            case_records.append(case_record)
            axis_names = ("x", "y", "z")
            for field_name, start, end in field_slices:
                if end - start <= 1:
                    continue
                for axis_idx in range(min(end - start, len(axis_names))):
                    component_name = f"{field_name}_{axis_names[axis_idx]}"
                    count = int(case_stats[f"{component_name}_numel"])
                    case_record[f"{component_name}_rmse"] = (
                        case_stats[f"{component_name}_squared_error"] / count
                    ) ** 0.5


        case_csv_path = os.path.join(self.prediction_save_path, "inference_case_rmse.csv")
        with open(case_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(case_records[0].keys()))
            writer.writeheader()
            writer.writerows(case_records)

        for case_record in case_records:
            rmse_text = ", ".join(
                f"{name}={value:.6e}" for name, value in case_record.items()
                if name.endswith("_rmse")
            )
            logger.info(f"case {case_record['trajectory']} RMSE: {rmse_text}")

        global_rmse = (
            self.predict_global_squared_error / self.predict_global_numel
        ) ** 0.5
        global_mae = self.predict_global_absolute_error / self.predict_global_numel
        global_relative_l2 = (
            self.predict_global_squared_error
            / max(self.predict_global_target_squared, 1.0e-16)
        ) ** 0.5

        rmse_values = torch.tensor(
            [record["rmse"] for record in self.predict_step_metrics],
            dtype=torch.float64,
        )
        summary = {
            "predict_global_rmse": global_rmse,
            "predict_global_mae": global_mae,
            "predict_global_relative_l2": global_relative_l2,
            "predict_step_rmse_mean": float(rmse_values.mean()),
            "predict_step_rmse_std": float(rmse_values.std(unbiased=False)),
            "predict_step_rmse_median": float(rmse_values.median()),
            "predict_step_rmse_max": float(rmse_values.max()),
        }

        for component_idx in range(len(self.predict_component_squared_error)):
            count = int(self.predict_component_numel[component_idx])
            summary[f"predict_component_{component_idx}_rmse"] = (
                float(self.predict_component_squared_error[component_idx]) / count
            ) ** 0.5
            summary[f"predict_component_{component_idx}_mae"] = (
                float(self.predict_component_absolute_error[component_idx]) / count
            )

        for name, value in summary.items():
            logger.info(f"{name}: {value:.6e}")

        # predictループではself.log()がサポートされていないため、
        # Lightningのロガーを介してWandBへ集計値を直接記録する。
        if self.logger is not None:
            self.logger.log_metrics(summary, step=self.global_step)

        logger.success(f"推論誤差統計を保存しました: {csv_path}")
        logger.success(f"ケース別・物理量別RMSEを保存しました: {case_csv_path}")

    def _save_predict_timing_statistics(self) -> None:
        """予測処理のみの実測時間（誤差計算・ファイル保存を除く）を保存する。"""
        if not self.predict_step_timings:
            return

        os.makedirs(self.prediction_save_path, exist_ok=True)
        timing_path = os.path.join(self.prediction_save_path, "inference_prediction_times.csv")
        with open(timing_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(self.predict_step_timings[0]))
            writer.writeheader()
            writer.writerows(self.predict_step_timings)

        case_times = {}
        for record in self.predict_step_timings:
            case_times.setdefault(record["trajectory"], []).append(record["prediction_time_seconds"])
        case_records = []
        for trajectory, times in sorted(case_times.items()):
            total = sum(times)
            case_records.append({
                "trajectory": trajectory,
                "steps": len(times),
                "prediction_time_seconds": total,
                "mean_prediction_time_seconds": total / len(times),
            })
            logger.info(
                f"case {trajectory} prediction time: total={total:.6f} s, "
                f"mean={total / len(times):.6f} s/step ({len(times)} steps)"
            )
        case_path = os.path.join(self.prediction_save_path, "inference_case_prediction_times.csv")
        with open(case_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(case_records[0]))
            writer.writeheader()
            writer.writerows(case_records)

        times = [record["prediction_time_seconds"] for record in self.predict_step_timings]
        summary = {
            "predict_total_prediction_time_seconds": sum(times),
            "predict_mean_prediction_time_seconds": sum(times) / len(times),
            "predict_min_prediction_time_seconds": min(times),
            "predict_max_prediction_time_seconds": max(times),
        }
        for name, value in summary.items():
            logger.info(f"{name}: {value:.6f}")
        if self.logger is not None:
            self.logger.log_metrics(summary, step=self.global_step)
        logger.success(f"ステップ別予測時間を保存しました: {timing_path}")
        logger.success(f"ケース別予測時間を保存しました: {case_path}")

    def on_predict_start(self) -> None:
        """モデル配置後にGPUピーク統計をリセットし、推論中の最大使用量を計測する。"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def _save_predict_memory_statistics(self) -> None:
        """推論中のPyTorch管理VRAMと、プロセス起動以降の最大RSSをMiBで保存する。"""
        summary = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            summary.update({
                "predict_peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
                "predict_peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
            })

        try:
            import resource
        except ImportError:
            logger.warning("この環境ではCPUの最大常駐メモリを取得できません。")
        else:
            # Linuxのru_maxrssはKiB、macOSはbytes。子プロセスは含まない。
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary["predict_process_peak_rss_mib"] = peak_rss / (
                1024 ** 2 if sys.platform == "darwin" else 1024
            )

        if not summary:
            return

        os.makedirs(self.prediction_save_path, exist_ok=True)
        memory_path = os.path.join(self.prediction_save_path, "inference_memory_statistics.csv")
        with open(memory_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

        for name, value in summary.items():
            logger.info(f"{name}: {value:.3f} MiB")
        if self.logger is not None:
            self.logger.log_metrics(summary, step=self.global_step)
        logger.success(f"最大メモリ使用量を保存しました: {memory_path}")

    def predict_step(self, batch: Batch):
        """
        純粋な推論実行（予測）ステップ。
        検証時と異なり、正解データが無い未来（あるいは評価用データ）に対して
        完全に自律回帰（Rollout）でシミュレーションを実行し、結果をファイル保存していく。
        """
        batch = batch.to(self.device, non_blocking=True)
        if batch.traj_index > self.current_pred_trajectory:
            self._reset_prediction_trajectory()

        # CUDAの非同期実行を同期し、予測処理のみの経過時間を計測する。
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prediction_started = perf_counter()

        # 予測の実行とキャッシュの自動更新
        (
            batch, predicted_outputs, target,
            self.last_pred_prediction, self.last_previous_data_pred_prediction,
        ) = self._make_prediction(
            batch, self.last_pred_prediction, self.last_previous_data_pred_prediction
        )
        
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prediction_seconds = perf_counter() - prediction_started
        self.predict_step_timings.append({
            "trajectory": int(self.current_pred_trajectory),
            "step": self.predict_timing_step,
            "prediction_time_seconds": prediction_seconds,
        })
        self.predict_timing_step += 1

        # 正解データが含まれる評価用推論では、誤差統計を逐次蓄積
        if target is not None:
            self._update_predict_error_statistics(predicted_outputs, target)

        # 予測した流体格子状態を、指定されたフォルダ（prediction_save_path）へXDMF追記保存
        self._save_batch_to_xdmf(
            self._batch_with_prediction(batch, predicted_outputs),
            self.prediction_save_path,
            self._get_frame_savename(batch, self.current_pred_trajectory),
            timestep=self.timestep,
        )
        # predict API expects a return value; return predicted outputs so trainer.predict() can collect them
        return predicted_outputs

    def _reset_predict_epoch_end(self):
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None
        self.current_pred_trajectory = 0
        self.predict_trajectory_step = 0
        self.predict_timing_step = 0
        self.predict_step_metrics.clear()
        self.predict_step_timings.clear()
        self.predict_global_squared_error = 0.0
        self.predict_global_absolute_error = 0.0
        self.predict_global_target_squared = 0.0
        self.predict_global_numel = 0
        self.predict_component_squared_error = None
        self.predict_component_absolute_error = None
        self.predict_component_numel = None
        self.predict_case_error_statistics.clear()

    def on_predict_epoch_end(self):
        self._save_predict_memory_statistics()
        self._save_predict_timing_statistics()
        self._save_predict_error_statistics()
        self._reset_predict_epoch_end()

    def on_save_checkpoint(self, checkpoint: dict):
        """モデルをファイル（.ckpt）に保存する際、実験追跡の一貫性を保つため WandB の実行IDも内部にパックする。"""
        if self.wandb_run_id is not None:
            checkpoint["wandb_run_id"] = self.wandb_run_id
        else:
            logger.warning("WandB の実行ID（run_id）が見つからないため、チェックポイントへのパッキングをスキップします。")

    def on_load_checkpoint(self, checkpoint):
        """チェックポイントのロード時に、パックされていた WandB の実行IDを自動復元する。"""
        self.wandb_run_id = checkpoint.get("wandb_run_id", None)

    def _get_frame_savename(
        self, batch: Batch, traj_idx: int, prefix: str = "graph"
    ) -> str:
        """メッシュデータをディスク保存する際のファイル名を決定する（固有IDがあればそれを使用、無ければ連番インデックス）。"""
        if hasattr(batch, "id") and batch.id[0] is not None:
            return f"{prefix}_{batch.id[0]}"
        else:
            return f"{prefix}_{traj_idx}"
