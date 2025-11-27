import os
import shutil
from typing import List, Optional, Tuple

import lightning as L
import meshio
import torch
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.training.parse_parameters import (
    get_gradient_method,
    get_loss,
    get_model,
    get_simulator,
)
from graphphysics.utils.loss import L2Loss, MultiLoss
from graphphysics.utils.meshio_mesh import convert_to_meshio_vtu
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.scheduler import CosineWarmupScheduler


def build_mask(param: dict, graph: Batch) -> torch.Tensor:
    """
    graph.x の node_type 列からマスクを作成するユーティリティ。
    mask は True の箇所が「マスク対象（観察不可）」を示すように返す。
    """
    # graph.x は (B,N,C) もしくは (N,C) の形で来る可能性があるため安全に取り出す
    if len(graph.x.shape) > 2:
        node_type = graph.x[:, 0, param["index"]["node_type_index"]]
    else:
        node_type = graph.x[:, param["index"]["node_type_index"]]
    mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
    mask = torch.logical_not(mask)
    return mask


class LightningModule(L.LightningModule):
    """
    PyTorch Lightning Module wrapper for graph-physics training pipeline.

    主な責務:
    - モデル / シミュレータの初期化
    - loss の計算およびログ
    - validation/predict 時の軌跡保存 (XDMF)
    - optimizer + CosineWarmupScheduler の設定
    """

    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
        trajectory_length: int = 599,
        timestep: float = 1.0,
        only_processor: bool = False,
        masks: List[NodeType] = [NodeType.NORMAL, NodeType.OUTFLOW],
        use_previous_data: bool = False,
        previous_data_start: Optional[int] = None,
        previous_data_end: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.param = parameters
        self.wandb_run_id: Optional[str] = None

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_str}")

        processor = get_model(param=parameters, only_processor=only_processor)
        self.model = get_simulator(param=parameters, model=processor, device=device_str)

        self.loss, self.loss_name = get_loss(param=parameters)
        logger.info(f"Using loss: {self.loss_name}")
        self.is_multiloss = isinstance(self.loss, MultiLoss)

        self.loss_masks = masks
        self.val_loss = L2Loss()
        self.gradient_method = get_gradient_method(param=parameters)

        self.learning_rate = float(learning_rate)
        self.num_steps = int(num_steps)
        self.warmup = int(warmup)

        # counters / buffers
        self.step_counter = 0
        self.first_step_losses: List[torch.Tensor] = []
        self.val_step_outputs: List[torch.Tensor] = []
        self.val_step_targets: List[torch.Tensor] = []
        self.trajectory_length = int(trajectory_length)
        self.timestep = float(timestep)
        self.current_val_trajectory = 0
        self.last_val_prediction: Optional[torch.Tensor] = None
        self.last_previous_data_prediction: Optional[torch.Tensor] = None

        # previous data handling
        self.use_previous_data = bool(parameters["model"].get("use_previous_data", use_previous_data))
        if not self.use_previous_data:
            self.previous_data_start = 0
            self.previous_data_end = 0
        else:
            if previous_data_start is None or previous_data_end is None:
                raise ValueError("use_previous_data=True の場合 previous_data_start/end を指定してください。")
            self.previous_data_start = int(previous_data_start)
            self.previous_data_end = int(previous_data_end)

        # visualization / prediction buffers
        self.trajectory_to_save: List[Batch] = []
        self.prediction_save_dir: str = "predictions"
        self.current_pred_trajectory = 0
        self.prediction_trajectory: List[Batch] = []
        self.last_pred_prediction: Optional[torch.Tensor] = None
        self.last_previous_data_pred_prediction: Optional[torch.Tensor] = None

    def forward(self, graph: Batch):
        return self.model(graph)

    def training_step(self, batch: Batch):
        batch = batch.to(self.device, non_blocking=True)
        node_type = batch.x[:, self.model.node_type_index]
        network_output, target_delta_normalized, _ = self.model(batch)

        if self.is_multiloss:
            network_output_physical = self.model.build_outputs(batch, network_output)
            target_physical = self.model.build_outputs(batch, target_delta_normalized)
            loss, train_losses = self.loss(
                graph=batch,
                target=target_delta_normalized,
                network_output=network_output,
                node_type=node_type,
                masks=self.loss_masks,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=self.gradient_method,
                return_all_losses=True,
            )
            for train_loss, loss_name in zip(train_losses, self.loss_name):
                self.log(f"train_loss - {loss_name}", train_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_multiloss", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.loss(
                graph=batch,
                target=target_delta_normalized,
                network_output=network_output,
                node_type=node_type,
                masks=self.loss_masks,
                gradient_method=self.gradient_method,
            )
            self.log(f"train_{self.loss_name}", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _save_trajectory_to_xdmf(
        self,
        trajectory: List[Batch],
        save_dir: str,
        archive_filename: str,
        timestep: float = 1.0,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        archive_path = os.path.join(save_dir, archive_filename)
        xdmf_filename = f"{archive_path}.xdmf"

        if not trajectory:
            logger.warning("No graphs to save for trajectory.")
            return

        init_mesh = convert_to_meshio_vtu(trajectory[0], add_all_data=True)
        points = init_mesh.points
        cells = init_mesh.cells

        try:
            with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
                writer.write_points_cells(points, cells)
                t = timestep if not self.use_previous_data else 2 * timestep
                for idx, graph in enumerate(trajectory):
                    mesh = convert_to_meshio_vtu(graph, add_all_data=True)
                    writer.write_data(t, point_data=mesh.point_data, cell_data=mesh.cell_data)
                    t += timestep
        except Exception as e:
            logger.error(f"Error saving trajectory mesh at epoch {getattr(self, 'current_epoch', -1)}: {e}")
            return

        # move generated h5 to same folder (meshio writes companion .h5 in cwd)
        try:
            shutil.move(
                src=os.path.join(os.getcwd(), os.path.basename(xdmf_filename).replace(".xdmf", ".h5")),
                dst=xdmf_filename.replace(".xdmf", ".h5"),
            )
        except Exception:
            # best-effort, ignore if move fails
            pass

        logger.info(f"Saved validation trajectory to {xdmf_filename}")

    def _reset_validation_trajectory(self) -> None:
        self.current_val_trajectory += 1
        self.last_val_prediction = None
        self.last_previous_data_prediction = None

    def _prev_width(self) -> int:
        return max(0, int(self.previous_data_end - self.previous_data_start))

    def _make_prediction(self, batch: Batch, last_prediction: Optional[torch.Tensor], last_previous_data_prediction: Optional[torch.Tensor]):
        batch = batch.clone()
        if last_prediction is not None:
            batch.x[:, self.model.output_index_start : self.model.output_index_end] = last_prediction.detach()
            if self.use_previous_data and self._prev_width() > 0 and last_previous_data_prediction is not None:
                w = self._prev_width()
                t = last_previous_data_prediction
                if t.dim() == 1:
                    t = t.unsqueeze(1)
                if t.shape[1] != w:
                    # Be defensive: some datasets / configs may expect previous-data
                    # width different from the model output width. Instead of
                    # raising (which aborts training), attempt a safe adjustment:
                    # - If t is wider than expected, trim extra columns.
                    # - If t is narrower, pad with zeros to match the expected width.
                    # Also emit a warning so the user can correct configuration if needed.
                    logger.warning(
                        f"previous feature width mismatch: want {w}, got {t.shape[1]}; "
                        "automatically trimming/padding to match expected width"
                    )
                    if t.shape[1] > w:
                        t = t[:, :w]
                    else:
                        pad = torch.zeros((t.shape[0], w - t.shape[1]), device=t.device, dtype=t.dtype)
                        t = torch.cat([t, pad], dim=1)
                batch.x[:, self.previous_data_start:self.previous_data_end] = t.detach()

        mask = build_mask(self.param, batch)
        target = batch.y
        current_output = batch.x[:, self.model.output_index_start : self.model.output_index_end]

        with torch.no_grad():
            _, _, predicted_outputs = self.model(batch)

        # enforce mask: keep ground-truth where mask True
        try:
            predicted_outputs[mask] = target[mask]
        except Exception:
            pass

        last_prediction = predicted_outputs
        if self.use_previous_data and self._prev_width() > 0:
            last_previous_data_prediction = predicted_outputs - current_output
        else:
            last_previous_data_prediction = None

        return batch, predicted_outputs, target, last_prediction, last_previous_data_prediction

    def validation_step(self, batch: Batch, batch_idx: int):
        batch = batch.to(self.device, non_blocking=True)

        if batch.traj_index > self.current_val_trajectory:
            self._reset_validation_trajectory()
            self.step_counter = 0

        (
            batch,
            predicted_outputs,
            target,
            self.last_val_prediction,
            self.last_previous_data_prediction,
        ) = self._make_prediction(batch, self.last_val_prediction, self.last_previous_data_prediction)

        if self.current_val_trajectory == 0:
            self.trajectory_to_save.append(batch)

        node_type = batch.x[:, self.model.node_type_index]
        self.val_step_outputs.append(predicted_outputs.cpu())
        self.val_step_targets.append(target.cpu())

        val_loss = self.val_loss(target, predicted_outputs, node_type, masks=self.loss_masks)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.step_counter == 0:
            rmse = torch.sqrt(((predicted_outputs - target) ** 2).mean()).detach().cpu()
            self.first_step_losses.append(rmse)
        self.step_counter += 1

    def _reset_validation_epoch_end(self) -> None:
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None
        self.trajectory_to_save.clear()
        self.step_counter = 0
        self.first_step_losses = []

    def on_validation_epoch_end(self) -> None:
        if not self.val_step_outputs or not self.val_step_targets:
            self._reset_validation_epoch_end()
            return

        predicteds = torch.cat(self.val_step_outputs, dim=0)
        targets = torch.cat(self.val_step_targets, dim=0)

        all_rollout_rmse = torch.sqrt(((predicteds - targets) ** 2).mean()).item()
        self.log("val_all_rollout_rmse", all_rollout_rmse, on_step=False, on_epoch=True, prog_bar=True)

        if self.first_step_losses:
            mean_first_step_loss = torch.stack(self.first_step_losses).mean().item()
            self.log("val_1step_rmse", mean_first_step_loss, on_epoch=True, prog_bar=True)

        save_dir = os.path.join("meshes", f"epoch_{self.current_epoch}")
        try:
            self._save_trajectory_to_xdmf(
                self.trajectory_to_save,
                save_dir,
                self._get_traj_savename(self.trajectory_to_save, self.current_val_trajectory, prefix=f"graph_epoch_{self.current_epoch}"),
                timestep=self.timestep,
            )
        except Exception as e:
            logger.error(f"Failed saving validation trajectory: {e}")

        self._reset_validation_epoch_end()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95))
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.num_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "train_loss", "interval": "step", "frequency": 1},
        }

    def _reset_prediction_trajectory(self) -> None:
        self.current_pred_trajectory += 1
        self.prediction_trajectory = []
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None

    def predict_step(self, batch: Batch):
        batch = batch.to(self.device, non_blocking=True)
        if batch.traj_index > self.current_pred_trajectory:
            # save current prediction trajectory then reset
            self._save_trajectory_to_xdmf(
                self.prediction_trajectory,
                self.prediction_save_dir,
                self._get_traj_savename(self.prediction_trajectory, self.current_pred_trajectory),
                timestep=self.timestep,
            )
            self._reset_prediction_trajectory()

        (
            batch,
            predicted_outputs,
            target,
            self.last_pred_prediction,
            self.last_previous_data_pred_prediction,
        ) = self._make_prediction(batch, self.last_pred_prediction, self.last_previous_data_pred_prediction)

        self.prediction_trajectory.append(batch)

    def _reset_predict_epoch_end(self) -> None:
        self.prediction_trajectory.clear()
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None
        self.current_pred_trajectory = 0

    def on_predict_epoch_end(self) -> None:
        if self.prediction_trajectory:
            self._save_trajectory_to_xdmf(
                self.prediction_trajectory,
                self.prediction_save_dir,
                self._get_traj_savename(self.prediction_trajectory, self.current_pred_trajectory),
                timestep=self.timestep,
            )
        self._reset_predict_epoch_end()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.wandb_run_id is not None:
            checkpoint["wandb_run_id"] = self.wandb_run_id
        else:
            logger.warning("No wandb run ID to save to checkpoint.")

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.wandb_run_id = checkpoint.get("wandb_run_id", None)

    def _get_traj_savename(self, traj: List[Batch], traj_idx: int, prefix: str = "graph") -> str:
        if not traj:
            return f"{prefix}_{traj_idx}"
        first = traj[0]
        if hasattr(first, "id") and getattr(first, "id"):
            try:
                tid = first.id[0]
                return f"{prefix}_{tid}"
            except Exception:
                pass
        return f"{prefix}_{traj_idx}"
