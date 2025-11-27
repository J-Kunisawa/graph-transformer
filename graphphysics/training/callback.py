import os
from typing import List, Optional, Tuple

import numpy as np
try:
    import pyvista as pv
except Exception:
    pv = None
import torch
from loguru import logger
from lightning.pytorch.callbacks import Callback
from torch_geometric.data import Data, Dataset

import wandb
from graphphysics.training.lightning_module import build_mask
from graphphysics.utils.pyvista_mesh import convert_to_pyvista_mesh

# Headless safety defaults
os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]


def _is_headless() -> bool:
    """環境変数で明示的にオフになっているか DISPLAY が無ければ headless とみなす。"""
    if os.environ.get("DISABLE_VIZ", "0") == "1":
        return True
    if os.environ.get("PYVISTA_OFF_SCREEN", "0") == "1":
        return True
    if not os.environ.get("DISPLAY"):
        return True
    return False


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Ensure an image is uint8 HxWxC."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        # normalize to 0-255 if float in 0..1 or scale linearly otherwise
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


class LogPyVistaPredictionsCallback(Callback):
    """
    Validation epoch end で PyVista 図を作り wandb に送るコールバック。

    - dataset: DataLoader ではなく Dataset（index 指定でサンプル取得するため）
    - indices: ログ対象のグローバルインデックス一覧
    - output_dir: 一時的に画像を保存したい場合のディレクトリ
    """

    def __init__(self, dataset: Dataset, indices: List[int], output_dir: str = "predictions"):
        super().__init__()
        self.dataset = dataset
        self.indices = list(indices)
        self.output_dir = output_dir

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Headless 環境では何もしない（呼び出しが高頻度なので早期リターン）
        if _is_headless():
            logger.debug("Headless environment detected: skipping PyVista logging.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        device = pl_module.device if hasattr(pl_module, "device") else torch.device("cpu")

        images = []
        ground_truth = []

        try:
            with torch.no_grad():
                for idx in self.indices:
                    sample = self.dataset[idx]
                    # dataset が (graph, mask) を返す可能性に対応
                    if isinstance(sample, tuple) and len(sample) >= 1:
                        graph = sample[0]
                    else:
                        graph = sample

                    graph = graph.to(device)

                    # model(graph) -> (network_out, target_delta_norm, outputs)
                    try:
                        _, _, predicted_outputs = pl_module(graph)
                    except Exception:
                        # 互換性保険: pl_module.predict_step / forward の違いに対応
                        predicted_outputs = None
                        try:
                            forward_ret = pl_module.forward(graph)
                            if isinstance(forward_ret, tuple) and len(forward_ret) >= 3:
                                predicted_outputs = forward_ret[2]
                            elif isinstance(forward_ret, torch.Tensor):
                                predicted_outputs = forward_ret
                        except Exception as e:
                            logger.error(f"Failed to obtain prediction for idx={idx}: {e}")

                    if predicted_outputs is None:
                        logger.warning(f"No prediction produced for idx={idx}; skipping visualization for this sample.")
                        continue

                    # set graph.x to predictions for visualization
                    graph_vis = graph.clone()
                    graph_vis.x = predicted_outputs

                    mesh_pred = convert_to_pyvista_mesh(graph_vis)
                    img = self._generate_pyvista_image(mesh_pred)
                    img_arr = _to_uint8_image(img)
                    if img_arr is not None:
                        images.append(wandb.Image(img_arr, caption=f"Prediction idx={idx}"))

                    # ground truth
                    gt_graph = graph.clone()
                    if hasattr(gt_graph, "y") and gt_graph.y is not None:
                        gt_graph.x = gt_graph.y
                        mesh_gt = convert_to_pyvista_mesh(gt_graph)
                        img = self._generate_pyvista_image(mesh_gt)
                        img_arr = _to_uint8_image(img)
                        if img_arr is not None:
                            ground_truth.append(wandb.Image(img_arr, caption=f"GroundTruth idx={idx}"))

            # log single images
            wandb_logger = getattr(trainer, "logger", None)
            if wandb_logger and hasattr(wandb_logger, "experiment"):
                if images:
                    wandb_logger.experiment.log({"pyvista_predictions": images})
                if ground_truth:
                    wandb_logger.experiment.log({"pyvista_ground_truth": ground_truth})
        except Exception as e:
            logger.error(f"Error in LogPyVistaPredictionsCallback.on_validation_epoch_end: {e}")

        # If multiple indices form a temporal sequence, also create videos
        if len(self.indices) > 1:
            try:
                frames_predictions = []
                frames_ground_truth = []
                predicted_outputs = None

                with torch.no_grad():
                    # iterate through a contiguous range (conservative)
                    for idx in range(self.indices[0], self.indices[-1] + 1):
                        sample = self.dataset[idx]
                        if isinstance(sample, tuple) and len(sample) >= 1:
                            graph = sample[0]
                        else:
                            graph = sample
                        graph = graph.to(device)

                        # carry-over from previous prediction if model expects recurrent usage
                        if predicted_outputs is not None:
                            out_slice_start = pl_module.model.output_index_start if hasattr(pl_module, "model") else 0
                            out_slice_end = pl_module.model.output_index_end if hasattr(pl_module, "model") else predicted_outputs.shape[1]
                            try:
                                graph.x[:, out_slice_start:out_slice_end] = predicted_outputs.detach()
                            except Exception:
                                pass

                        mask = build_mask(pl_module.param if hasattr(pl_module, "param") else {}, graph)
                        _, _, predicted_outputs = pl_module(graph)
                        if predicted_outputs is None:
                            continue

                        # enforce masked ground-truth where mask True
                        try:
                            target = graph.y
                            predicted_outputs[mask] = target[mask]
                        except Exception:
                            pass

                        # prediction frame
                        gvis = graph.clone()
                        gvis.x = predicted_outputs
                        mesh_pred = convert_to_pyvista_mesh(gvis)
                        img = self._generate_pyvista_image(mesh_pred)
                        arr = _to_uint8_image(img)
                        if arr is not None:
                            frames_predictions.append(arr)

                        # ground-truth frame
                        gvis_gt = graph.clone()
                        if hasattr(gvis_gt, "y") and gvis_gt.y is not None:
                            gvis_gt.x = gvis_gt.y
                            mesh_gt = convert_to_pyvista_mesh(gvis_gt)
                            img = self._generate_pyvista_image(mesh_gt)
                            arr = _to_uint8_image(img)
                            if arr is not None:
                                frames_ground_truth.append(arr)

                if frames_predictions:
                    # frames: (T, H, W, C) -> WandB wants (T, C, H, W)
                    vid_pred = np.stack(frames_predictions, axis=0)
                    vid_gt = np.stack(frames_ground_truth, axis=0) if frames_ground_truth else None

                    vid_pred = np.transpose(vid_pred, (0, 3, 1, 2))
                    if vid_gt is not None:
                        vid_gt = np.transpose(vid_gt, (0, 3, 1, 2))

                    if wandb_logger and hasattr(wandb_logger, "experiment"):
                        wandb_logger.experiment.log({"pyvista_predictions_video": wandb.Video(vid_pred, fps=4)})
                        if vid_gt is not None:
                            wandb_logger.experiment.log({"pyvista_ground_truth_video": wandb.Video(vid_gt, fps=4)})
            except Exception as e:
                logger.error(f"Error while creating PyVista videos: {e}")

    def _generate_pyvista_image(self, mesh) -> Optional[np.ndarray]:
        """PyVista でスクリーンショットを取得して numpy 配列で返す。headless なら None."""
        if _is_headless():
            return None
        try:
            if pv is None:
                logger.debug("pyvista not available: skipping render")
                return None
            p = pv.Plotter(off_screen=True)
            p.add_mesh(mesh, scalars=next(iter(mesh.point_data.keys())) if mesh.point_data else None, opacity=0.8)
            p.add_legend()
            img = p.screenshot(return_img=True)
            p.close()
            return img
        except Exception as e:
            logger.error(f"PyVista render failed: {e}")
            try:
                p.close()
            except Exception:
                pass
            return None
