import json
import warnings
import os

import torch
from absl import app, flags
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

import wandb
from graphphysics.external.aneurysm import build_features
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import get_dataset, get_preprocessing

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "prediction_project", "Name of the WandB project")
flags.DEFINE_string("model_path", None, "Path to the checkpoint (.ckpt) file")
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_string("predict_parameters_path", None, "Path to the training parameters JSON file")
flags.DEFINE_string("prediction_save_path", "predictions", "Path to where the predictions will be saved")


def _load_lightning_module_from_ckpt(model_path: str, parameters: dict, dataset) -> LightningModule:
    """
    Try multiple strategies to obtain a LightningModule from a checkpoint:
      1) LightningModule.load_from_checkpoint with common args (preferred)
      2) Fallback: torch.load checkpoint and instantiate LightningModule with safe defaults,
         then load state_dict with strict=False.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_len = getattr(dataset, "trajectory_length", getattr(dataset, "traj_len", 599))
    dt = getattr(dataset, "dt", getattr(dataset, "timestep", 1.0))

    # Preferred: try classmethod load_from_checkpoint (keeps Lightning semantics)
    try:
        logger.info("Attempting LightningModule.load_from_checkpoint(...)")
        lm = LightningModule.load_from_checkpoint(
            checkpoint_path=model_path,
            parameters=parameters,
            trajectory_length=traj_len,
            timestep=dt,
        )
        logger.info("Loaded LightningModule via load_from_checkpoint")
        return lm
    except Exception as e:
        logger.warning(f"load_from_checkpoint failed: {e}. Falling back to manual restore.")

    # Fallback: manual restore
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict") or ckpt

    # Instantiate with conservative defaults required by LightningModule.__init__
    try:
        lm = LightningModule(
            parameters=parameters,
            learning_rate=float(parameters.get("training", {}).get("learning_rate", 1e-3)),
            num_steps=int(parameters.get("training", {}).get("num_steps", 100000)),
            warmup=int(parameters.get("training", {}).get("warmup", 1000)),
            trajectory_length=int(traj_len),
            timestep=float(dt),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate LightningModule for manual restore: {e}")

    # load state dict permissively
    try:
        if isinstance(state, dict):
            # strip possible "model." prefixes from keys
            cleaned = {}
            for k, v in state.items():
                new_k = k
                if k.startswith("model.") and not k.startswith("model.model."):
                    new_k = k[len("model.") :]
                cleaned[new_k] = v
            lm.load_state_dict(cleaned, strict=False)
        else:
            # state might already be a state_dict-like object
            lm.load_state_dict(state, strict=False)
        logger.info("Restored LightningModule state_dict (strict=False)")
    except Exception as e:
        logger.warning(f"Failed to load state_dict into LightningModule: {e}")

    return lm


def main(argv):
    del argv

    if not FLAGS.predict_parameters_path:
        raise ValueError("The 'predict_parameters_path' flag must be provided.")

    parameters_path = FLAGS.predict_parameters_path
    logger.info(f"Opening prediction parameters from {parameters_path}")
    try:
        with open(parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_project_name = FLAGS.project_name
    model_path = FLAGS.model_path
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"model_path must point to an existing checkpoint file. Given: {model_path}")

    use_edge_feature = not FLAGS.no_edge_feature

    # Build preprocessing function
    preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        remove_noise=True,
        extra_node_features=build_features,
    )

    # Get predict dataset
    predict_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=True,
    )

    predict_dataloader_kwargs = {
        "dataset": predict_dataset,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 0,
    }

    predict_dataloader = DataLoader(**predict_dataloader_kwargs)

    prediction_save_path = FLAGS.prediction_save_path
    logger.info(f"Loading model from checkpoint: {model_path}")
    lightning_module = _load_lightning_module_from_ckpt(model_path, parameters, predict_dataset)
    lightning_module.to(device)

    # Initialize WandbLogger (best-effort; not required)
    try:
        wandb_run = wandb.init(project=wandb_project_name)
        wandb_logger = WandbLogger(experiment=wandb_run)
    except Exception as e:
        logger.warning(f"WandB init failed: {e}")
        wandb_logger = None

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        devices=1 if torch.cuda.is_available() else None,
        enable_model_summary=False,
        inference_mode=True,
    )

    # Start prediction
    logger.success("Starting prediction")
    trainer.predict(model=lightning_module, dataloaders=predict_dataloader)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    app.run(main)