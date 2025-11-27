import json
import os
import tempfile
import warnings
import shutil

import torch
import wandb
from absl import app, flags
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

from graphphysics.external.aneurysm import build_features
from graphphysics.training.callback import LogPyVistaPredictionsCallback
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import (
    get_dataset,
    get_num_workers,
    get_preprocessing,
)
from graphphysics.utils.progressbar import ColabProgressBar

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

# If the launcher sets `TMPDIR`, make Python's tempfile use it so that
# torch's multiprocessing shared-storage files are created in the same
# workspace-local directory instead of `/tmp`.
env_tmp_initial = os.environ.get("TMPDIR")
if env_tmp_initial:
    try:
        os.makedirs(env_tmp_initial, exist_ok=True)
        import tempfile as _tempfile

        _tempfile.tempdir = env_tmp_initial
    except Exception:
        pass

# Prefer 'fork' start method at import time where possible to avoid the
# heavier 'spawn' behavior that serializes callables and can generate many
# temporary files under `/tmp` during pickling. Use try/except to avoid
# raising if the start method is already set by the environment.
try:
    torch.multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate")
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("warmup", 1000, "Learning rate warmup steps")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_string(
    "model_save_name", None, "Name to save the checkpoint during training"
)
flags.DEFINE_string(
    "model_path", None, "Path to the checkpoint (.ckpt) to resume training from"
)
flags.DEFINE_bool(
    "resume_training", False, "Whether to resume an unfinished training or not"
)

flags.DEFINE_bool("use_previous_data", True, "Whether to use previous data or not")
flags.DEFINE_integer(
    "previous_data_start", 4, "Index of the start of the previous data in the features"
)
flags.DEFINE_integer(
    "previous_data_end", 7, "Index of the end of the previous data in the features"
)
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_string(
    "training_parameters_path", None, "Path to the training parameters JSON file"
)


def _init_wandb(project_name: str, resume: bool = False, run_id: str = None):
    try:
        if resume and run_id:
            run = wandb.init(project=project_name, id=run_id, resume="allow")
        else:
            run = wandb.init(project=project_name)
        return run
    except Exception as e:
        logger.warning(f"WandB initialization failed: {e}")
        return None


def main(argv):
    del argv

    # Check that the training parameters path is provided
    if not FLAGS.training_parameters_path:
        raise ValueError("The 'training_parameters_path' flag must be provided.")

    # Load training parameters from JSON file
    training_parameters_path = FLAGS.training_parameters_path
    logger.info(f"Opening training parameters from {training_parameters_path}")
    try:
        with open(training_parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    wandb_project_name = FLAGS.project_name
    num_epochs = FLAGS.num_epochs
    initial_lr = FLAGS.init_lr
    batch_size = FLAGS.batch_size
    warmup = FLAGS.warmup
    requested_num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_save_name = FLAGS.model_save_name
    model_path = FLAGS.model_path
    resume_training = FLAGS.resume_training
    use_edge_feature = not FLAGS.no_edge_feature
    use_previous_data = FLAGS.use_previous_data
    previous_data_start = FLAGS.previous_data_start
    previous_data_end = FLAGS.previous_data_end

    seed_everything(FLAGS.seed, workers=True)

    # Build preprocessing function
    train_preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        extra_node_features=build_features,
    )

    # Get training and validation datasets
    train_dataset = get_dataset(
        param=parameters,
        preprocessing=train_preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
    )

    val_preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        remove_noise=True,
        extra_node_features=build_features,
    )

    val_dataset = get_dataset(
        param=parameters,
        preprocessing=val_preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
        switch_to_val=True,
    )

    # choose num_workers based on dataset and environment
    num_workers = get_num_workers(param=parameters, default_num_workers=requested_num_workers)
    # If CUDA is not available, avoid many workers to reduce overhead
    if not torch.cuda.is_available():
        num_workers = min(num_workers, 2)

    train_dataloader_kwargs = {
        "dataset": train_dataset,
        "shuffle": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "exclude_keys": ["tetra"],
    }

    valid_dataloader_kwargs = {
        "dataset": val_dataset,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": num_workers,
    }

    if getattr(train_dataset, "type", None) == "h5":
        pin_memory = device.type == "cuda"
        train_dataloader_kwargs["pin_memory"] = pin_memory
        valid_dataloader_kwargs["pin_memory"] = pin_memory

    # Allow overriding DataLoader runtime options via environment for debugging
    # FORCE_PIN_MEMORY=0 or 1, FORCE_PERSISTENT_WORKERS=0 or 1, FORCE_PREFETCH_FACTOR=int
    env_pin = os.environ.get("FORCE_PIN_MEMORY")
    if env_pin is not None:
        try:
            val = int(env_pin)
            train_dataloader_kwargs["pin_memory"] = bool(val)
            valid_dataloader_kwargs["pin_memory"] = bool(val)
            logger.info(f"Overriding pin_memory via FORCE_PIN_MEMORY={val}")
        except Exception:
            logger.warning("Invalid FORCE_PIN_MEMORY value; ignoring")


    # Update arguments if num_workers > 0
    if num_workers > 0:
        train_dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )
        valid_dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )

    # Optionally override persistent_workers / prefetch via env for debugging
    env_pers = os.environ.get("FORCE_PERSISTENT_WORKERS")
    if env_pers is not None:
        try:
            v = int(env_pers)
            train_dataloader_kwargs["persistent_workers"] = bool(v)
            valid_dataloader_kwargs["persistent_workers"] = bool(v)
            logger.info(f"Overriding persistent_workers via FORCE_PERSISTENT_WORKERS={v}")
        except Exception:
            logger.warning("Invalid FORCE_PERSISTENT_WORKERS value; ignoring")

    env_pref = os.environ.get("FORCE_PREFETCH_FACTOR")
    if env_pref is not None:
        try:
            pf = int(env_pref)
            train_dataloader_kwargs["prefetch_factor"] = pf
            valid_dataloader_kwargs["prefetch_factor"] = pf
            logger.info(f"Overriding prefetch_factor via FORCE_PREFETCH_FACTOR={pf}")
        except Exception:
            logger.warning("Invalid FORCE_PREFETCH_FACTOR value; ignoring")

    # Create DataLoaders
    train_dataloader = DataLoader(**train_dataloader_kwargs)
    valid_dataloader = DataLoader(**valid_dataloader_kwargs)

    # Define or resume model
    num_steps = max(1, num_epochs * len(train_dataloader))

    prev_data_kwargs = {}
    if use_previous_data is True:
        prev_data_kwargs = {
            "use_previous_data": True,
            "previous_data_start": previous_data_start,
            "previous_data_end": previous_data_end,
        }

    lightning_module = None
    # load from checkpoint if provided
    if model_path and os.path.isfile(model_path):
        try:
            logger.info(f"Loading model from checkpoint: {model_path}")
            lightning_module = LightningModule.load_from_checkpoint(
                checkpoint_path=model_path,
                parameters=parameters,
                warmup=warmup,
                learning_rate=initial_lr,
                num_steps=num_steps,
                trajectory_length=getattr(train_dataset, "trajectory_length", 599),
                timestep=getattr(train_dataset, "dt", 1.0),
                **prev_data_kwargs,
            )
            logger.info(f"Resumed WandB run: {lightning_module.wandb_run_id}")
        except Exception as e:
            logger.warning(f"Failed to load LightningModule via classmethod: {e}. Falling back to fresh init and permissive restore.")
            # instantiate fresh and try permissive state load below
            lightning_module = LightningModule(
                parameters=parameters,
                learning_rate=initial_lr,
                num_steps=num_steps,
                warmup=warmup,
                trajectory_length=getattr(train_dataset, "trajectory_length", 599),
                timestep=getattr(train_dataset, "dt", 1.0),
                **prev_data_kwargs,
            )
            try:
                ckpt = torch.load(model_path, map_location=device)
                state = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
                if isinstance(state, dict):
                    # try to strip common prefixes
                    cleaned = {}
                    for k, v in state.items():
                        new_k = k
                        if k.startswith("model.") and not k.startswith("model.model."):
                            new_k = k[len("model.") :]
                        cleaned[new_k] = v
                    lightning_module.load_state_dict(cleaned, strict=False)
                else:
                    lightning_module.load_state_dict(state, strict=False)
                logger.info("Loaded checkpoint state into new LightningModule (strict=False).")
            except Exception as e2:
                logger.error(f"Failed to restore model state from checkpoint: {e2}")
    else:
        logger.info("Initializing new model")
        lightning_module = LightningModule(
            parameters=parameters,
            learning_rate=initial_lr,
            num_steps=num_steps,
            warmup=warmup,
            trajectory_length=getattr(train_dataset, "trajectory_length", 599),
            timestep=getattr(train_dataset, "dt", 1.0),
            **prev_data_kwargs,
        )

    # Initialize WandbLogger
    wandb_logger = None
    if resume_training and getattr(lightning_module, "wandb_run_id", None):
        run = _init_wandb(wandb_project_name, resume=True, run_id=lightning_module.wandb_run_id)
    else:
        run = _init_wandb(wandb_project_name, resume=False)

    if run is not None:
        wandb_logger = WandbLogger(experiment=run)
        lightning_module.wandb_run_id = run.id
    else:
        logger.warning("Proceeding without WandB logger.")

    # Configure checkpoint callback
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    if model_save_name:
        filename = model_save_name
    else:
        filename = "checkpoint-{epoch:02d}-{val_loss:.4f}"
    # save_top_k=3 requires a monitored metric to determine the top-k models.
    # The training module logs "val_loss" (see training/lightning_module.py),
    # so monitor that and pick the minimum value.
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # log basic config to wandb if available
    try:
        if run is not None:
            run.config.update(
                {
                    "architecture": parameters["model"]["type"],
                    "#_layers": parameters["model"]["message_passing_num"],
                    "#_neurons": parameters["model"]["hidden_size"],
                    "#_hops": parameters["dataset"]["khop"],
                    "max_lr": initial_lr,
                    "batch_size": batch_size,
                }
            )
    except Exception as e:
        logger.debug(f"Failed to update WandB config: {e}")

    # Configure Trainer
    # Build callbacks list; make PyVista logging optional via env var to allow
    # safe runs in headless / multiprocessing-sensitive environments.
    callbacks_list = [ColabProgressBar(), checkpoint_callback, lr_monitor]

    # Allow disabling PyVista visualization callback with env var
    # e.g. DISABLE_PYVISTA_CALLBACK=1 python -m graphphysics.train ...
    if os.environ.get("DISABLE_PYVISTA_CALLBACK", "0") != "1":
        try:
            callbacks_list.insert(2, LogPyVistaPredictionsCallback(dataset=val_dataset, indices=[1, 2, 3]))
        except Exception:
            # guard: if construction fails, continue without the callback
            logger.warning("Failed to add LogPyVistaPredictionsCallback; continuing without it.")

    # Lightning requires `devices` to be an int > 0 for CPUAccelerator.
    # Set devices=1 for CPU and GPUs by default (single-device training).
    trainer_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "max_epochs": num_epochs,
        "logger": wandb_logger,
        "callbacks": callbacks_list,
        "log_every_n_steps": 100,
        "gradient_clip_val": 1.0,
    }

    trainer = Trainer(**trainer_kwargs)

    # Fit
    try:
        if model_path and os.path.isfile(model_path) and resume_training:
            logger.success("Resuming training")
            trainer.fit(
                model=lightning_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                ckpt_path=model_path,
            )
        else:
            logger.success("Starting training")
            trainer.fit(
                model=lightning_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
            )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # attempt to save state for debugging
        try:
            tmp_ckpt = os.path.join(ckpt_dir, "crash_debug.pth")
            torch.save({"state_dict": lightning_module.state_dict()}, tmp_ckpt)
            logger.info(f"Saved debug state to {tmp_ckpt}")
        except Exception:
            pass


if __name__ == "__main__":
    # Prefer 'fork' start method to avoid heavy pickling / temporary module
    # generation that can occur with 'spawn' when cloudpickle serializes
    # dynamically created callables for child processes. Use a safe try/except
    # because the start method may already be set by the environment.
    try:
        torch.multiprocessing.set_start_method("fork", force=True)
    except Exception:
        # If setting 'fork' fails (rare on Linux), fall back to whatever is
        # already configured. We intentionally avoid forcing 'spawn' here to
        # reduce generation of temp files under /tmp.
        pass
    app.run(main)