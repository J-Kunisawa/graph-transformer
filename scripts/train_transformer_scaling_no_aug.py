#!/usr/bin/env python3
"""Run 300-epoch Transformer training sequentially across mesh sizes.

This launcher snapshots one generated config per mesh size, disables every
adjacency/data edge augmentation switch, and stops the queue on the first
failed training run.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "training_config" / "xdmf_transformer.json"
NODE_COUNTS = (2000, 5000, 10000, 20000, 40000)
NUM_EPOCHS = 300


def build_config(base: dict, node_count: int) -> dict:
    config = copy.deepcopy(base)

    model = config["model"]
    if model.get("type") != "transformer":
        raise ValueError(f"Expected transformer config, got {model.get('type')!r}")

    augmentation = model.setdefault("adjacency_augmentation", {})
    augmentation["enabled"] = False
    augmentation["random_edges_ratio"] = 0.0
    augmentation.setdefault("dilation", {})["enabled"] = False
    augmentation.setdefault("global_attention", {})["enabled"] = False

    dataset = config["dataset"]
    dataset_root = f"data/data_xdmf/mesh_re100/n={node_count}"
    dataset["train_path"] = f"{dataset_root}/train"
    dataset["val_path"] = f"{dataset_root}/valid"
    dataset["test_path"] = f"{dataset_root}/test"
    dataset["new_edges_ratio"] = 0.0

    # The request is for a full 300 epochs at every resolution.
    config.setdefault("training", {}).setdefault("early_stopping", {})[
        "enabled"
    ] = False
    return config


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> int:
    with BASE_CONFIG.open(encoding="utf-8") as config_file:
        base_config = json.load(config_file)

    run_id = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    generated_dir = PROJECT_ROOT / "training_config" / "generated" / run_id
    log_dir = PROJECT_ROOT / "logs" / f"transformer_scaling_no_aug_{run_id}"
    generated_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": f"{PROJECT_ROOT}:{environment.get('PYTHONPATH', '')}",
        }
    )

    log(f"Queue started: nodes={NODE_COUNTS}, epochs={NUM_EPOCHS}, run_id={run_id}")
    for node_count in NODE_COUNTS:
        config = build_config(base_config, node_count)
        config_path = generated_dir / f"xdmf_transformer_n{node_count}_no_aug.json"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2)
            config_file.write("\n")

        for split, path_key in (
            ("train", "train_path"),
            ("valid", "val_path"),
            ("test", "test_path"),
        ):
            data_path = PROJECT_ROOT / config["dataset"][path_key]
            if not data_path.is_dir():
                raise FileNotFoundError(f"Missing {split} data: {data_path}")

        checkpoint_name = (
            f"mesh_re100/n={node_count}/"
            f"re_100_transformer_mp10_noaug_300e_{run_id}"
        )
        job_log_path = log_dir / f"n={node_count}.log"
        command = [
            "nice",
            "-n",
            "10",
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--project_name=mesh_scaling",
            f"--training_parameters_path={config_path}",
            f"--model_save_name={checkpoint_name}",
            "--resume_training=false",
            f"--num_epochs={NUM_EPOCHS}",
            "--seed=42",
            "--batch_size=4",
            "--init_lr=0.001",
            "--warmup_ratio=0.02",
            "--num_workers=8",
            "--prefetch_factor=2",
            "--use_previous_data=false",
            "--previous_data_start=3",
            "--previous_data_end=6",
            "--no_edge_feature=false",
            "--use_partitioning=false",
            "--gradient_batch_size=2",
        ]

        log(f"Starting n={node_count}; job log: {job_log_path}")
        with job_log_path.open("w", encoding="utf-8") as job_log:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=job_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            log(f"FAILED n={node_count} with exit code {result.returncode}; queue stopped")
            return result.returncode
        log(f"Completed n={node_count}")

    log("Queue completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
