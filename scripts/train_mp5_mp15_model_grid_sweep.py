#!/usr/bin/env python3
"""Run the MP=5/15 GNN and Graph Transformer mesh-size sweep."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIGS = {
    "gnn": PROJECT_ROOT / "training_config" / "xdmf_gnn.json",
    "gt_noaug": PROJECT_ROOT / "training_config" / "xdmf_transformer.json",
    "gt_aug": PROJECT_ROOT / "training_config" / "xdmf_transformer.json",
}
MESSAGE_PASSING_COUNTS = (5, 15)
VARIANTS = ("gnn", "gt_noaug", "gt_aug")
NODE_COUNTS = (2000, 5000, 10000, 20000, 40000)
NUM_EPOCHS = 300
MAX_WANDB_ATTEMPTS = 3
WANDB_INIT_TIMEOUT_SECONDS = 300
WANDB_RETRY_DELAY_SECONDS = 30


def configure_gt_augmentation(model: dict, enabled: bool, depth: int) -> None:
    augmentation = model.setdefault("adjacency_augmentation", {})
    augmentation["enabled"] = enabled
    augmentation["random_edges_ratio"] = 0.2 if enabled else 0.0

    dilation = augmentation.setdefault("dilation", {})
    dilation.update(
        {
            "enabled": enabled,
            "hop": 2,
            # Apply dilation to the latter half of the Transformer stack.
            "start_layer": depth // 2,
            "head_fraction": 0.5,
        }
    )

    global_attention = augmentation.setdefault("global_attention", {})
    global_attention.update(
        {
            "enabled": enabled,
            "node_types": ["OBSTACLE"],
            "sample_ratio": 0.01 if enabled else 0.0,
        }
    )


def build_config(base: dict, variant: str, depth: int, node_count: int) -> dict:
    config = copy.deepcopy(base)
    model = config["model"]
    expected_type = "epd" if variant == "gnn" else "transformer"
    if model.get("type") != expected_type:
        raise ValueError(
            f"{variant} requires model.type={expected_type!r}, "
            f"got {model.get('type')!r}"
        )
    model["message_passing_num"] = depth

    if variant != "gnn":
        configure_gt_augmentation(model, enabled=variant == "gt_aug", depth=depth)

    dataset = config["dataset"]
    dataset_root = f"data/data_xdmf/mesh_re100/n={node_count}"
    dataset["train_path"] = f"{dataset_root}/train"
    dataset["val_path"] = f"{dataset_root}/valid"
    dataset["test_path"] = f"{dataset_root}/test"
    dataset["new_edges_ratio"] = 0.0

    # Run every requested job for the full 300 epochs.
    config.setdefault("training", {}).setdefault("early_stopping", {})[
        "enabled"
    ] = False
    return config


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def write_status(path: Path, state: dict) -> None:
    state["updated_at"] = timestamp()
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as status_file:
        json.dump(state, status_file, ensure_ascii=False, indent=2)
        status_file.write("\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-status",
        type=Path,
        help="Existing status.json; completed jobs are skipped.",
    )
    return parser.parse_args()


def is_wandb_initialization_error(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("rb") as log_file:
        size = log_file.seek(0, 2)
        log_file.seek(max(0, size - 1_000_000))
        log_tail = log_file.read().decode("utf-8", errors="replace")
    return (
        "wandb.errors.errors.CommError" in log_tail
        and "Run initialization" in log_tail
    )


def is_cuda_out_of_memory(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("rb") as log_file:
        size = log_file.seek(0, 2)
        log_file.seek(max(0, size - 1_000_000))
        log_tail = log_file.read().decode("utf-8", errors="replace")
    return (
        "torch.OutOfMemoryError" in log_tail
        or "CUDA out of memory" in log_tail
    )


def main() -> int:
    args = parse_args()
    base_configs = {}
    for variant, path in BASE_CONFIGS.items():
        with path.open(encoding="utf-8") as config_file:
            base_configs[variant] = json.load(config_file)

    if args.resume_status is not None:
        status_path = args.resume_status.resolve()
        state = json.loads(status_path.read_text(encoding="utf-8"))
        run_id = state["run_id"]
        jobs = state["jobs"]
        log_dir = status_path.parent
        generated_dir = PROJECT_ROOT / "training_config" / "generated" / run_id
        for job in jobs:
            if (
                job.get("status") == "failed"
                and job.get("log")
                and is_cuda_out_of_memory(Path(job["log"]))
            ):
                job["status"] = "skipped"
                job["skip_reason"] = "cuda_out_of_memory"
        state["status"] = "running"
        state["resumed_at"] = timestamp()
        log(
            f"Queue resuming: {sum(job['status'] == 'completed' for job in jobs)}/"
            f"{len(jobs)} jobs already completed, run_id={run_id}"
        )
    else:
        run_id = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        generated_dir = PROJECT_ROOT / "training_config" / "generated" / run_id
        log_dir = PROJECT_ROOT / "logs" / f"mp5_mp15_model_grid_sweep_{run_id}"
        generated_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        status_path = log_dir / "status.json"
        jobs = [
            {
                "message_passing": depth,
                "variant": variant,
                "nodes": node_count,
                "status": "pending",
            }
            for depth in MESSAGE_PASSING_COUNTS
            for variant in VARIANTS
            for node_count in NODE_COUNTS
        ]
        state = {
            "run_id": run_id,
            "status": "running",
            "epochs_per_job": NUM_EPOCHS,
            "jobs": jobs,
        }
    write_status(status_path, state)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{PROJECT_ROOT}:{environment.get('PYTHONPATH', '')}",
            "WANDB_RUN_GROUP": f"mp5_mp15_grid_sweep_{run_id}",
            "WANDB_INIT_TIMEOUT": str(WANDB_INIT_TIMEOUT_SECONDS),
        }
    )

    if args.resume_status is None:
        log(
            f"Queue started: {len(jobs)} jobs, message_passing={MESSAGE_PASSING_COUNTS}, "
            f"variants={VARIANTS}, nodes={NODE_COUNTS}, epochs={NUM_EPOCHS}, "
            f"run_id={run_id}"
        )
    for job_index, job in enumerate(jobs):
        if job["status"] in {"completed", "skipped"}:
            continue
        depth = job["message_passing"]
        variant = job["variant"]
        node_count = job["nodes"]
        config = build_config(base_configs[variant], variant, depth, node_count)
        config_path = generated_dir / f"mp{depth}_{variant}_n{node_count}.json"
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
            f"mesh_re100/n={node_count}/re_100_{variant}_mp{depth}_300e_{run_id}"
        )
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

        previous_attempts = int(job.get("attempts", 0))
        if job.get("status") == "failed" and previous_attempts == 0:
            previous_attempts = 1
        job.update(
            {
                "status": "running",
                "started_at": timestamp(),
                "config": str(config_path),
                "checkpoint": str(PROJECT_ROOT / "checkpoints" / f"{checkpoint_name}.ckpt"),
            }
        )
        job.pop("finished_at", None)
        job.pop("exit_code", None)
        state["current_job_index"] = job_index
        write_status(status_path, state)
        log(
            f"Starting job {job_index + 1}/{len(jobs)}: "
            f"mp={depth}, variant={variant}, n={node_count}"
        )

        result = None
        attempt = previous_attempts
        while attempt < MAX_WANDB_ATTEMPTS:
            attempt += 1
            retry_suffix = "" if attempt == 1 else f"_retry{attempt}"
            job_log_path = (
                log_dir / f"mp{depth}_{variant}_n{node_count}{retry_suffix}.log"
            )
            job["attempts"] = attempt
            job["log"] = str(job_log_path)
            job["status"] = "running"
            write_status(status_path, state)

            job_environment = environment.copy()
            job_environment["WANDB_NAME"] = (
                f"mp{depth}_{variant}_n{node_count}_{run_id}_attempt{attempt}"
            )
            log(
                f"Running job {job_index + 1}/{len(jobs)}, attempt "
                f"{attempt}/{MAX_WANDB_ATTEMPTS}"
            )
            with job_log_path.open("w", encoding="utf-8") as job_log:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=job_environment,
                    stdout=job_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            if result.returncode == 0:
                break
            if (
                attempt < MAX_WANDB_ATTEMPTS
                and is_wandb_initialization_error(job_log_path)
            ):
                job["status"] = "retrying"
                write_status(status_path, state)
                log(
                    f"W&B initialization failed; retrying in "
                    f"{WANDB_RETRY_DELAY_SECONDS} seconds"
                )
                time.sleep(WANDB_RETRY_DELAY_SECONDS)
                continue
            break

        job["finished_at"] = timestamp()
        if result is None:
            raise RuntimeError("Training subprocess was not started.")
        job["exit_code"] = result.returncode
        if result.returncode != 0:
            if is_cuda_out_of_memory(job_log_path):
                job["status"] = "skipped"
                job["skip_reason"] = "cuda_out_of_memory"
                write_status(status_path, state)
                log(
                    f"SKIPPED job {job_index + 1}/{len(jobs)} due to "
                    "CUDA out of memory; continuing queue"
                )
                continue
            job["status"] = "failed"
            state["status"] = "failed"
            write_status(status_path, state)
            log(
                f"FAILED job {job_index + 1}/{len(jobs)} with exit code "
                f"{result.returncode}; queue stopped"
            )
            return result.returncode

        job["status"] = "completed"
        write_status(status_path, state)
        log(f"Completed job {job_index + 1}/{len(jobs)}")

    state["status"] = "completed"
    state["skipped_jobs"] = sum(
        job.get("status") == "skipped" for job in jobs
    )
    state.pop("current_job_index", None)
    write_status(status_path, state)
    log("Queue completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
