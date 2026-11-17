#!/usr/bin/env python3
"""Predict every completed sweep model on its matching mesh test set."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--training-status", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training_status_path = args.training_status.resolve()
    training_state = json.loads(training_status_path.read_text(encoding="utf-8"))
    if training_state.get("status") != "completed":
        raise ValueError("Training sweep is not completed.")
    if any(
        job.get("status") not in {"completed", "skipped"}
        for job in training_state["jobs"]
    ):
        raise ValueError("Training status contains incomplete jobs.")

    training_run_id = training_state["run_id"]
    inference_run_id = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / f"predict_sweep_same_mesh_{inference_run_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    status_path = log_dir / "status.json"
    jobs = [
        {
            "message_passing": job["message_passing"],
            "variant": job["variant"],
            "nodes": job["nodes"],
            "status": "pending",
        }
        for job in training_state["jobs"]
        if job["status"] == "completed"
    ]
    state = {
        "run_id": inference_run_id,
        "training_run_id": training_run_id,
        "training_status": str(training_status_path),
        "training_skipped_jobs": [
            {
                "message_passing": job["message_passing"],
                "variant": job["variant"],
                "nodes": job["nodes"],
                "reason": job.get("skip_reason", "skipped"),
            }
            for job in training_state["jobs"]
            if job["status"] == "skipped"
        ],
        "status": "running",
        "jobs": jobs,
    }
    write_status(status_path, state)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{PROJECT_ROOT}:{environment.get('PYTHONPATH', '')}",
            "WANDB_MODE": "offline",
            "WANDB_INIT_TIMEOUT": "300",
            "WANDB_RUN_GROUP": f"sweep_same_mesh_{inference_run_id}",
        }
    )

    log(
        f"Prediction queue started: {len(jobs)} jobs from training "
        f"run {training_run_id}"
    )
    for job_index, job in enumerate(jobs):
        depth = job["message_passing"]
        variant = job["variant"]
        node_count = job["nodes"]
        model_stem = (
            f"re_100_{variant}_mp{depth}_300e_{training_run_id}"
        )
        config_path = (
            PROJECT_ROOT
            / "training_config"
            / "generated"
            / training_run_id
            / f"mp{depth}_{variant}_n{node_count}.json"
        )
        model_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "mesh_re100"
            / f"n={node_count}"
            / f"{model_stem}.ckpt"
        )
        output_dir = (
            PROJECT_ROOT
            / "predictions"
            / "mesh_re100"
            / f"n={node_count}"
            / model_stem
            / inference_run_id
        )
        job_log_path = (
            log_dir / f"mp{depth}_{variant}_n{node_count}.log"
        )

        if not config_path.is_file():
            raise FileNotFoundError(f"Missing prediction config: {config_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        expected_test_path = f"data/data_xdmf/mesh_re100/n={node_count}/test"
        if config["dataset"]["test_path"] != expected_test_path:
            raise ValueError(
                f"Unexpected test_path for n={node_count}: "
                f"{config['dataset']['test_path']}"
            )

        output_dir.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "predict.py"),
            "--project_name=mesh_scaling_predict",
            f"--predict_parameters_path={config_path}",
            f"--model_path={model_path}",
            f"--prediction_save_path={output_dir}",
            "--use_previous_data=false",
            "--previous_data_start=3",
            "--previous_data_end=6",
            "--no_edge_feature=false",
            "--use_test=true",
        ]

        job.update(
            {
                "status": "running",
                "started_at": timestamp(),
                "config": str(config_path),
                "model": str(model_path),
                "test_path": expected_test_path,
                "output": str(output_dir),
                "log": str(job_log_path),
            }
        )
        state["current_job_index"] = job_index
        write_status(status_path, state)
        log(
            f"Starting {job_index + 1}/{len(jobs)}: mp={depth}, "
            f"variant={variant}, n={node_count}"
        )

        job_environment = environment.copy()
        job_environment["WANDB_NAME"] = (
            f"predict_mp{depth}_{variant}_n{node_count}_{inference_run_id}"
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

        job["finished_at"] = timestamp()
        job["exit_code"] = result.returncode
        if result.returncode != 0:
            job["status"] = "failed"
            state["status"] = "failed"
            write_status(status_path, state)
            log(
                f"FAILED {job_index + 1}/{len(jobs)}: mp={depth}, "
                f"variant={variant}, n={node_count}, "
                f"exit_code={result.returncode}"
            )
            return result.returncode

        job["status"] = "completed"
        write_status(status_path, state)
        log(f"Completed {job_index + 1}/{len(jobs)}")

    state["status"] = "completed"
    state.pop("current_job_index", None)
    write_status(status_path, state)
    log("Prediction queue completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
