#!/usr/bin/env python3
"""Wait for a training sweep to finish, then launch matching predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 30


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-status", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_path = args.training_status.resolve()
    last_summary = None
    print(f"[{now()}] Waiting for training: {status_path}", flush=True)

    while True:
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"[{now()}] Status unavailable: {exc}", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        completed = sum(job.get("status") == "completed" for job in state["jobs"])
        skipped = sum(job.get("status") == "skipped" for job in state["jobs"])
        summary = (state.get("status"), completed, skipped, len(state["jobs"]))
        if summary != last_summary:
            print(
                f"[{now()}] Training status={summary[0]}, "
                f"completed={summary[1]}, skipped={summary[2]}, "
                f"total={summary[3]}",
                flush=True,
            )
            last_summary = summary

        if state.get("status") == "completed":
            break
        time.sleep(POLL_SECONDS)

    print(f"[{now()}] Training complete; starting prediction queue", flush=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "predict_completed_sweep_same_mesh.py"),
        f"--training-status={status_path}",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
