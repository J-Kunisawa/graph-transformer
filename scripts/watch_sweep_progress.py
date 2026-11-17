#!/usr/bin/env python3
"""Continuously show the active MP=5/15 sweep job in a terminal."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS = re.compile(
    r"Epoch\s+(?P<epoch>\d+):\s+"
    r"(?P<percent>\d+)%.*?"
    r"(?P<step>\d+)/(?P<steps>\d+).*?"
    r"loss=(?P<loss>[^,\]\s]+),\s*"
    r"val_loss=(?P<val_loss>[^\]\s]+)"
)


def latest_progress(log_path: Path) -> str:
    if not log_path.is_file():
        return "ログ生成待ち"
    with log_path.open("rb") as log_file:
        size = log_file.seek(0, 2)
        log_file.seek(max(0, size - 512_000))
        text = log_file.read().decode("utf-8", errors="replace")
    text = ANSI_ESCAPE.sub("", text)
    matches = list(PROGRESS.finditer(text))
    if not matches:
        return "初期化中"
    values = matches[-1].groupdict()
    return (
        f"epoch {int(values['epoch']) + 1}/300 | "
        f"step {values['step']}/{values['steps']} ({values['percent']}%) | "
        f"loss {values['loss']} | val_loss {values['val_loss']}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} STATUS_JSON", file=sys.stderr)
        return 2

    status_path = Path(sys.argv[1]).resolve()
    previous_job_index = None
    print(f"Monitoring: {status_path}", flush=True)

    while True:
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"\r状態ファイル待ち: {exc}\033[K", end="", flush=True)
            time.sleep(2)
            continue

        if state.get("status") in {"completed", "failed"}:
            print(
                f"\nQueue {state['status']}: "
                f"{sum(job.get('status') == 'completed' for job in state['jobs'])}/"
                f"{len(state['jobs'])} jobs completed",
                flush=True,
            )
            return 0 if state["status"] == "completed" else 1

        job_index = int(state.get("current_job_index", 0))
        job = state["jobs"][job_index]
        if job_index != previous_job_index:
            if previous_job_index is not None:
                print()
            print(
                f"Job {job_index + 1}/{len(state['jobs'])}: "
                f"MP={job['message_passing']} | {job['variant']} | "
                f"n={job['nodes']}",
                flush=True,
            )
            previous_job_index = job_index

        progress = latest_progress(Path(job.get("log", "")))
        now = datetime.now().astimezone().strftime("%H:%M:%S")
        print(f"\r[{now}] {progress}\033[K", end="", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
