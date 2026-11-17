#!/usr/bin/env python3
"""Plot per-step inference time against mesh resolution.

For every ``N=<nodes>`` directory, the newest inference run is selected for
each requested model.  Step zero is excluded by default because it includes
one-time CUDA/model warm-up overhead.

Example:
    python scripts/plot_mesh_resolution_inference_time.py
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "predictions/mesh_re100"
DEFAULT_OUTPUT = PROJECT_ROOT / "output_vis/mesh_resolution_inference_time.png"
DEFAULT_MODELS = (
    "GNN=re_100_gnn_s3",
    "Transformer=re_100_transformer_s3",
)


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=DIRECTORY")
    label, directory = value.split("=", 1)
    if not label or not directory or "/" in directory:
        raise argparse.ArgumentTypeError("model must be LABEL=DIRECTORY")
    return label, directory


def newest_timing_csv(model_dir: Path) -> Path:
    # Timestamped run directories (YYYYMMDD_HHMMSS) take precedence over a
    # legacy CSV stored directly below the model directory.
    candidates = [
        path
        for path in model_dir.glob("*/inference_prediction_times.csv")
        if re.fullmatch(r"\d{8}_\d{6}", path.parent.name)
    ]
    if candidates:
        return max(candidates, key=lambda path: path.parent.name)

    direct = model_dir / "inference_prediction_times.csv"
    if direct.is_file():
        return direct
    raise FileNotFoundError(
        f"No inference_prediction_times.csv below {model_dir}"
    )


def read_timing(path: Path, include_first_step: bool) -> tuple[float, int]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "prediction_time_seconds"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        values = [
            float(row["prediction_time_seconds"])
            for row in reader
            if include_first_step or int(row["step"]) > 0
        ]
    times = np.asarray(values)
    if not len(times):
        raise ValueError(f"{path}: no timing samples")
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError(f"{path}: timing values must be positive and finite")
    return float(times.mean()), len(times)


def collect(input_dir: Path, models: list[tuple[str, str]], include_first_step: bool):
    resolutions = []
    for path in input_dir.glob("N=*"):
        match = re.fullmatch(r"N=(\d+)", path.name)
        if match and path.is_dir():
            resolutions.append((int(match.group(1)), path))
    if not resolutions:
        raise FileNotFoundError(f"No N=<nodes> directories below {input_dir}")

    data = {label: [] for label, _ in models}
    selected = []
    for nodes, resolution_dir in sorted(resolutions):
        for label, model_name in models:
            model_dir = resolution_dir / model_name
            if not model_dir.is_dir():
                continue
            csv_path = newest_timing_csv(model_dir)
            mean, samples = read_timing(csv_path, include_first_step)
            data[label].append((nodes, mean))
            selected.append((nodes, label, mean, samples, csv_path))
    missing = [label for label, values in data.items() if not values]
    if missing:
        raise FileNotFoundError(f"No data found for models: {', '.join(missing)}")
    return data, selected


def plot(data, output: Path, include_first_step: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    for model_index, (label, values) in enumerate(data.items()):
        nodes = np.asarray([item[0] for item in values])
        mean_ms = np.asarray([item[1] for item in values]) * 1000
        ax.plot(
            nodes,
            mean_ms,
            marker=markers[model_index % len(markers)],
            markersize=6,
            linewidth=2,
            label=label,
        )

    all_nodes = sorted({nodes for values in data.values() for nodes, _ in values})
    ax.set_title("Inference time by mesh resolution")
    ax.set_xlabel("Number of mesh nodes")
    ax.set_ylabel("Mean inference time (ms/step)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_nodes, [f"{nodes:,}" for nodes in all_nodes], rotation=25)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    if not include_first_step:
        ax.text(
            0.98,
            0.03,
            "Warm-up step excluded",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        metavar="LABEL=DIRECTORY",
        help="Model directory to compare; repeat for multiple models",
    )
    parser.add_argument(
        "--include-first-step",
        action="store_true",
        help="Include step zero, which normally contains warm-up overhead",
    )
    args = parser.parse_args()
    models = args.model or [parse_model(value) for value in DEFAULT_MODELS]

    data, selected = collect(args.input, models, args.include_first_step)
    plot(data, args.output, args.include_first_step)
    for nodes, label, mean, samples, path in selected:
        print(
            f"{nodes:>6} nodes | {label:<11} | "
            f"{mean * 1000:8.3f} ms/step "
            f"(n={samples}) | {path}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
