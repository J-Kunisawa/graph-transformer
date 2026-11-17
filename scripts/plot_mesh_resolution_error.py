#!/usr/bin/env python3
"""Plot prediction RMSE against mesh resolution.

For every ``N=<nodes>`` directory, the newest inference run is selected for
each requested model.  The input of a run is ``inference_case_rmse.csv``.

Example:
    python scripts/plot_mesh_resolution_error.py
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
DEFAULT_OUTPUT = PROJECT_ROOT / "output_vis/mesh_resolution_error.png"
DEFAULT_MODELS = (
    "GNN=re_100_gnn_s3",
    "Transformer=re_100_transformer_s3",
)
METRICS = (
    ("overall_rmse", "Overall RMSE", "overall"),
    ("velocity_x_rmse", "Velocity x RMSE", "velocity_x"),
    ("velocity_y_rmse", "Velocity y RMSE", "velocity_y"),
    ("pressure_rmse", "Pressure RMSE", "pressure"),
)


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=DIRECTORY")
    label, directory = value.split("=", 1)
    if not label or not directory or "/" in directory:
        raise argparse.ArgumentTypeError("model must be LABEL=DIRECTORY")
    return label, directory


def newest_case_csv(model_dir: Path) -> Path:
    # Timestamped run directories (YYYYMMDD_HHMMSS) take precedence over a
    # legacy CSV stored directly below the model directory.
    candidates = [
        path
        for path in model_dir.glob("*/inference_case_rmse.csv")
        if re.fullmatch(r"\d{8}_\d{6}", path.parent.name)
    ]
    if candidates:
        return max(candidates, key=lambda path: path.parent.name)

    direct = model_dir / "inference_case_rmse.csv"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"No inference_case_rmse.csv below {model_dir}")


def read_mean_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {name for name, _, _ in METRICS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no cases")

    result = {}
    for name, _, _ in METRICS:
        values = np.asarray([float(row[name]) for row in rows])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{path}: {name} must contain positive finite values")
        result[name] = float(values.mean())
    return result


def collect(input_dir: Path, models: list[tuple[str, str]]):
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
            csv_path = newest_case_csv(model_dir)
            data[label].append((nodes, read_mean_metrics(csv_path)))
            selected.append((nodes, label, csv_path))
    missing = [label for label, values in data.items() if not values]
    if missing:
        raise FileNotFoundError(f"No data found for models: {', '.join(missing)}")
    return data, selected


def plot(data, output: Path) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    all_nodes = sorted({nodes for values in data.values() for nodes, _ in values})
    markers = ("o", "s", "^", "D")
    outputs = []

    for metric, title, suffix in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for model_index, (label, values) in enumerate(data.items()):
            nodes = np.asarray([item[0] for item in values])
            rmse = np.asarray([item[1][metric] for item in values])
            ax.plot(nodes, rmse, marker=markers[model_index % len(markers)],
                    markersize=6, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Number of mesh nodes")
        ax.set_ylabel("Mean case RMSE (log scale)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(all_nodes, [f"{nodes:,}" for nodes in all_nodes], rotation=25)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)

        metric_output = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
        fig.savefig(metric_output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(metric_output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Directory containing N=<nodes> subdirectories")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", action="append", type=parse_model,
                        metavar="LABEL=DIRECTORY",
                        help="Model directory to compare; repeat for multiple models")
    args = parser.parse_args()
    models = args.model or [parse_model(value) for value in DEFAULT_MODELS]

    data, selected = collect(args.input, models)
    outputs = plot(data, args.output)
    for nodes, label, path in selected:
        print(f"{nodes:>6} nodes | {label:<11} | {path}")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
