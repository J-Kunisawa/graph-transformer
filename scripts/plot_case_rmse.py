"""Compare per-case prediction RMSE from one or more inference CSV files."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "overall_rmse": "Overall",
    "velocity_rmse": "Velocity",
    "pressure_rmse": "Pressure",
    "velocity_x_rmse": "Velocity x",
    "velocity_y_rmse": "Velocity y",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", help="One legend label per CSV")
    parser.add_argument("--output", type=Path, default=Path("output_vis/case_rmse_comparison.png"))
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.csv):
        parser.error("--labels must contain one label per CSV")
    datasets = []
    for path in args.csv:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = {"trajectory", *METRICS} - set(reader.fieldnames or [])
            if missing:
                parser.error(f"{path}: missing columns {sorted(missing)}")
            rows = list(reader)
        if not rows:
            parser.error(f"{path}: no cases")
        cases = [int(row["trajectory"]) for row in rows]
        if len(set(cases)) != len(cases):
            parser.error(f"{path}: duplicate trajectory IDs")
        values = {int(row["trajectory"]): [float(row[key]) for key in METRICS] for row in rows}
        if not np.isfinite(list(values.values())).all():
            parser.error(f"{path}: non-finite RMSE")
        datasets.append(values)
    cases = sorted(set().union(*(data.keys() for data in datasets)))
    labels = args.labels or [str(path.parent) for path in args.csv]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(cases))
    width = 0.8 / len(datasets)
    for index, (key, title) in enumerate(METRICS.items()):
        ax = axes.flat[index]
        for j, (label, data) in enumerate(zip(labels, datasets)):
            y = [data[case][index] if case in data else np.nan for case in cases]
            ax.bar(x + (j - (len(datasets) - 1) / 2) * width, y, width, label=label)
        ax.set(title=title, xlabel="Trajectory ID", ylabel="RMSE")
        ax.set_xticks(x, cases)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes.flat[5].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, legend_labels, loc="center", frameon=False)
    fig.suptitle("Prediction RMSE by case", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
