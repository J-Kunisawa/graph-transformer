from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "predictions/mesh_re100/N=2000/re_100_gnn_s3/20261102_033930/inference_error_statistics.csv"
)
OUTPUT_PATH = CSV_PATH.with_name("inference_error_evolution.png")

df = pd.read_csv(CSV_PATH)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 12),
    sharex=True,
    constrained_layout=True,
)

# 1. 軌跡ごとのRMSE
for trajectory, group in df.groupby("trajectory"):
    group = group.sort_values("time")
    axes[0].plot(
        group["time"],
        group["rmse"],
        linewidth=0.9,
        alpha=0.55,
        label=f"Trajectory {trajectory}",
    )

rmse_stats = df.groupby("time")["rmse"].agg(["mean", "std"])
time = rmse_stats.index.to_numpy()

axes[0].plot(
    time,
    rmse_stats["mean"].to_numpy(),
    color="black",
    linewidth=2.5,
    label="Mean",
)

axes[0].set_title("Overall RMSE by trajectory")
axes[0].set_ylabel("RMSE")
axes[0].legend(ncol=4, fontsize=8, frameon=False)


# 2. 成分ごとのRMSE
components = {
    "Velocity x": ("component_0_rmse", "tab:red"),
    "Velocity y": ("component_1_rmse", "tab:green"),
    "Pressure": ("component_2_rmse", "tab:blue"),
}

for label, (column, color) in components.items():
    stats = df.groupby("time")[column].agg(["mean", "std"])

    x = stats.index.to_numpy()
    mean = stats["mean"].to_numpy()
    std = stats["std"].fillna(0).to_numpy()

    axes[1].plot(
        x,
        mean,
        color=color,
        linewidth=2,
        label=label,
    )
    axes[1].fill_between(
        x,
        np.maximum(mean - std, 0),
        mean + std,
        color=color,
        alpha=0.15,
    )

axes[1].set_title(
    "Component RMSE: mean across trajectories"
)
axes[1].set_ylabel("Component RMSE")
axes[1].legend(frameon=False)


# 3. 相対L2誤差
relative_stats = df.groupby("time")["relative_l2"].agg(
    ["mean", "std"]
)

x = relative_stats.index.to_numpy()
mean = relative_stats["mean"].to_numpy()
std = relative_stats["std"].fillna(0).to_numpy()

axes[2].plot(
    x,
    mean,
    color="tab:purple",
    linewidth=2,
    label="Mean relative L2",
)
axes[2].fill_between(
    x,
    np.maximum(mean - std, 0),
    mean + std,
    color="tab:purple",
    alpha=0.2,
    label="±1 SD",
)

# relative L2 = 1 の基準線
axes[2].axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label="Error = target norm",
)

axes[2].set_title("Relative L2 error")
axes[2].set_xlabel("Rollout time")
axes[2].set_ylabel("Relative L2")
axes[2].legend(frameon=False)


for ax in axes:
    ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Inference Error Evolution "
    f"({df['trajectory'].nunique()} trajectories)",
    fontsize=15,
)

fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {OUTPUT_PATH}")