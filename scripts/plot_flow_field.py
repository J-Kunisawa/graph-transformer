"""Plot saved 2D XDMF flow fields without rerunning inference."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xdmf", type=Path)
    parser.add_argument("--step", type=int, default=-1, help="Zero-based saved frame index; default: last")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default="Flow field")
    args = parser.parse_args()
    with meshio.xdmf.TimeSeriesReader(args.xdmf) as reader:
        points, cells = reader.read_points_cells()
        step = args.step if args.step >= 0 else reader.num_steps + args.step
        if not 0 <= step < reader.num_steps:
            parser.error(f"step must be between {-reader.num_steps} and {reader.num_steps - 1}")
        time, data, _ = reader.read_data(step)
    required = {"U_pred", "U_target", "p_pred", "p_target"}
    if required - data.keys():
        parser.error(f"Missing fields: {sorted(required - data.keys())}")
    triangles = [cell.data for cell in cells if cell.type == "triangle"]
    if not triangles:
        parser.error("A triangular mesh is required")
    if points.shape[1] > 2 and not np.allclose(points[:, 2], points[0, 2]):
        parser.error("Expected a 2D mesh in the xy plane")
    # Preserve the original connectivity so obstacles are not filled in.
    tri = mtri.Triangulation(points[:, 0], points[:, 1], np.concatenate(triangles))
    u_pred, u_true = data["U_pred"], data["U_target"]
    p_pred, p_true = data["p_pred"].ravel(), data["p_target"].ravel()
    fields = [
        (np.linalg.norm(u_true, axis=1), np.linalg.norm(u_pred, axis=1),
         np.linalg.norm(u_pred - u_true, axis=1), "Speed", "Velocity error |U_pred - U_target|"),
        (p_true, p_pred, np.abs(p_pred - p_true), "Pressure", "Pressure error |p_pred - p_target|"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 7), constrained_layout=True)
    for row, (truth, pred, error, name, error_title) in enumerate(fields):
        vmin, vmax = min(truth.min(), pred.min()), max(truth.max(), pred.max())
        for col, values in enumerate((truth, pred, error)):
            ax = axes[row, col]
            im = ax.tripcolor(tri, values, shading="gouraud", cmap="magma" if col == 2 else "viridis",
                              vmin=0 if col == 2 else vmin, vmax=max(float(error.max()), 1e-12) if col == 2 else vmax)
            ax.set_title(error_title if col == 2 else f"{name}: {'target' if col == 0 else 'prediction'}")
            ax.set_aspect("equal")
            ax.set(xlabel="x", ylabel="y")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"{args.title} | {args.xdmf.stem} | frame={step}, saved time={time:.4g}")
    output = args.output or args.xdmf.with_name(f"{args.xdmf.stem}_flow_step{step:04d}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved: {output} (frame={step}, time={time:.6g})")


if __name__ == "__main__":
    main()
