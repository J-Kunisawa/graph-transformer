"""Estimate St=fD/U from wake transverse velocity in a prediction XDMF.

Example:
    python scripts/calculate_strouhal.py
Edit the settings below to select the input and analysis conditions.
Command-line arguments can optionally override these settings.
Uses a detrended Hann-window FFT with sub-bin peak interpolation. Time values
are the exported XDMF times. Inspect the saved trace/spectrum before accepting
an estimate, especially when the record contains few shedding cycles.
"""
import argparse
import csv
import json
from pathlib import Path

import meshio
import numpy as np
from scipy.signal import detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==================== 設定（ここを編集） ====================
# 相対パスはプロジェクトルートを基準に解決します。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
XDMF_PATH = Path(
    'predictions/mesh_re100/N=40000/re_100_transformer_s3/20261106_022856/graph_0001.xdmf'
)
META_PATH = Path('dataset_config/meta_test.json')  # 円柱中心・直径を読むJSON
REFERENCE_VELOCITY = 1.0  # St=fD/U の U（データと同じ単位）
PROBE = None  # (x, y)。None: 円柱中心から下流へ直径の3倍の位置
START_TIME = None  # この時刻以降を解析。None: 全時刻（XDMFの時刻）
OUTPUT_DIR = None  # 保存先。None: XDMFと同じ場所の「ファイル名_strouhal」
# ==========================================================


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def dominant_frequency(times, values):
    dt = np.diff(times)
    if len(times) < 8 or not np.all(dt > 0) or not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-9):
        raise ValueError('At least eight uniformly spaced, increasing times are required')
    signal = detrend(values)
    if not np.all(np.isfinite(signal)) or np.std(signal) < 1e-10:
        raise ValueError('Probe signal is nonfinite or has no measurable variation')
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freq = np.fft.rfftfreq(len(signal), dt[0])
    k = 1 + np.argmax(power[1:])
    offset = 0.0
    if 1 < k < len(power) - 1:
        a, b, c = np.log(np.maximum(power[k-1:k+2], np.finfo(float).tiny))
        offset = float(np.clip(0.5 * (a-c) / (a-2*b+c), -.5, .5))
    return (k + offset) * freq[1], freq, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xdmf', type=Path, nargs='?', default=XDMF_PATH)
    parser.add_argument('--meta', type=Path, default=META_PATH)
    parser.add_argument('--velocity', type=float, default=REFERENCE_VELOCITY, help='Reference flow velocity in the same units as the data')
    parser.add_argument('--probe', type=float, nargs=2, default=PROBE, help='Requested x y; default: 3 diameters downstream of cylinder center')
    parser.add_argument('--start-time', type=float, default=START_TIME, help='Discard earlier exported times')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.xdmf = project_path(args.xdmf)
    args.meta = project_path(args.meta)
    if args.output_dir is not None:
        args.output_dir = project_path(args.output_dir)
    if args.velocity <= 0:
        parser.error('--velocity must be positive')
    geometry = json.loads(args.meta.read_text())['geometry']
    diameter = float(geometry['obstacle_diameter'])
    if diameter <= 0:
        raise ValueError('Diameter must be positive')
    probe = args.probe or [geometry['obstacle_center'][0] + 3 * diameter, geometry['obstacle_center'][1]]
    times, traces = [], []
    with meshio.xdmf.TimeSeriesReader(args.xdmf) as reader:
        points, _ = reader.read_points_cells()
        node = int(np.argmin(np.linalg.norm(points[:, :2] - probe, axis=1)))
        for step in range(reader.num_steps):
            time, fields, _ = reader.read_data(step)
            if args.start_time is None or time >= args.start_time:
                times.append(time)
                traces.append([fields['U_pred'][node, 1], fields['U_target'][node, 1]])
    times, traces = np.asarray(times), np.asarray(traces)
    out = args.output_dir or args.xdmf.parent / (args.xdmf.stem + '_strouhal')
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    rows = []
    for i, name in enumerate(('prediction', 'target')):
        f, frequencies, power = dominant_frequency(times, traces[:, i])
        row = dict(source=name, frequency=f, strouhal=f*diameter/args.velocity,
                   diameter=diameter, reference_velocity=args.velocity,
                   fft_bin_width=frequencies[1], observed_cycles=f*(times[-1]-times[0]),
                   time_start=times[0], time_end=times[-1], samples=len(times),
                   probe_node=node, probe_x=points[node, 0], probe_y=points[node, 1])
        rows.append(row)
        axes[0].plot(times, traces[:, i], label=name)
        axes[1].plot(frequencies, power/power.max(), label=f'{name}: St={row["strouhal"]:.5f}')
        print(f'{name}: f={f:.6g}, St={row["strouhal"]:.6g}, cycles={row["observed_cycles"]:.2f}')
    axes[0].set(xlabel='Exported time', ylabel='Wake transverse velocity')
    axes[1].set(xlabel='Frequency', ylabel='Normalized spectral power', xlim=(0, min(frequencies[-1], 4*max(r['frequency'] for r in rows))))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.savefig(out / 'strouhal.png', dpi=160)
    plt.close(fig)
    with (out / 'strouhal.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(out / 'probe_velocity.csv', np.column_stack((times, traces)), delimiter=',', header='time,v_prediction,v_target', comments='')
    print(f'Relative St error: {abs(rows[0]["strouhal"]/rows[1]["strouhal"]-1)*100:.3f}%')
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
