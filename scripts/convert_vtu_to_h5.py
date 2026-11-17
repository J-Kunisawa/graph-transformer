#!/usr/bin/env python
"""
VTU ファイルを学習用 HDF5 形式に変換するスクリプト。

複数ケースディレクトリ配下の時系列 VTU ファイルを読み込み、
メッシュ情報と物理量を抽出して HDF5 に保存します。
使用例:
    # 複数ケースから一括生成（推奨）
    python scripts/convert_vtu_to_h5.py \
      --case_dirs data/data_vtu/re_50-100/case{1..5} \
      --train_indices 0:20 \
      --valid_indices 20:30 \
      --test_indices 30:40 \
      --output_dir data/learning_datasets/re_50-100 \
      --create_meta
    
    # シンプル指定
    python scripts/convert_vtu_to_h5.py \
      --case_root data/data_vtu/re_50-100 \
      --case_numbers range(1,6) \
      --output_dir data/learning_datasets/re_50-100 \
      --train_ratio 0.7 --valid_ratio 0.15
"""

import argparse
import re
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

try:
    import h5py
except ImportError:
    raise RuntimeError("h5py is required")

try:
    import meshio
except ImportError:
    raise RuntimeError("meshio is required to read VTU files")


def _compress_points_and_build_inverse(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2D射影後の重複座標を統合し、元ノード->圧縮ノードの対応を返す。"""
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    return unique_points.astype(np.float32), inverse.astype(np.int64)


def _compress_point_array(
    arr: np.ndarray, inverse: np.ndarray, n_compressed_points: int
) -> np.ndarray:
    """元ノード配列を圧縮ノード配列へ集約（重複ノードは平均）。"""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    compressed = np.zeros((n_compressed_points, arr.shape[1]), dtype=np.float64)
    counts = np.zeros((n_compressed_points, 1), dtype=np.float64)

    np.add.at(compressed, inverse, arr.astype(np.float64))
    np.add.at(counts, inverse, 1.0)

    counts = np.clip(counts, a_min=1.0, a_max=None)
    compressed = compressed / counts
    return compressed.astype(np.float32)


def _align_to_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """配列を target_shape に合わせて切り詰め/0埋めする。"""
    target_rows, target_cols = target_shape

    if arr.shape[0] < target_rows:
        row_pad = np.zeros((target_rows - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        arr = np.vstack([arr, row_pad])
    else:
        arr = arr[:target_rows]

    if arr.shape[1] < target_cols:
        col_pad = np.zeros((arr.shape[0], target_cols - arr.shape[1]), dtype=arr.dtype)
        arr = np.hstack([arr, col_pad])
    else:
        arr = arr[:, :target_cols]

    return arr


def _parse_case_numbers(case_numbers: str) -> List[int]:
    """'1 2 3' / '1,2,3' / 'range(1,6[,step])' を int リストへ変換する。"""
    text = case_numbers.strip()
    range_match = re.fullmatch(
        r"range\(\s*(-?\d+)\s*,\s*(-?\d+)(?:\s*,\s*(-?\d+))?\s*\)", text
    )
    if range_match:
        start = int(range_match.group(1))
        stop = int(range_match.group(2))
        step = int(range_match.group(3)) if range_match.group(3) is not None else 1
        if step == 0:
            raise ValueError("range step must not be zero")
        return list(range(start, stop, step))

    tokens = [tok for tok in re.split(r"[\s,]+", text) if tok]
    if not tokens:
        raise ValueError("case_numbers is empty")
    return [int(tok) for tok in tokens]


def _build_projected_cells_from_vtu(
    vtu_path: str,
    inverse: np.ndarray,
    compressed_points: np.ndarray,
) -> np.ndarray:
    """元VTUのセル接続を2D投影後ノードへ写像して三角形セルを作る。"""
    mesh = meshio.read(vtu_path)
    triangles: List[Tuple[int, int, int]] = []

    for cell_block in mesh.cells:
        block = np.asarray(cell_block.data, dtype=np.int64)
        if block.size == 0:
            continue

        for conn in block:
            mapped = inverse[np.asarray(conn, dtype=np.int64)]
            # 投影で重なる上下層ノードを除去し、元順序を維持
            unique_conn = list(dict.fromkeys(mapped.tolist()))
            if len(unique_conn) < 3:
                continue

            # 多角形は扇分割で三角形化
            base = unique_conn[0]
            for i in range(1, len(unique_conn) - 1):
                tri = (base, unique_conn[i], unique_conn[i + 1])
                if len({tri[0], tri[1], tri[2]}) == 3:
                    triangles.append(tri)

    if triangles:
        tri_arr = np.asarray(triangles, dtype=np.int32)
        # 向き違いを同一視して重複除去
        tri_key = np.sort(tri_arr, axis=1)
        _, first_idx = np.unique(tri_key, axis=0, return_index=True)
        tri_arr = tri_arr[np.sort(first_idx)]
        print(
            f"[cells] {vtu_path}: projected connectivity triangulation -> "
            f"{tri_arr.shape[0]} triangles"
        )
        return tri_arr

    # セル復元に失敗した場合のみフォールバック
    cells = Delaunay(compressed_points[:, :2]).simplices.astype(np.int32)
    print(
        f"[cells] {vtu_path}: fallback to 2D Delaunay triangulation -> "
        f"{cells.shape[0]} triangles"
    )
    return cells


def _extract_time_from_mesh(mesh: "meshio.Mesh") -> Optional[float]:
    """meshio.Mesh から時刻情報を抽出する

    ParaView/OpenFOAM 由来VTUでよく使われるキーを優先して探索する。
    """
    time_keys = {"time", "timevalue", "timestep", "time_step"}

    # 1) field_data を最優先
    for key, value in mesh.field_data.items():
        if key.lower() not in time_keys:
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size > 0:
            return float(arr[0])

    # 2) point_data に単一値が入っているケースを補助的に許容
    for key, value in mesh.point_data.items():
        if key.lower() not in time_keys:
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size > 0:
            return float(arr[0])

    return None


def read_vtu_mesh_and_data(
    vtu_path: str,
    build_cells: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[float]]:
    """
    VTU ファイルからメッシュとデータを抽出
    
    Args:
        vtu_path: VTU ファイルのパス
        
    Returns:
        (points, cells, point_data_dict, time_value)
        - points: メッシュの節点座標 (N, dim)
        - cells: メッシュのセル情報 (M, cells_per_elem)
        - point_data_dict: 物理量データ {"field_name": (N,) or (N, C)}
        - time_value: 抽出できた時刻値（なければNone）
    """
    mesh = meshio.read(vtu_path)
    time_value = _extract_time_from_mesh(mesh)
    
    points = np.array(mesh.points, dtype=np.float32)
    # 3D座標で保存されているケースは2D (x, y) に射影して学習用に統一
    if points.ndim == 2 and points.shape[1] >= 3:
        points = points[:, :2]
    
    if build_cells:
        # 混在セルや3Dセルを含むVTUでは、先頭ブロックだけを使うと大半のノードが失われる。
        # 学習/可視化用には2D射影点群からDelaunay三角形分割を作り、全ノードを接続する。
        cell_types = [cell_block.type for cell_block in mesh.cells]
        supported_surface_types = {"triangle", "tetra"}
        if len(mesh.cells) == 0:
            cells = np.array([], dtype=np.int32)
        elif len(mesh.cells) == 1 and mesh.cells[0].type in supported_surface_types:
            cells = np.array(mesh.cells[0].data, dtype=np.int32)
        else:
            if points.shape[1] < 2:
                raise ValueError(
                    f"Delaunay triangulation requires at least 2D points, got shape {points.shape}."
                )
            cells = Delaunay(points[:, :2]).simplices.astype(np.int32)
            print(
                f"[read_vtu_mesh_and_data] {vtu_path}: mixed/unsupported cell types {cell_types} -> "
                f"2D Delaunay triangulation with {cells.shape[0]} triangles"
            )
    else:
        cells = np.array([], dtype=np.int32)
    
    # ポイントデータを抽出
    point_data_dict = {}
    for key, val in mesh.point_data.items():
        # 学習用には速度2成分のみを残し、nut は除外する
        if key == "nut":
            continue

        arr = np.array(val, dtype=np.float32)
        # 1次元なら2次元に
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if key in {"U", "velocity"} and arr.shape[1] >= 2:
            arr = arr[:, :2]
        point_data_dict[key] = arr
    
    return points, cells, point_data_dict, time_value


def collect_vtu_files_from_cases(
    case_dirs: List[str],
    vtu_pattern: str = "**/*.vtu",
) -> List[Tuple[str, List[str]]]:
    """
    ケースディレクトリから VTU を収集
    
    Args:
        case_dirs: ケースディレクトリのリスト
        vtu_pattern: VTU ファイルの探索パターン
        
    Returns:
        [(case_name, sorted_vtu_list), ...]
    """
    result = []
    for case_dir in case_dirs:
        case_path = Path(case_dir)
        vtu_files = sorted(case_path.glob(vtu_pattern))
        vtu_files_str = [str(f) for f in vtu_files]
        
        if vtu_files_str:
            case_name = case_path.name
            result.append((case_name, vtu_files_str))
    
    return result


def create_trajectory_h5(
    output_path: str,
    case_vtu_list: List[Tuple[str, List[str]]],
    meta_output_path: Optional[str] = None,
) -> Dict:
    """
    VTU から学習用 HDF5 を生成
    
    Args:
        output_path: 出力 HDF5 ファイルパス
        case_vtu_list: [(case_name, [vtu_file_list]), ...]
        meta_output_path: メタデータ出力パス
        
    Returns:
        メタデータ辞書
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    metadata = {
        "dt": 1.0,  # VTU側で時刻が抽出できない場合のデフォルト
        "trajectory_length": 0,
        "features": {
            "mesh_pos": {
                "type": "static",
                "shape": [],
                "dtype": "float32"
            },
            "cells": {
                "type": "static",
                "shape": [],
                "dtype": "int32"
            },
            "node_type": {
                "type": "static",
                "shape": [],
                "dtype": "int32"
            }
        }
    }
    
    all_fields = set()
    inferred_field_dims = {}
    inferred_dt: Optional[float] = None
    
    with h5py.File(output_path, "w") as h5f:
        for traj_idx, (case_name, vtu_files) in enumerate(case_vtu_list):
            traj_name = f"{traj_idx:02d}"
            traj_group = h5f.create_group(traj_name)
            
            n_frames = len(vtu_files)
            
            # 最初の VTU からメッシュ構造を取得
            points_raw, _cells, point_data_first, first_time_value = read_vtu_mesh_and_data(
                vtu_files[0],
                build_cells=False,
            )

            # 2D射影後に重複する前後層ノードを統合
            points, inverse = _compress_points_and_build_inverse(points_raw)
            if points.shape[0] < points_raw.shape[0]:
                print(
                    f"[{case_name}] compressed duplicated projected nodes: "
                    f"{points_raw.shape[0]} -> {points.shape[0]}"
                )

            # 元セル接続を投影後ノードへ写像して三角形化（穴形状を保持しやすい）
            cells = _build_projected_cells_from_vtu(vtu_files[0], inverse, points)

            # 最初のフレームのフィールドを同じ対応で圧縮
            compressed_point_data_first = {
                name: _compress_point_array(value, inverse, points.shape[0])
                for name, value in point_data_first.items()
            }
            n_points = points.shape[0]
            n_cells = cells.shape[0]
            spatial_dim = points.shape[1]
            cell_elem_size = cells.shape[1] if len(cells.shape) > 1 else 1
            
            # mesh_pos：時間ステップごとにコピー（通常は静的）
            mesh_pos_data = np.tile(points[np.newaxis, :, :], (n_frames, 1, 1))
            traj_group.create_dataset("mesh_pos", data=mesh_pos_data)
            
            # cells：時間ステップごとにコピー（通常は静的）
            cells_data = np.tile(cells[np.newaxis, :, :], (n_frames, 1, 1))
            traj_group.create_dataset("cells", data=cells_data)
            
            # node_type：簡易的に全て0（通常ノード）
            node_types = np.zeros((n_frames, n_points, 1), dtype=np.int32)
            traj_group.create_dataset("node_type", data=node_types)

            # node_type は特徴量としては末尾に来るので、メタデータ上の列位置も固定する
            # 生成する特徴量順: U(2), p(1), node_type(1), time(1)
            
            # 各時間ステップのフィールドデータを読み込み
            field_data_dict = {
                name: [value] for name, value in compressed_point_data_first.items()
            }
            frame_times: List[float] = [float(first_time_value)] if first_time_value is not None else []

            for vtu_file in vtu_files[1:]:
                _, _, point_data, time_value = read_vtu_mesh_and_data(vtu_file, build_cells=False)
                if time_value is not None:
                    frame_times.append(float(time_value))

                point_data = {
                    name: _compress_point_array(value, inverse, points.shape[0])
                    for name, value in point_data.items()
                }
                
                for field_name, field_arr in point_data.items():
                    all_fields.add(field_name)

                    field_data_dict.setdefault(field_name, []).append(field_arr)

            for field_name in compressed_point_data_first:
                all_fields.add(field_name)
            
            # フィールドデータを時系列スタック
            for field_name, data_list in field_data_dict.items():
                # サイズが異なるフレームは詰める（最初のフレームのサイズに合わせる）
                target_shape = data_list[0].shape
                aligned_list = [
                    data if data.shape == target_shape else _align_to_shape(data, target_shape)
                    for data in data_list
                ]
                
                field_array = np.stack(aligned_list, axis=0)
                traj_group.create_dataset(field_name, data=field_array)
                
                # フィールド次元を推測
                inferred_field_dims[field_name] = field_array.shape[2]
            
            print(f"[{case_name}] {n_frames} frames: {n_points} nodes, {n_cells} cells")

            # 最初に得られた有効な時刻列から dt を推定
            if inferred_dt is None and len(frame_times) >= 2:
                diffs = np.diff(np.asarray(frame_times, dtype=np.float64))
                positive_diffs = diffs[np.abs(diffs) > 0]
                if positive_diffs.size > 0:
                    inferred_dt = float(np.median(np.abs(positive_diffs)))
    
    # メタデータ更新
    metadata["trajectory_length"] = n_frames
    if inferred_dt is not None:
        metadata["dt"] = inferred_dt
    metadata["features"]["mesh_pos"]["shape"] = [n_frames, -1, spatial_dim]
    metadata["features"]["cells"]["shape"] = [n_frames, -1, cell_elem_size]
    metadata["features"]["node_type"]["shape"] = [n_frames, -1, 1]
    
    # ダイナミックフィールドを追加
    # 学習時の特徴量連結順を安定化するため、U(velocity) -> p(pressure) を優先
    filtered_fields = [field for field in all_fields if field != "nut"]
    ordered_fields: List[str] = []
    for preferred in ["U", "velocity", "p", "pressure"]:
        if preferred in filtered_fields and preferred not in ordered_fields:
            ordered_fields.append(preferred)
    for field in sorted(filtered_fields):
        if field not in ordered_fields:
            ordered_fields.append(field)

    for field_name in ordered_fields:
        dim = inferred_field_dims.get(field_name, 1)
        if field_name in {"U", "velocity"}:
            dim = 2
        metadata["features"][field_name] = {
            "type": "dynamic",
            "shape": [n_frames, -1, dim],
            "dtype": "float32"
        }
    
    metadata["field_names"] = ordered_fields
    
    # メタデータを保存
    if meta_output_path:
        os.makedirs(Path(meta_output_path).parent, exist_ok=True)
        with open(meta_output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {meta_output_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert VTU files to learning HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--case_dirs",
        type=str,
        nargs="+",
        help="Case directory paths (receives glob patterns)"
    )
    parser.add_argument(
        "--case_root",
        type=str,
        help="Root directory for cases (used with --case_numbers)"
    )
    parser.add_argument(
        "--case_numbers",
        type=str,
        help="Case numbers or range (e.g., '1 2 3' or 'range(1,6)')"
    )
    parser.add_argument(
        "--vtu_pattern",
        type=str,
        default="**/*.vtu",
        help="VTU file search pattern (default: **/*.vtu)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training data ratio"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="Validation data ratio"
    )
    parser.add_argument(
        "--create_meta",
        action="store_true",
        help="Create metadata JSON files"
    )
    
    args = parser.parse_args()
    
    # ケースディレクトリを決定
    if args.case_dirs:
        case_dirs = []
        for pattern in args.case_dirs:
            expanded = glob(pattern, recursive=True)
            case_dirs.extend(sorted(expanded))
    elif args.case_root and args.case_numbers:
        try:
            nums = _parse_case_numbers(args.case_numbers)
        except Exception as e:
            print(f"Error parsing case_numbers: {e}")
            return
        
        case_dirs = [
            str(Path(args.case_root) / f"case{num}")
            for num in nums
        ]
    else:
        parser.print_help()
        return
    
    print(f"Found {len(case_dirs)} case directories")
    
    # VTU ファイルを収集
    case_vtu_list = collect_vtu_files_from_cases(case_dirs, args.vtu_pattern)
    total_vtu = sum(len(vtus) for _, vtus in case_vtu_list)
    print(f"Collected {total_vtu} VTU files from {len(case_vtu_list)} cases")
    
    # train/valid/test に分割
    n_trajectories = len(case_vtu_list)
    n_train = int(n_trajectories * args.train_ratio)
    n_valid = int(n_trajectories * args.valid_ratio)
    
    train_list = case_vtu_list[:n_train]
    valid_list = case_vtu_list[n_train:n_train + n_valid]
    test_list = case_vtu_list[n_train + n_valid:]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nGenerating HDF5 files...")
    print(f"  Train: {len(train_list)} trajectories")
    print(f"  Valid: {len(valid_list)} trajectories")
    print(f"  Test:  {len(test_list)} trajectories")
    
    # train.h5
    if train_list:
        train_path = os.path.join(args.output_dir, "train.h5")
        train_meta_path = os.path.join(args.output_dir, "train_meta.json") if args.create_meta else None
        create_trajectory_h5(train_path, train_list, train_meta_path)
    
    # valid.h5
    if valid_list:
        valid_path = os.path.join(args.output_dir, "valid.h5")
        valid_meta_path = os.path.join(args.output_dir, "valid_meta.json") if args.create_meta else None
        create_trajectory_h5(valid_path, valid_list, valid_meta_path)
    
    # test.h5
    if test_list:
        test_path = os.path.join(args.output_dir, "test.h5")
        test_meta_path = os.path.join(args.output_dir, "test_meta.json") if args.create_meta else None
        create_trajectory_h5(test_path, test_list, test_meta_path)
    
    print(f"\n✓ Done! Output in {args.output_dir}")


if __name__ == "__main__":
    main()
