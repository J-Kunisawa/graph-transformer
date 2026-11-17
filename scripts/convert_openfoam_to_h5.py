"""
OpenFOAM結果をHDF5形式に変換するスクリプト

このスクリプトはOpenFOAMのシミュレーション結果を読み込み、
graph-physicsプロジェクトで使用できるHDF5形式に変換します。

必要なパッケージ:
- h5py
- numpy
- pyvista (OpenFOAMデータの読み込み用)

使用方法:
    python convert_openfoam_to_h5.py --openfoam_dir /path/to/openfoam/case --output_path output.h5

OpenFOAMケースの構造:
    case/
    ├── constant/
    │   └── polyMesh/
    ├── system/
    └── [time_directories]/
        ├── U (速度場)
        ├── p (圧力場)
        └── ...
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyvista as pv
from loguru import logger


def _ensure_foam_file(case_dir: str) -> str:
    """ケース内の .foam ファイルパスを返す（無ければ作成）。"""
    foam_files = [f for f in os.listdir(case_dir) if f.endswith(".foam")]
    if foam_files:
        return os.path.join(case_dir, foam_files[0])

    foam_path = os.path.join(case_dir, f"{os.path.basename(case_dir)}.foam")
    open(foam_path, "a").close()
    logger.info(f"Created .foam file: {foam_path}")
    return foam_path


def _extract_internal_mesh(mesh_obj):
    """MultiBlock から内部メッシュを安全に抽出。"""
    if not isinstance(mesh_obj, pv.MultiBlock):
        return mesh_obj

    for i in range(mesh_obj.n_blocks):
        block_name = mesh_obj.get_block_name(i)
        if isinstance(block_name, str) and "internal" in block_name.lower():
            return mesh_obj[i]

    return mesh_obj[0]


def _build_openfoam_reader(case_dir: str):
    """環境差を吸収してOpenFOAM readerを作成。"""
    foam_path = _ensure_foam_file(case_dir)
    reader = pv.get_reader(foam_path)
    return reader


def _set_reader_time(reader, time_dir: str) -> None:
    """pyvistaバージョン差に対応してreader時刻を設定。"""
    target = float(time_dir)
    time_values = getattr(reader, "time_values", None)
    if time_values is not None and len(time_values) > 0:
        nearest = min(time_values, key=lambda x: abs(float(x) - target))
    else:
        nearest = target

    if hasattr(reader, "set_active_time_value"):
        reader.set_active_time_value(float(nearest))
        return

    if hasattr(reader, "set_active_time_point") and time_values is not None and len(time_values) > 0:
        nearest_idx = int(np.argmin(np.abs(np.asarray(time_values, dtype=float) - float(nearest))))
        reader.set_active_time_point(nearest_idx)



def read_openfoam_mesh(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenFOAMメッシュを読み込む
    
    Args:
        case_dir: OpenFOAMケースのディレクトリパス
        
    Returns:
        points: メッシュの節点座標 (N, 3)
        cells: メッシュのセル情報 (M, cell_size)
    """
    logger.info(f"Reading OpenFOAM mesh from {case_dir}")
    
    foam_path = _ensure_foam_file(case_dir)
    logger.info(f"Using .foam file: {foam_path}")
    mesh = pv.read(foam_path)
    mesh = _extract_internal_mesh(mesh)
    
    points = np.array(mesh.points)
    
    # セル情報を取得
    cells_list = []
    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        cells_list.append(cell.point_ids)
    
    # セルのサイズが統一されているか確認
    cell_sizes = [len(cell) for cell in cells_list]
    if len(set(cell_sizes)) == 1:
        cells = np.array(cells_list)
    else:
        # セルサイズが異なる場合、最大サイズに合わせてパディング
        max_size = max(cell_sizes)
        cells = np.full((len(cells_list), max_size), -1, dtype=np.int32)
        for i, cell in enumerate(cells_list):
            cells[i, :len(cell)] = cell
    
    # 2Dメッシュの場合、Z座標を削除
    if np.allclose(points[:, 2], points[0, 2]):
        points = points[:, :2]
        logger.info("Detected 2D mesh, removing Z coordinate")
    
    logger.info(f"Mesh loaded: {points.shape[0]} points, {cells.shape[0]} cells")
    
    return points, cells


def read_openfoam_field(case_dir: str, time_dir: str, field_name: str) -> Optional[np.ndarray]:
    """
    OpenFOAMの場データを読み込む
    
    Args:
        case_dir: OpenFOAMケースのディレクトリパス
        time_dir: 時刻ディレクトリ名
        field_name: 読み込むフィールド名 (U, p など)
        
    Returns:
        field_data: フィールドデータ (N, dim) または None
    """
    field_path = os.path.join(case_dir, time_dir, field_name)
    
    if not os.path.exists(field_path):
        logger.warning(f"Field {field_name} not found in {time_dir}")
        return None
    
    try:
        # PyVistaでOpenFOAMケースを読み込む
        reader = _build_openfoam_reader(case_dir)
        _set_reader_time(reader, time_dir)

        # 不要な場（例: nut など）がメッシュと点数不一致でエラーを出すのを防ぐため、
        # 読み込み対象を必要なフィールドのみに絞り込む
        if hasattr(reader, "disable_all_point_arrays"):
            reader.disable_all_point_arrays()
        if hasattr(reader, "disable_all_cell_arrays"):
            reader.disable_all_cell_arrays()
        if hasattr(reader, "enable_point_array"):
            reader.enable_point_array(field_name)
        if hasattr(reader, "enable_cell_array"):
            reader.enable_cell_array(field_name)

        mesh = reader.read()
        
        # 内部メッシュを取得
        mesh = _extract_internal_mesh(mesh)
        
        # フィールドデータを取得
        if field_name in mesh.point_data:
            field_data = np.array(mesh.point_data[field_name])
        elif field_name in mesh.cell_data:
            # セルデータを節点データに変換
            field_data = mesh.cell_data_to_point_data()[field_name]
            field_data = np.array(field_data)
        else:
            logger.warning(f"Field {field_name} not found in mesh data")
            return None
        
        # 2Dデータの場合、Z成分を削除
        if field_data.ndim == 2 and field_data.shape[1] == 3:
            if np.allclose(field_data[:, 2], 0):
                field_data = field_data[:, :2]
                logger.debug(f"Removed Z component from {field_name}")
        
        logger.debug(f"Loaded field {field_name} from {time_dir}: shape {field_data.shape}")
        return field_data
        
    except Exception as e:
        logger.error(f"Error reading field {field_name} from {time_dir}: {e}")
        return None


def get_time_directories(case_dir: str) -> List[str]:
    """
    OpenFOAMケースから時刻ディレクトリのリストを取得
    
    Args:
        case_dir: OpenFOAMケースのディレクトリパス
        
    Returns:
        time_dirs: 時刻ディレクトリ名のリスト（ソート済み）
    """
    time_dirs = []
    
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path):
            try:
                # 数値として解釈できるディレクトリが時刻ディレクトリ
                float(item)
                time_dirs.append(item)
            except ValueError:
                continue
    
    # 時刻順にソート
    time_dirs.sort(key=lambda x: float(x))
    
    logger.info(f"Found {len(time_dirs)} time directories")
    return time_dirs


def determine_node_types(points: np.ndarray, mesh: pv.UnstructuredGrid) -> np.ndarray:
    """
    ノードタイプを決定する (境界条件に基づく)
    
    Args:
        points: メッシュの節点座標
        mesh: PyVistaメッシュオブジェクト
        
    Returns:
        node_types: ノードタイプ配列 (N, 1)
    """
    n_points = points.shape[0]
    node_types = np.zeros((n_points, 1), dtype=np.int32)
    
    # 境界上のノードを識別
    # 簡易的な実装: メッシュの外側の点を境界とする
    try:
        boundary = mesh.extract_surface()
        boundary_point_ids = boundary.point_data.get('vtkOriginalPointIds')
        
        if boundary_point_ids is not None:
            # 0: 通常ノード, 5: 境界ノード
            node_types[boundary_point_ids] = 5
            logger.info(f"Marked {len(boundary_point_ids)} boundary nodes")
    except Exception as e:
        logger.warning(f"Could not determine boundary nodes: {e}")
    
    return node_types


def convert_openfoam_to_h5(
    openfoam_cases: List[str],
    output_path: str,
    fields: List[str] = ["U", "p"],
    dt: Optional[float] = None,
) -> None:
    """
    OpenFOAMケースをHDF5形式に変換
    
    Args:
        openfoam_cases: OpenFOAMケースディレクトリのリスト
        output_path: 出力HDF5ファイルのパス
        fields: 読み込むフィールド名のリスト
        dt: タイムステップ (Noneの場合は自動検出)
    """
    logger.info(f"Converting {len(openfoam_cases)} OpenFOAM cases to {output_path}")

    # graph-physics で利用されるフィールド名に統一
    # OpenFOAM: U, p -> HDF5: velocity, pressure
    field_name_map = {
        "U": "velocity",
        "p": "pressure",
    }
    output_fields = [field_name_map.get(name, name) for name in fields]
    
    with h5py.File(output_path, "w") as h5f:
        
        for case_idx, case_dir in enumerate(openfoam_cases):
            logger.info(f"Processing case {case_idx}: {case_dir}")
            
            # タイムディレクトリを取得
            time_dirs = get_time_directories(case_dir)
            
            if len(time_dirs) == 0:
                logger.warning(f"No time directories found in {case_dir}, skipping")
                continue
            
            # タイムステップを計算
            if dt is None and len(time_dirs) >= 2:
                dt_calculated = float(time_dirs[1]) - float(time_dirs[0])
                logger.info(f"Auto-detected dt = {dt_calculated}")
            else:
                dt_calculated = dt if dt is not None else 1.0
            
            # メッシュを読み込む
            points, cells = read_openfoam_mesh(case_dir)
            
            # 最初の時刻のメッシュを読んでnode_types用のmeshオブジェクトを取得
            foam_path = _ensure_foam_file(case_dir)
            mesh = pv.read(foam_path)
            mesh = _extract_internal_mesh(mesh)
            
            node_types = determine_node_types(points, mesh)
            
            # 軌跡グループを作成
            traj_name = f"{case_idx:02d}"
            traj_group = h5f.create_group(traj_name)
            
            # 静的データを保存
            n_frames = len(time_dirs)
            
            # mesh_pos: (n_frames, n_points, dim)
            mesh_pos_data = np.tile(points[np.newaxis, :, :], (n_frames, 1, 1))
            traj_group.create_dataset("mesh_pos", data=mesh_pos_data.astype(np.float32))
            
            # cells: (n_frames, n_cells, cell_size)
            cells_data = np.tile(cells[np.newaxis, :, :], (n_frames, 1, 1))
            traj_group.create_dataset("cells", data=cells_data.astype(np.int32))
            
            # node_type: (n_frames, n_points, 1)
            node_type_data = np.tile(node_types[np.newaxis, :, :], (n_frames, 1, 1))
            traj_group.create_dataset("node_type", data=node_type_data.astype(np.int32))
            
            # 各時刻のフィールドデータを読み込む
            field_data_dict = {field: [] for field in output_fields}
            expected_points = points.shape[0]
            expected_shape = {}  # 各フィールドの期待形状を記録
            
            for time_idx, time_dir in enumerate(time_dirs):
                logger.debug(f"Reading time {time_dir} ({time_idx+1}/{n_frames})")
                
                for field_name in fields:
                    field_data = read_openfoam_field(case_dir, time_dir, field_name)
                    output_name = field_name_map.get(field_name, field_name)
                    
                    if field_data is not None:
                        # 1次元データを2次元に変換 (n_points,) -> (n_points, 1)
                        if field_data.ndim == 1:
                            field_data = field_data.reshape(-1, 1)
                        
                        # 最初のタイムステップで期待形状を決定
                        if output_name not in expected_shape:
                            expected_shape[output_name] = (expected_points, field_data.shape[1])
                            logger.info(
                                f"Field {output_name} expected shape: {expected_shape[output_name]}"
                            )

                        target_shape = expected_shape[output_name]
                        
                        # 点数が異なる場合の処理
                        if field_data.shape[0] != target_shape[0]:
                            logger.warning(
                                f"[{case_dir}] time={time_dir} field={output_name} "
                                f"points {field_data.shape[0]} != expected {target_shape[0]}; "
                                "auto-fixing by trim/pad"
                            )
                            if field_data.shape[0] > target_shape[0]:
                                field_data = field_data[:target_shape[0]]
                            else:
                                pad_rows = target_shape[0] - field_data.shape[0]
                                pad = np.zeros((pad_rows, field_data.shape[1]), dtype=field_data.dtype)
                                field_data = np.concatenate([field_data, pad], axis=0)
                        
                        # 次元数が異なる場合の処理
                        if field_data.shape[1] != target_shape[1]:
                            logger.warning(
                                f"[{case_dir}] time={time_dir} field={output_name} "
                                f"channels {field_data.shape[1]} != expected {target_shape[1]}; "
                                "resizing to match"
                            )
                            if field_data.shape[1] > target_shape[1]:
                                field_data = field_data[:, :target_shape[1]]
                            else:
                                pad_cols = target_shape[1] - field_data.shape[1]
                                pad = np.zeros((field_data.shape[0], pad_cols), dtype=field_data.dtype)
                                field_data = np.concatenate([field_data, pad], axis=1)
                        
                        field_data_dict[output_name].append(field_data.astype(np.float32))
                    else:
                        # データがない場合はゼロで埋める
                        if output_name in expected_shape:
                            shape = expected_shape[output_name]
                        else:
                            shape = (expected_points, 1)
                            expected_shape[output_name] = shape
                        field_data_dict[output_name].append(
                            np.zeros(shape, dtype=np.float32)
                        )
            
            # フィールドデータをHDF5に保存
            for field_name, field_list in field_data_dict.items():
                if len(field_list) > 0:
                    # すべて同じ形状になっているはずなので、スタック可能
                    try:
                        field_array = np.stack(field_list, axis=0)
                        traj_group.create_dataset(field_name, data=field_array)
                        logger.info(f"Saved field {field_name}: shape {field_array.shape}")
                    except ValueError as e:
                        logger.error(f"Failed to stack field {field_name}: {e}")
                        logger.error(f"Shapes in list: {[f.shape for f in field_list[:5]]}...")
                        raise
    
    logger.info(f"Successfully converted data to {output_path}")


def create_metadata_json(
    output_path: str,
    num_frames: int,
    spatial_dim: int = 2,
    cell_size: int = 3,
    dt: float = 0.01,
    fields: List[str] = ["U", "p"],
    field_dims_override: Optional[Dict[str, int]] = None,
) -> None:
    """
    メタデータJSONファイルを作成
    
    Args:
        output_path: 出力JSONファイルのパス
        num_frames: フレーム数
        spatial_dim: 空間次元 (2 or 3)
        dt: タイムステップ
        fields: フィールド名のリスト (例: ["U", "p"])
    """
    # フィールドの次元を決定
    field_dims = {}
    field_name_map = {
        "U": "velocity",
        "p": "pressure",
    }
    for field in fields:
        output_name = field_name_map.get(field, field)
        if field == "U":  # 速度
            field_dims[output_name] = spatial_dim
        elif field == "p":  # 圧力
            field_dims[output_name] = 1
        else:
            # デフォルトはスカラー
            field_dims[output_name] = 1

    # 実データから推定した次元があれば優先
    if field_dims_override:
        for name, dim in field_dims_override.items():
            if isinstance(dim, int) and dim > 0:
                field_dims[name] = dim
    
    output_fields = [field_name_map.get(name, name) for name in fields]

    metadata = {
        "dt": dt,
        "trajectory_length": num_frames,
        "features": {
            "cells": {
                "type": "static",
                "shape": [num_frames, -1, cell_size],
                "dtype": "int32"
            },
            "mesh_pos": {
                "type": "static",
                "shape": [num_frames, -1, spatial_dim],
                "dtype": "float32"
            },
            "node_type": {
                "type": "static",
                "shape": [num_frames, -1, 1],
                "dtype": "int32"
            }
        },
        "field_names": output_fields
    }
    
    # 動的フィールドを追加（すべて [T, N, C] で統一）
    for field, dim in field_dims.items():
        shape = [num_frames, -1, dim]
        metadata["features"][field] = {
            "type": "dynamic",
            "shape": shape,
            "dtype": "float32"
        }
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created metadata file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenFOAM simulation results to HDF5 format"
    )
    parser.add_argument(
        "--openfoam_cases",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to OpenFOAM case directory(ies)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["U", "p"],
        help="Fields to extract (default: U p)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step (auto-detected if not specified)",
    )
    parser.add_argument(
        "--create_meta",
        action="store_true",
        help="Create metadata JSON file",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Metadata JSON file path (default: output_path with .json extension)",
    )
    
    args = parser.parse_args()
    
    # OpenFOAMデータをHDF5に変換
    convert_openfoam_to_h5(
        openfoam_cases=args.openfoam_cases,
        output_path=args.output_path,
        fields=args.fields,
        dt=args.dt,
    )
    
    # メタデータファイルを作成
    if args.create_meta:
        # 最初のケースから情報を取得
        time_dirs = get_time_directories(args.openfoam_cases[0])
        num_frames = len(time_dirs)
        
        # 空間次元を検出
        points, cells = read_openfoam_mesh(args.openfoam_cases[0])
        spatial_dim = points.shape[1]
        cell_size = cells.shape[1]
        
        # タイムステップ
        if args.dt is None and len(time_dirs) >= 2:
            dt = float(time_dirs[1]) - float(time_dirs[0])
        else:
            dt = args.dt if args.dt is not None else 0.01
        
        meta_path = args.meta_path or args.output_path.replace(".h5", "_meta.json")
        
        # 動的フィールド次元を実データから推定（例: 3D座標でもUは2Dになるケースに対応）
        inferred_dims: Dict[str, int] = {}
        if len(time_dirs) > 0:
            first_time = time_dirs[0]
            field_name_map = {"U": "velocity", "p": "pressure"}
            for field in args.fields:
                arr = read_openfoam_field(args.openfoam_cases[0], first_time, field)
                if arr is None:
                    continue
                if arr.ndim == 1:
                    dim = 1
                elif arr.ndim >= 2:
                    dim = int(arr.shape[1])
                else:
                    dim = 1
                inferred_dims[field_name_map.get(field, field)] = dim

        create_metadata_json(
            output_path=meta_path,
            num_frames=num_frames,
            spatial_dim=spatial_dim,
            cell_size=cell_size,
            dt=dt,
            fields=args.fields,
            field_dims_override=inferred_dims,
        )


if __name__ == "__main__":
    main()
