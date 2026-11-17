"""
OpenFOAM結果をXDMF形式に変換するスクリプト（XDMFDataset用）

このスクリプトはOpenFOAMシミュレーション結果を読み込み、
XDMFDatasetクラスで使用できる形式に変換します。

各OpenFOAMケースは個別のXDMFファイルとして保存されます。

使用方法:
    python convert_openfoam_to_xdmf.py --openfoam_cases /path/to/case1 /path/to/case2 \
                                       --output_dir data/openfoam_xdmf/train \
                                       --fields U p

必要なパッケージ:
- meshio
- pyvista
- numpy
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import meshio
import numpy as np
import pyvista as pv
from loguru import logger


def get_time_directories(case_dir: str) -> List[str]:
    """OpenFOAMケースから時刻ディレクトリのリストを取得"""
    time_dirs = []
    
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path):
            try:
                float(item)
                time_dirs.append(item)
            except ValueError:
                continue
    
    time_dirs.sort(key=lambda x: float(x))
    logger.info(f"Found {len(time_dirs)} time directories in {case_dir}")
    return time_dirs


def convert_openfoam_case_to_xdmf(
    case_dir: str,
    output_path: str,
    fields: List[str] = ["U", "p"],
) -> Optional[float]:
    """
    1つのOpenFOAMケースをXDMF形式に変換
    
    Args:
        case_dir: OpenFOAMケースディレクトリ
        output_path: 出力XDMFファイルパス（拡張子なし）
        fields: 抽出するフィールド名
        
    Returns:
        dt: タイムステップ（Noneの場合は変換失敗）
    """
    logger.info(f"Converting {case_dir} to {output_path}.xdmf")
    
    try:
        # PyVistaでOpenFOAMケースを読み込む
        reader = pv.OpenFOAMReader(case_dir)
        
        # 利用可能な時刻を取得
        time_values = reader.time_values
        if len(time_values) == 0:
            logger.warning(f"No time values found in {case_dir}")
            return None
        
        logger.info(f"Found {len(time_values)} time steps")
        
        # タイムステップを計算
        if len(time_values) >= 2:
            dt = time_values[1] - time_values[0]
        else:
            dt = 1.0
        
        # XDMFファイルとして書き出す
        xdmf_path = f"{output_path}.xdmf"
        
        with meshio.xdmf.TimeSeriesWriter(xdmf_path) as writer:
            for time_idx, time_val in enumerate(time_values):
                logger.debug(f"Processing time step {time_idx+1}/{len(time_values)}: t={time_val}")
                
                # 時刻を設定してメッシュを読み込む
                reader.set_active_time_value(time_val)
                mesh_data = reader.read()
                
                # 内部メッシュを取得
                if isinstance(mesh_data, pv.MultiBlock):
                    for i in range(mesh_data.n_blocks):
                        if "internal" in mesh_data.get_block_name(i).lower():
                            mesh_data = mesh_data[i]
                            break
                    else:
                        mesh_data = mesh_data[0]
                
                # meshioのメッシュ形式に変換
                points = np.array(mesh_data.points)
                
                # 2Dメッシュの場合、Z座標を削除
                if np.allclose(points[:, 2], points[0, 2]):
                    points = points[:, :2]
                
                # セル情報を取得
                cells_dict = {}
                for i in range(mesh_data.n_cells):
                    cell = mesh_data.get_cell(i)
                    cell_type = cell.type
                    
                    # PyVistaのセルタイプをmeshioのセルタイプに変換
                    if cell_type == pv.CellType.TRIANGLE:
                        cell_name = "triangle"
                    elif cell_type == pv.CellType.QUAD:
                        cell_name = "quad"
                    elif cell_type == pv.CellType.TETRA:
                        cell_name = "tetra"
                    elif cell_type == pv.CellType.HEXAHEDRON:
                        cell_name = "hexahedron"
                    else:
                        continue
                    
                    if cell_name not in cells_dict:
                        cells_dict[cell_name] = []
                    cells_dict[cell_name].append(cell.point_ids)
                
                # リストを配列に変換
                cells = []
                for cell_name, cell_list in cells_dict.items():
                    cells.append((cell_name, np.array(cell_list)))
                
                # ポイントデータを取得
                point_data = {}
                for field_name in fields:
                    if field_name in mesh_data.point_data:
                        field_data = np.array(mesh_data.point_data[field_name])
                        
                        # 2Dの場合、ベクトルフィールドのZ成分を削除
                        if field_data.ndim == 2 and field_data.shape[1] == 3:
                            if points.shape[1] == 2:  # 2Dメッシュ
                                field_data = field_data[:, :2]
                        
                        point_data[field_name] = field_data
                    elif field_name in mesh_data.cell_data:
                        # セルデータを節点データに変換
                        mesh_data_with_point = mesh_data.cell_data_to_point_data()
                        field_data = np.array(mesh_data_with_point.point_data[field_name])
                        
                        if field_data.ndim == 2 and field_data.shape[1] == 3:
                            if points.shape[1] == 2:
                                field_data = field_data[:, :2]
                        
                        point_data[field_name] = field_data
                
                # meshioメッシュを作成
                mesh = meshio.Mesh(
                    points=points,
                    cells=cells,
                    point_data=point_data,
                )
                
                # XDMFに書き込み
                writer.write_points_cells(mesh.points, mesh.cells)
                writer.write_data(time_val, point_data=mesh.point_data)
        
        logger.info(f"Successfully created {xdmf_path}")
        return dt
        
    except Exception as e:
        logger.error(f"Error converting {case_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_multiple_cases_to_xdmf(
    openfoam_cases: List[str],
    output_dir: str,
    fields: List[str] = ["U", "p"],
    case_prefix: str = "case",
) -> float:
    """
    複数のOpenFOAMケースをXDMF形式に変換
    
    Args:
        openfoam_cases: OpenFOAMケースディレクトリのリスト
        output_dir: 出力ディレクトリ
        fields: 抽出するフィールド名
        case_prefix: ケース名のプレフィックス
        
    Returns:
        dt: 平均タイムステップ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dt_values = []
    
    for case_idx, case_dir in enumerate(openfoam_cases):
        output_path = os.path.join(output_dir, f"{case_prefix}_{case_idx:02d}")
        
        dt = convert_openfoam_case_to_xdmf(
            case_dir=case_dir,
            output_path=output_path,
            fields=fields,
        )
        
        if dt is not None:
            dt_values.append(dt)
    
    if len(dt_values) == 0:
        logger.error("No cases were successfully converted")
        return 1.0
    
    avg_dt = np.mean(dt_values)
    logger.info(f"Successfully converted {len(dt_values)}/{len(openfoam_cases)} cases")
    logger.info(f"Average time step: {avg_dt}")
    
    return avg_dt


def create_metadata_for_xdmf(
    output_path: str,
    sample_xdmf_path: str,
    fields: List[str] = ["U", "p"],
    dt: float = 0.01,
) -> None:
    """
    XDMFデータセット用のメタデータJSONファイルを作成
    
    Args:
        output_path: 出力JSONファイルのパス
        sample_xdmf_path: サンプルXDMFファイル（形状情報を取得）
        fields: フィールド名のリスト
        dt: タイムステップ
    """
    # サンプルXDMFファイルから情報を取得
    try:
        with meshio.xdmf.TimeSeriesReader(sample_xdmf_path) as reader:
            num_steps = reader.num_steps
            points, cells = reader.read_points_cells()
            spatial_dim = points.shape[1]
            
        logger.info(f"Detected {num_steps} time steps, spatial dimension: {spatial_dim}")
    except Exception as e:
        logger.warning(f"Could not read sample XDMF file: {e}")
        num_steps = 600
        spatial_dim = 2
    
    # フィールドの次元を決定
    field_dims = {}
    for field in fields:
        if field == "U":  # 速度
            field_dims[field] = spatial_dim
        elif field == "p":  # 圧力
            field_dims[field] = 1
        else:
            field_dims[field] = 1
    
    metadata = {
        "dt": dt,
        "trajectory_length": num_steps,
        "features": {}
    }
    
    # 動的フィールドを追加
    for field, dim in field_dims.items():
        metadata["features"][field] = {
            "type": "dynamic",
            "shape": [-1, dim],
            "dtype": "float32"
        }
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created metadata file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenFOAM cases to XDMF format for XDMFDataset"
    )
    parser.add_argument(
        "--openfoam_cases",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to OpenFOAM case directory(ies)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for XDMF files",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["U", "p"],
        help="Fields to extract (default: U p)",
    )
    parser.add_argument(
        "--case_prefix",
        type=str,
        default="case",
        help="Prefix for output case files (default: case)",
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
        help="Metadata JSON file path",
    )
    
    args = parser.parse_args()
    
    # OpenFOAMケースをXDMFに変換
    dt = convert_multiple_cases_to_xdmf(
        openfoam_cases=args.openfoam_cases,
        output_dir=args.output_dir,
        fields=args.fields,
        case_prefix=args.case_prefix,
    )
    
    # メタデータファイルを作成
    if args.create_meta:
        # 最初のXDMFファイルをサンプルとして使用
        sample_xdmf = os.path.join(args.output_dir, f"{args.case_prefix}_00.xdmf")
        
        if not os.path.exists(sample_xdmf):
            logger.error("No XDMF files were created, cannot generate metadata")
            return
        
        meta_path = args.meta_path or os.path.join(
            os.path.dirname(args.output_dir),
            "openfoam_xdmf_meta.json"
        )
        
        create_metadata_for_xdmf(
            output_path=meta_path,
            sample_xdmf_path=sample_xdmf,
            fields=args.fields,
            dt=dt,
        )


if __name__ == "__main__":
    main()
