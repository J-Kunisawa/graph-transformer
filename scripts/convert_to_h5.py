import json
import random
import re
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==============================================================================
#  ユーザー設定
# ==============================================================================
VTK_BASE_DIR = Path("data/data_vtu/ver5/re_80-400_20")
OUTPUT_TRAIN_H5 = Path("data/data_h5/re_80-400_20_test.h5")
OUTPUT_VAL_H5 = Path("data/data_h5/re_80-400_20_val.h5")
OUTPUT_META_PATH = Path("data/data_h5/re_80-400_20_meta.json")

TRAIN_RATIO = 1.0        # 訓練データの割合 (80%)
SHUFFLE_CASES = False    # シミュレーションケース単位のシャッフル (時系列は保持)
DT_VALUE = 0.01          # サンプリング時間間隔 (秒)
# ==============================================================================


def extract_2d_mesh(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    """OpenFOAMの3Dメッシュ(1セル厚み)からZ中央断面をスライスし、綺麗な2D三角形メッシュを抽出する"""
    if abs(mesh.bounds[5] - mesh.bounds[4]) < 1e-6:
        return mesh.extract_surface().triangulate()
    z_mid = (mesh.bounds[4] + mesh.bounds[5]) / 2.0
    return mesh.slice(normal="z", origin=(0, 0, z_mid)).triangulate()


def get_vtu_files_ordered(case_dir: Path) -> List[Path]:
    """物理時刻の昇順でVTUファイルを厳密にソートする (時系列崩壊・巻き戻り防止)"""
    vtu_files = list(case_dir.glob("**/*.vtu"))
    if not vtu_files:
        return []

    def extract_time(filepath: Path) -> float:
        # 1. VTUファイル内部の TimeValue メタデータ (最優先・Ground Truth)
        try:
            mesh = pv.read(filepath)
            for key in ["TimeValue", "time", "Time", "t"]:
                for data_source in [mesh.field_data, mesh.point_data]:
                    if key in data_source:
                        val = np.asarray(data_source[key]).flatten()
                        if val.size > 0 and not np.isnan(val[0]):
                            return float(val[0])
        except Exception:
            pass

        # 2. 親フォルダ名からの小数パース (例: case1_0.01/internal.vtu)
        match = re.search(r"_(\d+\.?\d*)$", filepath.parent.name) or re.search(r"(\d+\.?\d*)", filepath.parent.name)
        if match:
            try: return float(match.group(1))
            except ValueError: pass

        # 3. ファイル名からの小数パース (例: case_0.010000.vtu)
        match = re.search(r"(\d+\.?\d*)", filepath.name)
        return float(match.group(1)) if match else 0.0

    return sorted(vtu_files, key=lambda f: (extract_time(f), str(f)))


def assign_node_types(base_mesh: pv.PolyData, case_dir: Path) -> np.ndarray:
    """boundary/*.vtp ファイルから KDTree で近傍探索し、正確なノードタイプID配列を構築する"""
    points_2d = base_mesh.points[:, :2]
    node_types = np.zeros((len(points_2d), 1), dtype=np.int32)
    if len(points_2d) == 0:
        return node_types

    boundary_dir = case_dir / "boundary"
    if not boundary_dir.exists():
        boundary_dir = case_dir.parent / "boundary"
    if not boundary_dir.exists():
        return node_types

    tree = cKDTree(points_2d)
    # 割り当てルール (ファイル名小文字 : 正式な NodeType ID)
    # 優先順位の低い順に評価し、後から一致した重要なパッチで上書きします
    rules: List[Tuple[List[str], int]] = [
        # 6: WALL_BOUNDARY (風洞の上下壁など。大域的アテンションの要！)
        (["wall", "walls", "upper", "lower", "top", "bottom"], 6), 
        
        # 4: INFLOW (流入口。流速規定境界)
        (["inlet", "inflow"], 4),                                  
        
        # 5: OUTFLOW (流出口。自由流出境界)
        (["outlet", "outflow"], 5),                                
        
        # 2: AIRFOIL (翼型シミュレーション用・今回該当があれば)
        (["airfoil", "wing"], 2),                                  
        
        # 1: OBSTACLE (円柱などの障害物表面。最も重要！)
        (["obstacle", "cylinder", "body"], 1)                      
    ]

    for vtp_file in boundary_dir.glob("*.vtp"):
        fname_lower = vtp_file.stem.lower()
        target_id = next((tid for keywords, tid in rules if any(kw in fname_lower for kw in keywords)), None)
        if target_id is None:
            continue

        try:
            b_points = pv.read(vtp_file).points[:, :2]
            if len(b_points) > 0:
                dists, idxs = tree.query(b_points, k=1)
                node_types[idxs[dists < 1e-4]] = target_id
        except Exception as e:
            print(f"⚠️ 警告 ({vtp_file.name}): {e}")

    return node_types


def extract_field(mesh: pv.DataSet, names: List[str]) -> Tuple[str, np.ndarray]:
    """指定された候補名から物理量データセットを取り出し、適切な2次元配列として返す"""
    for name in names:
        if name in mesh.point_data:
            data = mesh.point_data[name]
            return name, data[:, np.newaxis] if data.ndim == 1 else data
    raise KeyError(f"必要な物理量が見つかりません: {names}")


def save_dataset(group: h5py.Group, name: str, data: np.ndarray, dtype: np.dtype):
    """同名データセットの衝突を回避して安全に保存する"""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data.astype(dtype))


def process_and_save(case_dirs: List[Path], output_h5: Path, desc: str):
    """シミュレーションデータを GNN 標準仕様に準拠した HDF5 として一括出力する"""
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as h5_file:
        for case_idx, case_dir in enumerate(tqdm(case_dirs, desc=desc)):
            vtu_files = get_vtu_files_ordered(case_dir)
            if not vtu_files:
                continue

            num_steps = len(vtu_files)
            sim_group = h5_file.create_group(f"sim_{case_idx:05d}")

            # 1. 動的物理量 (U, p) の時系列抽出
            u_list, p_list = [], []
            for vtu in vtu_files:
                mesh_2d = extract_2d_mesh(pv.read(vtu))
                u_name, u_data = extract_field(mesh_2d, ["U", "velocity"])
                p_name, p_data = extract_field(mesh_2d, ["p", "pressure"])
                u_list.append(u_data[:, :2])  # 2D速度成分のみ抽出
                p_list.append(p_data)

            save_dataset(sim_group, u_name, np.array(u_list), np.float32)
            save_dataset(sim_group, p_name, np.array(p_list), np.float32)

            # 2. 静的構造 (mesh_pos, cells, node_type) の時間方向へのブロードキャスト
            base_mesh = extract_2d_mesh(pv.read(vtu_files[0]))
            
            # 座標 [T, N, 2]
            pos_expanded = np.repeat(base_mesh.points[np.newaxis, :, :2], num_steps, axis=0)
            save_dataset(sim_group, "mesh_pos", pos_expanded, np.float32)

            # セル接続 [T, C, 3] (三角形要素)
            cells_single = base_mesh.faces.reshape(-1, 4)[:, 1:]
            cells_expanded = np.repeat(cells_single[np.newaxis, ...], num_steps, axis=0)
            save_dataset(sim_group, "cells", cells_expanded, np.int32)

            # ノードタイプ [T, N, 1]
            nt_single = assign_node_types(base_mesh, case_dir)
            if case_idx == 0:
                unique, counts = np.unique(nt_single, return_counts=True)
                print(f"\n📊 ノードタイプ構成比 [{case_dir.name}]: {dict(zip(unique, counts))}")
                
            nt_expanded = np.repeat(nt_single[np.newaxis, :, :], num_steps, axis=0)
            save_dataset(sim_group, "node_type", nt_expanded, np.int32)


def main():
    case_dirs = sorted([d for d in VTK_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("case")])
    if not case_dirs:
        print(f"❌ ディレクトリが見つかりません: {VTK_BASE_DIR}")
        return

    if SHUFFLE_CASES:
        random.seed(42)
        random.shuffle(case_dirs)

    split_idx = int(len(case_dirs) * TRAIN_RATIO)
    train_cases, val_cases = case_dirs[:split_idx], case_dirs[split_idx:]

    print(f"📦 総ケース数: {len(case_dirs)} (訓練: {len(train_cases)}, 検証: {len(val_cases)})")

    # メタデータの構築と保存
    base_mesh = extract_2d_mesh(pv.read(get_vtu_files_ordered(case_dirs[0])[0]))
    u_name, _ = extract_field(base_mesh, ["U", "velocity"])
    p_name, _ = extract_field(base_mesh, ["p", "pressure"])
    num_cells, num_nodes = base_mesh.faces.reshape(-1, 4).shape[0], base_mesh.points.shape[0]
    traj_len = len(get_vtu_files_ordered(case_dirs[0]))

    metadata = {
        "dt": DT_VALUE,
        "trajectory_length": traj_len,
        "field_names": ["cells", "mesh_pos", "node_type", u_name, p_name],
        "features": {
            "cells":     {"type": "static",  "shape": [traj_len, num_cells, 3], "dtype": "int32"},
            "mesh_pos":  {"type": "static",  "shape": [traj_len, num_nodes, 2], "dtype": "float32"},
            "node_type": {"type": "static",  "shape": [traj_len, num_nodes, 1], "dtype": "int32"},
            u_name:      {"type": "dynamic", "shape": [traj_len, num_nodes, 2], "dtype": "float32"},
            p_name:      {"type": "dynamic", "shape": [traj_len, num_nodes, 1], "dtype": "float32"}
        }
    }
    
    OUTPUT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    # 訓練・検証 HDF5 の生成
    process_and_save(train_cases, OUTPUT_TRAIN_H5, "製造中: 訓練データ (Train H5)")
    process_and_save(val_cases, OUTPUT_VAL_H5, "製造中: 検証データ (Val H5)")

    print("\n✨ 完全に整合・洗練されたHDF5データセットが完成しました！")


if __name__ == "__main__":
    main()