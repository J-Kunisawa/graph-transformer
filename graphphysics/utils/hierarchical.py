import json
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from torch_geometric.data import Data

from graphphysics.utils.torch_graph import meshdata_to_graph


def read_h5_metadata(
    dataset_path: str, meta_path: str
) -> Tuple[List[str], int, Dict[str, Any]]:
    """
    HDF5ファイルハンドルを開きっぱなしにせず、安全にクローズした状態で
    データセット（軌跡キー名一覧）とJSONメタデータを一括で読み込む。
    """
    # with構文のブロックを抜けると、h5ファイルハンドルは自動的にクローズされる
    with h5py.File(dataset_path, "r") as file_handle:
        datasets_index = list(file_handle.keys()) # 軌跡の名前（例: 'sample_000001' 等）のリストを取得

    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    return datasets_index, len(datasets_index), meta


def get_h5_dataset(
    dataset_path: str, meta_path: str
) -> Tuple[h5py.File, List[str], int, Dict[str, Any]]:
    """
    H5ファイルを開き、データセットのインデックスおよびハンドル、JSONメタデータを取得して返す。
    ※この関数ではハンドルを開いたまま返すため、利用側で明示的なクローズや管理が必要。
    """
    file_handle = h5py.File(dataset_path, "r")
    datasets_index = list(file_handle.keys())
    with open(meta_path, "r") as fp:
        meta = json.load(fp)
    return file_handle, datasets_index, len(datasets_index), meta


def get_traj_as_meshes(
    file_handle: h5py.File, traj_number: str, meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    特定の軌跡（シミュレーションケース全体）のデータをH5ファイルから一括ロードする。
    
    JSONメタデータで事前に定義された物理フィールドのデータ型（dtype）および形状（shape）に従って、
    ディスク上のバイナリを正確なテンソル構造へとリスト・辞書化して展開する。
    """
    features = file_handle[traj_number] # 指定された軌跡グループにアクセス
    meshes = {}

    # メタデータの宣言に基づいてループを回し、特徴量を展開
    for key, field in meta["features"].items():
        # features[key][()] によって、HDF5データセットの生の全フレーム配列をNumPy配列として一括メモリロード
        data = features[key][()].astype(field["dtype"])
        # [総フレーム数, ノード数, 物理次元] 等の正しい多次元形状へと整形
        data = data.reshape(field["shape"])
        meshes[key] = data

    return meshes


def get_frame_as_mesh(
    traj: Dict[str, np.ndarray],
    frame: int,
    targets: list[str] = None,
    frame_target: Optional[int] = None,
) -> Tuple[
    np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]
]:
    """
    ロード済みの軌跡（traj辞書）から、指定された特定の『フレーム（時刻 t）』のメッシュデータをスライス抽出する。
    
    もし訓練中などで `frame_target`（時刻 t+1）が指定されている場合は、
    未来のターゲット物理量（教師ラベル）と、時間変化する動的フィールドの次ステップ値も同時に切り出して仕分ける。
    """
    target_point_data = None
    next_data = None

    # 未来のターゲットステップ（t+1）の抽出ロジック
    if frame_target is not None:
        # モデルが直接予測すべき物理ターゲット（例: velocity）をスライス
        target_point_data = {key: traj[key][frame_target] for key in targets}
        
        # 予測対象（targets）ではないが、時間変化する動的特徴量（可動障害物の次ステップ座標など）の仕分け
        next_data = {
            key: traj[key][frame_target]
            for key in traj.keys()
            if key not in ["mesh_pos", "cells", "node_type"] and key not in targets
        }

    # 現時刻（t）の物理量（流速、圧力など）をまとめて辞書として切り出し
    point_data = {
        key: traj[key][frame]
        for key in traj.keys()
        if key not in ["mesh_pos", "cells", "node_type"]
    }
    # 境界条件を示すノード属性(node_type)は、時間不変であるため、最初のフレームのトポロジーで固定
    point_data["node_type"] = traj["node_type"][0]

    # ノードのデカルト座標(mesh_pos)および要素の接続情報(cells)を取得
    # メッシュ自体が時間変化して変形するタイプ(ndim > 1)か、静的格子（ndim <= 1）かでスライスを分岐
    mesh_pos = (
        traj["mesh_pos"][frame] if traj["mesh_pos"].ndim > 1 else traj["mesh_pos"]
    )
    cells = traj["cells"][frame] if traj["cells"].ndim > 1 else traj["cells"]

    return mesh_pos, cells, point_data, target_point_data, next_data


def get_frame_as_graph(
    traj: Dict[str, np.ndarray],
    frame: int,
    meta: Dict[str, Any],
    targets: list[str] = None,
    frame_target: Optional[int] = None,
) -> Data:
    """
    指定フレームのメッシュ配列（NumPy）を、PyTorch Geometricが要求する
    完全なグラフ表現オブジェクト（Data）へとコンバートする統合パイプライン関数。
    """
    # 1. NumPy配列形式としてメッシュ構成データをピンポイントで切り出し
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj, frame, targets, frame_target
    )
    
    # シミュレーション上の物理的な経過時間（time = ステップ数 × dt）を計算
    time = frame * meta.get("dt", 1)
    
    # 2. torch_graph.py の変換モジュールへ丸投げし、
    # ノード、エッジインデックス、デカルト座標、教師ラベルを内包した Data オブジェクトを新調して返す
    return meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        time=time,
        target=target,
        next_data=next_data,
    )