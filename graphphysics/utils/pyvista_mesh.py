from typing import List, Optional

import numpy as np
import pyvista as pv
from torch_geometric.data import Data


def convert_to_pyvista_mesh(
    graph: Data, 
    add_all_data: bool = False,
    feature_names: Optional[List[str]] = None,
) -> pv.PolyData:
    """
    PyTorch Geometric のグラフデータを、PyVista の 3Dポリゴンデータ形式（pv.PolyData）に変換する。
    主に、WandBなどのダッシュボードに流速コンター図のスクリーンショット（画像・動画）を転送するための前処理として使用される。

    パラメータ:
        graph (Data): 座標と物理特徴量を保持する PyG グラフオブジェクト。
        add_all_data (bool): True の場合、graph.x に格納されている全物理フィールドを、カラーバー描画用の頂点データとして登録する。
        feature_names (List[str], optional): x（入力特徴量）の各次元に割り当てる任意の変数名のリスト。
    """
    
    # 1. ノード座標（pos）の存在チェックと抽出
    if not hasattr(graph, "pos") or graph.pos is None:
        raise ValueError("グラフオブジェクトはノード位置を指定する 'pos' 属性を持つ必要があります。")

    # 座標テンソルをNumPy配列に変換
    vertices = graph.pos.cpu().numpy()
    num_coords = vertices.shape[1] # 幾何空間の次元数（2Dなら2、3Dなら3）
    
    # PyVista（VTKレンダラー）は原則3次元座標を要求するため、2Dデータの場合はZ座標を 0 で自動パディングする
    if num_coords < 3:
        padding = np.zeros((vertices.shape[0], 3 - num_coords), dtype=vertices.dtype)
        vertices = np.hstack([vertices, padding]) # 横方向に結合して [NumNodes, 3] の形状にする
    elif num_coords > 3:
        raise ValueError(f"サポートされていない頂点次元数です: {num_coords}")

    # 2. グラフエッジ（edge_index）の存在チェックと、VTKグラフィックス規格への超高速パッキング
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError("グラフオブジェクトはエッジのトポロジーを指定する 'edge_index' 属性を持つ必要があります。")

    # エッジインデックスを取得し転置 [NumEdges, 2 (始点ID, 終点ID)]
    edges = graph.edge_index.t().cpu().numpy()
    num_edges = edges.shape[0]
    
    # 【VTK / PyVistaの重要なデータ構造規則】
    # 線（Lines）のトポロジーを定義する際、各ペアの先頭に「その線を構成する点の数（今回は始点と終点の 2 点）」を格納する必要がある。
    # 例: エッジが [0, 1] と [2, 3] の場合、[2, 0, 1, 2, 2, 3] というフラットな1次元配列にする。
    # ここでは、すべての行の先頭に値「2」を持つ列ベクトルを一画生成し、エッジ配列の左側に結合（hstack）してから、1次元へ平坦化（flatten）する。
    lines = np.hstack([np.full((num_edges, 1), 2, dtype=np.int64), edges]).flatten()

    # 3. 3D幾何頂点と、パッキングしたエッジ配列を流し込んで、PyVistaのメッシュデータを新調
    mesh = pv.PolyData(vertices, lines=lines)

    # オプション：モデルのノード特徴量（流速ベクトルや圧力等）を、PyVistaメッシュの各頂点データ（point_data）に割り当て
    # これにより、レンダラーが頂点間の物理量を滑らかにグラデーション補間してカラーコンター図を生成できるようになる。
    if add_all_data and hasattr(graph, "x") and graph.x is not None:
        x_data = graph.x.cpu().numpy()
        for i in range(x_data.shape[1]):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"x{i}"
            mesh.point_data[name] = x_data[:, i]

        # 速度（Velocity）の絶対値を計算して追加
        vel_keys = [k for k in ["Velocity_X", "Velocity_Y", "Velocity_Z"] if k in mesh.point_data]
        if len(vel_keys) >= 2:
            mesh.point_data["Velocity_Magnitude"] = np.linalg.norm(
                np.stack([mesh.point_data[k] for k in vel_keys], axis=-1), axis=-1
            )

    return mesh