import copy
import os
import shutil
from typing import List, Optional

import h5py
import meshio
import numpy as np
from lxml import etree
from torch_geometric.data import Data


def convert_to_meshio_vtu(graph: Data, add_all_data: bool = False) -> meshio.Mesh:
    """
    PyTorch Geometric のグラフオブジェクトを Meshio のメッシュオブジェクトに変換します。
    予測値と正解値の物理フィールドを適切な VTK 配列（U_pred, U_target, p_pred, p_target）として保存し、
    両方が存在する場合は、予測値 − 正解値を U_error（ベクトル）と p_error（スカラー）として保存します。
    ParaView などのツール上でコンポーネント（X, Y, Magnitude）をプルダウンで直感的に切り替え・比較できるようにします。

    引数:
        graph (Data): 変換対象の PyG グラフデータ。
        add_all_data (bool, オプション): True の場合、graph.x や graph.y に含まれるすべての生特徴量を
                                         x0, x1... / y0, y1... として個別に点データに追加します。

    戻り値:
        meshio.Mesh: 変換された Meshio メッシュオブジェクト。
    """

    # 'pos' 属性（ノードの空間座標）が存在するか確認
    if not hasattr(graph, "pos") or graph.pos is None:
        raise ValueError("グラフはノード座標を持つ 'pos' 属性を含んでいる必要があります。")

    # ノード座標を抽出し、必ず3次元（3コンポーネント）であることを保証する
    vertices = graph.pos.cpu().numpy()
    num_coords = vertices.shape[1]
    if num_coords < 3:
        # 2Dデータの場合、Z成分に0をパディングして3D化する
        padding = np.zeros((vertices.shape[0], 3 - num_coords), dtype=vertices.dtype)
        vertices = np.hstack([vertices, padding])
    elif num_coords > 3:
        raise ValueError(f"サポートされていない頂点次元数です: {num_coords}")

    # 'face'（または 'tetra'）属性（メッシュの接続情報）が存在するか確認
    if not hasattr(graph, "face") or graph.face is None:
        raise ValueError("グラフは面の接続インデックスを持つ 'face' 属性を含んでいる必要があります。")

    # セル（要素）情報の抽出
    faces = (
        (graph.tetra if getattr(graph, "tetra", None) is not None else graph.face)
        .cpu()
        .numpy()
        .T
    )
    cells = [
        ("tetra" if getattr(graph, "tetra", None) is not None else "triangle", faces)
    ]

    # Meshio メッシュの作成
    mesh = meshio.Mesh(vertices, cells)
    num_nodes = vertices.shape[0]

    # 1. 【予測値フィールド (Prediction)】のパッキング
    if hasattr(graph, "x") and graph.x is not None and graph.x.shape[1] >= 2:
        x_data_np = graph.x.cpu().numpy()
        
        # 2D予測流速 (0, 1列目) を 3Dベクトル [ノード数, 3] へ拡張
        U_pred_3d = np.zeros((num_nodes, 3), dtype=np.float32)
        U_pred_3d[:, 0] = x_data_np[:, 0]  # u_pred (X成分)
        U_pred_3d[:, 1] = x_data_np[:, 1]  # v_pred (Y成分)
        mesh.point_data["U_pred"] = U_pred_3d

        # 予測圧力 (2列目) をスカラーとして格納
        if x_data_np.shape[1] >= 3:
            mesh.point_data["p_pred"] = x_data_np[:, 2].astype(np.float32)

    # 2. 【正解値フィールド (Target)】のパッキング
    if hasattr(graph, "y") and graph.y is not None and graph.y.shape[1] >= 2:
        y_data_np = graph.y.cpu().numpy()
        
        # 2D正解流速 (0, 1列目) を 3Dベクトル [ノード数, 3] へ拡張
        U_target_3d = np.zeros((num_nodes, 3), dtype=np.float32)
        U_target_3d[:, 0] = y_data_np[:, 0]  # u_target (X成分)
        U_target_3d[:, 1] = y_data_np[:, 1]  # v_target (Y成分)
        mesh.point_data["U_target"] = U_target_3d

        # 正解圧力 (2列目) をスカラーとして格納
        if y_data_np.shape[1] >= 3:
            mesh.point_data["p_target"] = y_data_np[:, 2].astype(np.float32)

    # 3. 【誤差フィールド】予測値 − 正解値（物理量の単位は元の場と同じ）
    for field in ("U", "p"):
        pred_key = f"{field}_pred"
        target_key = f"{field}_target"
        if pred_key in mesh.point_data and target_key in mesh.point_data:
            mesh.point_data[f"{field}_error"] = (
                mesh.point_data[pred_key] - mesh.point_data[target_key]
            )

    # 4. 【デバッグ用】add_all_dataがTrueの場合、生の全特徴量をチャンネルごとに個別展開
    if add_all_data and hasattr(graph, "x") and graph.x is not None:
        x_data = graph.x.cpu().numpy()
        for i in range(x_data.shape[1]):
            mesh.point_data[f"x{i}"] = x_data[:, i]

    if add_all_data and hasattr(graph, "y") and graph.y is not None:
        y_data = graph.y.cpu().numpy()
        for i in range(y_data.shape[1]):
            mesh.point_data[f"y{i}"] = y_data[:, i]

    return mesh

def vtu_to_xdmf(
    filename: str, files_list: List[str], timestep=1, remove_vtus: bool = True
) -> None:
    """
    ディスク上の独立した複数のVTUファイル（時系列データ）を、
    共通の幾何格子構造を持つ1つの高効率な XDMF/HDF5 アーカイブファイルへと圧縮統合する。
    """
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # 最初のファイルから不変であるメッシュの節点座標とセル構成を読み取る
    init_vtu = meshio.read(files_list[0])
    points = init_vtu.points
    cells = init_vtu.cells

    # meshioのTimeSeriesWriterを立ち上げる（裏でXMLファイルとHDF5バイナリが生成される）
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # 幾何形状（pointsとcells）はファイル全体の最初の一回だけ書き出す（データの重複を排除）
        writer.write_points_cells(points, cells)

        # 時系列リストをループして、各時刻の物理データ（スカラー・ベクトル場）のみを順次書き込み
        t = 0
        for file in files_list:
            mesh = meshio.read(file)
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # meshioの仕様によりH5ファイルがカレントディレクトリ（cwd）に作られるため、目的のディレクトリへ移動
    shutil.move(
        src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename
    )

    # 統合が完了した後、元となった肥大化した個別VTUファイルをディスクから削除して容量を解放
    if remove_vtus:
        for file in files_list:
            os.remove(file)


def meshes_to_xdmf(
    filename: str,
    meshes: List[meshio.Mesh],
    timestep=1,
) -> None:
    """
    メモリ上の複数の `meshio.Mesh` オブジェクトのリストから、
    一括で時系列 XDMF/HDF5 アーカイブファイル（.xdmf と .h5）を新規エクスポートする。
    """
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    points = meshes[0].points
    cells = meshes[0].cells

    # タイムシリーズライターを起動して一括パッキング
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        writer.write_points_cells(points, cells)

        t = 0
        for mesh in meshes:
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    shutil.move(
        src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename
    )


def append_mesh_to_xdmf(
    filename: str, mesh: meshio.Mesh, timestep: float = 1.0, compress=False
) -> None:
    """
    既存の大容量 XDMF/HDF5 時系列アーカイブに対し、現在の1フレーム分の物理データのみを
    過去のデータをメモリ（RAM）に一切ロードすることなく末尾に『超高速に追記（Append）』する。
    """
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # ファイルの存在チェック
    if not os.path.exists(h5_filename) or not os.path.exists(xdmf_filename):
        raise FileNotFoundError(f"追記対象の XDMF/HDF5 ファイルが見つかりません: {filename}")

    # 1. XMLファイル（.xdmf）を木構造としてパースし、時間発展コレクション要素を特定
    tree = etree.parse(xdmf_filename)
    root = tree.getroot()
    temporal_grid = root.find(
        ".//{*}Grid[@GridType='Collection'][@CollectionType='Temporal']"
    )
    if temporal_grid is None:
        raise ValueError("XDMFファイル内から時系列グリッドコレクション（Temporal Grid）を検出できませんでした。")

    # 登録されている全時間ステップのグリッドリストを取得
    time_grids = temporal_grid.findall("Grid")

    # 最後の時間ステップのグリッドから、現在の最終記録時刻（last_time）をパースし、新しい時刻（new_time）を決定
    last_grid = time_grids[-1]
    last_time = float(last_grid.find("Time").get("Value"))
    new_time = last_time + timestep

    # 2. HDF5バイナリ（.h5）を「追記（a: Append）モード」で開く
    with h5py.File(h5_filename, "a") as h5:
        existing_h5_keys = h5.keys()
        # 既存のキー名（例: 'data1', 'data2'）から、現在の最大連番インデックスを特定
        last_h5_key_idx = max(int(k.replace("data", "")) for k in existing_h5_keys) if existing_h5_keys else 0
        h5_new_keys_mapping = {}
        
        # 今回のステップの新しい頂点物理量データをH5の内部へ書き込み
        for i, (field_name, field_values) in enumerate(mesh.point_data.items()):
            new_key = f"data{last_h5_key_idx + (i+1)}" # 重複しない新しい連番キー名を割り当て
            
            # compress=True の場合は、Gzipによる可逆圧縮を施してディスク容量を節約
            if compress:
                h5.create_dataset(
                    new_key,
                    data=np.asarray(field_values),
                    chunks=True,               # 圧縮を有効化するために必須のチャンク設定
                    compression="gzip",        # 圧縮アルゴリズムとしてGzipを指定
                    compression_opts=4,        # 圧縮レベルを4（速度と圧縮率のベストバランス）に設定
                )
            else:
                # 圧縮なしの通常高速書き込み
                h5.create_dataset(new_key, data=np.asarray(field_values))
                
            # メタデータXMLを書き換えるために、物理量名とH5内のキーの対応マップを保持
            h5_new_keys_mapping[field_name] = new_key

    # 3. メタデータXML（.xdmf）への新しい時間ノードの追記
    # 直前の最終時間グリッド要素（last_grid）を深層コピー（deepcopy）して雛形にする
    new_grid = copy.deepcopy(last_grid)
    new_grid.find("Time").set("Value", str(new_time)) # 時刻属性を新しい時刻に更新

    # 複製した要素内の下位要素（AttributeやDataItem）を反復スキャンし、パス文字列を最新のH5キー名へ書き換え
    attributes_to_remove = []
    for attribute in new_grid.iter("Attribute"):
        attr_name = attribute.attrib.get("Name")
        if attr_name in h5_new_keys_mapping:
            next_h5_key = h5_new_keys_mapping[attr_name]
            for data_item in attribute.iter("DataItem"):
                data_item.text = f"{os.path.basename(h5_filename)}:/{next_h5_key}"
        else:
            attributes_to_remove.append(attribute)

    for attribute in attributes_to_remove:
        parent = attribute.getparent()
        if parent is not None:
            parent.remove(attribute)

    # 完成した新しい時間グリッドノードを XML のコレクションツリーの末尾へ追加（Append）
    temporal_grid.append(new_grid)
    
    # 変更されたツリーを UTF-8 でディスクへ書き戻し、保存完了
    tree.write(
        xdmf_filename, pretty_print=False, xml_declaration=False, encoding="UTF-8"
    )