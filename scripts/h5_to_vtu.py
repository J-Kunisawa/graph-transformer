import h5py
import numpy as np
import pyvista as pv
from pathlib import Path

# ========== 設定 ==========
H5_PATH = Path("data/data_h5/re_100-300_100_train.h5")
TARGET_CASE = "sim_00000"  # 可視化したいケース名
FRAME_IDX = 0              # 可視化したいタイムステップ（例: 0）
OUTPUT_VTU = Path("output_vis/re_100-300_100.vtp")
# ==========================

def main():
    with h5py.File(H5_PATH, "r") as f:
        if TARGET_CASE not in f:
            print(f"❌ ケース {TARGET_CASE} が見つかりません。利用可能: {list(f.keys())[:5]}")
            return
        
        group = f[TARGET_CASE]
        
        # 1. 座標データ (mesh_pos) の読み込みと 3D 化
        # shape例: [7, num_nodes, 2] -> 指定フレームを取り出す
        pos = group["mesh_pos"][FRAME_IDX]
        if pos.shape[1] == 2:
            # 2D座標の場合、Z=0 を追加して (num_nodes, 3) に変換（ParaView/VTKの必須仕様）
            pos_3d = np.hstack([pos, np.zeros((len(pos), 1), dtype=np.float32)])
        else:
            pos_3d = pos

        # 2. セル接続構造 (cells) の読み込みと VTK 規格への変換
        # shape例: [7, num_cells, 3] (三角形要素の場合)
        cells = group["cells"][FRAME_IDX]
        num_cells, num_verts = cells.shape
        
        # 各セルの先頭に「頂点数 (例: 3)」を結合してフラット化する [3, p0, p1, p2, 3, p3, p4, p5, ...]
        vtk_cells = np.hstack([np.full((num_cells, 1), num_verts, dtype=np.int32), cells]).flatten()

        # 3. PyVista ポリゴンメッシュの生成
        mesh = pv.PolyData(pos_3d, faces=vtk_cells)

        # 4. 物理量 (U, p など) をノードデータとして割り当て
        if "U" in group:
            u_data = group["U"][FRAME_IDX]  # shape: [num_nodes, 2] or [num_nodes, 3]
            if u_data.shape[1] == 2:
                u_data = np.hstack([u_data, np.zeros((len(u_data), 1), dtype=np.float32)])
            mesh.point_data["Velocity (U)"] = u_data
            
        if "p" in group:
            mesh.point_data["Pressure (p)"] = group["p"][FRAME_IDX].flatten()

    # --- 可視化と保存 ---
    print(f"✨ メッシュ構築完了: ノード数={mesh.n_points}, セル数={mesh.n_cells}")

    # (A) ParaView 用に .vtu ファイルとして保存
    OUTPUT_VTU.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(OUTPUT_VTU)
    print(f"📂 VTKファイルを保存しました: {OUTPUT_VTU} (ParaViewで開けます)")

    # # (B) Python上でインタラクティブに描画（GUI環境がある場合）
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, scalars="Velocity (U)", cmap="coolwarm", show_edges=True)
    # plotter.add_axes()
    # plotter.view_xy()  # 2D視点（Z軸方向から）
    # plotter.show()

if __name__ == "__main__":
    main()