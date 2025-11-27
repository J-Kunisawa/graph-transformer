import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional


def convert_to_pyvista_mesh(graph: Data, add_all_data: bool = False, scalar_name: Optional[str] = None) -> object:
    """
    PyTorch Geometric の Data を PyVista のメッシュに変換するユーティリティ。

    - 頂点は graph.pos から取得。2D の場合は z=0 でパディングする。
    - 面情報があれば triangle face を使って PolyData を作成する。
    - face が無ければ edge_index を用いてラインメッシュを作る。
    - tetra 等のボリュームセルはサポートせず（必要なら別関数で UnstructuredGrid を作成してください）。
    - add_all_data=True のときは graph.x / graph.y を point_data として追加します。
    - scalar_name を指定すると可視化用にその名前の point_data を優先して設定します
    """
    try:
        import pyvista as pv
    except Exception:
        raise ImportError("pyvista is required for convert_to_pyvista_mesh but failed to import")

    if not hasattr(graph, "pos") or graph.pos is None:
        raise ValueError("graph must have 'pos' attribute")

    pos = graph.pos.detach().cpu().numpy()
    if pos.ndim != 2:
        raise ValueError("graph.pos must be shape (N, D)")

    # ensure 3D coordinates
    if pos.shape[1] < 3:
        pad = np.zeros((pos.shape[0], 3 - pos.shape[1]), dtype=pos.dtype)
        points = np.hstack([pos, pad])
    elif pos.shape[1] == 3:
        points = pos
    else:
        raise ValueError(f"Unsupported coordinate dimension: {pos.shape[1]}")

    # Prefer triangle face if present
    faces_attr = getattr(graph, "face", None)
    if faces_attr is not None:
        faces = faces_attr.detach().cpu().numpy()
        # normalize shape to (M, 3)
        if faces.ndim == 2 and faces.shape[0] in (3, 4) and faces.shape[1] != faces.shape[0]:
            faces = faces.T
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("face must be (3, M) or (M, 3) for triangle meshes")
        n = faces.shape[0]
        # meshio / pyvista face encoding: [3, v0, v1, v2, 3, ...]
        faces_flat = np.hstack([np.full((n, 1), 3, dtype=np.int64), faces.astype(np.int64)]).astype(np.int64).flatten()
        mesh = pv.PolyData(points, faces=faces_flat)
    else:
        # fallback to edge_index -> create PolyData with lines
        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            # isolated points only
            mesh = pv.PolyData(points)
        else:
            ei = edge_index.detach().cpu()
            # expect shape (2, E)
            if ei.ndim == 2 and ei.shape[0] == 2:
                edges = ei.t().numpy().astype(np.int64)
            elif ei.ndim == 2 and ei.shape[1] == 2:
                edges = ei.numpy().astype(np.int64)
            else:
                raise ValueError("edge_index must be shape (2, E) or (E, 2)")
            # pyvista lines format: [2, a, b, 2, c, d, ...] flattened
            ne = edges.shape[0]
            lines = np.hstack([np.full((ne, 1), 2, dtype=np.int64), edges]).flatten()
            mesh = pv.PolyData(points, lines=lines)

    # add point-wise data if requested
    if add_all_data:
        if hasattr(graph, "x") and graph.x is not None:
            x_np = graph.x.detach().cpu().numpy()
            # add each column as separate array
            for i in range(x_np.shape[1]):
                mesh.point_data[f"x{i}"] = x_np[:, i]
        if hasattr(graph, "y") and graph.y is not None:
            y_np = graph.y.detach().cpu().numpy()
            for i in range(y_np.shape[1]):
                mesh.point_data[f"y{i}"] = y_np[:, i]

    # If user provided scalar_name, try to set active scalars
    if scalar_name and scalar_name in mesh.point_data:
        mesh.set_active_scalars(scalar_name)
    else:
        # try to pick a reasonable default scalar for visualization
        if mesh.point_data:
            first_key = next(iter(mesh.point_data.keys()))
            mesh.set_active_scalars(first_key)

    return mesh