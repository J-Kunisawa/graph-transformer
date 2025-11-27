from typing import Any, Dict, List, Optional, Tuple, Union

import meshio
import numpy as np
import torch
import torch_geometric.transforms as T
from meshio import Mesh
from torch_geometric.data import Data


def compute_k_hop_edge_index(
    edge_index: torch.Tensor,
    num_hops: int,
    num_nodes: int,
) -> torch.Tensor:
    """k-hop のエッジインデックスを稀疎行列で計算して返す。

    - edge_index: shape (2, E)
    - num_hops: 1 を指定した場合は入力をそのまま返す
    - num_nodes: ノード数
    """
    if num_hops <= 1:
        return edge_index.clone()

    device = edge_index.device
    values = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()

    # 初期は 1-hop の隣接（self ループは後で除去）
    adj_k = adj.clone()

    for _ in range(num_hops - 1):
        # sparse x sparse -> sparse の乗算
        adj_k = torch.sparse_coo_tensor(
            adj_k.indices(),
            adj_k.values(),
            adj_k.size(),
        ).coalesce()
        adj_k = adj_k + torch.sparse.mm(adj_k, adj)
        adj_k = adj_k.coalesce()

        # 値を 1 にしておく（存在のみを示すため）
        if adj_k.values().numel() > 0:
            vals = torch.ones_like(adj_k.values())
            adj_k = torch.sparse_coo_tensor(adj_k.indices(), vals, adj_k.size()).coalesce()

    # self-loop を除去
    idx = adj_k.indices()
    mask = idx[0] != idx[1]
    idx = idx[:, mask]
    vals = torch.ones(idx.shape[1], device=device)
    adj_k = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()

    return adj_k.indices()


def compute_k_hop_graph(
    graph: Data,
    num_hops: int,
    add_edge_features_to_khop: bool = False,
    device: str = "cpu",
    world_pos_index_start: int = 0,
    world_pos_index_end: int = 3,
) -> Data:
    """与えられたグラフを k-hop にリメッシュして返す。

    - add_edge_features_to_khop=True の場合は torch_geometric.transforms を使って edge_attr を計算する。
    - 戻り値は元グラフのコピーで edge_index/edge_attr を置き換える。
    """
    if num_hops <= 1:
        return graph

    edge_index = graph.edge_index
    if edge_index is None:
        raise ValueError("graph.edge_index is required to compute k-hop graph")

    num_nodes = getattr(graph, "num_nodes", graph.pos.size(0) if getattr(graph, "pos", None) is not None else None)
    if num_nodes is None:
        raise ValueError("num_nodes could not be inferred from graph")

    khop_idx = compute_k_hop_edge_index(edge_index=edge_index, num_hops=num_hops, num_nodes=int(num_nodes)).to(device)

    khop_graph = graph.clone()
    khop_graph.edge_index = khop_idx
    # Clear any existing edge_attr copied by clone. The original graph may
    # have edge_attr computed for the 1-hop mesh; after replacing
    # `edge_index` with the k-hop index we must remove stale edge_attr so
    # that the subsequent Cartesian/Distance transforms compute fresh
    # per-edge features with the correct length. Failing to clear this
    # leads to size mismatches when concatenating tensors inside
    # torch_geometric.transforms.Cartesian.
    khop_graph.edge_attr = None

    if add_edge_features_to_khop:
        transforms = [
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ]
        comp = T.Compose(transforms)
        # コンポーズは Data を受け取るため一時的に GPU->CPU -> GPU を行うことがある
        khop_graph = comp(khop_graph)

    return khop_graph


def _stack_point_arrays(point_data: Optional[Dict[str, np.ndarray]], expected_N: Optional[int] = None) -> torch.Tensor:
    """
    point_data の dict を受け取り、各キーの配列を列方向に結合して torch.Tensor を返す。
    配列は形状 (N,) または (N,k) を想定する。None ならゼロ配列を返す。
    """
    if not point_data:
        return torch.zeros((0, 0), dtype=torch.float32)

    arrays = []
    N = expected_N
    for key, arr in point_data.items():
        a = np.asarray(arr)
        # normalize 1D -> (len,1)
        if a.ndim == 1:
            a = a.reshape(-1, 1)

        # if 2D, try to ensure rows==N (number of points)
        if a.ndim == 2:
            if N is not None:
                if a.shape[0] == N:
                    pass
                elif a.shape[1] == N:
                    a = a.T
                else:
                    # ambiguous shape: provide informative error
                    raise ValueError(
                        f"Inconsistent point_data lengths for key '{key}': expected {N} rows, got {a.shape[0]} rows (shape={a.shape})."
                    )
            else:
                # no expected_N provided: prefer larger dimension as rows if sensible
                if a.shape[0] < a.shape[1]:
                    a = a.T
                # set N from first array
                N = a.shape[0]

        # after normalization, set N if still unknown
        if N is None:
            N = a.shape[0]

        if a.shape[0] != N:
            raise ValueError(f"Inconsistent point_data lengths for key '{key}': expected {N}, got {a.shape[0]}")

        arrays.append(a.astype(np.float32))

    if N is None:
        return torch.zeros((0, 0), dtype=torch.float32)

    stacked = np.hstack(arrays)
    return torch.tensor(stacked, dtype=torch.float32)


def meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]],
    time: Union[int, float] = 1,
    target: Optional[np.ndarray] = None,
    return_only_node_features: bool = False,
    id: Optional[str] = None,
    next_data: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Data:
    """メッシュデータを PyG Data に変換するユーティリティ。

    - points: (N, D)
    - cells: (K, M) または (M, K) のいずれか（関数内で整形）
    - point_data: dict で各キーは (N,) または (N,k) を想定
    - target: dict の場合は列結合して y にする。配列の場合はそのまま使用。
    """
    pts = np.asarray(points)
    if pts.ndim != 2:
        raise ValueError("points must be shape (N, D)")
    N = pts.shape[0]

    # node features
    x = _stack_point_arrays(point_data, expected_N=N)
    if x.numel() == 0:
        # fallback: time を唯一の特徴量として付与
        x = torch.full((N, 1), float(time), dtype=torch.float32)
    else:
        # append time as last column
        time_col = torch.full((N, 1), float(time), dtype=torch.float32)
        x = torch.cat([x, time_col], dim=1)

    if return_only_node_features:
        return x

    # target の扱い：配列もしくは dict
    y = None
    if target is not None:
        if isinstance(target, dict):
            y_np = np.hstack([np.asarray(v).reshape(N, -1) if np.asarray(v).ndim >= 2 else np.asarray(v).reshape(N, 1) for v in target.values()])
            y = torch.tensor(y_np, dtype=torch.float32)
        else:
            t = np.asarray(target)
            if t.ndim == 1:
                y = torch.tensor(t.reshape(N, 1), dtype=torch.float32)
            elif t.ndim == 2:
                y = torch.tensor(t, dtype=torch.float32)
            else:
                raise ValueError("Unsupported target ndim")

    # cells の整形： (K, M) または (M, K) -> 常に (M, K)
    c = np.asarray(cells)
    # Ensure cells has shape (num_cells, nodes_per_cell). Accept either
    # (num_cells, 3/4) or (3/4, num_cells) and normalize to (num_cells, k).
    if c.ndim != 2:
        raise ValueError("cells must be a 2D array")
    if c.shape[0] in (3, 4) and c.shape[1] not in (3, 4):
        # input is shaped (3, F) or (4, F) -> transpose to (F, 3/4)
        c = c.T

    c_t = torch.tensor(c, dtype=torch.long)

    tetra = None
    face = None
    if c_t.dim() == 2:
        # Now c_t is (num_cells, nodes_per_cell)
        if c_t.size(1) == 4:
            # Keep tetra in PyG convention: (4, num_tetra)
            tetra = c_t.t()  # (4, num_tets)
            # derive triangular faces from tetra (4 faces per tetra)
            f0 = c_t[:, [0, 1, 2]]
            f1 = c_t[:, [0, 1, 3]]
            f2 = c_t[:, [0, 2, 3]]
            f3 = c_t[:, [1, 2, 3]]
            faces_cat = torch.cat([f0, f1, f2, f3], dim=0)  # (4*num_tets, 3)
            # convert to PyG face convention: (3, num_faces)
            face = faces_cat.t()
        elif c_t.size(1) == 3:
            # triangles: convert to PyG face convention (3, num_faces)
            face = c_t.t()
        else:
            raise ValueError("cells must represent triangles (3) or tetrahedra (4) per row")

    # build a mapping from point_data keys to column ranges in x
    point_feature_mapping: Dict[str, Tuple[int, int]] = {}
    if point_data:
        col = 0
        for key, arr in point_data.items():
            a = np.asarray(arr)
            if a.ndim == 1:
                width = 1
            else:
                # if arr is 2D, its last dimension is features per node
                width = a.shape[1] if a.ndim == 2 else 1
            point_feature_mapping[key] = (col, col + int(width))
            col += int(width)

    data = Data(
        x=x,
        face=face,
        tetra=tetra,
        y=y,
        pos=torch.tensor(pts, dtype=torch.float32),
        id=id,
        next_data=next_data,
    )

    # attach metadata-derived helpers to the Data object for downstream code
    try:
        data.point_feature_mapping = point_feature_mapping
        data.feature_dims = {k: v[1] - v[0] for k, v in point_feature_mapping.items()}
    except Exception:
        # be permissive in case Data doesn't accept arbitrary attrs in some envs
        pass

    if meta is not None:
        try:
            data.meta = meta
        except Exception:
            pass
    return data


def mesh_to_graph(
    mesh: Mesh,
    time: Union[int, float] = 1,
    target_mesh: Optional[Mesh] = None,
    target_fields: Optional[List[str]] = None,
) -> Data:
    """meshio.Mesh から Data へ変換する簡易ラッパー。"""
    target = None
    if target_mesh is not None and target_fields:
        target_data = [target_mesh.point_data[f] for f in target_fields]
        target = np.hstack(target_data) if len(target_data) > 1 else target_data[0]

    # collect triangle/quad cells
    cells_list = []
    for k, v in mesh.cells_dict.items():
        if k in ("triangle", "quad", "tetra"):
            cells_list.append(v)
    if not cells_list:
        raise ValueError("mesh does not contain triangle/quad/tetra cells")
    cells = np.vstack(cells_list)

    return meshdata_to_graph(points=mesh.points, cells=cells, point_data=mesh.point_data, time=time, target=target)


def torch_graph_to_mesh(graph: Data, node_features_mapping: Dict[str, int]) -> Mesh:
    """PyG Data -> meshio.Mesh 変換。

    - node_features_mapping: {name: col_index} を使って point_data を構成する
    """
    point_data = {}
    for name, idx in node_features_mapping.items():
        if not hasattr(graph, "x") or graph.x is None:
            raise ValueError("graph.x is required to extract node features")
        point_data[name] = graph.x[:, idx].detach().cpu().numpy()

    # prefer face if present, else try tetra
    if getattr(graph, "face", None) is not None:
        faces = graph.face.detach().cpu().numpy()
        # If faces are in PyG convention (3, F), convert to (F, 3) for meshio
        if faces.ndim == 2 and faces.shape[0] in (3,) and faces.shape[1] >= 1:
            faces = faces.T
        cells = [("triangle", faces)]
    elif getattr(graph, "tetra", None) is not None:
        tets = graph.tetra.detach().cpu().numpy()
        # If tets are in PyG convention (4, T), convert to (T, 4)
        if tets.ndim == 2 and tets.shape[0] in (4,) and tets.shape[1] >= 1:
            tets = tets.T
        cells = [("tetra", tets)]
    else:
        raise ValueError("graph must have 'face' or 'tetra' to convert to mesh")

    pts = graph.pos.detach().cpu().numpy()
    return meshio.Mesh(points=pts, cells=cells, point_data=point_data)


def get_masked_indexes(graph: Data, masking_ratio: float = 0.15) -> torch.Tensor:
    """グラフノードのランダムマスクインデックスを返す（保持するインデックスを返す）。"""
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("graph.x is required to compute masked indexes")
    n = graph.x.size(0)
    num_keep = max(1, int(round((1.0 - float(masking_ratio)) * n)))
    perm = torch.randperm(n)
    return perm[:num_keep]