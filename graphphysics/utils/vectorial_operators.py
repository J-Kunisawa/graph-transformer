import warnings
from typing import Optional

import torch
from torch_geometric.data import Data


def _device_of_graph(graph: Data, device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if hasattr(graph, "pos") and graph.pos is not None:
        return graph.pos.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gradient_weighted_least_squares(
    graph: Data, field: torch.Tensor, device: Optional[str] = None
) -> torch.Tensor:
    """
    要素単位の加重最小二乗法で勾配を推定する（不整なメッシュ向け、VTKに近い手法）。
    戻り値: (N, F, D) -- 各ノードの各フィールド成分に対する空間次元での勾配
    """
    dev = _device_of_graph(graph, device)
    points = graph.pos.to(dev)  # (N, dim_x)
    field = field.to(dev)
    if field.ndim == 1:
        field = field.unsqueeze(1)  # (N, F)
    N = points.shape[0]
    dim_x = points.shape[1]
    F = field.shape[1]

    # 要素（面/体）情報を取得
    if getattr(graph, "face", None) is not None:
        elems = graph.face
    elif getattr(graph, "tetra", None) is not None:
        elems = graph.tetra
    else:
        raise ValueError("Graph must contain 'face' or 'tetra' for least squares gradient")

    elems = elems.to(dev)
    # normalize shape to (M, K) where K = D+1
    if elems.dim() == 2 and elems.shape[0] in (3, 4) and elems.shape[1] != elems.shape[0]:
        elems = elems.t()
    M = elems.shape[0]
    K = elems.shape[1]  # D+1

    # element node coordinates and field values
    elem_pts = points[elems]  # (M, K, dim_x)
    elem_field = field[elems]  # (M, K, F)

    # build A (M, D, dim_x) and B (M, D, F) relative to first vertex
    A = elem_pts[:, 1:, :] - elem_pts[:, :1, :]  # (M, D, dim_x)
    B = elem_field[:, 1:, :] - elem_field[:, :1, :]  # (M, D, F)

    # solve A @ Grad^T ≈ B  => Grad ≈ solution^T  (batched least squares)
    # torch.linalg.lstsq returns (..., n, k) solution for (..., m, n) * X = (..., m, k)
    try:
        sol = torch.linalg.lstsq(A, B).solution  # (M, dim_x, F)
    except Exception:
        # 古いPyTorch等でlstsqが使えない場合はpinvで近似
        A_flat = A.reshape(-1, A.shape[-2], A.shape[-1])
        B_flat = B.reshape(-1, B.shape[-2], B.shape[-1])
        sols = []
        for Ai, Bi in zip(A_flat, B_flat):
            # Ai: (D, dim_x), Bi: (D, F) -> X: (dim_x, F)
            try:
                X = torch.pinverse(Ai) @ Bi
            except Exception:
                X = torch.zeros((Ai.shape[1], Bi.shape[1]), device=dev, dtype=Bi.dtype)
            sols.append(X)
        sol = torch.stack(sols, dim=0)

    # sol: (M, dim_x, F) -> transpose -> (M, F, dim_x)
    grad_elems = sol.transpose(1, 2).contiguous()

    # element measure (area or volume) weighting
    D = K - 1
    if D == 2:
        v1 = A[:, 0, :]  # (M, dim_x)
        v2 = A[:, 1, :]
        if dim_x == 3:
            cross = torch.cross(v1, v2, dim=1)
            measure = 0.5 * torch.norm(cross, dim=1)  # area
        elif dim_x == 2:
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            measure = 0.5 * torch.abs(cross)
        else:
            measure = torch.ones(M, device=dev)
    elif D == 3:
        # A: (M,3,3) -> volume = |det(A)| / 6
        try:
            dets = torch.linalg.det(A)
            measure = torch.abs(dets) / 6.0
        except Exception:
            measure = torch.ones(M, device=dev)
    else:
        measure = torch.ones(M, device=dev)

    measure = measure.clamp(min=1e-12)

    # accumulate per node weighted by element measure
    gradients = torch.zeros((N, F, dim_x), device=dev, dtype=grad_elems.dtype)
    weights = torch.zeros((N, 1), device=dev, dtype=grad_elems.dtype)

    for i in range(K):
        idx = elems[:, i]  # (M,)
        # grad_elems * measure[:,None,None] -> (M,F,dim_x)
        gradients.index_add_(0, idx, grad_elems * measure[:, None, None])
        weights.index_add_(0, idx, measure[:, None])

    gradients = gradients / weights.clamp(min=1e-12).view(-1, 1, 1)
    return gradients


def compute_gradient_finite_differences(
    graph: Data, field: torch.Tensor, device: Optional[str] = None
) -> torch.Tensor:
    """
    エッジに基づく有限差分近似。各エッジでの勾配テンソルを近似してノードで平均化する。
    戻り値: (N, F, D)
    """
    dev = _device_of_graph(graph, device)
    pos = graph.pos.to(dev)
    if field.ndim == 1:
        field = field.unsqueeze(1)
    field = field.to(dev)

    edge_index = graph.edge_index.to(dev)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be shape (2, E)")

    # make undirected unique edges (i<->j considered same)
    ei = edge_index.t()
    # sort each row to make canonical, then unique
    sorted_pairs, _ = torch.sort(ei, dim=1)
    unique_pairs = torch.unique(sorted_pairs, dim=0)
    i = unique_pairs[:, 0]
    j = unique_pairs[:, 1]

    dx = pos[j] - pos[i]  # (E, D)
    du = field[j] - field[i]  # (E, F)
    distances2 = (dx ** 2).sum(dim=1).clamp(min=1e-12)  # (E,)

    # approximate gradient tensor on edge: grad ≈ (du ⊗ dx) / |dx|^2
    # du.unsqueeze(2): (E, F, 1), dx.unsqueeze(1): (E, 1, D) -> outer -> (E, F, D)
    grad_edges = (du.unsqueeze(2) * dx.unsqueeze(1)) / distances2.view(-1, 1, 1)

    # weighting by inverse squared distance
    weights = (1.0 / distances2).view(-1, 1, 1)  # (E,1,1)

    N = pos.shape[0]
    F = field.shape[1]
    D = pos.shape[1]

    grad = torch.zeros((N, F, D), device=dev, dtype=grad_edges.dtype)
    weight_sums = torch.zeros((N, F, D), device=dev, dtype=grad_edges.dtype)

    # accumulate contributions to both endpoints
    grad.index_add_(0, i, grad_edges * weights)
    grad.index_add_(0, j, grad_edges * weights)
    weight_sums.index_add_(0, i, weights.expand(-1, F, D))
    weight_sums.index_add_(0, j, weights.expand(-1, F, D))

    grad = grad / (weight_sums + 1e-12)
    return grad


def compute_gradient(
    graph: Data, field: torch.Tensor, method: str = "least_squares", device: Optional[str] = None
) -> torch.Tensor:
    method = method.lower() if method is not None else "least_squares"
    if method in ("least_squares", "least", "lstsq"):
        return compute_gradient_weighted_least_squares(graph, field, device=device)
    if method in ("finite_diff", "finite", "fd"):
        return compute_gradient_finite_differences(graph, field, device=device)
    raise ValueError(f"Unknown gradient method: {method}")


def compute_vector_gradient_product(
    graph: Data,
    field: torch.Tensor,
    gradient: Optional[torch.Tensor] = None,
    method: str = "finite_diff",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    ベクトル場 u とその勾配をかけ合わせた量（例: u · grad(u)）。
    field: (N, F)
    gradient: (N, F, D) が与えられなければ本関数で計算する。
    戻り値: (N, F)
    """
    dev = _device_of_graph(graph, device)
    field = field.to(dev)
    if gradient is None:
        gradient = compute_gradient(graph, field, method=method, device=device)
    else:
        gradient = gradient.to(dev)

    # einsum 'nf,nfd->nf'
    return torch.einsum("nf,nfd->nf", field, gradient)


def compute_divergence(
    graph: Data,
    field: torch.Tensor,
    gradient: Optional[torch.Tensor] = None,
    method: str = "finite_diff",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    ベクトル場の発散を計算する。
    field: (N, F) （通常 F == 空間次元 D のとき意味を持つ）
    gradient: (N, F, D) が与えられなければ本関数で計算する。
    戻り値: (N,) -- 各ノードでの発散（F==D の場合は各成分の対角和）
    """
    dev = _device_of_graph(graph, device)
    field = field.to(dev)
    if gradient is None:
        gradient = compute_gradient(graph, field, method=method, device=device)
    else:
        gradient = gradient.to(dev)

    # gradient: (N, F, D). divergence is sum_i grad_{i,i} when F and D correspond.
    # we take diagonal over dim1/dim2 and sum; if F!=D a best-effort is returned (min(F,D) diag)
    diag = torch.diagonal(gradient, dim1=1, dim2=2)  # (N, min(F,D))
    div = diag.sum(dim=-1)  # (N,)
    return div