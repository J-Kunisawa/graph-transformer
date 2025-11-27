import torch
from torch_geometric.data import Data
from typing import Optional


def _safe_col(x: torch.Tensor, idx: int, keep_dims: int = 1) -> torch.Tensor:
    """x の idx 列を安全に取り出し、存在しない場合はゼロ列を返す。"""
    N = x.size(0)
    if x.size(1) > idx:
        col = x[:, idx]
        return col.view(N, keep_dims) if col.dim() == 1 else col
    return torch.zeros((N, keep_dims), device=x.device, dtype=x.dtype)


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 2 else t.view(t.size(0), -1)


def build_features(graph: Data) -> Data:
    """
    panels 用の特徴量を構築する。

    - current_velocity: x[:,0:2] （無ければゼロ埋め）
    - pressure: x[:,3] -> (N,1)
    - levelset: x[:,4] -> (N,1)
    - nodetype: x[:,5] -> (N,1)
    - pos: graph.pos を可能な限りそのまま使用
    - target y の Z 成分を隠す（y[:,0:2] に切り詰め）
    """
    x = getattr(graph, "x", None)
    if x is None:
        return graph

    # current device / dtype
    device = x.device
    dtype = x.dtype
    N = x.size(0)

    # 現在速度（2次元）を安全に取得
    if x.size(1) >= 2:
        current_velocity = _ensure_2d(x[:, 0:2].to(dtype))
    else:
        current_velocity = torch.zeros((N, 2), device=device, dtype=dtype)

    # 各単一列フィーチャ（pressure, levelset, nodetype）
    pressure = _safe_col(x, 3, keep_dims=1).to(dtype)
    levelset = _safe_col(x, 4, keep_dims=1).to(dtype)
    nodetype = _safe_col(x, 5, keep_dims=1).to(dtype)

    # pos の取得（存在しない場合はゼロで埋める）
    pos = getattr(graph, "pos", None)
    if pos is None:
        pos = torch.zeros((N, 2), device=device, dtype=dtype)
    else:
        # pos が 1D の場合に備える
        if pos.dim() == 1:
            pos = pos.view(N, -1).to(dtype)
        else:
            pos = pos.to(dtype)

    # 最終的なノード特徴の結合（すべて 2D であることを前提）
    parts = [current_velocity, pressure, levelset, pos, nodetype]
    # 型・デバイス整合
    parts = [p.to(device).to(dtype) for p in parts]
    graph.x = torch.cat(parts, dim=1)

    # 目標 y の Vz を隠す（y が存在し、列が 2 より多ければ切り詰め）
    if hasattr(graph, "y") and graph.y is not None:
        y = graph.y
        if y.dim() == 2 and y.size(1) >= 2:
            graph.y = y[:, 0:2].to(dtype)
        else:
            graph.y = y.to(dtype)

    return graph