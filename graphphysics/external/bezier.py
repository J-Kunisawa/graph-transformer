import torch
from torch_geometric.data import Data
from typing import Optional

from graphphysics.utils.nodetype import NodeType


def add_bezier_node_type(graph: Data, feature_offset: int = 3) -> Data:
    """
    Bezier メッシュ向けに node_type 列を末尾に追加する。

    期待する入力 x の列:
      - feature_offset: ベジェ判定開始インデックス（デフォルト 3）
      - bn = x[:, feature_offset]         (ベースノードフラグ)
      - a1..a4 = x[:, feature_offset+1 ... feature_offset+4]

    出力:
      - graph.x に node_type 列を float -> long に変換して追加する（最後の列）
    安全性:
      - x の列数が足りない場合は存在しない要素を 0 として扱う
    """
    x = getattr(graph, "x", None)
    if x is None:
        return graph

    device = x.device
    dtype = x.dtype

    # helper to safely get a column or zero column
    def _col(idx: int) -> torch.Tensor:
        if x.size(1) > idx:
            return x[:, idx]
        return torch.zeros((x.size(0),), device=device, dtype=dtype)

    bn = _col(feature_offset)
    a1 = _col(feature_offset + 1)
    a2 = _col(feature_offset + 2)
    a3 = _col(feature_offset + 3)
    a4 = _col(feature_offset + 4)

    # Create node_type tensor (long) with default NORMAL (0)
    node_type = torch.zeros(bn.shape[0], dtype=torch.long, device=device)

    # Masks (use elementwise comparisons; cast masks to bool)
    wall_mask = (bn == 1.0) & (a1 == 0.0) & (a2 == 0.0) & (a3 == 0.0) & (a4 == 0.0)
    inflow_mask = (a1 == 1.0)
    outflow_mask = (a3 == 1.0)

    node_type[wall_mask] = int(NodeType.WALL_BOUNDARY)
    node_type[inflow_mask] = int(NodeType.INFLOW)
    node_type[outflow_mask] = int(NodeType.OUTFLOW)

    # append as a float column to keep x dtype consistent with downstream expectations
    # but store node_type values as integers in the last column (cast to same dtype)
    node_type_col = node_type.unsqueeze(1).to(dtype=dtype, device=device)
    graph.x = torch.cat((x, node_type_col), dim=1)

    return graph