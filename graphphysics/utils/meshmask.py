import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple, Optional


def _device_of_tensor(t: Optional[torch.Tensor]) -> torch.device:
    return t.device if isinstance(t, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_edges(
    edge_index: torch.Tensor, node_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    node_index に含まれるノードのみを残すよう edge_index と edge_attr をフィルタする。
    戻り値: (filtered_edge_index, filtered_edge_attr or None, boolean_mask_for_edges)
    """
    if edge_index is None:
        raise ValueError("edge_index must be provided")

    num_nodes = maybe_num_nodes(edge_index, None)
    device = _device_of_tensor(edge_index)

    node_index = node_index.to(device)
    cluster_index = torch.arange(node_index.size(0), device=device)

    # -1 で初期化されたマッピングテーブルを作る
    map_table = node_index.new_full((num_nodes,), -1)
    map_table[node_index] = cluster_index

    senders = edge_index[0].to(device)
    receivers = edge_index[1].to(device)

    mapped_senders = map_table[senders]
    mapped_receivers = map_table[receivers]

    edge_mask = (mapped_senders >= 0) & (mapped_receivers >= 0)
    mapped_senders = mapped_senders[edge_mask]
    mapped_receivers = mapped_receivers[edge_mask]

    filtered_ei = torch.stack([mapped_senders, mapped_receivers], dim=0)
    filtered_eattr = edge_attr.to(device)[edge_mask] if edge_attr is not None else None

    return filtered_ei, filtered_eattr, edge_mask


def build_masked_graph(
    graph: Data,
    selected_indexes: torch.Tensor,
) -> Tuple[Data, torch.Tensor]:
    """
    入力グラフの選択ノードのみを残した複製グラフを返す。
    - 元のグラフは変更しない。
    - 戻り値: (masked_graph, edges_boolean_mask)
    """
    if not isinstance(selected_indexes, torch.Tensor):
        selected_indexes = torch.tensor(selected_indexes, dtype=torch.long)

    # clone してインプレース変更を避ける
    mg = graph.clone()

    if getattr(graph, "edge_index", None) is None:
        # no edges: only select nodes
        mg.x = mg.x[selected_indexes]
        if getattr(mg, "pos", None) is not None:
            mg.pos = mg.pos[selected_indexes]
        mg.edge_index = None
        mg.edge_attr = None
        return mg, torch.tensor([], dtype=torch.bool)

    filtered_ei, filtered_eattr, edges_mask = filter_edges(graph.edge_index, selected_indexes, getattr(graph, "edge_attr", None))

    mg.x = mg.x[selected_indexes]
    if getattr(mg, "pos", None) is not None:
        mg.pos = mg.pos[selected_indexes]
    mg.edge_index = filtered_ei
    if filtered_eattr is not None:
        mg.edge_attr = filtered_eattr
    else:
        mg.edge_attr = None

    return mg, edges_mask


def reconstruct_graph(
    graph: Data,
    latent_masked_graph: Data,
    selected_indexes: torch.Tensor,
    node_mask_token: torch.nn.Parameter,
    edges_mask: torch.Tensor,
    edge_encoder: Optional[nn.Module] = None,
    edge_mask_token: Optional[torch.nn.Parameter] = None,
) -> Data:
    """
    マスクされた潜在グラフの出力を使って元グラフを再構築する。
    - masked ノードは latent_masked_graph.x の値を使い、マスクされたノードは node_mask_token で埋める。
    - edge_attr がある場合は edge_encoder を通し edge_mask_token で埋め、unmasked edge は latent_masked_graph.edge_attr で置換する。
    """
    device = _device_of_tensor(graph.x if hasattr(graph, "x") else None)

    n_nodes, feat_dim = graph.x.shape
    # prepare base features filled with node mask token
    node_token = node_mask_token.to(device)
    if node_token.numel() == 1:
        node_token = node_token.expand(feat_dim)
    node_token = node_token.reshape(1, -1)
    if node_token.size(1) != feat_dim:
        raise ValueError("node_mask_token dimension does not match graph node feature dim")

    features = node_token.expand(n_nodes, -1).clone()

    # place latent features for selected indexes
    sel = selected_indexes.to(device)
    features[sel] = latent_masked_graph.x.to(device)

    out_graph = graph.clone()
    out_graph.x = features

    # handle edges
    if getattr(graph, "edge_attr", None) is not None:
        orig_eattr = graph.edge_attr.to(device)
        n_edges = orig_eattr.size(0)

        # encode original edges if encoder provided
        if edge_encoder is not None:
            encoded = edge_encoder(orig_eattr)
        else:
            encoded = orig_eattr

        # prepare mask token for edges
        if edge_mask_token is None:
            edge_token = torch.zeros((1, encoded.size(1)), device=device, dtype=encoded.dtype)
        else:
            edge_token = edge_mask_token.to(device).reshape(1, -1)
            if edge_token.size(1) != encoded.size(1):
                raise ValueError("edge_mask_token dim mismatch with encoded edge feature dim")

        encoded = encoded.clone()
        encoded[:] = edge_token.expand(n_edges, -1)

        # replace unmasked edges with latent values (edges_mask is boolean over original edges)
        if latent_masked_graph.edge_attr is not None and edges_mask.numel() == n_edges:
            encoded[edges_mask.to(device)] = latent_masked_graph.edge_attr.to(device)

        out_graph.edge_attr = encoded
    else:
        out_graph.edge_attr = None

    return out_graph