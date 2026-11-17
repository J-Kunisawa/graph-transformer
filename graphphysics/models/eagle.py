"""EAGLE-style mesh transformer for the GraphPhysics PyG pipeline.

This module adapts the model introduced in:

    EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh
    Transformers (Janny et al., ICLR 2023)

The upstream implementation assumes a dense ``[batch, time, node, feature]``
layout and precomputed constrained k-means clusters.  This adaptation keeps the
same encode -> cluster pooling -> global attention -> graph retrieval design,
while exposing a one-step interface suitable for ``Simulator`` and variable
size PyG graphs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _make_mlp(
    input_size: int,
    output_size: int,
    hidden_size: int,
    *,
    layer_norm: bool = False,
    final_tanh: bool = False,
) -> nn.Sequential:
    if final_tanh:
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        ]
    else:
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
    if layer_norm:
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)


def _morton_order(pos: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """Return a deterministic spatially local node ordering.

    EAGLE normally consumes offline constrained-k-means clusters.  When those
    labels are unavailable, Morton ordering provides a cheap, deterministic
    fallback whose consecutive chunks remain spatially local.
    """

    if pos.ndim != 2 or pos.shape[0] == 0:
        raise ValueError("pos は [num_nodes, space_dim] の非空テンソルである必要があります。")

    used_dims = min(pos.shape[1], 3)
    selected = pos[:, :used_dims].detach()
    lower = selected.amin(dim=0, keepdim=True)
    scale = (selected.amax(dim=0, keepdim=True) - lower).clamp_min(1e-12)
    quantized = (((selected - lower) / scale) * ((1 << bits) - 1)).long()

    code = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    for bit in range(bits):
        for axis in range(used_dims):
            axis_bit = (quantized[:, axis] >> bit) & 1
            code |= axis_bit << (bit * used_dims + axis)
    return torch.argsort(code)


def build_spatial_clusters(
    pos: torch.Tensor, cluster_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build padded, balanced node clusters from coordinates.

    Returns:
        clusters: ``[num_clusters, cluster_size]`` local node indices. Padding
            entries are ``-1``.
        mask: Boolean tensor with the same shape; ``True`` marks real nodes.
    """

    if cluster_size <= 0:
        raise ValueError("cluster_size は正の整数である必要があります。")

    order = _morton_order(pos)
    num_nodes = pos.shape[0]
    num_clusters = math.ceil(num_nodes / cluster_size)
    clusters = torch.full(
        (num_clusters, cluster_size), -1, dtype=torch.long, device=pos.device
    )
    mask = torch.zeros(
        (num_clusters, cluster_size), dtype=torch.bool, device=pos.device
    )
    clusters.view(-1)[:num_nodes] = order
    mask.view(-1)[:num_nodes] = True
    return clusters, mask


def clusters_from_labels(
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert one cluster label per node into padded EAGLE memberships."""

    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("cluster labels は [num_nodes] の非空テンソルである必要があります。")
    if torch.any(labels < 0):
        raise ValueError("cluster labels に負の値は使用できません。")

    labels = labels.long()
    unique_labels = torch.unique(labels, sorted=True)
    groups = [torch.nonzero(labels == label, as_tuple=False).flatten() for label in unique_labels]
    max_size = max(group.numel() for group in groups)

    clusters = torch.full(
        (len(groups), max_size), -1, dtype=torch.long, device=labels.device
    )
    mask = torch.zeros_like(clusters, dtype=torch.bool)
    for cluster_index, group in enumerate(groups):
        clusters[cluster_index, : group.numel()] = group
        mask[cluster_index, : group.numel()] = True
    return clusters, mask


def _node_to_cluster(
    clusters: torch.Tensor, mask: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    if clusters.ndim != 2 or mask.shape != clusters.shape:
        raise ValueError("clusters と cluster_mask は同じ2次元形状である必要があります。")
    if clusters.shape[0] == 0 or not torch.all(mask.any(dim=1)):
        raise ValueError("空のクラスタは使用できません。")

    valid_nodes = clusters[mask]
    if valid_nodes.numel() != num_nodes:
        raise ValueError("各ノードはちょうど1つのクラスタに所属する必要があります。")
    if torch.any(valid_nodes < 0) or torch.any(valid_nodes >= num_nodes):
        raise ValueError("clusters に範囲外のノードインデックスがあります。")

    counts = torch.zeros(num_nodes, dtype=torch.long, device=clusters.device)
    counts.index_add_(0, valid_nodes, torch.ones_like(valid_nodes))
    if not torch.all(counts == 1):
        raise ValueError("clusters に重複または未割当のノードがあります。")

    cluster_ids = (
        torch.arange(clusters.shape[0], device=clusters.device)
        .unsqueeze(1)
        .expand_as(clusters)
    )
    mapping = torch.empty(num_nodes, dtype=torch.long, device=clusters.device)
    mapping[valid_nodes] = cluster_ids[mask]
    return mapping


class GraphNetworkBlock(nn.Module):
    """Residual graph-network building block used by EAGLE."""

    def __init__(
        self,
        node_input_size: int,
        node_output_size: int,
        edge_size: int,
        hidden_size: int,
        *,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.edge_mlp = _make_mlp(
            edge_size + 2 * node_input_size,
            edge_size,
            hidden_size,
            layer_norm=layer_norm,
        )
        self.node_mlp = _make_mlp(
            node_input_size + edge_size,
            node_output_size,
            hidden_size,
            layer_norm=layer_norm,
        )
        self.edge_size = edge_size

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        if edge_index.shape[1] == 0:
            updated_edges = edges
            aggregated = nodes.new_zeros((nodes.shape[0], self.edge_size))
        else:
            edge_input = torch.cat(
                [nodes[source], nodes[target], edges], dim=-1
            )
            updated_edges = self.edge_mlp(edge_input)
            aggregated = nodes.new_zeros((nodes.shape[0], self.edge_size))
            aggregated.index_add_(0, target, updated_edges)

        updated_nodes = self.node_mlp(torch.cat([nodes, aggregated], dim=-1))
        return updated_nodes, updated_edges


class FourierPositionEncoder(nn.Module):
    """Absolute and cluster-relative Fourier coordinate encoding."""

    def __init__(
        self, space_dim: int, num_frequencies: int = 8, frequency_start: int = -3
    ):
        super().__init__()
        if space_dim <= 0 or num_frequencies <= 0:
            raise ValueError("space_dim と num_frequencies は正である必要があります。")
        frequencies = (
            2.0
            ** torch.arange(
                frequency_start,
                frequency_start + num_frequencies,
                dtype=torch.float32,
            )
            * torch.pi
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.space_dim = space_dim
        self.embedding_size = space_dim * num_frequencies * 2
        self.node_embedding_size = self.embedding_size * 2

    def embed(self, pos: torch.Tensor) -> torch.Tensor:
        angles = pos.unsqueeze(-1) * self.frequencies.to(dtype=pos.dtype)
        encoded = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return encoded.flatten(start_dim=-2)

    def forward(
        self,
        pos: torch.Tensor,
        clusters: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        safe_clusters = clusters.clamp_min(0)
        member_pos = pos[safe_clusters]
        mask = cluster_mask.unsqueeze(-1).to(pos.dtype)
        centers = (member_pos * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        relative = centers.unsqueeze(1) - member_pos
        relative_embedding = self.embed(relative)
        node_relative = pos.new_zeros((pos.shape[0], self.embedding_size))
        node_relative[clusters[cluster_mask]] = relative_embedding[cluster_mask]

        node_embedding = torch.cat([self.embed(pos), node_relative], dim=-1)
        return node_embedding, self.embed(centers)


class GraphEncoder(nn.Module):
    """Encode node state and local mesh geometry with residual GNN layers."""

    def __init__(
        self,
        node_input_size: int,
        space_dim: int,
        latent_size: int,
        position_size: int,
        num_gnn_layers: int,
    ):
        super().__init__()
        self.node_encoder = _make_mlp(
            node_input_size, latent_size, latent_size, layer_norm=False
        )
        self.edge_encoder = _make_mlp(
            space_dim + 1, latent_size, latent_size, layer_norm=False
        )
        self.gnn_layers = nn.ModuleList(
            [
                GraphNetworkBlock(
                    node_input_size=latent_size + position_size,
                    node_output_size=latent_size,
                    edge_size=latent_size,
                    hidden_size=latent_size,
                    layer_norm=True,
                )
                for _ in range(num_gnn_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        position_encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = self.node_encoder(node_features)
        source, target = edge_index
        relative = pos[source] - pos[target]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        edge_geometry = torch.cat([relative, distance], dim=-1)
        edges = self.edge_encoder(edge_geometry)

        for layer in self.gnn_layers:
            node_input = torch.cat([nodes, position_encoding], dim=-1)
            node_update, edge_update = layer(node_input, edges, edge_index)
            nodes = nodes + node_update
            edges = edges + edge_update
        return nodes, edges


class GraphPooling(nn.Module):
    """Pool variable-size node groups into one token per cluster."""

    def __init__(
        self, latent_size: int, position_size: int, token_size: int
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_size + position_size,
            hidden_size=token_size,
            batch_first=True,
        )
        self.projection = _make_mlp(
            token_size, token_size, token_size, layer_norm=False
        )

    def forward(
        self,
        nodes: torch.Tensor,
        clusters: torch.Tensor,
        position_encoding: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_clusters = clusters.clamp_min(0)
        pooled_input = torch.cat(
            [nodes[safe_clusters], position_encoding[safe_clusters]], dim=-1
        )
        pooled_input = pooled_input * cluster_mask.unsqueeze(-1).to(pooled_input.dtype)

        output, _ = self.gru(pooled_input)
        lengths = cluster_mask.sum(dim=1).long()
        last = output[
            torch.arange(output.shape[0], device=output.device), lengths - 1
        ]
        return self.projection(last)


class ClusterAttentionBlock(nn.Module):
    """Pre-LN global self-attention over the pooled mesh tokens."""

    def __init__(
        self,
        token_size: int,
        cluster_position_size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        attention_size = token_size + cluster_position_size
        if attention_size % num_heads != 0:
            raise ValueError(
                "token_size + cluster_position_size は num_heads で割り切れる必要があります。"
            )

        self.norm_attention = nn.LayerNorm(token_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_projection = nn.Linear(attention_size, token_size)
        self.norm_mlp = nn.LayerNorm(token_size)
        self.mlp = _make_mlp(
            token_size, token_size, token_size, layer_norm=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, cluster_position: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm_attention(tokens)
        attention_input = torch.cat([normalized, cluster_position], dim=-1)
        attended, _ = self.attention(
            attention_input.unsqueeze(0),
            attention_input.unsqueeze(0),
            attention_input.unsqueeze(0),
            need_weights=False,
        )
        tokens = tokens + self.dropout(
            self.attention_projection(attended.squeeze(0))
        )
        return tokens + self.dropout(self.mlp(self.norm_mlp(tokens)))


class GraphRetriever(nn.Module):
    """Broadcast cluster tokens back to nodes and refine them on mesh edges."""

    def __init__(
        self,
        latent_size: int,
        token_size: int,
        position_size: int,
        output_size: int,
    ):
        super().__init__()
        retrieval_size = latent_size + token_size + position_size
        self.gnn = GraphNetworkBlock(
            node_input_size=retrieval_size,
            node_output_size=latent_size,
            edge_size=latent_size,
            hidden_size=latent_size,
            layer_norm=False,
        )
        self.output_mlp = _make_mlp(
            latent_size,
            output_size,
            latent_size,
            layer_norm=False,
            final_tanh=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        nodes: torch.Tensor,
        node_to_cluster: torch.Tensor,
        position_encoding: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_tokens = tokens[node_to_cluster]
        retrieval_input = torch.cat(
            [nodes, node_tokens, position_encoding], dim=-1
        )
        retrieved, _ = self.gnn(retrieval_input, edges, edge_index)
        return self.output_mlp(retrieved)


class EagleMeshTransformer(nn.Module):
    """One-step EAGLE mesh transformer.

    ``forward`` predicts the normalized state increment expected by
    :class:`graphphysics.models.simulator.Simulator`.
    """

    def __init__(
        self,
        node_input_size: int,
        output_size: int,
        *,
        space_dim: int = 2,
        latent_size: int = 128,
        token_size: int = 512,
        num_attention_layers: int = 4,
        num_gnn_layers: int = 4,
        num_heads: int = 4,
        cluster_size: int = 20,
        num_position_frequencies: int = 8,
        position_frequency_start: int = -3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if node_input_size <= 0 or output_size <= 0:
            raise ValueError("node_input_size と output_size は正である必要があります。")
        if latent_size <= 0 or token_size <= 0:
            raise ValueError("latent_size と token_size は正である必要があります。")
        if num_attention_layers <= 0 or num_gnn_layers <= 0:
            raise ValueError("attention/GNN layer 数は正である必要があります。")

        self.space_dim = space_dim
        self.output_size = output_size
        self.cluster_size = cluster_size
        self.position_encoder = FourierPositionEncoder(
            space_dim=space_dim,
            num_frequencies=num_position_frequencies,
            frequency_start=position_frequency_start,
        )
        node_position_size = self.position_encoder.node_embedding_size
        cluster_position_size = self.position_encoder.embedding_size

        self.encoder = GraphEncoder(
            node_input_size=node_input_size,
            space_dim=space_dim,
            latent_size=latent_size,
            position_size=node_position_size,
            num_gnn_layers=num_gnn_layers,
        )
        self.graph_pooling = GraphPooling(
            latent_size=latent_size,
            position_size=node_position_size,
            token_size=token_size,
        )
        self.attention = nn.ModuleList(
            [
                ClusterAttentionBlock(
                    token_size=token_size,
                    cluster_position_size=cluster_position_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_attention_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(token_size)
        self.retriever = GraphRetriever(
            latent_size=latent_size,
            token_size=token_size,
            position_size=node_position_size,
            output_size=output_size,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        clusters: Optional[torch.Tensor] = None,
        cluster_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features は [num_nodes, feature_dim] である必要があります。")
        if pos.ndim != 2 or pos.shape[0] != node_features.shape[0]:
            raise ValueError("pos は [num_nodes, space_dim] である必要があります。")
        if pos.shape[1] != self.space_dim:
            raise ValueError(
                f"設定 space_dim={self.space_dim} に対し pos の次元は {pos.shape[1]} です。"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index は [2, num_edges] である必要があります。")
        if edge_index.numel() and (
            torch.any(edge_index < 0)
            or torch.any(edge_index >= node_features.shape[0])
        ):
            raise ValueError("edge_index に範囲外のノードインデックスがあります。")

        if clusters is None:
            clusters, cluster_mask = build_spatial_clusters(
                pos, self.cluster_size
            )
        elif cluster_mask is None:
            cluster_mask = clusters >= 0
        clusters = clusters.to(device=node_features.device, dtype=torch.long)
        cluster_mask = cluster_mask.to(device=node_features.device, dtype=torch.bool)
        node_to_cluster = _node_to_cluster(
            clusters, cluster_mask, node_features.shape[0]
        )

        node_position, cluster_position = self.position_encoder(
            pos, clusters, cluster_mask
        )
        nodes, edges = self.encoder(
            node_features, pos, edge_index, node_position
        )
        tokens = self.graph_pooling(
            nodes, clusters, node_position, cluster_mask
        )
        for attention_block in self.attention:
            tokens = attention_block(tokens, cluster_position)
        tokens = self.final_norm(tokens)
        return self.retriever(
            tokens,
            nodes,
            node_to_cluster,
            node_position,
            edges,
            edge_index,
        )


# Keep a familiar entry point for users of the upstream EAGLE implementation.
Model = EagleMeshTransformer
