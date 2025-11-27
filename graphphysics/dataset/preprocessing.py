import math
import random
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric.transforms as T
from scipy.spatial import cKDTree
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from graphphysics.utils.nodetype import NodeType


def add_edge_features() -> List[Callable[[Data], Data]]:
    """Return transforms to compute basic edge features (Cartesian + Distance)."""
    return [T.Cartesian(norm=False), T.Distance(norm=False)]


def _3d_face_to_edge(graph: Data) -> Data:
    """Safely convert quad face (4 x M) into triangular faces if present."""
    face = getattr(graph, "face", None)
    if face is None:
        return graph
    if face.dim() != 2 or face.size(0) != 4:
        return graph
    graph.face = torch.cat(
        [
            face[0:3],
            face[1:4],
            torch.stack([face[2], face[3], face[0]], dim=0),
            torch.stack([face[3], face[0], face[1]], dim=0),
        ],
        dim=1,
    )
    return graph


def add_obstacles_next_pos(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
    node_type_index: int,
) -> Data:
    """
    Replace obstacle-next-position feature: compute displacement to next frame
    and broadcast mean displacement to non-obstacle nodes.
    """
    if not hasattr(graph, "x") or not hasattr(graph, "y"):
        return graph

    x = graph.x
    y = graph.y

    # safe slicing
    if world_pos_index_end <= world_pos_index_start:
        return graph

    world_pos = x[:, world_pos_index_start:world_pos_index_end]
    other_features = x[:, world_pos_index_end:]

    target_world_pos = y[:, world_pos_index_start:world_pos_index_end]
    if target_world_pos is None:
        return graph

    obstacle_disp = target_world_pos - world_pos

    # guard node_type index
    if node_type_index < 0 or node_type_index >= x.size(1):
        node_type_index = max(0, min(x.size(1) - 1, node_type_index))

    node_type = x[:, node_type_index].long()

    # If no obstacle nodes exist, use global mean
    mask_obstacle = node_type == NodeType.OBSTACLE
    if mask_obstacle.any():
        only_obstacle_disp = obstacle_disp[mask_obstacle]
        mean_obstacle_disp = only_obstacle_disp.mean(dim=0, keepdim=True)
    else:
        mean_obstacle_disp = obstacle_disp.mean(dim=0, keepdim=True)

    obstacle_disp[~mask_obstacle] = mean_obstacle_disp

    graph.x = torch.cat([world_pos, obstacle_disp, other_features], dim=1)
    return graph


def _close_pairs_ckdtree(X: torch.Tensor, max_d: float) -> torch.Tensor:
    """Return pairs (2,N) of indices within max_d using scipy cKDTree; return empty tensor if none."""
    if X.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    pts = X.cpu().numpy()
    try:
        pairs = cKDTree(pts).query_pairs(max_d, output_type="ndarray")
    except Exception:
        return torch.empty((2, 0), dtype=torch.long)
    if pairs.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(pairs.T).long()


def add_world_edges(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
    node_type_index: int,
    radius: float = 0.03,
) -> Data:
    """Add undirected edges between nearby (world) positions, filtering by node types."""
    if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
        return graph

    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    added_edges = _close_pairs_ckdtree(world_pos, radius).to(graph.x.device)

    if added_edges.numel() == 0:
        return graph

    # safe node type extraction
    node_type_index = max(0, min(node_type_index, graph.x.size(1) - 1))
    type_feat = graph.x[:, node_type_index].long()

    if type_feat.numel() == 0:
        return graph

    m1 = torch.gather(type_feat, 0, added_edges[0])
    m2 = torch.gather(type_feat, 0, added_edges[1])

    mask1 = (m1 == NodeType.OBSTACLE) & (m2 == NodeType.NORMAL)
    mask2 = (m1 == NodeType.NORMAL) & (m2 == NodeType.OBSTACLE)
    mask = mask1 | mask2

    if mask.sum() == 0:
        return graph

    added_edges = added_edges[:, mask]

    edge_index = torch.cat([added_edges, graph.edge_index], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

    graph.edge_index = edge_index
    return graph


def add_world_pos_features(
    graph: Data,
    world_pos_index_start: int,
    world_pos_index_end: int,
) -> Data:
    """Append relative world position and its norm to existing edge_attr."""
    if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
        return graph

    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    senders, receivers = graph.edge_index
    if world_pos.numel() == 0:
        return graph

    rel = world_pos[senders] - world_pos[receivers]
    rel_norm = torch.norm(rel, p=2, dim=-1, keepdim=True)

    # If edge_attr is None, initialize zeros with matching device/dtype
    if getattr(graph, "edge_attr", None) is None:
        graph.edge_attr = torch.zeros((rel.size(0), 0), device=rel.device, dtype=rel.dtype)

    graph.edge_attr = torch.cat(
        [graph.edge_attr, rel.type_as(graph.edge_attr), rel_norm.type_as(graph.edge_attr)], dim=-1
    )
    return graph


def add_noise(
    graph: Data,
    noise_index_start: Union[int, List[int]],
    noise_index_end: Union[int, List[int]],
    noise_scale: Union[float, List[float]],
    node_type_index: int,
    t: Optional[float] = None,
) -> Data:
    """Add gaussian noise to specified feature ranges; keep non-NORMAL nodes noise-free."""
    if isinstance(noise_index_start, int):
        noise_index_start = [noise_index_start]
    if isinstance(noise_index_end, int):
        noise_index_end = [noise_index_end]
    if isinstance(noise_scale, float):
        noise_scale = [noise_scale] * len(noise_index_start)

    if len(noise_index_start) != len(noise_index_end):
        raise ValueError("noise_index_start and noise_index_end must have same length")
    if len(noise_scale) != len(noise_index_start):
        raise ValueError("noise_scale must match noise_index ranges")

    if not hasattr(graph, "x"):
        return graph

    node_type_index = max(0, min(node_type_index, graph.x.size(1) - 1))
    node_type = graph.x[:, node_type_index]
    mask_not_normal = node_type != NodeType.NORMAL

    for start, end, scale in zip(noise_index_start, noise_index_end, noise_scale):
        feat = graph.x[:, start:end]
        scale_ = 10 * scale * (1 + math.cos(t * math.pi)) if t is not None else scale
        noise = torch.randn_like(feat) * float(scale_)
        noise[mask_not_normal] = 0
        graph.x[:, start:end] = feat + noise

    return graph


def compute_min_distance_to_type(
    graph: Data, target_type: NodeType, node_types: torch.Tensor
) -> torch.Tensor:
    """Compute min euclidean distance from each node to any node of target_type."""
    pos = getattr(graph, "pos", None)
    if pos is None or pos.numel() == 0:
        return torch.full((graph.num_nodes,), float("inf"))

    type_mask = node_types == target_type
    if not type_mask.any():
        return torch.full((graph.num_nodes,), float("inf"), device=pos.device, dtype=pos.dtype)

    pos_expanded = pos.unsqueeze(1)  # [N,1,3]
    pos_targets = pos[type_mask].unsqueeze(0)  # [1, M, 3]
    dists = torch.sqrt(((pos_expanded - pos_targets) ** 2).sum(dim=-1) + 1e-12)
    min_dists = dists.min(dim=1)[0]
    return min_dists


class Random3DRotate(BaseTransform):
    """Random 3D rotation applied to pos and selected xyz feature ranges."""

    def __init__(self, feature_indices: List[Tuple[int, int]] = None) -> None:
        self.feature_indices = feature_indices or []
        for s, e in self.feature_indices:
            assert e - s == 3, f"feature range must span 3 dims (xyz): {s}-{e}"

    def _get_random_angles(self):
        return [math.radians(random.uniform(-180, 180)) for _ in range(3)]

    def _build_rotation_matrix(self, alpha, beta, gamma):
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        cg, sg = math.cos(gamma), math.sin(gamma)
        mat = [
            [ca * cb, ca * sb * sg + sa * cg, -ca * sb * cg + sa * sg],
            [-sa * cb, -sa * sb * sg + ca * cg, sa * sb * cg + ca * sg],
            [sb, -cb * sg, cb * cg],
        ]
        return torch.tensor(mat, dtype=torch.float32)

    def _rotate_features(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        for start_idx, end_idx in self.feature_indices:
            feat = x[:, start_idx:end_idx]
            rotated = feat @ matrix.to(feat.device, feat.dtype)
            x[:, start_idx:end_idx] = rotated
        return x

    def forward(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        alpha, beta, gamma = self._get_random_angles()
        rot = self._build_rotation_matrix(alpha, beta, gamma)

        # rotate pos if present
        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos
            if pos.dim() == 1:
                pos = pos.view(-1, 3)
            assert pos.size(-1) == 3, "pos must be shape [N,3]"
            data.pos = (pos @ rot.to(pos.device, pos.dtype))

        # rotate specified xyz features
        if hasattr(data, "x") and data.x is not None and self.feature_indices:
            data.x = self._rotate_features(data.x, rot)

        # rotate target/y first 3 dims if present and compatible
        if hasattr(data, "y") and data.y is not None:
            if data.y.dim() == 2 and data.y.size(1) >= 3:
                tgt = data.y[:, 0:3]
                data.y[:, 0:3] = (tgt @ rot.to(tgt.device, tgt.dtype))

        return data


def build_preprocessing(
    noise_parameters: Optional[dict] = None,
    world_pos_parameters: Optional[dict] = None,
    add_edges_features: bool = True,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
) -> T.Compose:
    """Assemble preprocessing pipeline with sensible validation and ordering."""
    transforms: List[Callable[[Data], Data]] = []

    if extra_node_features is not None:
        if not isinstance(extra_node_features, list):
            extra_node_features = [extra_node_features]
        transforms.extend(extra_node_features)

    if world_pos_parameters is not None:
        transforms.extend(
            [
                partial(
                    add_obstacles_next_pos,
                    world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                    world_pos_index_end=world_pos_parameters["world_pos_index_end"],
                    node_type_index=world_pos_parameters["node_type_index"],
                ),
                T.FaceToEdge(remove_faces=False),
                partial(
                    add_world_edges,
                    world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                    world_pos_index_end=world_pos_parameters["world_pos_index_end"],
                    node_type_index=world_pos_parameters["node_type_index"],
                    radius=world_pos_parameters.get("radius", 0.03),
                ),
            ]
        )
        transforms.extend(add_edge_features())
    else:
        transforms.append(T.FaceToEdge(remove_faces=False))
        if add_edges_features:
            transforms.extend(add_edge_features())

    if noise_parameters is not None:
        noise_transform = partial(
            add_noise,
            noise_index_start=noise_parameters["noise_index_start"],
            noise_index_end=noise_parameters["noise_index_end"],
            noise_scale=noise_parameters["noise_scale"],
            node_type_index=noise_parameters["node_type_index"],
        )
        # insert after FaceToEdge to ensure face->edge done before noise if needed
        transforms.insert(1, noise_transform)

    if extra_edge_features is not None:
        if not isinstance(extra_edge_features, list):
            extra_edge_features = [extra_edge_features]
        transforms.extend(extra_edge_features)

    return T.Compose(transforms)