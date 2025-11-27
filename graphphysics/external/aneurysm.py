import torch
from torch_geometric.data import Data
from typing import Optional

from graphphysics.utils.nodetype import NodeType


def aneurysm_node_type(graph: Data, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Determine node types for an aneurysm mesh.
    - WALL_BOUNDARY where graph.x[:,3] == 1.0
    - INFLOW where pos[:,1] == 0.0 and pos[:,0] <= 0
    - OUTFLOW where pos[:,1] == 0.0 and pos[:,0] >= 0
    Returns a long tensor on the same device as graph.x by default.
    """
    if device is None:
        device = graph.x.device if hasattr(graph, "x") else torch.device("cpu")

    x = getattr(graph, "x", None)
    pos = getattr(graph, "pos", None)
    if x is None or pos is None:
        # fallback: empty tensor
        return torch.tensor([], dtype=torch.long, device=device)

    v_x = x[:, 0]
    wall_inputs = x[:, 3] if x.size(1) > 3 else torch.zeros_like(v_x)

    node_type = torch.zeros(v_x.shape, dtype=torch.long, device=device)

    wall_mask = wall_inputs == 1.0
    inflow_mask = (pos[:, 1] == 0.0) & (pos[:, 0] <= 0.0)
    outflow_mask = (pos[:, 1] == 0.0) & (pos[:, 0] >= 0.0)

    node_type[wall_mask] = NodeType.WALL_BOUNDARY
    node_type[inflow_mask] = NodeType.INFLOW
    node_type[outflow_mask] = NodeType.OUTFLOW

    return node_type.to(device)


def build_features(graph: Data) -> Data:
    """
    Augment graph.x with aneurysm-specific features:
      - acceleration (current - previous)
      - graph.pos
      - mean/min/max of next_acceleration magnitude (broadcasted)
      - node_type (as a feature column)
    Handles missing previous data gracefully by using zeros.
    """
    device = graph.x.device if hasattr(graph, "x") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    node_type = aneurysm_node_type(graph, device=device)

    # Determine velocity dimensionality using available metadata when possible.
    vel_dim = None

    # 1) Prefer explicit feature_dims attached to Data by meshdata_to_graph
    if hasattr(graph, "feature_dims") and isinstance(graph.feature_dims, dict):
        vel_dim = graph.feature_dims.get("velocity")

    # 2) Fallback to dataset meta if attached (meta.features.velocity.shape -> last element)
    if vel_dim is None and hasattr(graph, "meta") and isinstance(graph.meta, dict):
        try:
            f = graph.meta.get("features", {}).get("velocity", {})
            shape = f.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                # last dimension of the per-node feature shape
                cand = int(shape[-1])
                if cand > 0:
                    vel_dim = cand
        except Exception:
            vel_dim = None

    # 3) Then prefer graph.y shape if present
    if vel_dim is None and hasattr(graph, "y") and graph.y is not None:
        if graph.y.dim() == 1:
            vel_dim = 1
        else:
            vel_dim = graph.y.size(1)

    # 4) Last-resort: assume velocities occupy the first columns of graph.x (clamp to 3)
    if vel_dim is None:
        vel_dim = min(3, graph.x.size(1))

    current_velocity = graph.x[:, 0:vel_dim].to(torch.float32)
    target_velocity = graph.y[:, 0:vel_dim].to(torch.float32) if hasattr(graph, "y") and graph.y is not None else torch.zeros_like(current_velocity)

    # previous velocity: try multiple possible keys, fall back to zeros
    previous_velocity = None
    prev_data = getattr(graph, "previous_data", None)
    if isinstance(prev_data, dict):
        # common key used previously was "Vitesse" (French), try it
        if "Vitesse" in prev_data:
            previous_velocity = torch.as_tensor(prev_data["Vitesse"], device=device, dtype=torch.float32)
        else:
            # try to find a dynamic-like array in previous_data
            for k, v in prev_data.items():
                try:
                    arr = torch.as_tensor(v, device=device, dtype=torch.float32)
                    if arr.ndim == 2 and arr.size(1) >= 3:
                        previous_velocity = arr[:, 0:3]
                        break
                except Exception:
                    continue

    if previous_velocity is None:
        previous_velocity = torch.zeros_like(current_velocity, device=device, dtype=torch.float32)

    # per-node accelerations
    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    # zero out next_acceleration for non-inflow nodes
    not_inflow_mask = node_type != NodeType.INFLOW
    if next_acceleration.numel() > 0:
        next_acceleration = next_acceleration.clone()
        next_acceleration[not_inflow_mask] = 0.0

    # compute scalar statistics (magnitudes) over inflow nodes
    inflow_mask_bool = node_type == NodeType.INFLOW
    if inflow_mask_bool.any():
        inflow_next_accel = next_acceleration[inflow_mask_bool]
        magnitudes = torch.norm(inflow_next_accel, p=2, dim=1)
        mean_scalar = float(magnitudes.mean()) if magnitudes.numel() > 0 else 0.0
        min_scalar = float(magnitudes.min()) if magnitudes.numel() > 0 else 0.0
        max_scalar = float(magnitudes.max()) if magnitudes.numel() > 0 else 0.0
    else:
        mean_scalar = 0.0
        min_scalar = 0.0
        max_scalar = 0.0

    N = graph.x.size(0)
    mean_next_accel_col = torch.full((N, 1), mean_scalar, device=device, dtype=torch.float32)
    min_next_accel_col = torch.full((N, 1), min_scalar, device=device, dtype=torch.float32)
    max_next_accel_col = torch.full((N, 1), max_scalar, device=device, dtype=torch.float32)

    node_type_col = node_type.to(device).unsqueeze(1).to(torch.float32)

    # safe concatenation: ensure shapes match
    parts = [graph.x.to(torch.float32), acceleration.to(torch.float32), graph.pos.to(torch.float32), mean_next_accel_col, min_next_accel_col, max_next_accel_col, node_type_col]
    graph.x = torch.cat([p if p.ndim == 2 else p.unsqueeze(1) for p in parts], dim=1)

    return graph