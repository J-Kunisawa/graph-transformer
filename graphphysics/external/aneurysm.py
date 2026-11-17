import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType


def aneurysm_node_type(graph: Data) -> torch.Tensor:
    device = graph.x.device
    v_x = graph.x[:, 0]
    wall_inputs = graph.x[:, 3]
    node_type = torch.zeros(v_x.shape, device=device)

    wall_mask = wall_inputs == 1.0

    inflow_mask = torch.logical_and(graph.pos[:, 1] == 0.0, graph.pos[:, 0] <= 0)

    outflow_mask = torch.logical_and(graph.pos[:, 1] == 0.0, graph.pos[:, 0] >= 0)

    node_type[wall_mask] = NodeType.WALL_BOUNDARY
    node_type[inflow_mask] = NodeType.INFLOW
    node_type[outflow_mask] = NodeType.OUTFLOW

    return node_type


def build_features(graph: Data) -> Data:
    device = graph.x.device
    node_type = aneurysm_node_type(graph)

    current_velocity = graph.x[:, 0:3]
    target_velocity = graph.y[:, 0:3]

    if getattr(graph, "previous_data", None) is not None:
        if isinstance(graph.previous_data, dict):
            previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)[:, 0:3]
        else:
            previous_velocity = graph.previous_data[:, 0:3]
    else:
        previous_velocity = current_velocity.clone()

    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    not_inflow_mask = node_type != NodeType.INFLOW
    next_acceleration[not_inflow_mask] = 0
    next_acceleration_unique = next_acceleration.unique()

    mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
        next_acceleration_unique
    )
    min_next_accel = torch.ones(node_type.shape, device=device) * torch.min(
        next_acceleration_unique
    )
    max_next_accel = torch.ones(node_type.shape, device=device) * torch.max(
        next_acceleration_unique
    )

    graph.x = torch.cat(
        (
            graph.x,
            acceleration,
            graph.pos,
            mean_next_accel.unsqueeze(1),
            min_next_accel.unsqueeze(1),
            max_next_accel.unsqueeze(1),
            node_type.unsqueeze(1),
        ),
        dim=1,
    )

    return graph