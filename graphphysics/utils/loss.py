import enum

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.vectorial_operators import (
    compute_divergence,
    compute_gradient,
    compute_vector_gradient_product,
)


def _device_of(tensor):
    return tensor.device if tensor is not None and isinstance(tensor, torch.Tensor) else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _prepare_mask_for_loss(
    network_output: torch.Tensor,
    node_type: torch.Tensor,
    masks: list,
    selected_indexes: torch.Tensor = None,
):
    """
    node_type に基づくマスクを作成する。
    - masks: list of NodeType or ints
    - selected_indexes: nodes to keep (will be excluded from loss) - optional
    """
    device = _device_of(node_type)
    if not isinstance(masks, (list, tuple)):
        masks = [masks]

    # build boolean mask where True means included in loss
    include_mask = torch.zeros_like(node_type, dtype=torch.bool, device=device)
    for m in masks:
        mval = int(m) if not isinstance(m, torch.Tensor) else m
        include_mask = include_mask | (node_type == mval)

    if selected_indexes is not None:
        n = network_output.shape[0]
        idx = torch.arange(n, device=device)
        # selected_indexes may be on CPU; move to device
        sel = selected_indexes.to(device)
        # nodes_mask True where node is NOT selected (i.e., usable for loss)
        nodes_mask = ~torch.isin(idx, sel)
        include_mask = include_mask & nodes_mask

    return include_mask


class L2Loss(_Loss):
    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output, node_type, masks, selected_indexes)
        errors = ((network_output - target) ** 2)[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output))
        return torch.mean(errors)


class L1SmoothLoss(_Loss):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)

    @property
    def __name__(self):
        return "L1Smooth"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output, node_type, masks, selected_indexes)
        loss_per_elem = F.smooth_l1_loss(network_output, target, reduction="none", beta=self.beta)
        errors = loss_per_elem[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output))
        return torch.mean(errors)


class GradientL2Loss(_Loss):
    @property
    def __name__(self):
        return "GradientL2Loss"

    def forward(
        self,
        graph: Data,
        target_physical: torch.Tensor,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        target_gradient: torch.Tensor = None,
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output_physical, node_type, masks, selected_indexes)

        if network_output_gradient is None:
            network_output_gradient = compute_gradient(graph, network_output_physical, method=gradient_method)
        if target_gradient is None:
            target_gradient = compute_gradient(graph, target_physical, method=gradient_method)

        errors = ((network_output_gradient - target_gradient) ** 2)[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output_physical))
        return torch.mean(errors)


class ConvectionL2Loss(_Loss):
    @property
    def __name__(self):
        return "ConvectionL2Loss"

    def forward(
        self,
        graph: Data,
        target_physical: torch.Tensor,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        target_gradient: torch.Tensor = None,
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output_physical, node_type, masks, selected_indexes)

        network_output_convection = compute_vector_gradient_product(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        target_convection = compute_vector_gradient_product(
            graph,
            target_physical,
            gradient=target_gradient,
            method=gradient_method,
        )
        errors = ((network_output_convection - target_convection) ** 2)[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output_physical))
        return torch.mean(errors)


class DivergenceL2Loss(_Loss):
    @property
    def __name__(self):
        return "DivergenceL2Loss"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output_physical, node_type, masks, selected_indexes)
        divergence = compute_divergence(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        errors = (divergence ** 2)[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output_physical))
        return torch.mean(errors)


class DivergenceL1Loss(_Loss):
    @property
    def __name__(self):
        return "DivergenceL1Loss"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output_physical, node_type, masks, selected_indexes)
        divergence = compute_divergence(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        errors = torch.abs(divergence)[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output_physical))
        return torch.mean(errors)


class DivergenceL1SmoothLoss(_Loss):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)

    @property
    def __name__(self):
        return "DivergenceL1Smooth"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list,
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        mask = _prepare_mask_for_loss(network_output_physical, node_type, masks, selected_indexes)
        divergence = compute_divergence(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        zeros = torch.zeros_like(divergence)
        errors = F.smooth_l1_loss(divergence, zeros, reduction="none", beta=self.beta)[mask]
        if errors.numel() == 0:
            return torch.tensor(0.0, device=_device_of(network_output_physical))
        return torch.mean(errors)


class MultiLoss(_Loss):
    def __init__(self, losses, weights, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        self.weights = weights

    @property
    def __name__(self):
        return "MultiLoss"

    def forward(
        self,
        graph: Data = None,
        network_output_physical: torch.Tensor = None,
        target_physical: torch.Tensor = None,
        gradient_method: str = None,
        return_all_losses: bool = False,
        **kwargs
    ) -> torch.Tensor:
        # prepare optional gradients
        network_output_gradient = None
        target_gradient = None
        if gradient_method is not None:
            network_output_gradient = compute_gradient(graph=graph, field=network_output_physical, method=gradient_method)
            target_gradient = compute_gradient(graph=graph, field=target_physical, method=gradient_method)

        losses = []
        for w, loss in zip(self.weights, self.losses):
            val = loss(
                graph=graph,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=gradient_method,
                network_output_gradient=network_output_gradient,
                target_gradient=target_gradient,
                **kwargs
            )
            losses.append(w * val)

        total = sum(losses) if losses else torch.tensor(0.0, device=_device_of(network_output_physical))
        if return_all_losses:
            return total, losses
        return total


class LossType(enum.Enum):
    L2LOSS = L2Loss
    L1SMOOTHLOSS = L1SmoothLoss
    GRADIENTL2LOSS = GradientL2Loss
    CONVECTIONL2LOSS = ConvectionL2Loss
    DIVERGENCEL2LOSS = DivergenceL2Loss
    DIVERGENCEL1LOSS = DivergenceL1Loss
    DIVERGENCEL1SMOOTHLOSS = DivergenceL1SmoothLoss