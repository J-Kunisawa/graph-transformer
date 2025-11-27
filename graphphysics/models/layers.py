import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

try:
    import dgl.sparse as dglsp
    from dgl.sparse import SparseMatrix

    HAS_DGL_SPARSE = True
except ImportError:
    HAS_DGL_SPARSE = False
    dglsp = None
    SparseMatrix = Any  # Use Any as a placeholder for SparseMatrix


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies RMS normalization over the last dimension.
    """

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__()
        self.d = int(d)
        self.p = float(p)
        self.eps = float(eps)
        self.bias = bool(bias)

        self.scale = nn.Parameter(torch.ones(self.d))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(self.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x / math.sqrt(max(1, d_x))
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


ACTIVATION = {"relu": nn.ReLU, "gelu": nn.GELU}


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
    act: str = "relu",
) -> nn.Module:
    """
    Build a simple MLP with optional RMSNorm on the output.
    """
    assert nb_of_layers >= 2, "nb_of_layers must be >= 2"
    if act not in ACTIVATION:
        raise NotImplementedError(f"Activation '{act}' not supported")
    act_cls = ACTIVATION[act]

    layers = [nn.Linear(in_size, hidden_size), act_cls()]
    for _ in range(nb_of_layers - 2):
        layers.extend([nn.Linear(hidden_size, hidden_size), act_cls()])
    layers.append(nn.Linear(hidden_size, out_size))
    if layer_norm:
        layers.append(RMSNorm(out_size))
    return nn.Sequential(*layers)


class GatedMLP(nn.Module):
    """Gated MLP: activation(linear1(x)) * linear2(x)"""

    def __init__(self, in_size: int, hidden_size: int, expansion_factor: int):
        super().__init__()
        self.linear1 = nn.Linear(in_size, expansion_factor * hidden_size)
        self.linear2 = nn.Linear(in_size, expansion_factor * hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.activation(self.linear1(x))
        right = self.linear2(x)
        return left * right


def build_gated_mlp(in_size: int, hidden_size: int, out_size: int, expansion_factor: int = 3) -> nn.Module:
    layers = [RMSNorm(in_size), GatedMLP(in_size, hidden_size, expansion_factor), nn.Linear(hidden_size * expansion_factor, out_size)]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """
    Running normalizer with safe None handling.

    Usage:
      norm = Normalizer(size)
      x_norm = norm(x)         # returns None if x is None
      x_denorm = norm.inverse(x_norm)
    """

    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**5,
        std_epsilon: float = 1e-8,
        name: str = "Normalizer",
        device: Optional[Union[str, torch.device]] = "cuda",
    ):
        super().__init__()
        # normalize device arg
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.name = name
        self._max_accumulations = int(max_accumulations)
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=dev)

        # running stats stored as buffers on device
        self.register_buffer("_acc_count", torch.tensor(0.0, device=dev))
        self.register_buffer("_num_accumulations", torch.tensor(0.0, device=dev))
        self.register_buffer("_acc_sum", torch.zeros((1, int(size)), dtype=torch.float32, requires_grad=False, device=dev))
        self.register_buffer("_acc_sum_squared", torch.zeros((1, int(size)), dtype=torch.float32, requires_grad=False, device=dev))

    def forward(self, batched_data: Optional[torch.Tensor], accumulate: bool = True) -> Optional[torch.Tensor]:
        # Propagate None safely
        if batched_data is None:
            return None

        if accumulate:
            if self._num_accumulations < self._max_accumulations:
                # detach to avoid tracking history
                try:
                    self._accumulate(batched_data.detach())
                except Exception:
                    # guard against unexpected shapes/types
                    pass

        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if normalized_batch_data is None:
            return None
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: Optional[torch.Tensor]):
        if batched_data is None:
            return
        if batched_data.numel() == 0:
            return
        count = float(batched_data.shape[0])
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_sum = torch.sum(batched_data ** 2, dim=0, keepdim=True)
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_sum
        self._acc_count += count
        self._num_accumulations += 1.0

    def _mean(self) -> torch.Tensor:
        safe_count = torch.max(self._acc_count, torch.tensor(1.0, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self) -> torch.Tensor:
        safe_count = torch.max(self._acc_count, torch.tensor(1.0, device=self._acc_count.device))
        variance = self._acc_sum_squared / safe_count - self._mean() ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.max(std, self._std_epsilon)

    def get_variable(self) -> Dict[str, Any]:
        return {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }


def scaled_query_key_softmax(q: torch.Tensor, k: torch.Tensor, att_mask) -> torch.Tensor:
    """
    Compute scaled query-key softmax. If sparse DGL mask present, use dgl.sparse ops.
    """
    scaling_factor = math.sqrt(max(1.0, float(k.size(1))))
    q = q / scaling_factor

    if att_mask is not None and HAS_DGL_SPARSE:
        attn = dglsp.bsddmm(att_mask, q, k.transpose(1, 0))
        attn = attn.softmax()
    else:
        attn = q @ k.transpose(-2, -1)
        attn = torch.softmax(attn, dim=-1)
    return attn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask=None,
    return_attention: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    attn = scaled_query_key_softmax(q, k, att_mask=att_mask)

    if att_mask is not None and HAS_DGL_SPARSE:
        y = dglsp.bspmm(attn, v)
    else:
        y = attn @ v

    if return_attention:
        return y, attn
    return y


class Attention(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, num_heads=4, use_proj_bias: bool = True, use_separate_proj_weight: bool = True):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.hidden_size = int(output_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(input_dim, self.hidden_size, bias=use_proj_bias)
        self.k_proj = nn.Linear(input_dim, self.hidden_size, bias=use_proj_bias)
        self.v_proj = nn.Linear(input_dim, self.hidden_size, bias=use_proj_bias)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=use_proj_bias)

        if not use_separate_proj_weight:
            with torch.no_grad():
                self.k_proj.weight = self.q_proj.weight
                self.v_proj.weight = self.q_proj.weight

    def forward(self, x: torch.Tensor, adj, return_attention: bool = False):
        N = x.size(0)
        query, key, value = x, x, x

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # reshape to (N, head_dim, num_heads) as in original implementation
        q = q.reshape(N, self.head_dim, self.num_heads)
        k = k.reshape(N, self.head_dim, self.num_heads)
        v = v.reshape(N, self.head_dim, self.num_heads)

        if return_attention:
            y, attn = scaled_dot_product_attention(q, k, v, adj, return_attention=True)
        else:
            y = scaled_dot_product_attention(q, k, v, adj)

        out = self.proj(y.reshape(N, -1))
        if return_attention:
            return out, attn
        return out


class Transformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, activation_layer: torch.nn.Module = nn.ReLU, use_proj_bias: bool = True, use_separate_proj_weight: bool = True):
        super().__init__()
        self.attention = Attention(input_dim=input_dim, output_dim=output_dim, num_heads=num_heads, use_proj_bias=use_proj_bias, use_separate_proj_weight=use_separate_proj_weight)
        self.activation = activation_layer()
        self.norm1 = RMSNorm(output_dim)
        self.norm2 = RMSNorm(output_dim)
        self.gated_mlp = build_gated_mlp(in_size=output_dim, hidden_size=output_dim, out_size=output_dim)
        self.use_adjacency = HAS_DGL_SPARSE

    def forward(self, x: torch.Tensor, adj, return_attention: bool = False):
        if not self.use_adjacency:
            adj = None

        if return_attention:
            x_, attn = self.attention(self.norm1(x), adj, return_attention=True)
            x = x + x_
        else:
            x = x + self.attention(self.norm1(x), adj, return_attention=False)

        x = x + self.gated_mlp(self.norm2(x))
        if return_attention:
            return x, attn
        return x


class GraphNetBlock(MessagePassing):
    def __init__(self, hidden_size: int, nb_of_layers: int = 4, layer_norm: bool = True):
        super().__init__(aggr="add", flow="source_to_target")
        edge_input_dim = 3 * hidden_size
        node_input_dim = 2 * hidden_size
        self.edge_block = build_mlp(in_size=edge_input_dim, hidden_size=hidden_size, out_size=hidden_size, nb_of_layers=nb_of_layers, layer_norm=layer_norm)
        self.node_block = build_mlp(in_size=node_input_dim, hidden_size=hidden_size, out_size=hidden_size, nb_of_layers=nb_of_layers, layer_norm=layer_norm)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], size: int = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # prepare node features for edge update
        row, col = edge_index
        x_i = x[col]  # target
        x_j = x[row]  # source
        edge_attr_in = edge_attr if edge_attr is not None else torch.zeros((row.size(0), x.size(1)), device=x.device, dtype=x.dtype)
        edge_attr_ = self.edge_update(edge_attr_in, x_i, x_j)

        x_ = self.propagate(edge_index, x=x, edge_attr=edge_attr_, size=(x.size(0), x.size(0)))

        if edge_attr is None:
            edge_attr = edge_attr_.clone()
        else:
            edge_attr = edge_attr + edge_attr_
        x = x + x_

        return x, edge_attr

    def edge_update(self, edge_attr: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
        return self.edge_block(edge_input)

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_block(node_input)