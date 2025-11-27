import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from graphphysics.models.layers import GraphNetBlock, Transformer, build_mlp
import graphphysics.models.transolver as Transolver

try:
    import dgl.sparse as dglsp

    HAS_DGL_SPARSE = True
except Exception as e:
    HAS_DGL_SPARSE = False
    dglsp = None
    logger.warning(
        f"DGL sparse not available — falling back to torch_geometric TransformerConv. Reason: {e}"
    )


class EncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode GNN.

    - Encoder: node/edge MLP -> hidden
    - Processor: repeated GraphNetBlock
    - Decoder: hidden -> output
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        only_processor: bool = False,
    ):
        super().__init__()
        self.only_processor = only_processor
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=int(node_input_size),
                hidden_size=self.hidden_size,
                out_size=self.hidden_size,
            )
            self.edges_encoder = build_mlp(
                in_size=int(edge_input_size),
                hidden_size=self.hidden_size,
                out_size=self.hidden_size,
            )
            self.decode_module = build_mlp(
                in_size=self.hidden_size,
                hidden_size=self.hidden_size,
                out_size=self.output_size,
                layer_norm=False,
            )

        self.processor_list = nn.ModuleList(
            [GraphNetBlock(hidden_size=self.hidden_size) for _ in range(int(message_passing_num))]
        )

    def forward(self, graph: Data) -> torch.Tensor:
        edge_index = getattr(graph, "edge_index", None)
        if self.only_processor:
            x = graph.x
            edge_attr = getattr(graph, "edge_attr", None)
        else:
            x = self.nodes_encoder(graph.x)
            edge_attr = self.edges_encoder(getattr(graph, "edge_attr", torch.zeros((edge_index.size(1) if edge_index is not None else 0, 0), device=graph.x.device)))

        for block in self.processor_list:
            x, edge_attr = block(x, edge_index, edge_attr)

        if self.only_processor:
            return x
        return self.decode_module(x)


class EncodeTransformDecode(nn.Module):
    """
    Encoder + Transformer-based processor + Decoder.

    Uses DGL sparse attention if available, otherwise torch_geometric.TransformerConv.
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        only_processor: bool = False,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.only_processor = only_processor
        self.output_size = int(output_size)

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=int(node_input_size),
                hidden_size=self.hidden_size,
                out_size=self.hidden_size,
            )
            self.decode_module = build_mlp(
                in_size=self.hidden_size,
                hidden_size=self.hidden_size,
                out_size=self.output_size,
                layer_norm=False,
            )

        if HAS_DGL_SPARSE:
            self.processor_list = nn.ModuleList(
                [
                    Transformer(
                        input_dim=self.hidden_size,
                        output_dim=self.hidden_size,
                        num_heads=int(num_heads),
                        use_proj_bias=use_proj_bias,
                        use_separate_proj_weight=use_separate_proj_weight,
                    )
                    for _ in range(int(message_passing_num))
                ]
            )
        else:
            # fallback: TransformerConv from torch_geometric
            self.processor_list = nn.ModuleList(
                [
                    TransformerConv(
                        in_channels=self.hidden_size,
                        out_channels=self.hidden_size,
                        heads=int(num_heads),
                        concat=False,
                        beta=True,
                    )
                    for _ in range(int(message_passing_num))
                ]
            )

    def forward(self, graph: Data) -> torch.Tensor:
        x = graph.x if self.only_processor else self.nodes_encoder(graph.x)
        edge_index = getattr(graph, "edge_index", None)

        if HAS_DGL_SPARSE:
            # build sparse adjacency for DGL path; robustly handle missing edge_index
            adj = None
            if edge_index is not None:
                try:
                    adj = dglsp.spmatrix(indices=edge_index, shape=(x.shape[0], x.shape[0]))
                except Exception:
                    adj = None
            for block in self.processor_list:
                x = block(x, adj)
        else:
            for block in self.processor_list:
                x = block(x, edge_index)

        if self.only_processor:
            return x
        return self.decode_module(x)


class TransolverProcessor(nn.Module):
    """
    Adapter for Transolver++ model.

    Expects graph.x (N,C), optional graph.pos (N,3).
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.model = Transolver.Model(
            space_dim=0,
            n_layers=int(message_passing_num),
            n_hidden=self.hidden_size,
            dropout=float(dropout),
            n_head=int(num_heads),
            act="gelu",
            mlp_ratio=int(mlp_ratio),
            fun_dim=int(node_input_size),
            out_dim=int(output_size),
            slice_num=int(slice_num),
            ref=int(ref),
            unified_pos=bool(unified_pos),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        x = graph.x.unsqueeze(0)  # (1, N, C)
        pos = graph.pos.unsqueeze(0) if getattr(graph, "pos", None) is not None else None
        condition = getattr(graph, "condition", None)
        out = self.model.forward(x, pos, condition)
        return out.squeeze(0)