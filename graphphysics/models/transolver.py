# --------------------------------------------------------------------------------------
# DISCLAIMER:
# This file is adapted from the TransolverPlusPlus GitHub repository:
# https://github.com/thuml/Transolver_plus/tree/main
# Retrieved on September 17, 2025.
# --------------------------------------------------------------------------------------

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from typing import Optional

from graphphysics.models.layers import build_mlp


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Gumbel-Softmax sampler with optional straight-through hard sampling."""
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / float(tau)
    y = F.softmax(y, dim=-1)
    if hard:
        _, idx = y.max(dim=-1)
        y_hard = torch.zeros_like(y).scatter_(-1, idx.unsqueeze(-1), 1.0)
        y = (y_hard - y).detach() + y
    return y


class Physics_Attention_1D_Eidetic(nn.Module):
    """Physics-aware attention block using learned slice tokens and Gumbel softmax."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, slice_num: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # projection that computes slice logits per head
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        returns: [B, N, C]
        """
        B, N, C = x.shape

        # project and reshape -> [B, H, N, D]
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # slice weights: [B, H, N, G]
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), tau=1.0, hard=False)
        slice_norm = slice_weights.sum(2)  # [B, H, G]

        # aggregate tokens
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        # normalize across nodes
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].clamp_min(1e-8))

        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)

        # scaled dot-product attention on slice tokens
        out_slice = F.scaled_dot_product_attention(q_slice, k_slice, v_slice)
        # project back to nodes using slice_weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class Transolver_plus_block(nn.Module):
    """Single Transolver block: attention + MLP with residuals and optional final projection."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D_Eidetic(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // max(1, num_heads), dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = build_mlp(
            in_size=hidden_dim,
            hidden_size=hidden_dim * mlp_ratio,
            out_size=hidden_dim,
            nb_of_layers=2,
            layer_norm=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        # attention with optional checkpointing for training memory tradeoff
        if self.training:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=True) + fx
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = fx + self.Attn(self.ln_1(fx))
            fx = fx + self.mlp(self.ln_2(fx))

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Model(nn.Module):
    """
    Transolver-like model wrapper.

    Args:
      space_dim: auxiliary spatial dim appended to inputs
      n_layers: number of blocks
      n_hidden: hidden dimension
      dropout: dropout rate
      n_head: number of attention heads
      act: activation
      mlp_ratio: expansion in MLP
      fun_dim: input feature dim
      out_dim: output dim
      slice_num: number of slices for eidetic attention
      ref, unified_pos: optional grid-related behavior (kept for compatibility)
    """

    def __init__(
        self,
        space_dim: int = 1,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 1,
        fun_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
    ):
        super().__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = int(ref)
        self.unified_pos = bool(unified_pos)

        input_dim = fun_dim + (self.ref ** 3 if self.unified_pos else space_dim)
        self.preprocess = build_mlp(
            in_size=input_dim,
            hidden_size=n_hidden * 2,
            out_size=n_hidden,
            nb_of_layers=2,
            layer_norm=False,
            act=act,
        )

        self.n_hidden = int(n_hidden)
        self.space_dim = int(space_dim)
        self.embedding = nn.Linear(3, n_hidden)
        self.blocks = nn.ModuleList(
            [
                Transolver_plus_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(int(n_layers))
            ]
        )
        # small learned placeholder to break symmetry
        self.placeholder = nn.Parameter((1.0 / max(1, n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def get_grid(self, my_pos: torch.Tensor) -> torch.Tensor:
        """Construct a reference grid used when unified_pos is enabled."""
        batchsize = my_pos.shape[0]
        device = my_pos.device
        gridx = torch.linspace(-1.5, 1.5, self.ref, device=device).reshape(1, self.ref, 1, 1, 1).repeat(batchsize, 1, self.ref, self.ref, 1)
        gridy = torch.linspace(0, 2, self.ref, device=device).reshape(1, 1, self.ref, 1, 1).repeat(batchsize, self.ref, 1, self.ref, 1)
        gridz = torch.linspace(-4, 4, self.ref, device=device).reshape(1, 1, 1, self.ref, 1).repeat(batchsize, self.ref, self.ref, 1, 1)
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).reshape(batchsize, self.ref**3, 3)
        # distances from each pos to grid points
        pos = torch.sqrt(((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2).sum(dim=-1)).reshape(batchsize, my_pos.shape[1], -1).contiguous()
        return pos

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor], condition: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [B, N, fun_dim (+ space/unified features)]
        pos: optional [B, N, 3]
        condition: optional condition tensor
        """
        fx = None
        if pos is not None and self.unified_pos:
            new_pos = self.get_grid(pos)
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if condition is not None:
            cond_emb = self.embedding(condition).unsqueeze(1)
            fx = fx + cond_emb

        for block in self.blocks:
            fx = block(fx)
        return fx