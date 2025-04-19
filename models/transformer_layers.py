from __future__ import annotations

import math
import os
from typing import Optional, Callable, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

# --------------------------------------------------------------------------- #
# 1.  Geometric and delay‑likelihood aware multi‑head self‑attention          #
# --------------------------------------------------------------------------- #
class GeomAwareSelfAttention(nn.Module):
    """Multi‑head attention with geometric and delay‑likelihood biases.

    Args
    ----
    d_model:            Transformer hidden size.
    n_heads:            Number of attention heads.
    bias_hidden_dim:    Width of the two‑layer MLPs used for the bias terms.
    dropout:            Dropout applied to the attention weights & projections.
    learnable_geom:     Turn the geometric bias on/off.
    learnable_delay:    Turn the delay‑likelihood bias on/off.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        bias_hidden_dim: int = 64,
        dropout: float = 0.0,
        learnable_geom: bool = True,
        learnable_delay: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.n_heads = n_heads

        # Linear projections (bias=False to mirror stock Transformer)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # ------------------------- bias networks ------------------------- #
        self.use_geom = learnable_geom
        self.use_delay = learnable_delay

        if self.use_geom:
            self.geom_mlp = nn.Sequential(
                nn.Linear(2, bias_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bias_hidden_dim, 1),
            )
        if self.use_delay:
            self.delay_mlp = nn.Sequential(
                nn.Linear(4, bias_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bias_hidden_dim, 2),  # → shape, rate (raw)
            )

    # ------------------------------------------------------------------- #
    # forward                                                            #
    # ------------------------------------------------------------------- #
    def forward(
        self,
        src: torch.Tensor,               # (B, N, d_model)
        coords: torch.Tensor,            # (B, N, 4)  [x, y, z, t]
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = src.shape

        # 1) Linear projections & head split ---------------------------- #
        q = self.q_proj(src).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(src).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(src).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if self.use_geom or self.use_delay:
            # 2) ---------- build bias matrix (B, N, N) -------------------- #
            bias = 0.0

            # pairwise spatial distance & time difference
            # coords_spatial: (B, N, 3) ; coords_time: (B, N)
            coords_spatial = coords[..., :3]
            coords_time = coords[..., 3]

            # delta: (B, N, N, 3)
            delta = coords_spatial[:, :, None, :] - coords_spatial[:, None, :, :]
            dist = torch.norm(delta, dim=-1)  # (B, N, N)

            dt = coords_time[:, :, None] - coords_time[:, None, :]  # (B, N, N)

            if self.use_geom:
                geom_feat = torch.stack((dist, dt), dim=-1)  # (B, N, N, 2)
                bias = bias + self.geom_mlp(geom_feat).squeeze(-1)  # (B, N, N)

            if self.use_delay:
                # Use the coordinate of DOM j as input to the delay MLP
                xyz_j = coords.unsqueeze(1)[..., :3].expand(-1, N, -1, -1)            # broadcast to (B,N,N,3)
                # Concatenate distance (scalar) and key coordinates (3-vector)
                delay_feat = torch.cat((dist.unsqueeze(-1), xyz_j), dim=-1)  # (B, N, N, 4)
                raw = self.delay_mlp(delay_feat)
                shape = F.softplus(raw[..., 0]) + 1e-4
                rate = F.softplus(raw[..., 1]) + 1e-4

                dt_pos = dt.abs().clamp_min(1e-4)
                bias = bias + Gamma(shape, rate).log_prob(dt_pos)

            attn = attn + bias.unsqueeze(1)  # broadcast to heads

        # 3) masks ------------------------------------------------------- #
        if src_mask is not None:
            attn = attn.masked_fill(src_mask.bool(), float("-inf"))
        if src_key_padding_mask is not None:
            pad = src_key_padding_mask[:, None, None, :].expand(-1, self.n_heads, N, -1)  # (B, H, N, N)
            attn = attn.masked_fill(pad.bool(), float("-inf"))

        weights = self.attn_drop(F.softmax(attn, dim=-1))  # (B, H, N, N)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, d_model)
        return self.out_proj(out)

# --------------------------------------------------------------------------- #
# 2.  Encoder layer                                                          #
# --------------------------------------------------------------------------- #
class BiasTransformerEncoderLayer(nn.Module):
    """Drop‑in replacement for `nn.TransformerEncoderLayer` (batch_first=True)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "relu",
        layer_norm_eps: float = 1e-5,
        bias_hidden_dim: int = 64,
        learnable_geom: bool = True,
        learnable_delay: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(activation, str):
            act_fn: Callable = nn.ReLU() if activation == "relu" else nn.GELU()
        else:
            act_fn = activation

        self.self_attn = GeomAwareSelfAttention(
            d_model,
            nhead,
            bias_hidden_dim=bias_hidden_dim,
            dropout=dropout,
            learnable_geom=learnable_geom,
            learnable_delay=learnable_delay,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        coords: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = src + self.drop(self.self_attn(src, coords, src_mask, src_key_padding_mask))
        src = self.norm1(src)
        src = src + self.drop(self.ffn(src))
        src = self.norm2(src)
        return src

# --------------------------------------------------------------------------- #
# 3.  Stacked encoder                                                        #
# --------------------------------------------------------------------------- #
class BiasTransformerEncoder(nn.Module):
    """`nn.TransformerEncoder` clone that threads `coords` through every layer."""

    def __init__(self, layer: BiasTransformerEncoderLayer, num_layers: int, *, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        coords: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for l in self.layers:
            src = l(src, coords, src_mask, src_key_padding_mask)
        return self.norm(src) if self.norm is not None else src