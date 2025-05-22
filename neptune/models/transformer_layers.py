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
# 1.  Geometric aware multi‑head self‑attention                          #
# --------------------------------------------------------------------------- #
class GeomAwareSelfAttention(nn.Module):
    """Multi‑head attention with geometric biases.

    Args
    ----
    d_model:            Transformer hidden size.
    n_heads:            Number of attention heads.
    bias_hidden_dim:    Width of the two‑layer MLP used for the geometric bias.
    dropout:            Dropout applied to the attention weights & projections.
    learnable_geom:     Turn the geometric bias on/off.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        bias_hidden_dim: int = 64,
        dropout: float = 0.0,
        learnable_geom: bool = True,
        num_fourier_features: int = 32,
        c_ice: float = 0.2306,  # speed of light in ice, m/ns (c_vacuum=0.29979 / 1.3 refractive index)
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
        self.num_fourier_features = num_fourier_features
        self.c_squared = c_ice**2

        if self.use_geom:
            if self.num_fourier_features <= 0 or self.num_fourier_features % 2 != 0:
                raise ValueError(
                    "If use_geom is True, num_fourier_features must be a positive even number."
                )
            self.geom_mlp = nn.Sequential(
                nn.Linear(self.num_fourier_features, bias_hidden_dim), # Input is now num_fourier_features
                nn.ReLU(inplace=True),
                nn.Linear(bias_hidden_dim, 1),
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

        if self.use_geom:
            # 2) ---------- build bias matrix (B, N, N) -------------------- #
            bias = 0.0 # Initialize bias term

            # Calculate pairwise spacetime interval ds^2 and its Fourier encoding
            coords_spatial = coords[..., :3]  # (B, N, 3)
            coords_time = coords[..., 3]    # (B, N)

            # Pairwise differences for spatial coordinates
            delta_spatial_x = coords_spatial[:, :, None, 0] - coords_spatial[:, None, :, 0] # (B, N, N)
            delta_spatial_y = coords_spatial[:, :, None, 1] - coords_spatial[:, None, :, 1] # (B, N, N)
            delta_spatial_z = coords_spatial[:, :, None, 2] - coords_spatial[:, None, :, 2] # (B, N, N)

            # Squared spatial distance dr^2 = dx^2 + dy^2 + dz^2
            dr_sq = delta_spatial_x**2 + delta_spatial_y**2 + delta_spatial_z**2  # (B, N, N)

            # Pairwise time difference dt
            delta_t = coords_time[:, :, None] - coords_time[:, None, :]  # (B, N, N)

            # Spacetime interval ds^2 = c^2 * dt^2 - dr^2
            ds_sq = self.c_squared * (delta_t**2) - dr_sq  # (B, N, N)

            # Fourier encoding for ds_sq
            # div_term determines the frequencies for sine and cosine components
            div_term = torch.exp(
                torch.arange(0, self.num_fourier_features, 2, device=src.device, dtype=src.dtype) *
                -(math.log(10000.0) / self.num_fourier_features)
            )  # Shape: (num_fourier_features / 2)

            # Expand ds_sq and div_term for broadcasting
            # ds_sq_expanded shape: (B, N, N, 1)
            # div_term_expanded shape: (1, 1, 1, num_fourier_features / 2)
            fourier_args = ds_sq.unsqueeze(-1) * div_term.view(1, 1, 1, -1)
            # fourier_args shape: (B, N, N, num_fourier_features / 2)

            fourier_feat = torch.cat((torch.sin(fourier_args), torch.cos(fourier_args)), dim=-1)
            # fourier_feat shape: (B, N, N, num_fourier_features)

            geom_contribution = self.geom_mlp(fourier_feat).squeeze(-1)  # (B, N, N)
            bias = bias + geom_contribution # Add new contribution to bias

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
class RelativePosTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with relative position bias. Drop‑in replacement for `nn.TransformerEncoderLayer` (batch_first=True)."""

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
        pre_norm: bool = False,
        num_fourier_features: int = 32, # New parameter with default
        c_ice: float = 0.2306,          # New parameter with default
    ) -> None:
        super().__init__()

        if isinstance(activation, str):
            act_fn: Callable = nn.ReLU() if activation == "relu" else nn.GELU()
        else:
            act_fn = activation

        self.pre_norm = pre_norm

        self.self_attn = GeomAwareSelfAttention(
            d_model,
            nhead,
            bias_hidden_dim=bias_hidden_dim,
            dropout=dropout,
            learnable_geom=learnable_geom,
            num_fourier_features=num_fourier_features, # Pass to GeomAwareSelfAttention
            c_ice=c_ice,                               # Pass to GeomAwareSelfAttention
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
        if self.pre_norm:
            src_norm = self.norm1(src)
            attn_out = self.self_attn(src_norm, coords, src_mask, src_key_padding_mask)
            src = src + self.drop(attn_out)

            src_norm = self.norm2(src)
            ffn_out = self.ffn(src_norm)
            src = src + self.drop(ffn_out)
        else:
            src = src + self.drop(self.self_attn(src, coords, src_mask, src_key_padding_mask))
            src = self.norm1(src)
            src = src + self.drop(self.ffn(src))
            src = self.norm2(src)
        return src

# --------------------------------------------------------------------------- #
# 3.  Stacked encoder                                                        #
# --------------------------------------------------------------------------- #
class RelativePosTransformerEncoder(nn.Module):
    """`nn.TransformerEncoder` clone using `RelativePosTransformerEncoderLayer` that threads `coords` through every layer."""

    def __init__(self, layer: RelativePosTransformerEncoderLayer, num_layers: int, *, norm: Optional[nn.Module] = None):
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