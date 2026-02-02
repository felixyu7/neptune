# model.py
"""
Neptune neutrino event reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, List, Tuple

from .transformers import (
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
    VarlenTransformerEncoder,
    VarlenTransformerEncoderLayer,
)
from torch.nn import RMSNorm
from .tokenizer import FPSTokenizer


# ============================================================================
# Helper functions for varlen attention
# ============================================================================

def compute_cu_seqlens(batch_ids: Tensor) -> Tuple[Tensor, int]:
    """
    Compute cumulative sequence lengths from batch indices.

    Args:
        batch_ids: [N] tensor of batch indices (should be sorted/contiguous)

    Returns:
        cu_seqlens: [B+1] cumulative sequence lengths, e.g., [0, n1, n1+n2, ...]
        max_seqlen: maximum sequence length in the batch
    """
    counts = torch.bincount(batch_ids)  # [B]
    cu_seqlens = F.pad(counts.cumsum(0), (1, 0))  # [B+1], prepend 0
    max_seqlen = int(counts.max().item())
    return cu_seqlens.to(torch.int32), max_seqlen


def packed_global_pool(x: Tensor, batch_ids: Tensor, num_batches: int) -> Tensor:
    """
    Mean pool packed sequences to get [B, D] output using scatter operations.

    Args:
        x: [total_tokens, D] packed features
        batch_ids: [total_tokens] batch indices
        num_batches: B (number of batches)

    Returns:
        pooled: [B, D] mean-pooled features
    """
    D = x.size(-1)
    # Sum pooling via scatter_add
    summed = torch.zeros(num_batches, D, device=x.device, dtype=x.dtype)
    summed.scatter_add_(0, batch_ids.unsqueeze(-1).expand(-1, D), x)
    # Count per batch
    counts = torch.bincount(batch_ids, minlength=num_batches).float().clamp(min=1.0)
    return summed / counts.unsqueeze(-1)


class PointTransformerEncoder(nn.Module):
    """Encoder wrapper with centroid-aware inputs."""
    def __init__(
        self,
        token_dim=768,
        num_layers=12,
        num_heads=12,
        hidden_dim=2048,
        dropout=0.1,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.token_dim = token_dim

        enc_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            drop_path_rate=0.0,
        )
        self.centroid_mlp = nn.Sequential(
            nn.Linear(4, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        self.layers = NeptuneTransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
        )
        self.norm = RMSNorm(token_dim)

    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        # Run encoder with centroid-aware inputs
        attn_pad = (~masks) if masks is not None else None
        centroid_emb = self.centroid_mlp(centroids.to(tokens.dtype))
        if masks is not None:
            centroid_emb = centroid_emb * masks.to(dtype=centroid_emb.dtype).unsqueeze(-1)
        x = self.layers(tokens + centroid_emb, centroids, src_key_padding_mask=attn_pad)
        x = self.norm(x)
        if masks is None:
            return x.mean(dim=1)
        weights = masks.to(dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom.squeeze(1)


class NeptuneModel(nn.Module):
    """
      Note: coords are expected as [N, 4] = [x, y, z, t]
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_patches: int = 128,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        output_dim: int = 3,
        k_neighbors: int = 8,   # only for fps
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        tokenizer_cfg: Dict[str, Any] = dict(tokenizer_kwargs or {})
        mlp_layers_cfg = tokenizer_cfg.pop("mlp_layers", [256, 512, 768])

        tokenizer_dropout = tokenizer_cfg.pop("dropout", dropout)

        self.tokenizer = FPSTokenizer(
            feature_dim=in_channels,
            max_tokens=num_patches,
            token_dim=token_dim,
            mlp_layers=mlp_layers_cfg,
            k_neighbors=k_neighbors,
            dropout=tokenizer_dropout,
            **tokenizer_cfg,
        )

        self.encoder = PointTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )

    def forward(
        self,
        coords: Tensor,       # [N,4] -> [x, y, z, t]
        features: Tensor,     # [N,F]
        batch_ids: Tensor,    # [N]
    ) -> Tensor:
        spatial = coords[:, :3]
        times = coords[:, 3].unsqueeze(-1)

        tokens, centroids, masks = self.tokenizer(spatial, features, batch_ids, times)
        global_feat = self.encoder(tokens, centroids, masks)
        return self.head(global_feat)


# ============================================================================
# Varlen model components
# ============================================================================

class VarlenPointTransformerEncoder(nn.Module):
    """
    Encoder for varlen format - operates on packed sequences.

    Unlike PointTransformerEncoder which uses padded [B,S,D] format,
    this operates on packed [total_tokens, D] format throughout.
    """

    def __init__(
        self,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim

        enc_layer = VarlenTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            drop_path_rate=0.0,
        )

        # Coordinate embedding MLP (same as original PointTransformerEncoder)
        self.coord_mlp = nn.Sequential(
            nn.Linear(4, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )

        self.layers = VarlenTransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
        )
        self.norm = RMSNorm(token_dim)

    def forward(
        self,
        tokens: Tensor,        # [total_tokens, token_dim]
        coords: Tensor,        # [total_tokens, 4]
        batch_ids: Tensor,     # [total_tokens]
        cu_seqlens: Tensor,    # [B+1]
        max_seqlen: int,
        num_batches: int
    ) -> Tensor:
        """
        Args:
            tokens: Packed features [total_tokens, token_dim]
            coords: Packed 4D coordinates [total_tokens, 4]
            batch_ids: Batch indices [total_tokens]
            cu_seqlens: Cumulative sequence lengths [B+1]
            max_seqlen: Max sequence length
            num_batches: Number of batches B

        Returns:
            global_features: [B, token_dim]
        """
        # Add coordinate embedding
        coord_emb = self.coord_mlp(coords.to(tokens.dtype))  # [N, token_dim]
        x = tokens + coord_emb

        # Run transformer layers
        x = self.layers(x, coords, cu_seqlens, max_seqlen)
        x = self.norm(x)

        # Global pooling: packed -> [B, token_dim]
        return packed_global_pool(x, batch_ids, num_batches)


class NeptuneVarlenModel(nn.Module):
    """
    Neptune model variant using varlen attention.

    Key differences from NeptuneModel:
    - No tokenization (FPS/kNN) - uses all points directly
    - No padding - uses packed sequences with cu_seqlens
    - Requires GPU (CUDA A100+) and BF16/FP16
    - Feature projection MLPs are retained

    Input/output interface matches NeptuneModel for easy switching.

    Note: coords are expected as [N, 4] = [x, y, z, t]
    """

    def __init__(
        self,
        in_channels: int = 6,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        output_dim: int = 3,
        mlp_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.token_dim = token_dim

        if mlp_layers is None:
            mlp_layers = [256, 512, 768]

        # Feature projection MLP (replaces tokenizer's MLP1)
        mlp = []
        in_dim = in_channels
        for out_dim in mlp_layers:
            mlp += [nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = out_dim
        mlp += [nn.Linear(in_dim, token_dim)]
        self.feature_mlp = nn.Sequential(*mlp)

        # Token refinement MLP (equivalent to tokenizer's MLP2)
        self.refine_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim)
        )

        # Varlen encoder
        self.encoder = VarlenPointTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )

    def forward(
        self,
        coords: Tensor,       # [N, 4] -> [x, y, z, t]
        features: Tensor,     # [N, F]
        batch_ids: Tensor,    # [N]
    ) -> Tensor:
        """
        Args:
            coords: [N, 4] coordinates (x, y, z, t)
            features: [N, F] point features
            batch_ids: [N] batch indices (should be sorted/contiguous)

        Returns:
            output: [B, output_dim] predictions
        """
        # Validate GPU requirement
        if not coords.is_cuda:
            raise RuntimeError(
                "NeptuneVarlenModel requires CUDA. "
                "Use NeptuneModel for CPU/MPS support."
            )

        # Compute cu_seqlens from batch_ids
        cu_seqlens, max_seqlen = compute_cu_seqlens(batch_ids)
        num_batches = cu_seqlens.size(0) - 1

        # Feature projection (no downsampling)
        tokens = self.feature_mlp(features)  # [N, token_dim]
        tokens = self.refine_mlp(tokens)     # [N, token_dim]

        # Encoder with varlen attention
        global_feat = self.encoder(
            tokens, coords, batch_ids, cu_seqlens, max_seqlen, num_batches
        )

        return self.head(global_feat)
