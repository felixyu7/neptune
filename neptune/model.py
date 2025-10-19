# model.py
"""
Neptune neutrino event reconstruction.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any

from .transformers import (
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
)
from torch.nn import RMSNorm
from .tokenizer import FPSTokenizer, LearnedImportanceTokenizer, TokenLearnerTokenizer


class PointTransformerEncoder(nn.Module):
    """Encoder wrapper with centroid-aware inputs."""
    def __init__(
        self,
        token_dim=768,
        num_layers=12,
        num_heads=12,
        hidden_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.token_dim = token_dim

        enc_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.centroid_mlp = nn.Sequential(
            nn.Linear(4, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.layers = NeptuneTransformerEncoder(
            enc_layer,
            num_layers=num_layers,
        )
        self.norm = RMSNorm(token_dim)

    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        # Run encoder with centroid-aware inputs
        attn_pad = (~masks) if masks is not None else None
        centroid_emb = self.centroid_mlp(centroids.to(tokens.dtype))
        if masks is not None:
            centroid_emb = centroid_emb * masks.to(dtype=centroid_emb.dtype).unsqueeze(-1)
        x = self.layers(tokens + centroid_emb, src_key_padding_mask=attn_pad)
        x = self.norm(x)
        if masks is None:
            return x.mean(dim=1)
        weights = masks.to(dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom.squeeze(1)


class NeptuneModel(nn.Module):
    """
    Next-gen Neptune:
      - tokenizer_type: "fps" | "learned_importance" | "tokenlearner"
      - coords are expected as [N, 4] = [x, y, z, t]
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_patches: int = 128,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.1,
        output_dim: int = 3,
        tokenizer_type: str = "learned_importance",
        k_neighbors: int = 8,   # only for fps
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        tokenizer_cfg: Dict[str, Any] = dict(tokenizer_kwargs or {})
        mlp_layers_cfg = tokenizer_cfg.pop("mlp_layers", [256, 512, 768])

        if tokenizer_type == "fps":
            k_val = tokenizer_cfg.pop("k_neighbors", k_neighbors)
            self.tokenizer = FPSTokenizer(
                feature_dim=in_channels,
                max_tokens=num_patches,
                token_dim=token_dim,
                mlp_layers=mlp_layers_cfg,
                k_neighbors=k_val,
                **tokenizer_cfg,
            )
        elif tokenizer_type == "learned_importance":
            tokenizer_cfg.pop("k_neighbors", None)
            self.tokenizer = LearnedImportanceTokenizer(
                feature_dim=in_channels,
                max_tokens=num_patches,
                token_dim=token_dim,
                mlp_layers=mlp_layers_cfg,
                **tokenizer_cfg,
            )
        elif tokenizer_type == "tokenlearner":
            tokenizer_cfg.pop("k_neighbors", None)
            self.tokenizer = TokenLearnerTokenizer(
                feature_dim=in_channels,
                max_tokens=num_patches,
                token_dim=token_dim,
                mlp_layers=mlp_layers_cfg,
                **tokenizer_cfg,
            )
        else:
            raise ValueError(
                f"Unknown tokenizer type: {tokenizer_type}. Expected 'fps', 'learned_importance', or 'tokenlearner'."
            )

        self.encoder = PointTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim, bias=False), nn.GELU(), nn.Dropout(dropout),
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
