# model.py
"""
Neptune (next-gen): transformer for point-cloud neutrino event reconstruction.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .transformers import (
    SwiGLU,                  # from your file
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
)
from .tokenizer import PointCloudTokenizerV1, PointCloudTokenizerV2


class AttentionPool(nn.Module):
    """Masked attention pooling for global feature."""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # x: [B,S,D], mask: [B,S] True=valid
        B, S, D = x.shape
        q = self.query.view(1, 1, D)        # [1,1,D]
        scores = (x * q).sum(-1) / (D ** 0.5)    # [B,S]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,S,1]
        return (x * w).sum(dim=1)                       # [B,D]


class PointTransformerEncoder(nn.Module):
    """Encoder wrapper with centroid-aware 4D RoPE and optional spacetime bias."""
    def __init__(
        self,
        token_dim=768,
        num_layers=12,
        num_heads=12,
        hidden_dim=3072,
        dropout=0.1,
        use_spacetime_bias: bool = True,
        spacetime_bias_layers: int = 2,
        bias_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.token_dim = token_dim

        enc_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        bias_kwargs = bias_kwargs or {}
        self.layers = NeptuneTransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            use_spacetime_bias=use_spacetime_bias,
            spacetime_bias_layers=min(spacetime_bias_layers, num_layers),
            bias_kwargs=bias_kwargs,
        )
        self.norm = nn.RMSNorm(token_dim)
        self.pool = AttentionPool(token_dim)

    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        # Run encoder with centroid-aware 4D RoPE (optional relative bias enabled via config)
        attn_pad = (~masks) if masks is not None else None
        x = self.layers(tokens, centroids, src_key_padding_mask=attn_pad, valid_mask=masks)
        x = self.norm(x)
        # Global pooling (attention pooling; masked)
        return self.pool(x, masks)


class NeptuneModel(nn.Module):
    """
    Next-gen Neptune:
      - tokenizer_type: "v1" | "v2" | "v3" (default "v3")
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
        tokenizer_type: str = "v2",
        use_spacetime_bias: bool = False,
        spacetime_bias_layers: int = 2,
        bias_kwargs: Optional[dict] = None,
        k_neighbors: int = 8,   # only for v1
    ):
        super().__init__()

        if tokenizer_type == "v1":
            self.tokenizer = PointCloudTokenizerV1(
                feature_dim=in_channels, max_tokens=num_patches, token_dim=token_dim, mlp_layers=[256,512,768], k_neighbors=k_neighbors
            )
        elif tokenizer_type == "v2":
            self.tokenizer = PointCloudTokenizerV2(
                feature_dim=in_channels, max_tokens=num_patches, token_dim=token_dim, mlp_layers=[256,512,768]
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        self.encoder = PointTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_spacetime_bias=use_spacetime_bias,
            spacetime_bias_layers=spacetime_bias_layers,
            bias_kwargs=bias_kwargs,
        )
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )

    def forward(
        self,
        coords: Tensor,       # [N,4] or [N,3] (+ time inferred)
        features: Tensor,     # [N,F]
        batch_ids: Tensor,    # [N]
        times: Optional[Tensor] = None,
    ) -> Tensor:
        spatial = coords[:, :3] if coords.size(1) >= 3 else coords
        if times is None:
            if coords.size(1) >= 4: times = coords[:, 3:4]
            elif features is not None and features.size(1) > 0: times = features[:, -1:].clone()
            else: times = spatial.new_zeros((spatial.size(0), 1))

        tokens, centroids, masks = self.tokenizer(spatial, features, batch_ids, times)
        global_feat = self.encoder(tokens, centroids, masks)
        return self.head(global_feat)
