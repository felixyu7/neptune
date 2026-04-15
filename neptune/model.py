# model.py
"""
Neptune neutrino event reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, List

from .transformers import (
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
)
from torch.nn import RMSNorm
from .tokenizer import FPSTokenizer


class AttentionPool(nn.Module):
    """Cross-attention pooling with a learned query."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * (dim ** -0.5))
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, S, D] encoder output
            mask: [B, S] bool (True = valid token)
        Returns:
            [B, D] global feature vector
        """
        B, S, D = x.shape
        q = self.q.expand(B, -1, -1)                          # [B, 1, D]
        k, v = self.kv(x).chunk(2, dim=-1)                    # each [B, S, D]

        attn = torch.bmm(q, k.transpose(1, 2)) * (D ** -0.5) # [B, 1, S]
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).squeeze(1)                   # [B, D]
        return self.proj(out)


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
        pool_type="mean",
    ):
        super().__init__()
        self.token_dim = token_dim
        self.pool_type = pool_type

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

        if pool_type == "attention":
            self.pool = AttentionPool(token_dim)

    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        # Run encoder with centroid-aware inputs
        attn_pad = (~masks) if masks is not None else None
        centroid_emb = self.centroid_mlp(centroids.to(tokens.dtype))
        if masks is not None:
            centroid_emb = centroid_emb * masks.to(dtype=centroid_emb.dtype).unsqueeze(-1)
        x = self.layers(tokens + centroid_emb, centroids, src_key_padding_mask=attn_pad)
        x = self.norm(x)

        if self.pool_type == "attention":
            return self.pool(x, masks)

        # Mean pooling (default)
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
        pool_type: str = "mean",
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
            pool_type=pool_type,
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


class SharedBackbone(nn.Module):
    """Shared point-cloud trunk: tokenizer + centroid embedding + N transformer layers.

    Returns pre-pool token features for downstream expert heads. Coordinate
    embedding is added once here; expert heads should not re-add it.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_patches: int = 128,
        token_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 672,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        k_neighbors: int = 8,
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
        self.centroid_mlp = nn.Sequential(
            nn.Linear(4, token_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        enc_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, drop_path_rate=0.0,
        )
        self.layers = NeptuneTransformerEncoder(
            enc_layer, num_layers=num_layers, drop_path_rate=drop_path_rate,
        )

    def forward(self, coords: Tensor, features: Tensor, batch_ids: Tensor):
        spatial = coords[:, :3]
        times = coords[:, 3].unsqueeze(-1)
        tokens, centroids, masks = self.tokenizer(spatial, features, batch_ids, times)

        attn_pad = (~masks) if masks is not None else None
        centroid_emb = self.centroid_mlp(centroids.to(tokens.dtype))
        if masks is not None:
            centroid_emb = centroid_emb * masks.to(dtype=centroid_emb.dtype).unsqueeze(-1)
        x = self.layers(tokens + centroid_emb, centroids, src_key_padding_mask=attn_pad)
        return x, centroids, masks


class ExpertHead(nn.Module):
    """Specialized expert head: M transformer layers + RMSNorm + pool + output MLP.

    Operates on pre-pool token features from a SharedBackbone. Does not re-add
    coordinate embedding (already baked in by the backbone).
    """

    def __init__(
        self,
        token_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        pool_type: str = "mean",
    ):
        super().__init__()
        self.pool_type = pool_type
        enc_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, drop_path_rate=0.0,
        )
        self.layers = NeptuneTransformerEncoder(
            enc_layer, num_layers=num_layers, drop_path_rate=drop_path_rate,
        )
        self.norm = RMSNorm(token_dim)
        if pool_type == "attention":
            self.pool = AttentionPool(token_dim)
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim),
        )

    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor]) -> Tensor:
        attn_pad = (~masks) if masks is not None else None
        x = self.layers(tokens, centroids, src_key_padding_mask=attn_pad)
        x = self.norm(x)

        if self.pool_type == "attention":
            global_feat = self.pool(x, masks)
        elif masks is None:
            global_feat = x.mean(dim=1)
        else:
            weights = masks.to(dtype=x.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            global_feat = (x * weights).sum(dim=1) / denom.squeeze(1)

        return self.head(global_feat)


class NeptuneMoEModel(nn.Module):
    """
    Physics-motivated mixture of experts with shared backbone + specialized heads.

    All events pass through a shared backbone that extracts low-level features from
    the full dataset. Specialized expert heads then do task-specific deeper processing.
    Routing is supervised via morphology classification (ground-truth labels).

    Expert DAG:
        Backbone (shared trunk, all events)
        ├─ Morphology head (6-class router, supervised)
        ├─ Energy: contained (cascade+starting) / uncontained (throughgoing+stopping+bundle)
        └─ Direction: cascade / low-E track / high-E track
        Noise events (uncontained=4) are excluded from energy/direction experts.

    Training: soft routing (all experts run, weighted by router probabilities).
        Noise events are masked from energy/direction losses.
    Inference: hard routing (only relevant expert path fires per event).
        Noise events (P(uncontained) > noise_threshold) get NaN energy/direction.
    """

    CONTAINED_ENERGY = [0, 1]       # cascade, starting track
    UNCONTAINED_ENERGY = [2, 3, 5]  # throughgoing, stopping, bundle
    CASCADE_DIR = [0]               # cascade only
    TRACK_DIR = [1, 2, 3, 5]        # starting, throughgoing, stopping, bundle
    NOISE = [4]                     # uncontained — excluded from energy/direction

    def __init__(
        self,
        backbone: nn.Module,
        morphology_head: nn.Module,
        energy_experts: Dict[str, nn.Module],
        direction_experts: Dict[str, nn.Module],
        energy_gate_threshold: float = 10000.0,
        noise_threshold: float = 0.8,
    ):
        super().__init__()
        self.backbone = backbone
        self.morphology_head = morphology_head
        self.energy_experts = nn.ModuleDict(energy_experts)
        self.direction_experts = nn.ModuleDict(direction_experts)
        self.energy_gate_threshold = nn.Parameter(
            torch.tensor(energy_gate_threshold), requires_grad=False
        )
        self.noise_threshold = noise_threshold
        self.hard_route = False

    def train(self, mode=True):
        super().train(mode)
        self.hard_route = not mode
        return self

    def _soft_forward(
        self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor]
    ) -> Tensor:
        """Soft routing: all experts process all events, outputs weighted by P(morph)."""
        morph_logits = self.morphology_head(tokens, centroids, masks)
        morph_probs = F.softmax(morph_logits.float(), dim=-1).clamp(min=1e-6)

        # Energy experts
        p_cont = morph_probs[:, self.CONTAINED_ENERGY].sum(-1, keepdim=True)
        p_uncont = morph_probs[:, self.UNCONTAINED_ENERGY].sum(-1, keepdim=True)
        e_cont = self.energy_experts['contained'](tokens, centroids, masks)
        e_uncont = self.energy_experts['uncontained'](tokens, centroids, masks)
        energy_pred = p_cont * e_cont + p_uncont * e_uncont

        # Direction experts (track sub-routing via predicted-energy gate)
        p_cas = morph_probs[:, self.CASCADE_DIR].sum(-1, keepdim=True)
        p_track = morph_probs[:, self.TRACK_DIR].sum(-1, keepdim=True)
        log_threshold = torch.log10(self.energy_gate_threshold)
        gate = torch.sigmoid(energy_pred[:, 0:1] - log_threshold)

        d_cas = self.direction_experts['cascade'](tokens, centroids, masks)
        d_low = self.direction_experts['low_track'](tokens, centroids, masks)
        d_high = self.direction_experts['high_track'](tokens, centroids, masks)
        dir_pred = p_cas * d_cas + p_track * (1 - gate) * d_low + p_track * gate * d_high

        return torch.cat([morph_logits, energy_pred, dir_pred], dim=-1)

    def _hard_forward(
        self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor]
    ) -> Tensor:
        """Hard routing: only relevant expert path fires per event."""
        B = tokens.size(0)
        device = tokens.device

        morph_logits = self.morphology_head(tokens, centroids, masks)
        morph_probs = F.softmax(morph_logits.float(), dim=-1)
        morph_class = morph_logits.argmax(dim=-1)
        dtype = morph_logits.dtype  # match expert output dtype (may differ from tokens under autocast)

        e_dim = self.energy_experts['contained'].head[-1].out_features
        d_dim = self.direction_experts['cascade'].head[-1].out_features
        energy_pred = torch.full((B, e_dim), float('nan'), device=device, dtype=dtype)
        dir_pred = torch.full((B, d_dim), float('nan'), device=device, dtype=dtype)

        noise = morph_probs[:, self.NOISE].sum(-1) > self.noise_threshold

        def _run(expert, mask):
            if mask.any():
                return expert(tokens[mask], centroids[mask],
                              masks[mask] if masks is not None else None)
            return None

        # Energy routing
        cont_mask = torch.isin(morph_class, torch.tensor(self.CONTAINED_ENERGY, device=device)) & ~noise
        uncont_mask = torch.isin(morph_class, torch.tensor(self.UNCONTAINED_ENERGY, device=device)) & ~noise

        out = _run(self.energy_experts['contained'], cont_mask)
        if out is not None:
            energy_pred[cont_mask] = out
        out = _run(self.energy_experts['uncontained'], uncont_mask)
        if out is not None:
            energy_pred[uncont_mask] = out

        # Direction routing
        cas_mask = torch.isin(morph_class, torch.tensor(self.CASCADE_DIR, device=device)) & ~noise
        track_mask = torch.isin(morph_class, torch.tensor(self.TRACK_DIR, device=device)) & ~noise

        out = _run(self.direction_experts['cascade'], cas_mask)
        if out is not None:
            dir_pred[cas_mask] = out

        if track_mask.any():
            log_threshold = torch.log10(self.energy_gate_threshold)
            track_idx = track_mask.nonzero(as_tuple=True)[0]
            high_e = energy_pred[track_idx, 0] >= log_threshold
            for sub, key in [(~high_e, 'low_track'), (high_e, 'high_track')]:
                idx = track_idx[sub]
                if idx.numel() > 0:
                    sub_mask = torch.zeros(B, dtype=torch.bool, device=device)
                    sub_mask[idx] = True
                    dir_pred[idx] = self.direction_experts[key](
                        tokens[sub_mask], centroids[sub_mask],
                        masks[sub_mask] if masks is not None else None,
                    )

        return torch.cat([morph_logits, energy_pred, dir_pred], dim=-1)

    def forward(
        self, coords: Tensor, features: Tensor, batch_ids: Tensor
    ) -> Tensor:
        tokens, centroids, masks = self.backbone(coords, features, batch_ids)
        if self.hard_route:
            return self._hard_forward(tokens, centroids, masks)
        return self._soft_forward(tokens, centroids, masks)

