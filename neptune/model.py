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
    SwiGLU,
    RoPE4D,
    DropPath,
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


class MoEFFN(nn.Module):
    """DeepSeek-style MoE FFN: 1 shared SwiGLU + N routed SwiGLUs.

    Drop-in replacement for a transformer block's FFN. Shared expert runs for every
    token; routed experts are weighted by externally supplied routing probabilities
    (physics-supervised via morphology in our case).

    Output = shared(x) + Σ_c w_c · routed_c(x)    (soft routing)
           = shared(x) + routed[argmax(w_per_event)](x)    (hard routing, per-event dispatch)
    """

    def __init__(
        self,
        dim: int,
        shared_hidden: int,
        routed_hidden: int,
        num_routed: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.num_routed = num_routed
        self.shared = SwiGLU(dim, shared_hidden, dropout, bias)
        self.routed = nn.ModuleList(
            [SwiGLU(dim, routed_hidden, dropout, bias) for _ in range(num_routed)]
        )

    def forward(
        self, x: Tensor, routing_weights: Tensor, hard_route: bool = False
    ) -> Tensor:
        """
        Args:
            x: [B, K, D]
            routing_weights: [B, num_routed] — softmax probabilities (detached, temp-applied upstream)
            hard_route: if True, dispatch one routed expert per event by argmax
        """
        shared_out = self.shared(x)
        if hard_route:
            top = routing_weights.argmax(dim=-1)  # [B]
            routed_out = torch.zeros_like(shared_out)
            for c in range(self.num_routed):
                mask = (top == c)
                if mask.any():
                    routed_out[mask] = self.routed[c](x[mask])
        else:
            # Stack all routed outputs and weighted-sum (soft routing)
            stacked = torch.stack([e(x) for e in self.routed], dim=1)  # [B, N, K, D]
            w = routing_weights.to(stacked.dtype).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
            routed_out = (stacked * w).sum(dim=1)
        return shared_out + routed_out


class MoETransformerLayer(nn.Module):
    """NeptuneTransformerEncoderLayer with MoEFFN replacing SwiGLU.

    Attention block is identical to NeptuneTransformerEncoderLayer (pre-norm, 4D RoPE,
    scaled dot-product attention, LayerScale + DropPath). FFN block uses MoEFFN with
    a single LayerScale applied to the combined shared+routed output (faithful to
    DeepSeek's single-residual formulation).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        shared_hidden_dim: int,
        routed_hidden_dim: int,
        num_routed_experts: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        rope_scales=(1.0, 1.0, 1.0, 0.2),
        rope_base: int = 10000,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0
        assert self.head_dim % 2 == 0

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        self.rope = RoPE4D(dim=self.head_dim, scales=rope_scales, base=rope_base)

        self.ffn = MoEFFN(
            dim=d_model,
            shared_hidden=shared_hidden_dim,
            routed_hidden=routed_hidden_dim,
            num_routed=num_routed_experts,
            dropout=dropout,
            bias=bias,
        )

        self.gamma_1 = nn.Parameter(1e-5 * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(1e-5 * torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        routing_weights: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        hard_route: bool = False,
    ) -> Tensor:
        B, S, _ = src.shape

        # Attention block (identical to NeptuneTransformerEncoderLayer)
        x = src
        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm).reshape(B, S, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q, centroids)
        k = self.rope(k, centroids)

        attn_mask = None
        if src_key_padding_mask is not None:
            allow = ~src_key_padding_mask.to(torch.bool).to(q.device)
            attn_mask = allow.unsqueeze(1).unsqueeze(2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        attn_output = self.out_proj(attn_output)
        x = x + self.drop_path1(self.dropout(self.gamma_1 * attn_output))

        # FFN block: MoE FFN with shared + routed
        x_norm = self.norm2(x)
        ff_output = self.ffn(x_norm, routing_weights, hard_route=hard_route)
        x = x + self.drop_path2(self.gamma_2 * ff_output)
        return x


class MoETransformerEncoder(nn.Module):
    """Stack of MoETransformerLayers with staggered drop-path rates."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        shared_hidden_dim: int,
        routed_hidden_dim: int,
        num_routed_experts: int,
        num_layers: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if num_layers == 1:
            drop_rates = [drop_path_rate]
        else:
            drop_rates = [drop_path_rate * i / (num_layers - 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            MoETransformerLayer(
                d_model=d_model, nhead=nhead,
                shared_hidden_dim=shared_hidden_dim,
                routed_hidden_dim=routed_hidden_dim,
                num_routed_experts=num_routed_experts,
                dropout=dropout, drop_path_rate=drop_rates[i],
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        routing_weights: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        hard_route: bool = False,
    ) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, centroids, routing_weights, src_key_padding_mask, hard_route)
        return x


class NeptuneMoEModel(nn.Module):
    """DeepSeek-faithful MoE for IceCube event reconstruction.

    Structure:
        SharedBackbone (standard transformer stack) → pre-pool token features
        MorphRouter (ExpertHead, output=6) → morph logits + physics-supervised routing
        MoETransformerEncoder (per-layer MoE FFNs: 1 shared + N routed, morph-routed)
        Final RMSNorm + pool → pooled features
        Simple task heads: energy_head (D→D→2), dir_head (D→D→dir_dim)

    Training: soft routing — all N routed FFNs run at every MoE layer, outputs
      weighted by softmax(morph_logits / T). Routing weights are detached so the
      router is trained purely by morphology CE loss. Temperature prevents
      routing collapse as the morph classifier becomes peaky.
    Inference (model.eval() → hard_route=True): per-event dispatch — at each MoE
      layer, only the shared FFN + top-1 routed FFN fire per event. Events with
      argmax morph class == 4 (uncontained/noise) get NaN energy/direction.

    Output format [B, 6+2+dir_dim] is preserved for MoELoss compatibility.
    """

    NOISE_CLASS = 4  # uncontained: excluded from energy/direction losses and masked at inference

    def __init__(
        self,
        backbone: nn.Module,
        morph_router: nn.Module,
        expert_stack: nn.Module,
        final_norm: nn.Module,
        pool: Optional[nn.Module],
        energy_head: nn.Module,
        dir_head: nn.Module,
        route_temperature: float = 1.5,
        detach_routing: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.morph_router = morph_router
        self.expert_stack = expert_stack
        self.final_norm = final_norm
        self.pool = pool
        self.energy_head = energy_head
        self.dir_head = dir_head
        self.route_temperature = route_temperature
        self.detach_routing = detach_routing
        self.hard_route = False

    def train(self, mode=True):
        super().train(mode)
        self.hard_route = not mode
        return self

    def _pool(self, features: Tensor, masks: Optional[Tensor]) -> Tensor:
        if self.pool is not None:
            return self.pool(features, masks)
        if masks is None:
            return features.mean(dim=1)
        w = masks.to(dtype=features.dtype).unsqueeze(-1)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (features * w).sum(dim=1) / denom.squeeze(1)

    def forward(
        self, coords: Tensor, features: Tensor, batch_ids: Tensor
    ) -> Tensor:
        # 1. Shared backbone
        tokens, centroids, masks = self.backbone(coords, features, batch_ids)

        # 2. Morph router → routing signal (+ morphology prediction)
        morph_logits = self.morph_router(tokens, centroids, masks)

        # 3. Routing weights: fp32 softmax + temperature + clamp, cast to token dtype, detach
        w = F.softmax(morph_logits.float() / self.route_temperature, dim=-1).clamp_min(1e-6)
        w = w.to(tokens.dtype)
        if self.detach_routing:
            w = w.detach()

        # 4. MoE expert stack (per-layer shared + routed FFNs)
        attn_pad = (~masks) if masks is not None else None
        x = self.expert_stack(
            tokens, centroids, w,
            src_key_padding_mask=attn_pad,
            hard_route=self.hard_route,
        )
        x = self.final_norm(x)

        # 5. Pool
        pooled = self._pool(x, masks)

        # 6. Simple task heads
        energy_pred = self.energy_head(pooled)
        dir_pred = self.dir_head(pooled)

        # 7. Noise masking at inference (argmax-based)
        if self.hard_route:
            noise = morph_logits.argmax(dim=-1) == self.NOISE_CLASS
            if noise.any():
                energy_pred = energy_pred.clone()
                dir_pred = dir_pred.clone()
                energy_pred[noise] = float('nan')
                dir_pred[noise] = float('nan')

        return torch.cat([morph_logits, energy_pred, dir_pred], dim=-1)

