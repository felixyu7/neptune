# model.py
"""
Neptune neutrino event reconstruction.
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any

from .transformers import (
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
    RMSNorm,
)
from .tokenizer import FPSTokenizer
from .packing import pack, build_block_mask, unpack, FLEX_AVAILABLE


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
        layerscale_init=1e-5,
        attn_impl="auto",
    ):
        super().__init__()
        self.token_dim = token_dim
        self.pool_type = pool_type
        # "auto": packed flex on CUDA, padded SDPA on CPU (B=1 there, nothing to
        # pack). "padded": always padded. "packed": force packed where supported.
        # Dense batches and overflow fall back to padded inside pack().
        self.attn_impl = attn_impl
        # The "auto" packed path is only safe once a compiled flex path is active;
        # eager flex_attention materializes the full packed N×N and can OOM. Set
        # True by compile_encoder() on success, cleared on failure/fallback.
        self.packed_flex_ready = False

        enc_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            drop_path_rate=0.0,
            layerscale_init=layerscale_init,
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

    def _use_packed(self, src: Tensor, masks: Optional[Tensor]) -> bool:
        """Packed flex path is used only when it pays off and is supported."""
        if masks is None or not FLEX_AVAILABLE or self.attn_impl == "padded":
            return False
        if self.attn_impl == "packed":
            # flex_attention has no CPU backward; under grad on CPU it raises at
            # the forward call. Fall back to padded there even when forced.
            return src.is_cuda or not torch.is_grad_enabled()
        # "auto": GPU packs only when the compiled flex path is active; CPU (B=1)
        # and uncompiled/failed-compile runs stay on padded SDPA.
        return src.is_cuda and self.packed_flex_ready

    def _encode(self, src: Tensor, centroids: Tensor, masks: Optional[Tensor]) -> Tensor:
        """Run the transformer stack, packed (block-diagonal flex) or padded.

        Pack/unpack use data-dependent shapes and stay eager here; only
        ``self.layers`` (the inner encoder) is compiled.
        """
        if self._use_packed(src, masks):
            packed = pack(src, centroids, masks)
            if packed is not None:  # None == empty/dense -> padded fallback below
                packed_tokens, packed_centroids, doc_id, pack_idx = packed
                block_mask = build_block_mask(doc_id, packed_tokens.shape[1])
                B, S = masks.shape
                # doc_id (= pack_idx // S) indexes original events 0..B-1, so
                # num_docs=B lets DropPath sample one decision per event.
                out = self.layers(packed_tokens, packed_centroids,
                                  block_mask=block_mask, doc_id=doc_id, num_docs=B)
                return unpack(out, pack_idx, B, S)
        attn_pad = (~masks) if masks is not None else None
        return self.layers(src, centroids, src_key_padding_mask=attn_pad)

    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        # Run encoder with centroid-aware inputs (absolute position embedding)
        centroid_emb = self.centroid_mlp(centroids.to(tokens.dtype))
        if masks is not None:
            centroid_emb = centroid_emb * masks.to(dtype=centroid_emb.dtype).unsqueeze(-1)
        x = self._encode(tokens + centroid_emb, centroids, masks)
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
        layerscale_init: float = 1e-5,
        attn_impl: str = "auto",
        compile_encoder: bool = True,
        compile_options: Optional[Dict[str, Any]] = None,
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
            layerscale_init=layerscale_init,
            attn_impl=attn_impl,
        )
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )

        # Auto-compile the encoder for faster train/inference. Compilation is
        # lazy (deferred to the first forward) so it targets the model's final
        # device, and falls back to the uncompiled encoder if it ever fails —
        # so any caller, including downstream inference pipelines, gets the
        # speedup for free with no code changes and no risk of a hard failure.
        self.encoder_compiled = False
        if compile_encoder:
            self.encoder_compiled = self.compile_encoder(**(compile_options or {}))

    def forward(
        self,
        coords: Tensor,       # [N,4] -> [x, y, z, t]
        features: Tensor,     # [N,F]
        batch_ids: Tensor,    # [N]
    ) -> Tensor:
        spatial = coords[:, :3]
        times = coords[:, 3].unsqueeze(-1)

        tokens, centroids, masks = self.tokenizer(spatial, features, batch_ids, times)
        global_feat = self._run_encoder(tokens, centroids, masks)
        return self.head(global_feat)

    def _run_encoder(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor]) -> Tensor:
        """Run the encoder, guarding every compiled call so a backend failure at
        runtime falls back to the uncompiled encoder instead of crashing.

        The guard stays armed (not just on the first forward) because lazy +
        dynamic compilation traces each distinct path on first use: a dense first
        batch compiles the padded path, while the packed/flex path only compiles
        when a later sparse batch hits it. Wrapping every call until the first
        failure (after which compile is disabled, so the wrapper is skipped)
        catches that deferred failure too. try/except is free on the happy path."""
        if not self.encoder_compiled:
            return self.encoder(tokens, centroids, masks)
        try:
            return self.encoder(tokens, centroids, masks)
        except Exception as exc:  # compile/backend failure at runtime
            warnings.warn(
                f"Neptune encoder torch.compile failed at runtime "
                f"({type(exc).__name__}: {exc}); falling back to the "
                "uncompiled encoder."
            )
            self.encoder_compiled = False
            # Route "auto" back to padded SDPA: eager flex would OOM on large
            # packed batches, so the retry must not stay on packed.
            self.encoder.packed_flex_ready = False
            try:
                self.encoder.layers._compiled_call_impl = None  # revert to eager
            except Exception:
                pass
            return self.encoder(tokens, centroids, masks)

    def compile_encoder(self, mode: str = "default", dynamic: bool = True,
                        **compile_kwargs: Any) -> bool:
        """Compile the inner transformer stack for faster train/inference.

        Only ``self.encoder.layers`` (the ``NeptuneTransformerEncoder`` layer
        loop) is compiled. The surrounding ``PointTransformerEncoder`` stays eager
        because pack/unpack use data-dependent shapes (``nonzero``), and the FPS
        tokenizer uses a custom op plus data-dependent control flow — both would
        force graph breaks. ``dynamic=True`` lets one graph serve the varying
        packed length ``[1, N, D]`` and the varying padded batch ``[B, 128, D]``
        without recompiling per shape. Compilation uses ``nn.Module.compile`` (in
        place), so ``state_dict`` keys are unchanged and existing checkpoints stay
        loadable. Benchmarks show ~1.5x train/inference on GPU and ~1.2x on CPU;
        see the perf report.

        Compilation is *lazy*: ``torch.compile`` traces on the first forward, so
        it targets the model's final device (e.g. after ``.to('cuda')``) rather
        than the construction-time device. If any compiled forward fails at
        runtime (missing backend, lowering error, ...), the model transparently
        reverts to the uncompiled encoder (see :meth:`_run_encoder`).

        Called automatically from ``__init__`` (``compile_encoder=True``); may
        also be invoked manually.

        Returns:
            True if compilation was set up, False if torch.compile is unavailable
            in this build (the model then runs uncompiled, exactly as before).
        """
        if not hasattr(torch, "compile") or not hasattr(self.encoder.layers, "compile"):
            warnings.warn(
                "torch.compile is unavailable in this PyTorch build; "
                "running the Neptune encoder uncompiled."
            )
            self.encoder_compiled = False
            return False
        try:
            self.encoder.layers.compile(mode=mode, dynamic=dynamic, **compile_kwargs)
        except Exception as exc:  # setup-time failure (rare; tracing is lazy)
            warnings.warn(
                f"Neptune encoder torch.compile setup failed "
                f"({type(exc).__name__}: {exc}); falling back to the uncompiled "
                "encoder."
            )
            try:
                self.encoder.layers._compiled_call_impl = None
            except Exception:
                pass
            self.encoder_compiled = False
            return False
        # _run_encoder guards every compiled forward so a lazy runtime failure
        # (on either the padded or the packed/flex path) falls back to eager.
        self.encoder_compiled = True
        self.encoder.packed_flex_ready = True
        return True
