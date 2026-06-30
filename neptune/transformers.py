from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import flex_attention


class RMSNorm(nn.Module):
    """Drop-in for nn.RMSNorm that casts weight to input dtype so the fused
    bf16/fp16 kernel is used under autocast (stdlib keeps weight fp32 and
    falls back to an unfused path)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor, doc_id: Optional[torch.Tensor] = None,
                num_docs: Optional[int] = None) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        if doc_id is None:
            # Padded path: one stochastic-depth decision per sample (batch dim).
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            binary_tensor = random_tensor.floor()
            return x.div(keep_prob) * binary_tensor
        # Packed path: x is [1, N, D]. Sample one decision per original event,
        # then gather to tokens by doc_id so every token from an event shares it
        # — matching the per-event granularity of the padded path.
        event_mask = (keep_prob + torch.rand((num_docs, 1), dtype=x.dtype, device=x.device)).floor().div(keep_prob)
        token_mask = event_mask.index_select(0, doc_id).unsqueeze(0)  # [1, N, 1]
        return x * token_mask


class FourierPositionEncoder(nn.Module):
    """Absolute position encoding via log-spaced sinusoidal (Fourier) features.

    Lifts the spatial coordinates (x, y, z) through sin/cos at multiple
    log-spaced frequencies before a small MLP, mitigating the spectral bias of a
    raw-coordinate MLP (Tancik et al. 2020 / NeRF) so the encoder can represent
    sharp position dependence (boundaries, depth structure). Time is dropped:
    events are time-centered so absolute t is ~meaningless, and relative time is
    handled by RoPE in attention. It is a pure, deterministic function of
    position (identical for MC and data) and runs eagerly outside the compiled
    encoder, so there are no torch.compile/dynamic-shape concerns.
    """

    def __init__(self, token_dim, in_dim=3, num_bands=20, freq_min=3.0,
                 freq_max=180.0, axis_scales=(1.0, 1.0, 1.0), dropout=0.1):
        super().__init__()
        assert num_bands >= 1, f"num_bands must be >= 1 (got {num_bands})"
        assert len(axis_scales) == in_dim, (
            f"axis_scales must have length in_dim={in_dim} (got {len(axis_scales)})"
        )
        self.in_dim = in_dim
        self.num_bands = num_bands

        # Log-spaced angular frequencies in [freq_min, freq_max] (rad per coord unit).
        if num_bands == 1:
            freqs = torch.tensor([float(freq_min)])
        else:
            exponents = torch.arange(num_bands, dtype=torch.float32) / (num_bands - 1)
            freqs = float(freq_min) * (float(freq_max) / float(freq_min)) ** exponents
        self.register_buffer("freqs", freqs)                                   # (num_bands,)
        self.register_buffer("axis_scales",
                             torch.tensor(axis_scales, dtype=torch.float32))   # (in_dim,)

        feat_dim = in_dim * num_bands * 2  # sin + cos per (axis, band)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, coords):
        # coords: (B, S, >=in_dim); use only the first in_dim axes (drops time).
        # Trig in fp32 for accuracy; the MLP follows the surrounding autocast dtype.
        xyz = coords[..., :self.in_dim].float()                            # (B, S, in_dim)
        scaled = self.freqs.unsqueeze(0) * self.axis_scales.unsqueeze(1)   # (in_dim, num_bands)
        ang = xyz.unsqueeze(-1) * scaled                                   # (B, S, in_dim, num_bands)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)      # (B, S, in_dim*2*num_bands)
        return self.mlp(feats)


class RoPE4D(nn.Module):
    """
    4D Rotary Position Embedding for (x, y, z, t) coordinates.

    Implements standard axis-aligned 4D RoPE (maximal toral construction).
    Each coordinate axis gets dedicated rotation planes, ensuring the four
    generators B_x, B_y, B_z, B_t are linearly independent.

    Pair j rotates dims (j, D/2+j) using contiguous chunk+cat on the last
    dim — each side of the rotation is a contiguous half, which is much
    faster than a strided (2j, 2j+1) layout.
    """

    def __init__(self, dim, scales=(1.0, 1.0, 1.0, 1.0), base=10000):
        super().__init__()
        assert dim % 2 == 0, f"RoPE4D dim must be even (got {dim})"
        assert dim >= 8, f"RoPE4D requires dim >= 8 for 4 axes (got {dim})"
        self.dim = dim
        self.scales = scales
        self.base = base

        # Divide dim into M = dim/2 planes
        num_planes = dim // 2

        # Allocate at least one plane per axis first (ensures linear independence)
        base_allocation = [1, 1, 1, 1]
        remaining = num_planes - 4

        # Distribute remaining planes round-robin
        for i in range(remaining):
            base_allocation[i % 4] += 1

        self.num_planes_per_axis = base_allocation

        # Build concatenated frequency vector for vectorized computation
        all_freqs = []
        for n_planes, scale in zip(base_allocation, scales):
            all_freqs.append(self._build_freqs(n_planes, scale))
        self.register_buffer("freqs", torch.cat(all_freqs))  # (D/2,)

        # Build coordinate selection indices: coord_select[plane] = axis_index
        coord_select = []
        for axis_idx, n_planes in enumerate(base_allocation):
            coord_select.extend([axis_idx] * n_planes)
        self.register_buffer("coord_select", torch.tensor(coord_select, dtype=torch.long))

    def _build_freqs(self, num_bands, scale):
        """Build log-spaced frequency bands: omega_min * rho^(l/(L-1))."""
        if num_bands == 0:
            return torch.zeros(0)
        if num_bands == 1:
            return torch.tensor([1.0 / self.base]) * scale

        exponents = torch.arange(num_bands, dtype=torch.float32) / (num_bands - 1)
        freqs = (1.0 / self.base) * (self.base ** exponents)
        return freqs * scale

    def compute_tables(self, coords, dtype=None):
        """Precompute (cos, sin) rotation tables for a given centroid tensor.

        Returns a pair of (B, 1, S, D/2) tensors. Trig is always evaluated in
        float32 for accuracy; if `dtype` is supplied (e.g. the model dtype),
        the tables are cast to it once so per-layer rotation can stay in that
        dtype and skip an fp32 round-trip.

        Every layer in NeptuneTransformerEncoder sees the same centroids, so
        the encoder computes these once and shares them across all layers.
        """
        coords_f = coords if coords.dtype == torch.float32 else coords.float()
        coord_per_plane = coords_f.index_select(dim=-1, index=self.coord_select)
        angles = (coord_per_plane * self.freqs).unsqueeze(1)  # (B, 1, S, D/2)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        if dtype is not None and dtype != cos_a.dtype:
            cos_a = cos_a.to(dtype)
            sin_a = sin_a.to(dtype)
        return cos_a, sin_a

    def forward(self, x, coords, tables=None):
        """
        Apply 4D rotations to Q or K based on spatial-temporal coordinates.

        Rotation: (a, b) -> (a cos θ - b sin θ, a sin θ + b cos θ) with
        pair j = (j, D/2+j).

        Args:
            x: (B, H, S, D) queries or keys
            coords: (B, S, 4) coordinates (x, y, z, t)
            tables: optional precomputed (cos, sin) pair from compute_tables.
                    Avoids recomputing the transcendentals for every layer/Q/K.
        Returns:
            Rotated tensor (B, H, S, D)
        """
        if tables is None:
            tables = self.compute_tables(coords, dtype=x.dtype)
        cos_a, sin_a = tables

        # Fast path: same-dtype tables — stay entirely in x's dtype.
        if cos_a.dtype == x.dtype:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat(
                (x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a),
                dim=-1,
            )

        # Mixed-dtype fallback: do the math in fp32, cast back.
        x_f = x if x.dtype == torch.float32 else x.float()
        x1, x2 = x_f.chunk(2, dim=-1)
        out = torch.cat(
            (x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a),
            dim=-1,
        )
        return out if out.dtype == x.dtype else out.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, bias=True):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in (self.w13, self.w2):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.dropout(self.w2(F.silu(a) * b))


class NeptuneTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 layer_norm_eps=1e-5, bias=True, ff_bias=False, qk_norm=True,
                 rope_scales=(180.0, 180.0, 180.0, 40.0), rope_base=60, drop_path_rate=0.0,
                 layerscale_init=1e-5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE4D (got {self.head_dim})"

        # Multi-head attention components
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._initialize_weights()

        # RMSNorm
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)

        # Pre-RoPE QK-norm
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=layer_norm_eps)

        # SwiGLU FFN
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=ff_bias)

        # 4D RoPE
        self.rope = RoPE4D(dim=self.head_dim, scales=rope_scales, base=rope_base)

        # LayerScale: learnable per-channel scaling for residual branches
        self.layerscale_init = layerscale_init
        self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(d_model))

        # Store hyperparameters for cloning
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.ff_bias = ff_bias
        self.rope_scales = rope_scales
        self.rope_base = rope_base

        # Dropout (residual-branch only; attention-matrix dropout is intentionally
        # unused — flex_attention can't express it, so dropping it keeps the
        # packed and padded paths regularized identically).
        self.dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.drop_path_rate = drop_path_rate

    def _initialize_weights(self):
        """Initialize weights using modern industry-standard practices."""
        # Q, K, V projections: Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        
        # Output projection: Xavier/Glorot but may be scaled later for deep networks
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _attn(self, q, k, v, attn_mask, block_mask):
        """Self-attention dispatch.

        ``block_mask`` (a flex BlockMask) selects the packed/varlen path used on
        GPU — block-diagonal attention over a single packed sequence. Otherwise
        the standard padded SDPA path runs. Attention-matrix dropout is not used
        on either path (flex_attention can't express it), so the two paths stay
        identically regularized; residual/FFN dropout and DropPath still apply.
        """
        if block_mask is not None:
            return flex_attention(q, k, v, block_mask=block_mask)
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    def forward(self, src, centroids, src_key_padding_mask=None,
                rope_tables=None, attn_mask=None, block_mask=None,
                doc_id=None, num_docs=None):
        batch_size, seq_len, _ = src.shape

        # Self-attention block with pre-norm
        x = src
        x_norm = self.norm1(x)

        # Compute Q, K, V
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, S, D)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply 4D RoPE to Q and K using spatial-temporal coordinates
        q = self.rope(q, centroids, tables=rope_tables)
        k = self.rope(k, centroids, tables=rope_tables)

        # Use the pre-built mask from the encoder when available; otherwise
        # build one from the per-layer key-padding mask (single-layer callers).
        # Skipped entirely on the packed path, which masks via block_mask.
        if attn_mask is None and block_mask is None:
            attn_mask = self._prepare_attention_mask(src_key_padding_mask, q.device)

        # Apply attention (padded SDPA or packed block-diagonal flex)
        attn_output = self._attn(q, k, v, attn_mask, block_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        # Residual connection with LayerScale
        x = x + self.drop_path1(self.dropout(self.gamma_1 * attn_output), doc_id, num_docs)

        # FFN block with pre-norm
        x_norm = self.norm2(x)
        ff_output = self.ffn(x_norm)
        x = x + self.drop_path2(self.gamma_2 * ff_output, doc_id, num_docs)

        return x
    
    def _prepare_attention_mask(self, key_padding_mask, device):
        """Convert key padding mask to SDPA format where True = allowed to attend."""
        if key_padding_mask is None:
            return None
        
        # Convert MHA semantics (True = padding) to SDPA semantics (True = allowed)
        allow = ~key_padding_mask.to(torch.bool).to(device)  # (B, S_k)
        return allow.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_k) for broadcasting


class NeptuneTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, drop_path_rate=0.0):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_layers == 1:
            drop_rates = [drop_path_rate]
        else:
            drop_rates = [drop_path_rate * float(i) / (num_layers - 1) for i in range(num_layers)]
        # Create separate instances for each layer to avoid weight sharing
        self.layers = nn.ModuleList([
            self._get_cloned_layer(encoder_layer, drop_path_rate=drop_rates[i]) for i in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def _get_cloned_layer(self, module, drop_path_rate=0.0):
        """Create a new layer with the same parameters"""
        if isinstance(module, NeptuneTransformerEncoderLayer):
            return NeptuneTransformerEncoderLayer(
                d_model=module.d_model,
                nhead=module.nhead,
                dim_feedforward=module.ffn.w13.out_features // 2,
                dropout=module.dropout.p,
                layer_norm_eps=module.layer_norm_eps,
                bias=module.bias,
                ff_bias=module.ff_bias,
                qk_norm=module.qk_norm,
                rope_scales=module.rope_scales,
                rope_base=module.rope_base,
                drop_path_rate=drop_path_rate,
                layerscale_init=module.layerscale_init,
            )
        else:
            import copy
            return copy.deepcopy(module)

    def forward(self, src, centroids, src_key_padding_mask=None, block_mask=None,
                doc_id=None, num_docs=None):
        # All layers share the same (dim, scales, base) RoPE parameters and
        # receive the same centroids, so the cos/sin tables are identical.
        # Pre-cast tables to src.dtype so per-layer rotation skips the fp32
        # round-trip on bf16/fp16 paths.
        first = self.layers[0] if self.layers else None
        rope_tables = (
            first.rope.compute_tables(centroids, dtype=src.dtype)
            if first is not None else None
        )
        # Build the SDPA mask once instead of rebuilding it per layer. On the
        # packed path attention is masked by block_mask, so no SDPA mask is built.
        attn_mask = (
            first._prepare_attention_mask(src_key_padding_mask, src.device)
            if first is not None and block_mask is None else None
        )

        output = src
        for mod in self.layers:
            output = mod(
                output,
                centroids,
                rope_tables=rope_tables,
                attn_mask=attn_mask,
                block_mask=block_mask,
                doc_id=doc_id,
                num_docs=num_docs,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


