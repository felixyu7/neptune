from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm  # Assume availability (PyTorch >= 2.1)


class FourDimRoPE(nn.Module):
    """
    Rotary positional embedding that operates directly on (x, y, z, t) coordinates.
    Each attention head is split into four equal sub-blocks that are rotated using the
    corresponding coordinate, preserving RoPE's relative property in continuous space.
    """

    def __init__(self, dim: int, base: float = 10000.0, learnable_axis_scale: bool = True):
        super().__init__()
        assert dim % 8 == 0, f"RoPE head dim must be divisible by 8 (got dim={dim})"
        self.dim = dim
        self.axis_dim = dim // 4
        assert self.axis_dim % 2 == 0, f"Each axis chunk must be even (got axis_dim={self.axis_dim})"

        inv_freqs = []
        for _ in range(4):
            freq_indices = torch.arange(0, self.axis_dim, 2).float()
            inv_freq = base ** (-freq_indices / self.axis_dim)
            inv_freqs.append(inv_freq)
        self.register_buffer("inv_freq", torch.stack(inv_freqs, dim=0))  # [4, axis_dim/2]

        if learnable_axis_scale:
            self.axis_scale = nn.Parameter(torch.ones(4))
        else:
            self.register_buffer("axis_scale", torch.ones(4))
        self.learnable_axis_scale = learnable_axis_scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (B, H, S, Dh)
            positions: (B, S, 4) containing absolute (x, y, z, t)
        Returns:
            Rotated q and k tensors with the same shape as inputs.
        """
        if positions is None:
            raise ValueError("positions must be provided for FourDimRoPE")

        cos_cache, sin_cache = self._compute_trig_cache(positions)
        return self._rotate(q, cos_cache, sin_cache), self._rotate(k, cos_cache, sin_cache)

    def _compute_trig_cache(self, positions: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pos = positions.float()  # (B, S, 4)
        cos_per_axis: List[torch.Tensor] = []
        sin_per_axis: List[torch.Tensor] = []
        device = pos.device
        dtype = pos.dtype

        for axis in range(4):
            coord = pos[..., axis] * self.axis_scale[axis]
            inv_freq = self.inv_freq[axis].to(device=device, dtype=dtype)
            angles = coord.unsqueeze(-1) * inv_freq  # (B, S, axis_dim/2)
            cos = torch.cos(angles).unsqueeze(1)  # (B, 1, S, axis_dim/2)
            sin = torch.sin(angles).unsqueeze(1)  # (B, 1, S, axis_dim/2)
            cos_per_axis.append(cos)
            sin_per_axis.append(sin)

        return cos_per_axis, sin_per_axis

    def _rotate(
        self,
        x: torch.Tensor,
        cos_per_axis: List[torch.Tensor],
        sin_per_axis: List[torch.Tensor],
    ) -> torch.Tensor:
        original_dtype = x.dtype
        chunks = x.split(self.axis_dim, dim=-1)
        rotated_chunks = []

        for axis, chunk in enumerate(chunks):
            chunk = chunk.float()
            cos = cos_per_axis[axis].to(chunk.dtype)
            sin = sin_per_axis[axis].to(chunk.dtype)

            x_even = chunk[..., 0::2]
            x_odd = chunk[..., 1::2]

            rotated_even = x_even * cos - x_odd * sin
            rotated_odd = x_odd * cos + x_even * sin

            combined = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
            rotated_chunks.append(combined)

        return torch.cat(rotated_chunks, dim=-1).to(original_dtype)


class RelativeSpacetimeBias(nn.Module):
    """
    Computes a relative attention bias based on the Minkowski line element between tokens.
    Implements the ds definition from the provided description and projects Fourier features
    into a scalar bias term.
    """

    def __init__(
        self,
        num_freq: int = 16,
        light_speed: float = 1.0,
        clip_value: float = 4.0,
        scale_divisor: float = 1024.0,
    ):
        super().__init__()
        self.light_speed = light_speed
        self.clip_value = clip_value
        self.scale_divisor = scale_divisor

        freqs = torch.logspace(0, num_freq - 1, num_freq, base=2.0)
        self.register_buffer("freqs", freqs)  # [num_freq]
        self.proj = nn.Linear(num_freq * 2, 1)

    def forward(self, positions: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Args:
            positions: (B, S, 4)
            valid_mask: optional (B, S) boolean mask where True denotes a valid token.
        Returns:
            bias: (B, S, S) additive bias or None if S == 0.
        """
        if positions is None or positions.numel() == 0:
            return None

        pos = positions.float()
        B, S, _ = pos.shape
        if S == 0:
            return None

        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, S, S, 4)
        dx, dy, dz, dt = diff.unbind(dim=-1)
        c_dt = self.light_speed * dt
        ds2 = c_dt.square() - dx.square() - dy.square() - dz.square()

        ds = torch.sign(ds2) * torch.sqrt(ds2.abs() + 1e-12)
        ds = ds.clamp(min=-self.clip_value, max=self.clip_value)
        ds = ds / self.scale_divisor

        angles = ds.unsqueeze(-1) * self.freqs.to(ds.device, ds.dtype)  # (B, S, S, num_freq)
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        bias = self.proj(features).squeeze(-1)  # (B, S, S)

        if valid_mask is not None:
            valid_pairs = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
            bias = bias.masked_fill(~valid_pairs, 0.0)

        return bias


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with Xavier/Glorot for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize SwiGLU weights using Xavier/Glorot initialization."""
        for module in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # SwiGLU: SiLU(W1 @ x) ⊙ (W3 @ x), then W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class NeptuneTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        layer_norm_eps=1e-5,
        bias=False,
        rope_base=10000.0,
        learnable_rope_scale=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert self.head_dim % 8 == 0, f"head_dim must be divisible by 8 for 4D RoPE (got {self.head_dim})"
        
        # Multi-head attention components - using bias=False for stability
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights using industry-standard practices
        self._initialize_weights()

        # RMSNorm
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        
        # Use SwiGLU instead of regular FFN
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=bias)
        
        # 4D RoPE
        self.rope = FourDimRoPE(dim=self.head_dim, base=rope_base, learnable_axis_scale=learnable_rope_scale)
        
        # Store hyperparameters for cloning
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.rope_base = rope_base
        self.learnable_rope_scale = learnable_rope_scale
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

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

    def forward(self, src, positions, src_key_padding_mask=None, relative_bias=None):
        batch_size, seq_len, _ = src.shape
        
        # Self-attention block with pre-norm
        x = src
        x_norm = self.norm1(x)
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, S, D)
        
        # Apply 4D RoPE to Q and K
        q, k = self.rope(q, k, positions)
        
        # Prepare attention mask
        attn_mask = self._prepare_attention_mask(src_key_padding_mask, q.device, q.dtype)
        if relative_bias is not None:
            relative_bias = relative_bias.to(dtype=q.dtype, device=q.device).unsqueeze(1)  # (B, 1, S, S)
            attn_mask = relative_bias if attn_mask is None else (relative_bias + attn_mask)
        
        # Apply attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False  # We handle causality manually
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # FFN block with pre-norm
        x_norm = self.norm2(x)
        ff_output = self.ffn(x_norm)
        x = x + ff_output  # Dropout already applied in SwiGLU
        
        return x
    
    def _prepare_attention_mask(self, key_padding_mask, device, dtype):
        """Convert key padding mask to an additive mask with -inf on padded keys."""
        if key_padding_mask is None:
            return None
        
        padding = key_padding_mask.to(torch.bool).to(device)
        if padding.ndim != 2:
            raise ValueError("key_padding_mask must have shape (B, S)")
        mask = torch.zeros(
            padding.size(0),
            1,
            padding.size(1),
            padding.size(1),
            device=device,
            dtype=dtype,
        )
        mask_value = torch.finfo(dtype).min
        mask = mask.masked_fill(padding.unsqueeze(1).unsqueeze(2), mask_value)
        return mask


class NeptuneTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        use_spacetime_bias: bool = False,
        spacetime_bias_layers: int = 0,
        bias_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        # Create separate instances for each layer to avoid weight sharing
        self.layers = nn.ModuleList([
            self._get_cloned_layer(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        self.spacetime_bias_layers = min(spacetime_bias_layers, num_layers) if use_spacetime_bias else 0
        self.relative_bias = RelativeSpacetimeBias(**(bias_kwargs or {})) if use_spacetime_bias else None
        
        # Apply depth-scaled initialization for residual connections
        self._apply_depth_scaled_init()

    def _get_cloned_layer(self, module):
        """Create a new layer with the same parameters"""
        if isinstance(module, NeptuneTransformerEncoderLayer):
            return NeptuneTransformerEncoderLayer(
                d_model=module.d_model,
                nhead=module.nhead,
                dim_feedforward=module.ffn.w1.out_features,
                dropout=module.dropout.p,
                layer_norm_eps=module.layer_norm_eps,
                bias=module.bias,
                rope_base=module.rope_base,
                learnable_rope_scale=module.learnable_rope_scale,
            )
        else:
            import copy
            return copy.deepcopy(module)

    def _apply_depth_scaled_init(self):
        """Apply depth-scaled initialization for stable deep network training."""
        # Scale down residual output projections by 1/sqrt(2*N) for stable training
        # Factor of 2 accounts for two residual connections per layer (attn + ffn)
        scale_factor = 1.0 / (2.0 * self.num_layers) ** 0.5
        
        for layer in self.layers:
            if isinstance(layer, NeptuneTransformerEncoderLayer):
                # Scale attention output projection
                with torch.no_grad():
                    layer.out_proj.weight.mul_(scale_factor)
                
                # Scale FFN down projection (w2 in SwiGLU)
                with torch.no_grad():
                    layer.ffn.w2.weight.mul_(scale_factor)

    def forward(self, src, positions, src_key_padding_mask=None, valid_mask=None):
        output = src
        bias = None
        if self.relative_bias is not None and positions is not None:
            bias = self.relative_bias(positions, valid_mask=valid_mask)
        
        for idx, mod in enumerate(self.layers):
            layer_bias = bias if (bias is not None and idx < self.spacetime_bias_layers) else None
            output = mod(
                output,
                positions,
                src_key_padding_mask=src_key_padding_mask,
                relative_bias=layer_bias,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
