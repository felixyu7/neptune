from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Standard sequence-wise RoPE."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even (got dim={dim})")
        inv_freq = base ** (-torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq)

    def _cached_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos, sin = self._cached_cos_sin(seq_len, q.device, q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class LayerNormNoBias(nn.Module):
    """LayerNorm variant retaining scale but no bias."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.weight.numel(),), weight=self.weight, bias=None, eps=self.eps)


class GeGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.output_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.input_proj(x)
        gate, value = up.chunk(2, dim=-1)
        return self.dropout(self.output_proj(F.gelu(gate) * value))


class NeptuneTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        layer_norm_eps=1e-5,
        rope_base=10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE (got {self.head_dim})"

        # Multi-head attention components (biasless)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Initialize weights using industry-standard practices
        self._initialize_weights()

        # LayerNorm without bias
        self.norm1 = LayerNormNoBias(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNormNoBias(d_model, eps=layer_norm_eps)

        # Use GeGLU instead of regular FFN
        self.ffn = GeGLU(d_model, dim_feedforward, dropout)

        # Standard RoPE
        self.rope = RotaryEmbedding(dim=self.head_dim, base=rope_base)

        # Store hyperparameters for cloning
        self.layer_norm_eps = layer_norm_eps
        self.rope_base = rope_base

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def _initialize_weights(self):
        """Initialize weights using modern industry-standard practices."""
        # Q, K, V projections: Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.qkv_proj.weight)

        # Output projection: Xavier/Glorot but may be scaled later for deep networks
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, src, src_key_padding_mask=None):
        batch_size, seq_len, _ = src.shape

        # Self-attention block with pre-norm
        x = src
        x_norm = self.norm1(x)

        # Compute Q, K, V
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, S, D)

        # Apply standard RoPE to Q and K
        q, k = self.rope(q, k)

        # Prepare attention mask
        attn_mask = self._prepare_attention_mask(src_key_padding_mask, q.device, q.dtype)

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
        x = x + ff_output  # Dropout already applied in GeGLU

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
    ):
        super().__init__()
        # Create separate instances for each layer to avoid weight sharing
        self.layers = nn.ModuleList([
            self._get_cloned_layer(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

        # Apply depth-scaled initialization for residual connections
        self._apply_depth_scaled_init()

    def _get_cloned_layer(self, module):
        """Create a new layer with the same parameters"""
        if isinstance(module, NeptuneTransformerEncoderLayer):
            return NeptuneTransformerEncoderLayer(
                d_model=module.d_model,
                nhead=module.nhead,
                dim_feedforward=module.ffn.hidden_dim,
                dropout=module.dropout.p,
                layer_norm_eps=module.layer_norm_eps,
                rope_base=module.rope_base,
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
                
                # Scale FFN down projection
                with torch.no_grad():
                    layer.ffn.output_proj.weight.mul_(scale_factor)

    def forward(self, src, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output
