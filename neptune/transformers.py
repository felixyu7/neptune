import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm  # Assume availability (PyTorch >= 2.1)

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even (got dim={dim})"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute cos and sin values up to max_seq_len for efficiency
        self._set_cos_sin_cache(seq_len=self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Pre-compute and cache cos/sin values for efficient lookup."""
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Cache cos and sin values with persistent=False to save checkpoint space
        # (they can be deterministically recreated from inv_freq and max_seq_len)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        original_dtype = x.dtype
        
        # Handle edge case where seq_len > max_seq_len by extending cache
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Efficiently slice pre-computed values instead of recomputing
        cos_freq = self.cos_cached[:seq_len]
        sin_freq = self.sin_cached[:seq_len]
        
        return self.apply_rotary_pos_emb(x, cos_freq, sin_freq, original_dtype)
    
    def apply_rotary_pos_emb(self, x, cos, sin, original_dtype):
        # x shape: (..., seq_len, dim)
        seq_len = x.shape[-2]
        
        # Ensure cos and sin have the right shape for broadcasting
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) if x.ndim == 4 else cos[:seq_len, :]
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0) if x.ndim == 4 else sin[:seq_len, :]
        
        # Convert to float for computation
        x = x.float()
        
        # Split into even and odd indices
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_odd * cos + x_even * sin
        
        # Interleave back
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        out = out.flatten(-2)
        
        return out.to(original_dtype)


class GeGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Initialize with Xavier/Glorot for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize GeGLU weights using Xavier/Glorot initialization."""
        for module in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # GeGLU: GELU(W1 @ x) ⊙ (W3 @ x), then W2
        return self.dropout(self.w2(F.gelu(self.w1(x)) * self.w3(x)))


class NeptuneTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, 
                 layer_norm_eps=1e-5, bias=False, rope_max_seq_len=512, rope_base=10000):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert self.head_dim % 2 == 0, f"head_dim must be even for RoPE (got {self.head_dim})"
        
        # Multi-head attention components - using bias=False for stability
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights using industry-standard practices
        self._initialize_weights()

        # RMSNorm
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        
        # Use GeGLU instead of regular FFN
        self.ffn = GeGLU(d_model, dim_feedforward, dropout, bias=bias)
        
        # RoPE
        self.rope = RoPE(dim=self.head_dim, max_seq_len=rope_max_seq_len, base=rope_base)
        
        # Store hyperparameters for cloning
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.rope_max_seq_len = rope_max_seq_len
        self.rope_base = rope_base
        
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
        
        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)
        
        # Prepare attention mask
        attn_mask = self._prepare_attention_mask(src_key_padding_mask, q.device)
        
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
    
    def _prepare_attention_mask(self, key_padding_mask, device):
        """Convert key padding mask to SDPA format where True = allowed to attend."""
        if key_padding_mask is None:
            return None
        
        # Convert MHA semantics (True = padding) to SDPA semantics (True = allowed)
        allow = ~key_padding_mask.to(torch.bool).to(device)  # (B, S_k)
        return allow.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_k) for broadcasting


class NeptuneTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # Create separate instances for each layer to avoid weight sharing
        self.layers = nn.ModuleList([
            self._get_cloned_layer(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        
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
                rope_max_seq_len=module.rope_max_seq_len,
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
                
                # Scale FFN down projection (w2 in GeGLU)
                with torch.no_grad():
                    layer.ffn.w2.weight.mul_(scale_factor)

    def forward(self, src, src_key_padding_mask=None):
        output = src
        
        for mod in self.layers:
            output = mod(
                output, 
                src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output