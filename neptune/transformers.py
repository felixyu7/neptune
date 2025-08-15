import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        device = x.device
        original_dtype = x.dtype
        
        # Generate position indices
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Create rotation matrices
        cos_freq = freqs.cos()
        sin_freq = freqs.sin()
        
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
        
        # Apply rotation - CORRECTED FORMULA
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_odd * cos + x_even * sin
        
        # Interleave back
        out = torch.stack([rotated_even, rotated_odd], dim=-1)
        out = out.flatten(-2)
        
        return out.to(original_dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: SiLU(W1 @ x) ⊙ (W3 @ x), then W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class NeptuneTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, 
                 activation="gelu", layer_norm_eps=1e-5, bias=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Multi-head attention components - using bias=False for stability
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize output projection to zero for better training stability
        nn.init.zeros_(self.out_proj.weight)
        if bias:
            nn.init.zeros_(self.out_proj.bias)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Use SwiGLU instead of regular FFN
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=bias)
        
        # RoPE
        self.rope = RoPE(dim=self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
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
        attn_mask = self._prepare_attention_mask(
            batch_size, seq_len, is_causal, src_key_padding_mask, q.device
        )
        
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
    
    def _prepare_attention_mask(self, batch_size, seq_len, is_causal, key_padding_mask, device):
        """Prepare and combine attention masks"""
        attn_mask = None
        
        # Causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 
                diagonal=1
            )
            attn_mask = causal_mask
        
        # Key padding mask
        if key_padding_mask is not None:
            # key_padding_mask shape: (B, S) - True for valid positions
            padding_mask = ~key_padding_mask.bool()  # Invert: True for positions to mask
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            
            if attn_mask is None:
                attn_mask = padding_mask
            else:
                # Combine masks
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
                attn_mask = attn_mask.expand(batch_size, 1, -1, -1)  # (B, 1, S, S)
                attn_mask = attn_mask | padding_mask.expand(-1, -1, seq_len, -1)
        
        return attn_mask


class NeptuneTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # Create separate instances for each layer to avoid weight sharing
        self.layers = nn.ModuleList([
            self._get_cloned_layer(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def _get_cloned_layer(self, module):
        """Create a new layer with the same parameters"""
        if isinstance(module, NeptuneTransformerEncoderLayer):
            return NeptuneTransformerEncoderLayer(
                d_model=module.d_model,
                nhead=module.nhead,
                dim_feedforward=module.ffn.w1.out_features,
                dropout=module.dropout.p
            )
        else:
            import copy
            return copy.deepcopy(module)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        
        for mod in self.layers:
            output = mod(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal
            )

        if self.norm is not None:
            output = self.norm(output)

        return output