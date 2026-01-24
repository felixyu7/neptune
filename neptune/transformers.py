import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm  # Assume availability (PyTorch >= 2.1)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

class RoPE4D(nn.Module):
    """
    4D Rotary Position Embedding for (x, y, z, t) coordinates.

    Implements standard axis-aligned 4D RoPE (maximal toral construction).
    Each coordinate axis gets dedicated rotation planes, ensuring the four
    generators B_x, B_y, B_z, B_t are linearly independent.

    Requirements:
        - dim >= 8 (need at least one 2x2 plane per axis)
        - scales should be chosen so max_angle < 2π for local injectivity
          Example: if x ∈ [0, X_max], ensure s_x * X_max * ω_max < 2π
    """

    def __init__(self, dim, scales=(1.0, 1.0, 1.0, 1.0), base=10000):
        """
        Args:
            dim: Head dimension (must be even and >= 8 for valid 4D RoPE)
            scales: (s_x, s_y, s_z, s_t) scaling factors for coordinate axes
            base: Base for log-spaced frequency schedule
        """
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

        # Build log-spaced frequencies for each axis
        self.register_buffer("freqs_x", self._build_freqs(self.num_planes_per_axis[0], scales[0]))
        self.register_buffer("freqs_y", self._build_freqs(self.num_planes_per_axis[1], scales[1]))
        self.register_buffer("freqs_z", self._build_freqs(self.num_planes_per_axis[2], scales[2]))
        self.register_buffer("freqs_t", self._build_freqs(self.num_planes_per_axis[3], scales[3]))

    def _build_freqs(self, num_bands, scale):
        """Build log-spaced frequency bands: omega_min * rho^(l/(L-1))."""
        if num_bands == 0:
            return torch.zeros(0)
        if num_bands == 1:
            return torch.tensor([1.0 / self.base]) * scale

        exponents = torch.arange(num_bands, dtype=torch.float32) / (num_bands - 1)
        freqs = (1.0 / self.base) * (self.base ** exponents)
        return freqs * scale

    def forward(self, x, coords):
        """
        Apply 4D rotations to Q or K based on spatial-temporal coordinates.

        Args:
            x: (B, H, S, D) queries or keys
            coords: (B, S, 4) coordinates (x, y, z, t)
        Returns:
            Rotated tensor (B, H, S, D)
        """
        B, H, S, D = x.shape
        original_dtype = x.dtype

        # Reshape to planes: (B, H, S, M, 2) where M = D/2
        x_planes = x.float().reshape(B, H, S, D // 2, 2)
        x_complex = torch.view_as_complex(x_planes.contiguous())  # (B, H, S, M)

        # Extract coordinates
        coords = coords.float()
        x_c, y_c, z_c, t_c = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]

        plane_idx = 0

        # Rotate planes for x-axis
        if self.num_planes_per_axis[0] > 0:
            angles = x_c.unsqueeze(-1) * self.freqs_x  # (B, S, L_x)
            angles = angles.unsqueeze(1)  # (B, 1, S, L_x) broadcast over heads
            rot = torch.polar(torch.ones_like(angles), angles)
            end = plane_idx + self.num_planes_per_axis[0]
            x_complex[..., plane_idx:end] = x_complex[..., plane_idx:end] * rot
            plane_idx = end

        # Rotate planes for y-axis
        if self.num_planes_per_axis[1] > 0:
            angles = y_c.unsqueeze(-1) * self.freqs_y
            angles = angles.unsqueeze(1)
            rot = torch.polar(torch.ones_like(angles), angles)
            end = plane_idx + self.num_planes_per_axis[1]
            x_complex[..., plane_idx:end] = x_complex[..., plane_idx:end] * rot
            plane_idx = end

        # Rotate planes for z-axis
        if self.num_planes_per_axis[2] > 0:
            angles = z_c.unsqueeze(-1) * self.freqs_z
            angles = angles.unsqueeze(1)
            rot = torch.polar(torch.ones_like(angles), angles)
            end = plane_idx + self.num_planes_per_axis[2]
            x_complex[..., plane_idx:end] = x_complex[..., plane_idx:end] * rot
            plane_idx = end

        # Rotate planes for t-axis
        if self.num_planes_per_axis[3] > 0:
            angles = t_c.unsqueeze(-1) * self.freqs_t
            angles = angles.unsqueeze(1)
            rot = torch.polar(torch.ones_like(angles), angles)
            end = plane_idx + self.num_planes_per_axis[3]
            x_complex[..., plane_idx:end] = x_complex[..., plane_idx:end] * rot

        # Convert back to real representation
        x_real = torch.view_as_real(x_complex).reshape(B, H, S, D)
        return x_real.to(original_dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, bias=True):
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
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 layer_norm_eps=1e-5, bias=True, rope_scales=(1.0, 1.0, 1.0, 0.2),
                 rope_base=10000, drop_path_rate=0.0):
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

        # SwiGLU FFN
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=bias)

        # 4D RoPE
        self.rope = RoPE4D(dim=self.head_dim, scales=rope_scales, base=rope_base)

        # LayerScale: learnable per-channel scaling for residual branches
        self.gamma_1 = nn.Parameter(1e-5 * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(1e-5 * torch.ones(d_model))

        # Store hyperparameters for cloning
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.rope_scales = rope_scales
        self.rope_base = rope_base

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout
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

    def forward(self, src, centroids, src_key_padding_mask=None):
        batch_size, seq_len, _ = src.shape

        # Self-attention block with pre-norm
        x = src
        x_norm = self.norm1(x)

        # Compute Q, K, V
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, S, D)

        # Apply 4D RoPE to Q and K using spatial-temporal coordinates
        q = self.rope(q, centroids)
        k = self.rope(k, centroids)

        # Prepare attention mask
        attn_mask = self._prepare_attention_mask(src_key_padding_mask, q.device)

        # Apply attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)  # (B, S, H, D)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        # Residual connection with LayerScale
        x = x + self.drop_path1(self.dropout(self.gamma_1 * attn_output))

        # FFN block with pre-norm
        x_norm = self.norm2(x)
        ff_output = self.ffn(x_norm)
        x = x + self.drop_path2(self.gamma_2 * ff_output)

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
                dim_feedforward=module.ffn.w1.out_features,
                dropout=module.dropout.p,
                layer_norm_eps=module.layer_norm_eps,
                bias=module.bias,
                rope_scales=module.rope_scales,
                rope_base=module.rope_base,
                drop_path_rate=drop_path_rate,
            )
        else:
            import copy
            return copy.deepcopy(module)

    def forward(self, src, centroids, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, centroids, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
