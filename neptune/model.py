"""
Neptune: A transformer-based point cloud processing model for neutrino event reconstruction.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional
from .transformers import NeptuneTransformerEncoder, NeptuneTransformerEncoderLayer
from .tokenizer import PointCloudTokenizerV1, PointCloudTokenizerV2

class CentroidEncoder(nn.Module):
    """MLP for encoding position information into tokens."""
    
    def __init__(self, in_dim=4, hidden_dims=[64, 256, 768], out_dim=768):
        super().__init__()
        layers = []
        last_dim = in_dim
        # Build MLP with Linear -> GELU -> RMSNorm pattern
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.GELU(),
                nn.RMSNorm(hidden_dim)
            ])
            last_dim = hidden_dim
        # Final projection layer
        layers.append(nn.Linear(last_dim, out_dim)) 
        self.mlp = nn.Sequential(*layers)

    def forward(self, centroids: Tensor) -> Tensor:
        """Args: centroids [B, N, 4]"""
        return self.mlp(centroids)


class PointTransformerEncoder(nn.Module):
    """Transformer encoder for point cloud tokens."""
    
    def __init__(self, token_dim=768, num_layers=12, num_heads=12,
                 hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        
        # Position embedding component
        self.pos_embed = CentroidEncoder(out_dim=token_dim)

        # Custom Neptune transformer
        encoder_layer = NeptuneTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.layers = NeptuneTransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output normalization
        self.norm = nn.RMSNorm(token_dim)
    
    def forward(self, tokens, centroids, masks=None):
        """Process tokens through transformer architecture."""
        # Add positional embeddings
        pos_embed_out = self.pos_embed(centroids)
        tokens = tokens + pos_embed_out

        # Apply transformer layers
        if masks is not None:
            attention_masks = ~masks
            tokens = self.layers(tokens, src_key_padding_mask=attention_masks)
        else:
            tokens = self.layers(tokens)
        
        # Global mean pooling
        if masks is not None:
            # Apply mask to prevent pooling over padding tokens
            masked_tokens = tokens * masks.unsqueeze(-1).float()
            # Compute mean only over valid tokens
            valid_counts = masks.sum(dim=1, keepdim=True).float()
            global_features = masked_tokens.sum(dim=1) / torch.clamp(valid_counts, min=1.0)
        else:
            global_features = torch.mean(tokens, dim=1)
        
        # Apply final normalization
        return self.norm(global_features)


class NeptuneModel(nn.Module):
    """
    Main model class.
    Args:
        in_channels: Number of input feature channels per point
        num_patches: Maximum number of tokens after point cloud tokenization
        token_dim: Dimension of transformer tokens
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Dimension of transformer feed-forward network
        dropout: Dropout rate
        output_dim: Dimension of output (task-dependent)
        k_neighbors: Number of neighbors for point aggregation
        mlp_layers: List of dimensions for tokenizer MLP layers
        tokenizer_type: Type of tokenizer ('point_cloud', 'gumbel_softmax', or 'lightweight')
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
        k_neighbors: int = 16,
        mlp_layers: List[int] = [256, 512, 768],
        tokenizer_type: str = "point_cloud"
    ):
        super().__init__()
        
        # Model Components - choose tokenizer type
        if tokenizer_type == "v1":
            self.tokenizer = PointCloudTokenizerV1(
                feature_dim=in_channels,
                max_tokens=num_patches,
                token_dim=token_dim,
                k_neighbors=k_neighbors,
                mlp_layers=mlp_layers
            )
        elif tokenizer_type == "v2":
            self.tokenizer = PointCloudTokenizerV2(
                feature_dim=in_channels,
                max_tokens=num_patches,
                token_dim=token_dim,
                mlp_layers=mlp_layers
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Supported: 'point_cloud', 'gumbel_softmax', 'lightweight'")
        
        self.encoder = PointTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Classification/regression head
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )

    def forward(
        self,
        coords: Tensor,
        features: Tensor,
        batch_ids: Tensor,
        times: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the Neptune model.
        
        Args:
            coords: Point coordinates [N, 4] with columns [x, y, z, t]
            features: Point features [N, in_channels]
            batch_ids: Batch indices for each point [N]
            times: Optional per-point time feature [N, 1]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        spatial_coords = coords[:, :3] if coords.size(1) >= 3 else coords

        if times is None:
            if coords.size(1) >= 4:
                times = coords[:, 3:4]
            elif features is not None and features.size(1) > 0:
                times = features[:, -1:].clone()
            else:
                times = spatial_coords.new_zeros((spatial_coords.size(0), 1))

        # 1. Tokenize point cloud
        tokens, centroids, masks = self.tokenizer(spatial_coords, features, batch_ids, times)
        
        # 2. Encode with transformer
        global_features = self.encoder(tokens, centroids, masks)
        
        # 3. Apply classification/regression head
        output = self.classifier(global_features)
        
        return output
