"""
Neptune: A transformer-based point cloud processing model for neutrino event reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
from .transformers import NeptuneTransformerEncoder, NeptuneTransformerEncoderLayer
import fpsample


class AttentionPooling(nn.Module):
    """Efficient attention pooling for neighborhood feature aggregation."""
    
    def __init__(self, token_dim: int, use_position: bool = True):
        super().__init__()
        self.token_dim = token_dim
        self.use_position = use_position
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        
        # Linear projections for keys and values
        self.key_proj = nn.Linear(token_dim, token_dim)
        self.value_proj = nn.Linear(token_dim, token_dim)
        
        # Optional positional encoding for spatial-temporal relationships
        if use_position:
            self.pos_encoding = nn.Linear(4, token_dim)  # 4D: x, y, z, time
        
        self.scale = token_dim ** -0.5
        
    def forward(self, input_features: Tensor, positions: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply attention pooling to input features.
        
        Args:
            input_features: [batch_size, seq_len, token_dim] or [num_tokens, k_neighbors, token_dim]
            positions: [batch_size, seq_len, 4] or [num_tokens, k_neighbors, 4] - spatial-temporal coordinates
            mask: [batch_size, seq_len] - mask for valid tokens (for global pooling)
            
        Returns:
            pooled_features: [batch_size, token_dim] or [num_tokens, token_dim]
        """
        if input_features.dim() == 3 and mask is not None:
            # Global pooling mode: [batch_size, seq_len, token_dim]
            batch_size, _, token_dim = input_features.shape
            
            # Single learnable query for global pooling
            query = self.query.expand(batch_size, 1, token_dim)  # [batch_size, 1, token_dim]
            
            # Project input features to keys and values
            keys = self.key_proj(input_features)    # [batch_size, seq_len, token_dim]
            values = self.value_proj(input_features) # [batch_size, seq_len, token_dim]
            
            # Add positional encoding to keys if available
            if self.use_position and positions is not None:
                pos_embed = self.pos_encoding(positions)  # [batch_size, seq_len, token_dim]
                keys = keys + pos_embed
                
            # Compute attention scores: query @ keys^T
            scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # [batch_size, 1, seq_len]
            
            # Apply mask to prevent attention to padding tokens
            if mask is not None:
                # mask: True = valid, False = padding
                # Need to mask out positions where mask is False (padding tokens)
                attention_mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
                scores = scores.masked_fill(~attention_mask, float('-inf'))
            
            # Apply softmax to get attention weights
            weights = F.softmax(scores, dim=-1)  # [batch_size, 1, seq_len]
            
            # Apply weights to values for pooled output
            pooled = torch.matmul(weights, values)  # [batch_size, 1, token_dim]
            
            return pooled.squeeze(1)  # [batch_size, token_dim]
            
        else:
            # Neighborhood pooling mode: [num_tokens, k_neighbors, token_dim]
            num_tokens, _, token_dim = input_features.shape
            
            # Expand learnable query for all tokens
            query = self.query.expand(num_tokens, 1, token_dim)  # [num_tokens, 1, token_dim]
            
            # Project neighborhood features to keys and values
            keys = self.key_proj(input_features)    # [num_tokens, k_neighbors, token_dim]
            values = self.value_proj(input_features) # [num_tokens, k_neighbors, token_dim]
            
            # Add positional encoding to keys if available (relative positions)
            if self.use_position and positions is not None:
                pos_embed = self.pos_encoding(positions)  # [num_tokens, k_neighbors, token_dim]
                keys = keys + pos_embed
                
            # Compute attention scores: query @ keys^T
            scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # [num_tokens, 1, k_neighbors]
            
            # Apply softmax to get attention weights
            weights = F.softmax(scores, dim=-1)  # [num_tokens, 1, k_neighbors]
            
            # Apply weights to values for pooled output
            pooled = torch.matmul(weights, values)  # [num_tokens, 1, token_dim]
            
            return pooled.squeeze(1)  # [num_tokens, token_dim]


class PointCloudTokenizer(nn.Module):
    """Converts point cloud data into tokens for transformer processing."""
    
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: List[int] = [256, 512, 768],
                 k_neighbors: int = 16, 
                 pool_method: str = 'max'):
        super().__init__()
        if pool_method not in ['max', 'mean', 'attention']:
            raise ValueError(f"pool_method must be 'max', 'mean', or 'attention', got {pool_method}")
            
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.pool_method = pool_method
        
        # Per-Point Feature MLP
        mlp = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim
        mlp.append(nn.Linear(in_dim, token_dim))
        self.mlp = nn.Sequential(*mlp)
        
        # Neighborhood Aggregation MLP
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )
        
        # Attention pooling for neighborhood aggregation
        if pool_method == 'attention':
            self.neighborhood_attention = AttentionPooling(
                token_dim=token_dim,
                use_position=True
            )
        
    def forward(self, coordinates: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_indices = coordinates[:, 0].long()
        xyz = coordinates[:, 1:4] 
        time = coordinates[:, 4:5]

        batch_size = batch_indices.max().item() + 1 if coordinates.numel() > 0 else 0
        if batch_size == 0:
            # Handle empty input
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=coordinates.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=coordinates.device, dtype=coordinates.dtype)
            empty_masks = torch.zeros((0, self.max_tokens), device=coordinates.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        all_tokens = []
        all_centroids = []
        all_masks = []
        
        # Process each point cloud in the batch individually
        for batch_idx in range(batch_size):
            batch_mask_indices = batch_indices == batch_idx
            batch_xyz = xyz[batch_mask_indices]  # [M, 3]
            batch_features_data = features[batch_mask_indices]  # [M, F]
            batch_time = time[batch_mask_indices]  # [M, 1]
            num_points = batch_xyz.shape[0]
            
            if num_points == 0:
                # Handle empty point clouds within the batch
                batch_tokens = torch.zeros(self.max_tokens, self.token_dim, device=coordinates.device, dtype=features.dtype)
                batch_centroids = torch.zeros(self.max_tokens, 4, device=coordinates.device, dtype=coordinates.dtype)
                batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
                all_tokens.append(batch_tokens)
                all_centroids.append(batch_centroids)
                all_masks.append(batch_mask_valid.unsqueeze(0))
                continue

            # Combine spatial coordinates and time for FPS and neighborhood search: [M, 4]
            points_for_sampling = torch.cat([batch_xyz, batch_time], dim=-1)
            
            # Apply Per-Point MLP
            all_point_features = self.mlp(batch_features_data)
            
            if num_points <= self.max_tokens:
                # Use all points if fewer than max_tokens
                batch_centroids = points_for_sampling
                batch_tokens = all_point_features
                num_valid_tokens = num_points
            else:
                # Select centroids using Farthest Point Sampling (bucket_fps_kdline_sampling)
                fps_indices = fpsample.bucket_fps_kdline_sampling(points_for_sampling.detach().cpu().numpy(), self.max_tokens, h=3)
                batch_centroids = points_for_sampling[fps_indices]  # [max_tokens, 4]
                
                # Find k-Nearest Neighbors for each centroid
                dist_matrix = torch.cdist(batch_centroids, points_for_sampling)
                k = min(self.k_neighbors, num_points)
                _, knn_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)
                
                # Gather features
                flat_knn_indices = knn_indices.view(-1)
                gathered_features = all_point_features[flat_knn_indices]
                neighborhood_features = gathered_features.view(self.max_tokens, k, self.token_dim)
                                
                # Pool features
                if self.pool_method == 'max':
                    pooled_features = torch.max(neighborhood_features, dim=1)[0]
                elif self.pool_method == 'mean':
                    pooled_features = torch.mean(neighborhood_features, dim=1)
                else: # self.pool_method == 'attention'
                    # Compute relative positions for attention pooling
                    neighbor_positions = points_for_sampling[knn_indices]  # [max_tokens, k, 4]
                    centroid_positions = batch_centroids.unsqueeze(1)      # [max_tokens, 1, 4]
                    relative_positions = neighbor_positions - centroid_positions  # [max_tokens, k, 4]
                    
                    # Apply attention pooling with positional information
                    pooled_features = self.neighborhood_attention(
                        neighborhood_features, relative_positions
                    )
                
                # Apply aggregation MLP
                batch_tokens = self.neighborhood_mlp(pooled_features)
                num_valid_tokens = self.max_tokens
            
            # Sort tokens by time coordinate for meaningful RoPE positioning
            time_coords = batch_centroids[:num_valid_tokens, 3]  # Extract time dimension for valid tokens
            time_sort_indices = torch.argsort(time_coords)
            batch_centroids[:num_valid_tokens] = batch_centroids[:num_valid_tokens][time_sort_indices]
            batch_tokens[:num_valid_tokens] = batch_tokens[:num_valid_tokens][time_sort_indices]
            
            # Padding
            if num_valid_tokens < self.max_tokens:
                num_padding = self.max_tokens - num_valid_tokens
                pad_tokens = torch.zeros((num_padding, self.token_dim), device=batch_tokens.device, dtype=batch_tokens.dtype)
                batch_tokens = torch.cat([batch_tokens, pad_tokens], dim=0)
                pad_centroids = torch.zeros((num_padding, 4), device=batch_centroids.device, dtype=batch_centroids.dtype)
                batch_centroids = torch.cat([batch_centroids, pad_centroids], dim=0)
            
            # Create boolean mask
            batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
            batch_mask_valid[:num_valid_tokens] = True
            
            all_tokens.append(batch_tokens)
            all_centroids.append(batch_centroids)
            all_masks.append(batch_mask_valid.unsqueeze(0))
        
        # Stack results
        tokens = torch.stack(all_tokens, dim=0)
        centroids = torch.stack(all_centroids, dim=0)
        masks = torch.cat(all_masks, dim=0)
        return tokens, centroids, masks


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
        
        # Global attention pooling
        self.global_pooling = AttentionPooling(
            token_dim=token_dim,
            use_position=True
        )
        
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
        
        # Global attention pooling
        global_features = self.global_pooling(tokens, centroids, masks)
        
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
        pool_method: Pooling method for neighborhood aggregation ('max', 'mean', or 'attention')
        mlp_layers: List of dimensions for tokenizer MLP layers
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
        pool_method: str = 'max',
        mlp_layers: List[int] = [256, 512, 768]
    ):
        super().__init__()
        
        # Model Components
        self.tokenizer = PointCloudTokenizer(
            feature_dim=in_channels,
            max_tokens=num_patches,
            token_dim=token_dim,
            mlp_layers=mlp_layers,
            k_neighbors=k_neighbors,
            pool_method=pool_method
        )
        
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

    def forward(self, coords: Tensor, features: Tensor) -> Tensor:
        """
        Forward pass through the Neptune model.
        
        Args:
            coords: Point coordinates [N, 5] where columns are [batch_idx, x, y, z, t]
            features: Point features [N, in_channels]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # 1. Tokenize point cloud
        tokens, centroids, masks = self.tokenizer(coords, features)
        
        # 2. Encode with transformer
        global_features = self.encoder(tokens, centroids, masks)
        
        # 3. Apply classification/regression head
        output = self.classifier(global_features)
        
        return output