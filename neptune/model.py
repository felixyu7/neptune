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

class PointCloudTokenizer(nn.Module):
    """Converts point cloud data into tokens for transformer processing."""
    
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: List[int] = [256, 512, 768],
                 k_neighbors: int = 16):
        super().__init__()
            
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        
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
                                
                # Pool features using max pooling
                pooled_features = torch.max(neighborhood_features, dim=1)[0]
                
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
        mlp_layers: List[int] = [256, 512, 768]
    ):
        super().__init__()
        
        # Model Components
        self.tokenizer = PointCloudTokenizer(
            feature_dim=in_channels,
            max_tokens=num_patches,
            token_dim=token_dim,
            mlp_layers=mlp_layers,
            k_neighbors=k_neighbors
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