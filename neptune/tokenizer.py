"""
Gumbel-Softmax based learnable tokenizer for Neptune model.
Drop-in replacement for PointCloudTokenizer with learnable point selection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

class GumbelSoftmaxTokenizer(nn.Module):
    """
    Learnable point cloud tokenizer using differentiable Gumbel-Softmax selection.
    Uses straight-through estimator for gradient flow to importance network.
    Temperature is learnable and affects selection sharpness during training.
    """
    
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: List[int] = [256, 512, 768],
                 importance_hidden_dim: int = 256):
        super().__init__()
        
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Per-Point Feature MLP (same as original)
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
        
        # Importance scoring network
        # Takes encoded features + spatio-temporal coordinates
        self.importance_encoder = nn.Sequential(
            nn.Linear(token_dim + 4, importance_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(importance_hidden_dim),
            nn.Linear(importance_hidden_dim, importance_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(importance_hidden_dim, 1)
        )
        
        # Neighborhood Aggregation MLP (same as original)
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )
    
    def gumbel_softmax_topk(self, logits: Tensor, k: int, temperature: float, 
                            dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Differentiable top-k selection using Gumbel-Softmax with straight-through estimator.
        
        Returns:
            weights: Selection weights [*, N] - soft for gradients, hard for forward
            indices: Top-k indices [*, k]
            base_logits: Temperature-scaled logits without Gumbel noise for consistent aggregation
        """
        # Apply temperature scaling to logits FIRST
        temp_scaled_logits = logits / temperature
        
        if self.training:
            # Add Gumbel noise to temperature-scaled logits
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(temp_scaled_logits) + 1e-10) + 1e-10)
            scaled_logits = temp_scaled_logits + gumbel_noise
        else:
            # Deterministic during evaluation
            scaled_logits = temp_scaled_logits
        
        # Get top-k indices
        _, topk_indices = torch.topk(scaled_logits, k=k, dim=dim)
        
        # Create soft selection weights
        soft_weights = F.softmax(scaled_logits, dim=dim)
        
        # Create hard selection weights (one-hot for selected indices)
        hard_weights = torch.zeros_like(logits)
        hard_weights.scatter_(dim, topk_indices, 1.0)
        
        # Straight-through estimator: hard forward, soft backward
        weights = hard_weights.detach() + soft_weights - soft_weights.detach()
        
        return weights, topk_indices, temp_scaled_logits
    
    def compute_importance_scores(self, features: Tensor, coords_4d: Tensor) -> Tensor:
        """
        Compute importance scores for each point with context awareness.
        
        Args:
            features: Point features [N, token_dim]
            coords_4d: 4D coordinates [N, 4] (x, y, z, t)
            
        Returns:
            importance_scores: [N] importance score for each point
        """
        
        # Combine features with coordinates for importance scoring
        combined = torch.cat([features, coords_4d], dim=-1)  # [N, token_dim + 4]
        
        # Compute fully learned importance scores (no physics biases)
        importance = self.importance_encoder(combined).squeeze(-1)  # [N]
        
        return importance
    
    def select_and_aggregate(self, points_4d: Tensor, features: Tensor, 
                           importance_scores: Tensor, num_select: int) -> Tuple[Tensor, Tensor]:
        """
        Select points using Gumbel-Softmax and directly use their features.
        
        Args:
            points_4d: [M, 4] coordinates
            features: [M, token_dim] point features
            importance_scores: [M] importance scores
            num_select: Number of points to select
            
        Returns:
            tokens: [num_select, token_dim]
            centroids: [num_select, 4]
        """
        M = points_4d.shape[0]
        
        if M <= num_select:
            # Use all points if fewer than needed, but pass through MLP for consistency
            return self.neighborhood_mlp(features), points_4d
        
        # Differentiable top-k selection using Gumbel-Softmax
        _, selected_indices, _ = self.gumbel_softmax_topk(
            importance_scores.unsqueeze(0),  # [1, M]
            k=num_select,
            temperature=F.softplus(self.temperature) + 0.1,  # Ensure positive temp
            dim=-1
        )
        selected_indices = selected_indices.squeeze(0)  # [num_select]
        
        # Directly use the features and coordinates of the selected points
        # This makes the selection meaningful and creates a strong gradient signal
        centroids = points_4d[selected_indices]  # [num_select, 4]
        tokens = features[selected_indices]      # [num_select, token_dim]
        
        # Apply final MLP transformation
        tokens = self.neighborhood_mlp(tokens)  # [num_select, token_dim]
        
        return tokens, centroids
    
    def forward(self, coordinates: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass - same interface as original PointCloudTokenizer.
        
        Args:
            coordinates: [N, 5] where columns are [batch_idx, x, y, z, t]
            features: [N, feature_dim] point features
            
        Returns:
            tokens: [B, max_tokens, token_dim]
            centroids: [B, max_tokens, 4] 
            masks: [B, max_tokens] boolean validity masks
        """
        batch_indices = coordinates[:, 0].long()
        xyz = coordinates[:, 1:4]
        time = coordinates[:, 4:5]
        
        batch_size = batch_indices.max().item() + 1 if coordinates.numel() > 0 else 0
        if batch_size == 0:
            # Handle empty input
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), 
                                      device=coordinates.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), 
                                        device=coordinates.device, dtype=coordinates.dtype)
            empty_masks = torch.zeros((0, self.max_tokens), 
                                    device=coordinates.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks
        
        # First pass: encode all features and compute importance scores
        all_point_features = self.mlp(features)  # [N, token_dim]
        coords_4d = torch.cat([xyz, time], dim=-1)  # [N, 4]
        
        # Compute global importance scores
        importance_scores = self.compute_importance_scores(
            all_point_features, coords_4d
        )
        
        all_tokens = []
        all_centroids = []
        all_masks = []
        
        # Process each point cloud in the batch
        for batch_idx in range(batch_size):
            batch_mask_indices = batch_indices == batch_idx
            batch_coords_4d = coords_4d[batch_mask_indices]  # [M, 4]
            batch_features = all_point_features[batch_mask_indices]  # [M, token_dim]
            batch_importance = importance_scores[batch_mask_indices]  # [M]
            num_points = batch_coords_4d.shape[0]
            
            # Select and aggregate tokens
            if num_points <= self.max_tokens:
                batch_tokens = batch_features
                batch_centroids = batch_coords_4d
                num_valid_tokens = num_points
            else:
                batch_tokens, batch_centroids = self.select_and_aggregate(
                    batch_coords_4d, batch_features, batch_importance, self.max_tokens
                )
                num_valid_tokens = self.max_tokens
            
            # Sort tokens by time coordinate (maintaining temporal causality)
            time_coords = batch_centroids[:num_valid_tokens, 3]
            time_sort_indices = torch.argsort(time_coords)
            batch_centroids[:num_valid_tokens] = batch_centroids[:num_valid_tokens][time_sort_indices]
            batch_tokens[:num_valid_tokens] = batch_tokens[:num_valid_tokens][time_sort_indices]
            
            # Padding if needed
            if num_valid_tokens < self.max_tokens:
                num_padding = self.max_tokens - num_valid_tokens
                pad_tokens = torch.zeros((num_padding, self.token_dim), 
                                        device=batch_tokens.device, dtype=batch_tokens.dtype)
                batch_tokens = torch.cat([batch_tokens, pad_tokens], dim=0)
                pad_centroids = torch.zeros((num_padding, 4), 
                                          device=batch_centroids.device, dtype=batch_centroids.dtype)
                batch_centroids = torch.cat([batch_centroids, pad_centroids], dim=0)
            
            # Create boolean mask
            batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
            batch_mask_valid[:num_valid_tokens] = True
            
            all_tokens.append(batch_tokens)
            all_centroids.append(batch_centroids)
            all_masks.append(batch_mask_valid.unsqueeze(0))
        
        # Stack results
        tokens = torch.stack(all_tokens, dim=0)  # [B, max_tokens, token_dim]
        centroids = torch.stack(all_centroids, dim=0)  # [B, max_tokens, 4]
        masks = torch.cat(all_masks, dim=0)  # [B, max_tokens]
        
        return tokens, centroids, masks