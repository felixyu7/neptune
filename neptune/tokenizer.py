"""
Gumbel-Softmax based learnable tokenizer for Neptune model.
Drop-in replacement for PointCloudTokenizer with learnable point selection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
import math

class GumbelSoftmaxTokenizer(nn.Module):
    """
    Learnable point cloud tokenizer using Gumbel-Softmax sampling.
    Drop-in replacement for PointCloudTokenizer with the same interface.
    """
    
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: List[int] = [256, 512, 768],
                 k_neighbors: int = 16,
                 temperature: float = 1.0,
                 temperature_min: float = 0.1,
                 hard_sampling: bool = False,
                 importance_hidden_dim: int = 256):
        super().__init__()
        
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.hard_sampling = hard_sampling
        self.temperature_min = temperature_min
        
        # Temperature parameter (learnable)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
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
        
    
    @property
    def temperature(self):
        """Get current temperature value."""
        return torch.exp(self.log_temperature).clamp(min=self.temperature_min)
    
    def gumbel_softmax_topk(self, logits: Tensor, k: int, temperature: float, 
                            hard: bool = False, dim: int = -1) -> Tuple[Tensor, Tensor]:
        """
        Gumbel-Softmax sampling for top-k selection.
        
        Returns:
            weights: Soft/hard selection weights [*, N]
            indices: Top-k indices [*, k]
        """
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        perturbed_logits = (logits + gumbel_noise) / temperature
        
        # Get top-k indices
        topk_values, topk_indices = torch.topk(perturbed_logits, k=k, dim=dim)
        
        if hard:
            # Hard selection with straight-through estimator
            weights = torch.zeros_like(logits)
            weights.scatter_(dim, topk_indices, 1.0)
            
            # Compute soft weights for gradient flow
            soft_weights = F.softmax(perturbed_logits, dim=dim)
            topk_soft = torch.gather(soft_weights, dim, topk_indices)
            
            # Straight-through: forward is hard, backward is soft
            weights_at_topk = torch.zeros_like(logits)
            weights_at_topk.scatter_(dim, topk_indices, topk_soft)
            weights = weights.detach() + weights_at_topk - weights_at_topk.detach()
        else:
            # Soft selection
            weights = F.softmax(perturbed_logits, dim=dim)
        
        return weights, topk_indices
    
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
        Select points using Gumbel-Softmax and aggregate neighborhoods.
        
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
            # Use all points if fewer than needed
            return features, points_4d
        
        # Gumbel-Softmax top-k selection
        selection_weights, selected_indices = self.gumbel_softmax_topk(
            importance_scores.unsqueeze(0),  # [1, M]
            k=num_select,
            temperature=self.temperature,
            hard=self.hard_sampling,
            dim=-1
        )
        selected_indices = selected_indices.squeeze(0)  # [num_select]
        selection_weights = selection_weights.squeeze(0)  # [M]
        
        # Get selected centroids
        centroids = points_4d[selected_indices]  # [num_select, 4]
        
        # Neighborhood aggregation
        # Compute spatio-temporal distances from centroids to all points (4D: x,y,z,t)
        dist_matrix = torch.cdist(centroids, points_4d)  # [num_select, M]
        
        # Find k-nearest neighbors (natural spatial-temporal proximity)
        k = min(self.k_neighbors, M)
        _, knn_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)  # [num_select, k]
        
        # Gather and aggregate features
        gathered_features = features[knn_indices.reshape(-1)]  # [num_select * k, token_dim]
        neighborhood_features = gathered_features.view(num_select, k, self.token_dim)
        
        # Max pooling over neighborhoods
        pooled_features, _ = torch.max(neighborhood_features, dim=1)  # [num_select, token_dim]
        
        # Apply aggregation MLP
        tokens = self.neighborhood_mlp(pooled_features)  # [num_select, token_dim]
        
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
            
            if num_points == 0:
                # Handle empty point clouds
                batch_tokens = torch.zeros(self.max_tokens, self.token_dim, 
                                          device=coordinates.device, dtype=features.dtype)
                batch_centroids = torch.zeros(self.max_tokens, 4, 
                                            device=coordinates.device, dtype=coordinates.dtype)
                batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, 
                                              device=coordinates.device)
                all_tokens.append(batch_tokens)
                all_centroids.append(batch_centroids)
                all_masks.append(batch_mask_valid.unsqueeze(0))
                continue
            
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
    
    def anneal_temperature(self, current_step: int, total_steps: int):
        """
        Anneal temperature during training for gradually harder selection.
        Should be called during training loop.
        """
        progress = min(1.0, current_step / total_steps)
        target_log_temp = math.log(self.temperature_min)
        current_log_temp = self.log_temperature.item()
        new_log_temp = current_log_temp * (1 - progress) + target_log_temp * progress
        self.log_temperature.data.fill_(new_log_temp)
    
    def enable_hard_sampling(self):
        """Switch to hard sampling mode (for inference or late training)."""
        self.hard_sampling = True
    
    def disable_hard_sampling(self):
        """Switch to soft sampling mode (for early training)."""
        self.hard_sampling = False


def create_neptune_with_gumbel(
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
    temperature: float = 5.0,
    temperature_min: float = 0.1,
    importance_hidden_dim: int = 256
):
    """
    Factory function to create Neptune model with Gumbel-Softmax tokenizer.
    
    Example usage:
        model = create_neptune_with_gumbel(in_channels=6, output_dim=3)
    """
    from .neptune import NeptuneModel, PointTransformerEncoder
    
    class NeptuneGumbelModel(NeptuneModel):
        def __init__(self, **kwargs):
            # Extract Gumbel-specific parameters
            gumbel_params = {
                'temperature': kwargs.pop('temperature', 5.0),
                'temperature_min': kwargs.pop('temperature_min', 0.1),
                'importance_hidden_dim': kwargs.pop('importance_hidden_dim', 256)
            }
            
            # Initialize base model
            super().__init__(**kwargs)
            
            # Replace tokenizer with Gumbel-Softmax version
            self.tokenizer = GumbelSoftmaxTokenizer(
                feature_dim=kwargs.get('in_channels', 6),
                max_tokens=kwargs.get('num_patches', 128),
                token_dim=kwargs.get('token_dim', 768),
                mlp_layers=kwargs.get('mlp_layers', [256, 512, 768]),
                k_neighbors=kwargs.get('k_neighbors', 16),
                **gumbel_params
            )
    
    return NeptuneGumbelModel(
        in_channels=in_channels,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_dim=output_dim,
        k_neighbors=k_neighbors,
        mlp_layers=mlp_layers,
        temperature=temperature,
        temperature_min=temperature_min,
        importance_hidden_dim=importance_hidden_dim
    )

# Training utilities
class GumbelScheduler:
    """
    Scheduler for temperature annealing and hard sampling transition.
    """
    def __init__(self, model, total_steps: int, warmup_steps: int = 1000,
                 hard_sampling_after: float = 0.7):
        """
        Args:
            model: Neptune model with GumbelSoftmaxTokenizer
            total_steps: Total training steps
            warmup_steps: Steps before starting temperature annealing
            hard_sampling_after: Fraction of training after which to use hard sampling
        """
        self.model = model
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hard_sampling_threshold = int(hard_sampling_after * total_steps)
        self.current_step = 0
    
    def step(self):
        """Call after each training step."""
        self.current_step += 1
        
        if self.current_step > self.warmup_steps:
            # Start annealing after warmup
            anneal_steps = self.current_step - self.warmup_steps
            anneal_total = self.total_steps - self.warmup_steps
            self.model.tokenizer.anneal_temperature(anneal_steps, anneal_total)
        
        if self.current_step == self.hard_sampling_threshold:
            # Switch to hard sampling
            self.model.tokenizer.enable_hard_sampling()
            print(f"Switched to hard sampling at step {self.current_step}")
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.model.tokenizer.temperature.item()


def compute_gumbel_regularization_loss(model):
    """
    Compute regularization losses for Gumbel-Softmax training.
    
    Args:
        model: Neptune model with GumbelSoftmaxTokenizer
        
    Returns:
        Dictionary of regularization losses
    """
    reg_losses = {}
    
    # Note: These losses would need to be computed during forward pass
    # This is a placeholder showing the structure
    # In practice, you'd modify the tokenizer to return these values
    
    # Entropy regularization (encourages exploration)
    # reg_losses['entropy'] = -selection_entropy_weight * selection_entropy
    
    # Diversity regularization (prevents mode collapse)
    # reg_losses['diversity'] = -diversity_weight * selection_diversity
    
    # Temperature regularization (optional - prevents temperature from growing)
    temp = model.tokenizer.temperature
    reg_losses['temperature'] = 0.001 * torch.abs(torch.log(temp))
    
    return reg_losses