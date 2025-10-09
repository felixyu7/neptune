import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
import fpsample

class PointCloudTokenizerV1(nn.Module):
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
        
        
    def forward(
        self,
        coords: Tensor,
        features: Tensor,
        batch_ids: Tensor,
        times: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if features is None:
            raise ValueError("features must be provided for tokenization")

        time_column = times
        if time_column is None:
            if features.size(1) > 0:
                time_column = features[:, -1:].clone()
            else:
                time_column = coords.new_zeros((coords.size(0), 1))

        batch_size = int(batch_ids.max().item() + 1) if batch_ids.numel() > 0 else 0
        if batch_size == 0:
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=coords.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=coords.device, dtype=coords.dtype)
            empty_masks = torch.zeros((0, self.max_tokens), device=coords.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        all_tokens = []
        all_centroids = []
        all_masks = []
        
        # Process each point cloud in the batch individually
        for batch_idx in range(batch_size):
            batch_mask_indices = batch_ids == batch_idx
            batch_xyz = coords[batch_mask_indices]  # [M, 3]
            batch_features_data = features[batch_mask_indices]  # [M, F]
            batch_time = time_column[batch_mask_indices]  # [M, 1]
            num_points = batch_xyz.shape[0]
            
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
            batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coords.device)
            batch_mask_valid[:num_valid_tokens] = True
            
            all_tokens.append(batch_tokens)
            all_centroids.append(batch_centroids)
            all_masks.append(batch_mask_valid.unsqueeze(0))
        
        # Stack results
        tokens = torch.stack(all_tokens, dim=0)
        centroids = torch.stack(all_centroids, dim=0)
        masks = torch.cat(all_masks, dim=0)
        return tokens, centroids, masks

class PointCloudTokenizerV2(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: list = [256, 512, 768],
                 tau: float = 2.0):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))  # Learnable temperature parameter

        # Coordinate processing
        self.coord_norm = nn.LayerNorm(4)  # Normalize coordinates
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),  # Dedicated network for coordinates
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Per-point feature MLP (maps input features + spatial features to token_dim)
        layers = []
        in_dim = feature_dim + 64  # Features + spatial encoding
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, token_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Simple importance scorer (no global context)
        self.importance_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // 2, 1)
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        batch_ids: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            coords: Tensor [N, 3] with spatial coordinates.
            features: Tensor [N, F] containing per-point features.
            batch_ids: Tensor [N] with batch indices.
            times: Optional tensor [N, 1] with per-point time values.
        Returns:
            tokens: Tensor [B, max_tokens, token_dim] - selected tokens.
            centroids: Tensor [B, max_tokens, 4] - [x, y, z, time] for each token.
            mask: Tensor [B, max_tokens] (bool) - True for valid positions.
        """
        if coords.numel() == 0:
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=features.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=features.device, dtype=coords.dtype)
            empty_mask = torch.zeros((0, self.max_tokens), device=features.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_mask

        time_column = times
        if time_column is None:
            if features.size(1) > 0:
                time_column = features[:, -1:].clone()
            else:
                time_column = coords.new_zeros((coords.size(0), 1))

        coords_with_time = torch.cat([coords, time_column], dim=-1)
        batch_size = int(batch_ids.max().item() + 1) if batch_ids.numel() > 0 else 0

        # Normalize and encode coordinates
        coords_features = self.coord_norm(coords_with_time)  # Normalize coordinates
        spatial_features = self.spatial_encoder(coords_features)  # [N, 64]
        combined_input = torch.cat([features, spatial_features], dim=1)  # [N, F+64]
        
        # Encode all points at once
        point_feats = self.mlp(combined_input)  # [N, token_dim]
        
        # Compute importance scores directly from point features
        importance_scores = self.importance_head(point_feats).squeeze(-1)  # [N]

        tokens_list: List[torch.Tensor] = []
        centroids_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        for b in range(batch_size):
            idx_mask = batch_ids == b
            pts_coords = coords_with_time[idx_mask]
            pts_features = point_feats[idx_mask]
            batch_scores = importance_scores[idx_mask]
            num_points = pts_features.shape[0]

            if num_points == 0:
                tokens = torch.zeros((self.max_tokens, self.token_dim), device=features.device, dtype=point_feats.dtype)
                cents = torch.zeros((self.max_tokens, 4), device=coords.device, dtype=coords.dtype)
                mask = torch.zeros((self.max_tokens,), device=features.device, dtype=torch.bool)
                tokens_list.append(tokens)
                centroids_list.append(cents)
                mask_list.append(mask)
                continue

            if num_points <= self.max_tokens:
                selected_feats = pts_features
                selected_coords = pts_coords
                num_valid = num_points
            else:
                if self.training:
                    _, topk_indices = torch.topk(batch_scores, self.max_tokens, largest=True)
                    selected_feats_hard = pts_features[topk_indices]
                    selected_coords_hard = pts_coords[topk_indices]

                    soft_weights = torch.softmax(batch_scores / self.tau, dim=0)
                    selected_feats_soft = torch.matmul(soft_weights.unsqueeze(0), pts_features).squeeze(0)
                    selected_coords_soft = torch.matmul(soft_weights.unsqueeze(0), pts_coords).squeeze(0)

                    selected_feats_soft = selected_feats_soft.unsqueeze(0).expand(self.max_tokens, -1)
                    selected_coords_soft = selected_coords_soft.unsqueeze(0).expand(self.max_tokens, -1)

                    selected_feats = selected_feats_hard + selected_feats_soft - selected_feats_soft.detach()
                    selected_coords = selected_coords_hard + selected_coords_soft - selected_coords_soft.detach()
                else:
                    _, topk_indices = torch.topk(batch_scores, self.max_tokens, largest=True)
                    selected_feats = pts_features[topk_indices]
                    selected_coords = pts_coords[topk_indices]
                num_valid = self.max_tokens

            sort_idx = torch.argsort(selected_coords[:num_valid, 3])
            selected_coords[:num_valid] = selected_coords[:num_valid][sort_idx]
            selected_feats[:num_valid] = selected_feats[:num_valid][sort_idx]

            if num_valid < self.max_tokens:
                pad_tokens = torch.zeros((self.max_tokens - num_valid, self.token_dim), device=features.device, dtype=point_feats.dtype)
                pad_coords = torch.zeros((self.max_tokens - num_valid, 4), device=coords.device, dtype=coords.dtype)
                selected_feats = torch.cat([selected_feats, pad_tokens], dim=0)
                selected_coords = torch.cat([selected_coords, pad_coords], dim=0)

            mask = torch.zeros((self.max_tokens,), device=features.device, dtype=torch.bool)
            mask[:num_valid] = True

            tokens_list.append(selected_feats)
            centroids_list.append(selected_coords)
            mask_list.append(mask)

        tokens = torch.stack(tokens_list, dim=0)
        centroids = torch.stack(centroids_list, dim=0)
        mask = torch.stack(mask_list, dim=0)

        return tokens, centroids, mask
