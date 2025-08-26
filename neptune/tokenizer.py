import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightPointSelector(nn.Module):
    """Lightweight importance-based point selector with hybrid selection for gradient flow."""
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

    def forward(self, coordinates: torch.Tensor, features: torch.Tensor):
        """
        Args:
            coordinates: Tensor of shape [N, 5] with columns [batch_index, x, y, z, time].
            features: Tensor of shape [N, F] containing per-point features.
        Returns:
            tokens: Tensor [B, max_tokens, token_dim] - selected tokens.
            centroids: Tensor [B, max_tokens, 4] - [x, y, z, time] for each token.
            mask: Tensor [B, max_tokens] (bool) - True for valid positions.
        """
        if coordinates.numel() == 0:
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=features.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=features.device, dtype=coordinates.dtype)
            empty_mask = torch.zeros((0, self.max_tokens), device=features.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_mask

        batch_indices = coordinates[:, 0].long()
        xyz = coordinates[:, 1:4]
        time = coordinates[:, 4:5]
        batch_size = batch_indices.max().item() + 1

        # Normalize and encode coordinates
        coords_features = coordinates[:, 1:]  # [N, 4] - remove batch index
        coords_features = self.coord_norm(coords_features)  # Normalize coordinates
        spatial_features = self.spatial_encoder(coords_features)  # [N, 64]
        combined_input = torch.cat([features, spatial_features], dim=1)  # [N, F+64]
        
        # Encode all points at once
        point_feats = self.mlp(combined_input)  # [N, token_dim]
        
        # Compute importance scores directly from point features
        importance_scores = self.importance_head(point_feats).squeeze(-1)  # [N]

        tokens_list = []
        centroids_list = []
        mask_list = []

        for b in range(batch_size):
            idx_mask = (batch_indices == b)
            pts_xyz = xyz[idx_mask]            
            pts_time = time[idx_mask]          
            pts_features = point_feats[idx_mask]  # [M, token_dim]
            pts_coords = torch.cat([pts_xyz, pts_time], dim=1)  # [M, 4]
            batch_scores = importance_scores[idx_mask]  # [M]
            M = pts_features.shape[0]

            if M == 0:
                tokens = torch.zeros((self.max_tokens, self.token_dim), device=features.device, dtype=point_feats.dtype)
                cents = torch.zeros((self.max_tokens, 4), device=coordinates.device, dtype=coordinates.dtype)
                mask = torch.zeros((self.max_tokens,), device=features.device, dtype=torch.bool)
                tokens_list.append(tokens)
                centroids_list.append(cents)
                mask_list.append(mask)
                continue

            if M <= self.max_tokens:
                # Use all points
                selected_feats = pts_features  # [M, token_dim]
                selected_coords = pts_coords   # [M, 4]
                num_valid = M
            else:
                # Differentiable selection using straight-through estimator
                if self.training:
                    # Training: straight-through estimator for differentiable top-k
                    
                    # 1. Hard selection (forward pass - discrete)
                    _, topk_indices = torch.topk(batch_scores, self.max_tokens, largest=True)
                    selected_feats_hard = pts_features[topk_indices]  # [max_tokens, token_dim]
                    selected_coords_hard = pts_coords[topk_indices]   # [max_tokens, 4]
                    
                    # 2. Soft selection (backward pass - differentiable)
                    soft_weights = torch.softmax(batch_scores / self.tau, dim=0)  # [M]
                    selected_feats_soft = torch.matmul(soft_weights.unsqueeze(0), pts_features).squeeze(0)  # [token_dim]
                    selected_coords_soft = torch.matmul(soft_weights.unsqueeze(0), pts_coords).squeeze(0)   # [4]
                    
                    # Expand to match hard selection shape
                    selected_feats_soft = selected_feats_soft.unsqueeze(0).expand(self.max_tokens, -1)
                    selected_coords_soft = selected_coords_soft.unsqueeze(0).expand(self.max_tokens, -1)
                    
                    # 3. Combine using straight-through trick
                    # Note: detach() only affects the soft part, preserving gradients through soft_weights
                    selected_feats = selected_feats_hard + selected_feats_soft - selected_feats_soft.detach()
                    selected_coords = selected_coords_hard + selected_coords_soft - selected_coords_soft.detach()
                else:
                    # Inference: use standard top-k for efficiency
                    _, topk_indices = torch.topk(batch_scores, self.max_tokens, largest=True)
                    selected_feats = pts_features[topk_indices]  # [max_tokens, token_dim]
                    selected_coords = pts_coords[topk_indices]   # [max_tokens, 4]
                num_valid = self.max_tokens

            # Sort selected tokens by time
            if num_valid > 0:
                time_vals = selected_coords[:, 3]
                sort_idx = torch.argsort(time_vals)
                selected_feats = selected_feats[sort_idx]
                selected_coords = selected_coords[sort_idx]

            # Pad to max_tokens if needed
            if num_valid < self.max_tokens:
                pad_count = self.max_tokens - num_valid
                pad_feat = torch.zeros((pad_count, self.token_dim), device=features.device, dtype=selected_feats.dtype)
                pad_coord = torch.zeros((pad_count, 4), device=features.device, dtype=selected_coords.dtype)
                selected_feats = torch.cat([selected_feats, pad_feat], dim=0)
                selected_coords = torch.cat([selected_coords, pad_coord], dim=0)

            # Create mask (True for valid tokens, False for padding)
            mask = torch.zeros((self.max_tokens,), device=features.device, dtype=torch.bool)
            mask[:num_valid] = True

            tokens_list.append(selected_feats)
            centroids_list.append(selected_coords)
            mask_list.append(mask)

        tokens = torch.stack(tokens_list, dim=0)       # [B, max_tokens, token_dim]
        centroids = torch.stack(centroids_list, dim=0) # [B, max_tokens, 4]
        mask = torch.stack(mask_list, dim=0)           # [B, max_tokens]

        return tokens, centroids, mask

class GumbelSoftmaxTokenizer(nn.Module):
    """Optimized version: Downsamples point cloud into tokens using differentiable Gumbel-Softmax sampling."""
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: list = [256, 512, 768],
                 k_neighbors: int = 16, 
                 tau: float = 1.0):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.tau = tau  # Gumbel-Softmax temperature

        # Per-point feature MLP (maps input features to token_dim)
        layers = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
            in_dim = out_dim
        # Final layer to token_dim (no ReLU here, let raw features pass to selection)
        layers.append(nn.Linear(in_dim, token_dim))
        self.mlp = nn.Sequential(*layers)

        # Learnable queries for Gumbel-Softmax sampling (W in R^{max_tokens x token_dim})
        self.selection_weights = nn.Parameter(torch.randn(max_tokens, token_dim) * 0.02)

        # Neighborhood aggregation MLP (optional, similar to original)
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )

    def forward(self, coordinates: torch.Tensor, features: torch.Tensor):
        """
        Args:
            coordinates: Tensor of shape [N, 5] with columns [batch_index, x, y, z, time].
            features: Tensor of shape [N, F] containing per-point features.
        Returns:
            tokens: Tensor [B, max_tokens, token_dim] - output token features per batch.
            centroids: Tensor [B, max_tokens, 4] - [x, y, z, time] for each token (padded for empty slots).
            mask: Tensor [B, max_tokens] (bool) - True for valid token positions (False for padding).
        """
        if coordinates.numel() == 0:
            # Handle empty input (no points)
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=features.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=features.device, dtype=coordinates.dtype)
            empty_mask = torch.zeros((0, self.max_tokens), device=features.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_mask

        batch_indices = coordinates[:, 0].long()
        xyz = coordinates[:, 1:4]   # spatial coordinates
        time = coordinates[:, 4:5]  # time coordinate (will keep as [N,1] for concatenation)
        batch_size = batch_indices.max().item() + 1

        # Lists to collect outputs for each batch
        tokens_list = []
        centroids_list = []
        mask_list = []

        # Process each cloud in the batch independently
        for b in range(batch_size):
            idx_mask = (batch_indices == b)
            pts_xyz = xyz[idx_mask]            # [M, 3]
            pts_time = time[idx_mask]          # [M, 1]
            pts_features = features[idx_mask]  # [M, F]
            M = pts_xyz.shape[0]              # number of points in this batch element

            # Embed per-point features to token_dim
            point_feats = self.mlp(pts_features)  # [M, token_dim]

            # Combine spatial coords and time for neighbor search
            pts_coords = torch.cat([pts_xyz, pts_time], dim=1)  # [M, 4]

            if M == 0:
                # No points in this batch (edge case)
                # create padded outputs with no valid tokens
                tokens = torch.zeros((self.max_tokens, self.token_dim), device=features.device, dtype=point_feats.dtype)
                cents = torch.zeros((self.max_tokens, 4), device=pts_coords.device, dtype=pts_coords.dtype)
                mask = torch.zeros((self.max_tokens,), device=features.device, dtype=torch.bool)
                tokens_list.append(tokens)
                centroids_list.append(cents)
                mask_list.append(mask)
                continue

            if M <= self.max_tokens:
                # If points are fewer than max_tokens, use all points directly
                selected_feats = point_feats  # [M, token_dim]
                selected_coords = pts_coords  # [M, 4]
                num_valid = M
            else:
                # We'll pick `self.max_tokens` indices sequentially with masking
                K = self.max_tokens
                y_list = []
                avail = torch.ones(M, dtype=torch.bool, device=features.device)
                feats_T = point_feats.transpose(0, 1)  # [token_dim, M]

                all_scores = torch.matmul(self.selection_weights, feats_T)  # [K, M]

                for k in range(K):
                    # scores: [M] for this token slot
                    scores = all_scores[k]  # [M] - no computation, just indexing!
                    # mask out already selected points
                    scores = scores.masked_fill(~avail, float('-inf'))

                    # handle degenerate case: if all masked (shouldn't happen), break
                    if not torch.isfinite(scores).any():
                        break

                    # Straight-through Gumbel-Softmax: one-hot forward, soft gradients
                    y = F.gumbel_softmax(scores, tau=self.tau, hard=True, dim=0)  # [M]
                    y_list.append(y)

                    # Update the availability mask for the next iteration
                    # This operation does not need to be differentiable
                    with torch.no_grad():
                        selected_idx = y.argmax()  # Direct index instead of boolean mask
                        avail[selected_idx] = False

                if not y_list:
                    # Handle case where loop broke immediately
                    selected_feats = torch.zeros((0, self.token_dim), device=point_feats.device, dtype=point_feats.dtype)
                    selected_coords = torch.zeros((0, 4), device=pts_coords.device, dtype=pts_coords.dtype)
                    num_valid = 0
                else:
                    selection_matrix = torch.stack(y_list, dim=0)  # [K_selected, M]

                    # Differentiable selection via matrix multiplication
                    selected_feats = torch.matmul(selection_matrix, point_feats)  # [K_selected, token_dim]
                    selected_coords = torch.matmul(selection_matrix, pts_coords) # [K_selected, 4]
                    num_valid = selected_feats.shape[0]

            # ====== Optional neighborhood aggregation (recommended) ======
            if M > 0 and num_valid > 0 and self.k_neighbors > 0:
                k = min(self.k_neighbors, M)
                with torch.no_grad():
                    # distances in XYZ for neighbor search
                    dist_matrix = torch.cdist(selected_coords[:, :3], pts_xyz)  # [num_valid, M]
                    _, knn_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)  # [num_valid, k]

                gathered = point_feats[knn_indices.reshape(-1)]            # [num_valid*k, token_dim]
                neighborhood_features = gathered.view(num_valid, k, self.token_dim)       # [num_valid, k, C]
                pooled = neighborhood_features.max(dim=1)[0]                      # [num_valid, C]
                aggregated_feats = self.neighborhood_mlp(pooled)                      # [num_valid, C]
                
                # Add neighborhood information to the selected features (preserving gradient path)
                selected_feats = selected_feats + aggregated_feats

            # Sort tokens by time coordinate (ascending) for the valid tokens
            if num_valid > 0:
                time_vals = selected_coords[:, 3]  # time column of shape [num_valid]
                sort_idx = torch.argsort(time_vals)
                # Apply the sort to all tokens (they're all valid at this point)
                selected_feats = selected_feats[sort_idx]
                selected_coords = selected_coords[sort_idx]

            # Pad tokens/coords if fewer than max_tokens
            if num_valid < self.max_tokens:
                pad_count = self.max_tokens - num_valid
                pad_feat = torch.zeros((pad_count, self.token_dim), device=features.device, dtype=selected_feats.dtype)
                pad_coord = torch.zeros((pad_count, 4), device=features.device, dtype=selected_coords.dtype)
                selected_feats = torch.cat([selected_feats, pad_feat], dim=0)
                selected_coords = torch.cat([selected_coords, pad_coord], dim=0)
            elif num_valid > self.max_tokens:
                # If more than max_tokens, truncate to max_tokens to maintain shape
                selected_feats = selected_feats[:self.max_tokens]
                selected_coords = selected_coords[:self.max_tokens]
                num_valid = self.max_tokens

            # Create mask for valid tokens (True for actual tokens, False for padding)
            mask = torch.zeros((self.max_tokens,), device=features.device, dtype=torch.bool)
            mask[:min(num_valid, self.max_tokens)] = True

            tokens_list.append(selected_feats)
            centroids_list.append(selected_coords)
            mask_list.append(mask)

        # Stack results for all batch elements
        tokens = torch.stack(tokens_list, dim=0)       # [B, max_tokens, token_dim]
        centroids = torch.stack(centroids_list, dim=0) # [B, max_tokens, 4]
        mask = torch.stack(mask_list, dim=0)           # [B, max_tokens]

        return tokens, centroids, mask