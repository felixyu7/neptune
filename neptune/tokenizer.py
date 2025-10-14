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
    """
    Vectorized V2 tokenizer:
      - Processes all points at once (GPU-friendly).
      - Segmented top-k per batch without Python loops.
      - Straight-through path so the importance scorer learns end-to-end.
      - Optional Gumbel-Top-k noise during training.
      - Fully vectorized k-NN neighborhood aggregation after selection.

    Output (unchanged API):
        tokens   : [B, max_tokens, token_dim] (neighbor-aggregated)
        centroids: [B, max_tokens, 4]     (x,y,z,time)
        mask     : [B, max_tokens] (bool) True for valid tokens
    """
    def __init__(
        self,
        feature_dim: int,
        max_tokens: int = 128,
        token_dim: int = 768,
        mlp_layers: list = [256, 512, 768],
        tau: float = 2.0,
        use_gumbel_topk: bool = True,
        k_neighbors: int = 16,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.tau = nn.Parameter(torch.tensor(float(tau)))  # learnable temperature
        self.use_gumbel_topk = use_gumbel_topk
        self.k_neighbors = int(k_neighbors)

        # Coordinate processing (keep structure close to original V2)
        self.coord_norm = nn.LayerNorm(4)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True)
        )

        # Per-point feature MLP: [features || spatial] -> token_dim
        layers: List[nn.Module] = []
        in_dim = feature_dim + 64
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, token_dim)]
        self.mlp = nn.Sequential(*layers)

        # Importance scorer
        self.importance_head = nn.Sequential(
            nn.Linear(token_dim, max(32, token_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, token_dim // 2), 1),
        )

        # Neighborhood Aggregation MLP (vectorized), mirrors V1 but batched
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )

    @staticmethod
    def _segmented_layout(batch_ids: Tensor, N: int):
        """
        Create a batched layout (no Python loops):
            - counts[b]: number of items in batch b
            - offsets[b]: start position of batch b in the sorted index list
            - perm: indices that sort items by batch
            - b_sorted: batch id per perm index
            - pos_in_b: position within each batch segment for each perm index
        """
        device = batch_ids.device
        B = int(batch_ids.max().item() + 1) if batch_ids.numel() > 0 else 0
        counts = torch.bincount(batch_ids, minlength=B) if B > 0 else torch.zeros(0, dtype=torch.long, device=device)
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        perm = torch.argsort(batch_ids)  # [N]
        b_sorted = batch_ids[perm]       # [N]
        pos_in_b = torch.arange(N, device=device) - offsets[b_sorted]  # [N]
        max_per_b = int(counts.max().item()) if B > 0 else 0
        return B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b

    def _segmented_topk(self, scores: Tensor, batch_ids: Tensor, k: int, pad_idx: int):
        """
        Vectorized per-batch top-k:
          scores: [N], batch_ids: [N], returns:
            sel_idx : [B, k_eff] global indices in [0..N] (pad_idx for invalid)
            sel_mask: [B, k_eff] True where valid
            layout  : (B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b)
        """
        device = scores.device
        N = scores.numel()
        B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b = self._segmented_layout(batch_ids, N)

        if B == 0:
            sel_idx = torch.full((0, 0), pad_idx, device=device, dtype=torch.long)
            sel_mask = torch.zeros((0, 0), device=device, dtype=torch.bool)
            return sel_idx, sel_mask, (B, counts, offsets, perm, b_sorted, pos_in_b, 0)

        # Build dense [B, max_per_b] tables for indices and scores
        idx_table = torch.full((B, max_per_b), pad_idx, device=device, dtype=torch.long)
        idx_table[b_sorted, pos_in_b] = perm  # map row/col -> global index

        neg_inf = torch.finfo(scores.dtype).min
        S = torch.full((B, max_per_b), neg_inf, device=device, dtype=scores.dtype)
        S[b_sorted, pos_in_b] = scores[perm]

        k_eff = min(k, max_per_b) if max_per_b > 0 else 0
        if k_eff == 0:
            sel_idx = torch.full((B, 0), pad_idx, device=device, dtype=torch.long)
            sel_mask = torch.zeros((B, 0), device=device, dtype=torch.bool)
            return sel_idx, sel_mask, (B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b)

        top_vals, top_pos = torch.topk(S, k=k_eff, dim=1)        # [B, k_eff]
        sel_idx = torch.gather(idx_table, dim=1, index=top_pos)  # [B, k_eff]
        sel_mask = top_pos < counts.unsqueeze(1)                 # invalid rows (no points) -> False

        return sel_idx, sel_mask, (B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b)

    def _segmented_softmax(self, scores: Tensor, layout, tau: Tensor) -> Tensor:
        """
        Per-batch softmax over all points (for ST gradient path), vectorized.
        Returns weights w: [N], with sum_b w_b = 1 for each batch b.
        """
        B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b = layout
        device = scores.device
        N = scores.numel()
        if B == 0 or N == 0:
            return torch.zeros_like(scores)

        neg_inf = torch.finfo(scores.dtype).min
        S = torch.full((B, max_per_b), neg_inf, device=device, dtype=scores.dtype)
        S[b_sorted, pos_in_b] = scores[perm]  # [B, max_per_b]

        # Numerical stability and temperature
        S = S / tau.clamp(min=1e-6)
        Smax = torch.max(S, dim=1, keepdim=True).values
        Sexp = torch.exp(S - Smax)
        # Zero out padded columns (positions >= counts[b])
        row_pos = torch.arange(max_per_b, device=device).unsqueeze(0).expand(B, -1)
        valid = row_pos < counts.unsqueeze(1)
        Sexp = Sexp * valid.to(Sexp.dtype)

        Z = Sexp.sum(dim=1, keepdim=True).clamp(min=1e-12)
        W = Sexp / Z  # [B, max_per_b]
        # Map back to flat [N]
        w = torch.zeros(N, device=device, dtype=scores.dtype)
        w[perm] = W[b_sorted, pos_in_b].to(scores.dtype)
        return w

    @staticmethod
    def _sample_gumbel(shape, device, dtype):
        # Gumbel(0,1) = -log(-log(U))
        u = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(u.clamp_(1e-6, 1.0 - 1e-6)))

    def forward(
        self,
        coords: Tensor,        # [N, 3]
        features: Tensor,      # [N, F]
        batch_ids: Tensor,     # [N]
        times: Optional[Tensor] = None,  # [N, 1]
    ) -> Tuple[Tensor, Tensor, Tensor]:

        device = features.device
        dtype = features.dtype
        N = coords.size(0)
        if N == 0:
            B = int(batch_ids.max().item() + 1) if batch_ids.numel() > 0 else 0
            return (
                torch.zeros((B, self.max_tokens, self.token_dim), device=device, dtype=dtype),
                torch.zeros((B, self.max_tokens, 4), device=device, dtype=coords.dtype),
                torch.zeros((B, self.max_tokens), device=device, dtype=torch.bool),
            )

        # Compose (x,y,z,t)
        if times is None:
            if features.size(1) > 0:
                times = features[:, -1:].clone()
            else:
                times = coords.new_zeros((N, 1))
        xyzt = torch.cat([coords, times], dim=-1)  # [N,4]

        # Per-point encoding
        spatial = self.spatial_encoder(self.coord_norm(xyzt))
        point_feats = self.mlp(torch.cat([features, spatial], dim=1))   # [N, D]
        scores = self.importance_head(point_feats).squeeze(-1)          # [N]

        # Optional Gumbel-Top-k logits for exploration (training only)
        logits = scores
        if self.training and self.use_gumbel_topk:
            logits = logits + self._sample_gumbel(logits.shape, device, logits.dtype)

        # --- Vectorized segmented top-k selection per batch ---
        pad_idx = N  # sentinel index for padding
        sel_idx, sel_mask, layout = self._segmented_topk(
            scores=logits, batch_ids=batch_ids, k=self.max_tokens, pad_idx=pad_idx
        )  # [B, k_eff], [B, k_eff]
        B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b = layout
        k_eff = sel_idx.size(1)  # <= max_tokens

        # Gather selected centroids (+ padded row for pad_idx)
        xyzt_pad = torch.cat([xyzt, xyzt.new_zeros(1, 4)], dim=0)
        sel_xyzt_hard = xyzt_pad[sel_idx]          # [B, k_eff, 4]

        # --- Fully vectorized neighborhood aggregation around selected centroids ---
        # Build dense [B, max_per_b, *] tables for all points in each batch
        # Map flat arrays by batch into padded layout
        if max_per_b > 0 and k_eff > 0 and self.k_neighbors > 0:
            # Tables for coordinates and point features
            xyzt_table = xyzt.new_zeros((B, max_per_b, 4))
            xyzt_table[b_sorted, pos_in_b] = xyzt[perm]
            feats_table = point_feats.new_zeros((B, max_per_b, self.token_dim))
            feats_table[b_sorted, pos_in_b] = point_feats[perm]

            # Valid mask for points per batch
            row_pos = torch.arange(max_per_b, device=device).unsqueeze(0).expand(B, -1)
            valid_pts = row_pos < counts.unsqueeze(1)  # [B, max_per_b]

            # Distances [B, k_eff, max_per_b] using broadcasted squared L2 in 4D (x,y,z,t)
            # Use broadcasting to avoid torch.cdist dtype/backends limitations
            diff = sel_xyzt_hard.unsqueeze(2) - xyzt_table.unsqueeze(1)  # [B, k_eff, max_per_b, 4]
            dists = (diff * diff).sum(dim=-1)
            # Mask out invalid points so they aren't selected
            large = torch.finfo(dists.dtype).max
            dists = torch.where(valid_pts.unsqueeze(1), dists, dists.new_full((), large))

            # Top-k neighbors per selected centroid (per batch)
            k_nn = int(min(self.k_neighbors, max_per_b))
            if k_nn > 0:
                _, nn_idx = torch.topk(dists, k=k_nn, largest=False, dim=2)  # [B, k_eff, k]
                # Gather neighbor features: [B, k_eff, k, D]
                feats_exp = feats_table.unsqueeze(1).expand(B, k_eff, max_per_b, self.token_dim)
                nn_idx_exp = nn_idx.unsqueeze(-1).expand(B, k_eff, k_nn, self.token_dim)
                neigh_feats = torch.gather(feats_exp, 2, nn_idx_exp)
                # Pool (max) over neighbors -> [B, k_eff, D]
                pooled = neigh_feats.max(dim=2).values
                # Aggregation MLP
                sel_tokens_hard = self.neighborhood_mlp(pooled)
                # Zero-out entries where selection is invalid for a given batch
                sel_tokens_hard = sel_tokens_hard * sel_mask.to(dtype).unsqueeze(-1)
            else:
                sel_tokens_hard = point_feats.new_zeros((B, k_eff, self.token_dim))
        else:
            # Fallback to per-point features if no neighbors available
            point_feats_pad = torch.cat([point_feats, point_feats.new_zeros(1, self.token_dim)], dim=0)
            sel_tokens_hard = point_feats_pad[sel_idx]  # [B, k_eff, D]

        # Pad to [B, max_tokens, ...] if needed
        if k_eff < self.max_tokens:
            pad_t = torch.zeros((B, self.max_tokens - k_eff, self.token_dim), device=device, dtype=dtype)
            pad_c = torch.zeros((B, self.max_tokens - k_eff, 4), device=device, dtype=xyzt.dtype)
            pad_m = torch.zeros((B, self.max_tokens - k_eff), device=device, dtype=torch.bool)
            sel_tokens_hard = torch.cat([sel_tokens_hard, pad_t], dim=1)
            sel_xyzt_hard = torch.cat([sel_xyzt_hard, pad_c], dim=1)
            sel_mask = torch.cat([sel_mask, pad_m], dim=1)

        # --- Straight-through soft path for gradient to importance ---
        # Per-batch softmax over all points (optionally uses same noisy logits)
        soft_w = self._segmented_softmax(logits, layout, tau=self.tau)  # [N], sum_b w_b = 1
        # Soft aggregates per batch
        soft_feat = torch.zeros((B, self.token_dim), device=device, dtype=dtype)
        soft_cent = torch.zeros((B, 4), device=device, dtype=xyzt.dtype)
        # index_add_: accumulate w * value by batch
        soft_feat.index_add_(0, batch_ids, (soft_w.unsqueeze(-1) * point_feats).to(soft_feat.dtype))
        soft_cent.index_add_(0, batch_ids, (soft_w.unsqueeze(-1) * xyzt).to(soft_cent.dtype))

        # Repeat soft aggregates across token dimension and apply ST estimator
        soft_feat_rep = soft_feat.unsqueeze(1).expand(B, self.max_tokens, self.token_dim)
        soft_cent_rep = soft_cent.unsqueeze(1).expand(B, self.max_tokens, 4)

        # Straight-through composition: forward uses hard; backward flows through soft
        valid_counts = sel_mask.sum(dim=1, keepdim=True).clamp(min=1)
        st_scale = (sel_mask.float() / valid_counts).unsqueeze(-1)
        tokens = sel_tokens_hard + (soft_feat_rep - soft_feat_rep.detach()) * st_scale.to(dtype)
        centroids = sel_xyzt_hard + (soft_cent_rep - soft_cent_rep.detach()) * st_scale.to(sel_xyzt_hard.dtype)

        return tokens, centroids, sel_mask
