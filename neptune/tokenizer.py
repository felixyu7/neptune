import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional

class FPSTokenizer(nn.Module):
    """Tokenizes batched point clouds via FPS centroids and k-NN feature pooling."""
    
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

    @staticmethod
    def _segmented_layout(batch_ids: Tensor, N: int):
        device = batch_ids.device
        B = int(batch_ids.max().item() + 1) if batch_ids.numel() > 0 else 0
        counts = (
            torch.bincount(batch_ids, minlength=B)
            if B > 0
            else torch.zeros(0, dtype=torch.long, device=device)
        )
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        perm = torch.argsort(batch_ids)
        b_sorted = batch_ids[perm]
        pos_in_b = torch.arange(N, device=device) - offsets[b_sorted]
        max_per_b = int(counts.max().item()) if B > 0 else 0
        return B, counts, offsets, perm, b_sorted, pos_in_b, max_per_b

    @staticmethod
    def _segmented_topk(scores: Tensor, layout, k: int, pad_idx: int):
        B, counts, _, perm, b_sorted, pos_in_b, max_per_b = layout
        device = scores.device
        if B == 0:
            sel_idx = torch.full((0, 0), pad_idx, dtype=torch.long, device=device)
            sel_mask = torch.zeros((0, 0), dtype=torch.bool, device=device)
            return sel_idx, sel_mask

        idx_table = torch.full((B, max_per_b), pad_idx, dtype=torch.long, device=device)
        idx_table[b_sorted, pos_in_b] = perm

        neg_inf = torch.finfo(scores.dtype).min
        S = torch.full((B, max_per_b), neg_inf, dtype=scores.dtype, device=device)
        S[b_sorted, pos_in_b] = scores[perm]

        k_eff = min(k, max_per_b) if max_per_b > 0 else 0
        if k_eff == 0:
            sel_idx = torch.full((B, 0), pad_idx, dtype=torch.long, device=device)
            sel_mask = torch.zeros((B, 0), dtype=torch.bool, device=device)
            return sel_idx, sel_mask

        _, top_pos = torch.topk(S, k=k_eff, dim=1)
        sel_idx = torch.gather(idx_table, dim=1, index=top_pos)
        counts_exp = counts.unsqueeze(1)
        sel_mask = top_pos < counts_exp
        return sel_idx, sel_mask

    def _batched_fps(self, points: Tensor, batch_ids: Tensor):
        device = points.device
        N = points.size(0)
        layout = self._segmented_layout(batch_ids, N)
        B, counts, _, _, _, _, _ = layout
        pad_idx = N

        sel_idx = torch.full((B, self.max_tokens), pad_idx, dtype=torch.long, device=device)
        sel_mask = torch.zeros((B, self.max_tokens), dtype=torch.bool, device=device)
        if B == 0 or N == 0:
            return sel_idx, sel_mask

        active = counts > 0
        if not active.any():
            return sel_idx, sel_mask

        sums = torch.zeros((B, points.size(1)), dtype=points.dtype, device=device)
        sums.index_add_(0, batch_ids, points)
        counts_safe = counts.clamp(min=1).unsqueeze(1).to(points.dtype)
        means = sums / counts_safe
        means_expanded = means[batch_ids]
        dist2_mean = ((points - means_expanded) ** 2).sum(dim=1)

        neg_inf = torch.finfo(points.dtype).min
        min_d2 = torch.full((N,), float("inf"), dtype=points.dtype, device=device)

        active_points = active[batch_ids]
        min_d2[~active_points] = neg_inf

        for t in range(self.max_tokens):
            if not active.any():
                break

            scores = dist2_mean if t == 0 else min_d2
            sel_t, mask_t = self._segmented_topk(scores, layout, k=1, pad_idx=pad_idx)
            sel_t = sel_t.squeeze(1)
            mask_t = mask_t.squeeze(1)
            mask_t = mask_t & active
            sel_idx[:, t] = torch.where(mask_t, sel_t, sel_idx[:, t])
            sel_mask[:, t] = mask_t

            if not mask_t.any():
                break

            valid_batches = mask_t
            if not valid_batches.any():
                active = torch.zeros_like(active)
                break

            batch_centroids = torch.zeros((B, points.size(1)), dtype=points.dtype, device=device)
            batch_centroids[valid_batches] = points[sel_t[valid_batches]]
            centroid_per_point = batch_centroids[batch_ids]
            d2_new = ((points - centroid_per_point) ** 2).sum(dim=1)
            point_valid = valid_batches[batch_ids]
            min_d2 = torch.where(point_valid, torch.minimum(min_d2, d2_new), min_d2)

            chosen_points = sel_t[valid_batches]
            min_d2[chosen_points] = neg_inf

            active = active & ((t + 1) < counts)
            min_d2[~active[batch_ids]] = neg_inf

        return sel_idx, sel_mask

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

        xyzt = torch.cat([coords, time_column], dim=-1)
        point_feats = self.mlp(features)

        centroid_indices, centroid_mask = self._batched_fps(xyzt, batch_ids)
        pad_idx = coords.size(0)

        tokens = torch.zeros(
            (batch_size, self.max_tokens, self.token_dim),
            device=coords.device,
            dtype=point_feats.dtype,
        )
        centroids = torch.zeros(
            (batch_size, self.max_tokens, 4),
            device=coords.device,
            dtype=coords.dtype,
        )

        if centroid_mask.any():
            batch_grid = torch.arange(batch_size, device=coords.device).unsqueeze(1).expand(-1, self.max_tokens)
            valid_positions = centroid_mask
            valid_batches = batch_grid[valid_positions]
            valid_slots = torch.arange(self.max_tokens, device=coords.device).unsqueeze(0).expand(batch_size, -1)
            valid_slots = valid_slots[valid_positions]
            flat_indices = centroid_indices[valid_positions]

            centroids[valid_batches, valid_slots] = xyzt[flat_indices]

            k = min(self.k_neighbors, coords.size(0))
            dist = torch.cdist(xyzt[flat_indices], xyzt, p=2)
            mismatch = valid_batches.unsqueeze(1) != batch_ids.unsqueeze(0)
            dist.masked_fill_(mismatch, float("inf"))

            knn_dists, knn_indices = torch.topk(dist, k=k, dim=1, largest=False)
            knn_valid = torch.isfinite(knn_dists)

            gathered = point_feats[knn_indices]
            if not knn_valid.all():
                fill_value = torch.finfo(gathered.dtype).min
                gathered = gathered.masked_fill(~knn_valid.unsqueeze(-1), fill_value)

            pooled = gathered.max(dim=1).values
            aggregated = self.neighborhood_mlp(pooled)

            tokens[valid_batches, valid_slots] = aggregated

        return tokens, centroids, centroid_mask

class LearnedImportanceTokenizer(nn.Module):
    """Learns per-point importance, samples top-k tokens per batch, and returns features, centroids, and a validity mask."""
    def __init__(
        self,
        feature_dim: int,
        max_tokens: int = 128,
        token_dim: int = 768,
        mlp_layers: list = [256, 512, 768],
        tau: float = 2.0,
        use_gumbel_topk: bool = True,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.tau = nn.Parameter(torch.tensor(float(tau)))  # learnable temperature
        self.use_gumbel_topk = use_gumbel_topk

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

        _, top_pos = torch.topk(S, k=k_eff, dim=1)        # [B, k_eff]
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

        # Gather hard selections (+ a padded row to index when pad_idx appears)
        point_feats_pad = torch.cat([point_feats, point_feats.new_zeros(1, self.token_dim)], dim=0)
        xyzt_pad = torch.cat([xyzt, xyzt.new_zeros(1, 4)], dim=0)
        sel_tokens_hard = point_feats_pad[sel_idx]  # [B, k_eff, D]
        sel_xyzt_hard = xyzt_pad[sel_idx]          # [B, k_eff, 4]

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
