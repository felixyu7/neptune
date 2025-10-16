import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional

class FPSTokenizer(nn.Module):
    """
    GPU-native, vectorized tokenizer for point clouds:
      - Per-point MLP
      - If num_points <= max_tokens: use all points (sorted by time), no neighborhood MLP
      - Else: FPS on 4D (x,y,z,t) to select max_tokens centroids, kNN pooling, neighborhood MLP
      - Fully batched and vectorized (no Python loop over batch), only a small loop over K=max_tokens for FPS
    """

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

        # Per-Point Feature MLP (shared)
        mlp = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
            in_dim = out_dim
        mlp += [nn.Linear(in_dim, token_dim)]
        self.mlp = nn.Sequential(*mlp)

        # Neighborhood Aggregation MLP (used only when num_points > max_tokens)
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )

    # ---------- helpers (vectorized) ----------

    @staticmethod
    def _pack_flat_to_padded(points4: Tensor,  # [N,4]
                              feats: Tensor,   # [N,F]
                              batch_idx: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Convert flat lists to padded [B, Nmax, *] without Python loops.
        Returns:
          padded_points4: [B, Nmax, 4]
          padded_feats:   [B, Nmax, F]
          valid_mask:     [B, Nmax] (bool)
          counts:         [B] number of points per batch
        """
        device = points4.device
        B = int(batch_idx.max().item()) + 1 if points4.numel() > 0 else 0
        if B == 0:
            return (torch.zeros(0, 0, 4, device=device, dtype=points4.dtype),
                    torch.zeros(0, 0, feats.size(-1), device=device, dtype=feats.dtype),
                    torch.zeros(0, 0, device=device, dtype=torch.bool),
                    torch.zeros(0, device=device, dtype=torch.long))

        counts = torch.bincount(batch_idx, minlength=B)                     # [B]
        Nmax = int(counts.max().item())
        N = points4.size(0)

        # Sort by batch so groups are contiguous
        sort_idx = torch.argsort(batch_idx, stable=True)
        b_sorted  = batch_idx[sort_idx]                                     # [N]
        p_sorted  = points4[sort_idx]                                       # [N,4]
        f_sorted  = feats[sort_idx]                                         # [N,F]

        # Position within each batch: pos = arange(N) - offsets[b_sorted]
        offsets = torch.cumsum(counts, dim=0) - counts                      # [B]
        pos_sorted = torch.arange(N, device=device) - offsets[b_sorted]     # [N]

        # Allocate and scatter
        padded_points4 = torch.zeros(B, Nmax, 4, device=device, dtype=points4.dtype)
        padded_feats   = torch.zeros(B, Nmax, feats.size(-1), device=device, dtype=feats.dtype)
        padded_points4[b_sorted, pos_sorted] = p_sorted
        padded_feats[b_sorted, pos_sorted]   = f_sorted

        # Valid mask
        valid_mask = (torch.arange(Nmax, device=device)[None, :] < counts[:, None])  # [B,Nmax]
        return padded_points4, padded_feats, valid_mask, counts

    @staticmethod
    def _batched_fps(points4: Tensor,           # [B, N, 4]
                     valid_mask: Tensor,        # [B, N] bool
                     K: int) -> Tuple[Tensor, Tensor]:
        """
        Batched FPS in PyTorch (Euclidean in 4D). Vectorized across batch.
        Returns:
          idx:     [B, K] long indices into N
          validK:  [B, K] bool (True for real selections; False for padded)
        """
        B, N, D = points4.shape
        device = points4.device

        if N == 0 or B == 0 or K == 0:
            return (torch.zeros(B, 0, device=device, dtype=torch.long),
                    torch.zeros(B, 0, device=device, dtype=torch.bool))

        # Number of valid points per batch
        counts = valid_mask.sum(dim=1)                        # [B]
        K_eff  = torch.clamp(counts, max=K)                   # [B]

        # Distances to the selected set: initialize to +inf for valids, 0 for invalids.
        inf = torch.tensor(float('inf'), device=device, dtype=points4.dtype)
        min_dists = torch.full((B, N), inf, device=device, dtype=points4.dtype)
        min_dists = min_dists.masked_fill(~valid_mask, 0.0)

        # Deterministic start: point with largest 4D norm among valids
        norms = (points4.square().sum(dim=2)).masked_fill(~valid_mask, -float('inf'))  # [B,N]
        first = torch.argmax(norms, dim=1)                                             # [B]

        idx = torch.zeros(B, K, device=device, dtype=torch.long)
        validK = torch.arange(K, device=device)[None, :] < K_eff[:, None]              # [B,K]

        # Iterative greedy FPS across K (no loop over B).
        last = first  # [B]
        for i in range(K):
            # Update min distances using the newly chosen centroids
            # Gather centroid coords: [B, D]
            c = points4[torch.arange(B, device=device), last]                          # [B,D]
            # dists to all points: [B,N]
            d = (points4 - c[:, None, :]).square().sum(dim=2)
            # Only update where valid points exist
            d = d.masked_fill(~valid_mask, inf)
            min_dists = torch.minimum(min_dists, d)

            # Record selection (even if i >= K_eff for some batches; will be ignored by validK)
            idx[:, i] = last

            if i + 1 < K:
                # Next farthest among valids (argmax of min_dists)
                # For batches with no valids (counts=0), argmax will be 0 but validK will mask it later.
                last = torch.argmax(min_dists, dim=1)

        return idx, validK

    # ---------- main forward ----------

    def forward(
        self,
        coords: Tensor,        # [N, 3]
        features: Tensor,      # [N, F]
        batch_ids: Tensor,     # [N]
        times: Optional[Tensor] = None,  # [N, 1]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
          coords:    [N, 3] spatial coordinates (x, y, z)
          features:  [N, F]
          batch_ids: [N] batch indices
          times:     [N, 1] optional time coordinates
        Returns:
          tokens:    [B, max_tokens, token_dim]
          centroids: [B, max_tokens, 4]   (x,y,z,t)
          masks:     [B, max_tokens] bool
        """
        device = coords.device
        dtype_p = coords.dtype
        dtype_f = features.dtype

        if coords.numel() == 0:
            empty_tokens    = torch.zeros((0, self.max_tokens, self.token_dim), device=device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4),              device=device, dtype=coords.dtype)
            empty_masks     = torch.zeros((0, self.max_tokens),                 device=device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        batch_idx = batch_ids.long()
        xyz = coords[:, :3]
        if times is None:
            times = coords.new_zeros((coords.size(0), 1))
        points4 = torch.cat([xyz, times], dim=-1)                                   # [N,4]

        # Per-point features (shared across both branches)
        point_feats = self.mlp(features)                                         # [N, token_dim]

        # Pack to padded [B, Nmax, *]
        P, Fp, valid_mask, counts = self._pack_flat_to_padded(points4, point_feats, batch_idx)
        B, Nmax, _ = P.shape

        # Identify small vs large batches
        small_batch = counts <= self.max_tokens                                  # [B]
        large_batch = ~small_batch

        # ---- Small case: use all points, sort by time, pad to max_tokens ----
        # Sort times; send invalid to +inf so they go to the end
        times_full = P[..., 3]                                                   # [B,Nmax]
        times_for_sort = times_full.masked_fill(~valid_mask, float('inf'))
        sorted_idx = torch.argsort(times_for_sort, dim=1)                       # [B, Nmax]
        K_eff = min(Nmax, self.max_tokens)
        sort_idx_smallN = sorted_idx[:, :K_eff]                                  # [B, K_eff]

        # Pad indices to max_tokens if Nmax < max_tokens
        if K_eff < self.max_tokens:
            pad_idx = torch.zeros(B, self.max_tokens - K_eff, device=device, dtype=torch.long)
            sort_idx_smallN = torch.cat([sort_idx_smallN, pad_idx], dim=1)      # [B, max_tokens]

        # Gather per-point tokens & centroids (already zero-padded for invalids)
        gather_idx_KD = sort_idx_smallN.unsqueeze(-1).expand(-1, -1, self.token_dim)  # [B,max_tokens,T]
        gather_idx_4  = sort_idx_smallN.unsqueeze(-1).expand(-1, -1, 4)               # [B,max_tokens,4]
        tokens_small  = Fp.gather(dim=1, index=gather_idx_KD)                          # [B,max_tokens,T]
        cents_small   = P.gather(dim=1, index=gather_idx_4)                            # [B,max_tokens,4]
        masks_small   = (torch.arange(self.max_tokens, device=device)[None, :] <
                        counts.clamp(max=self.max_tokens)[:, None])                    # [B,K] True for valid tokens

        # Zero-out the padded tail explicitly to be safe
        tokens_small = tokens_small * masks_small.unsqueeze(-1)
        cents_small  = cents_small  * masks_small.unsqueeze(-1)

        # ---- Large case: FPS + kNN + neighborhood MLP, then sort by time ----
        # FPS indices (vectorized across batch)
        fps_idx, validK = self._batched_fps(P, valid_mask, self.max_tokens)          # [B,K], [B,K]
        # Selected centroids
        gather_idx_4 = fps_idx.unsqueeze(-1).expand(-1, -1, 4)
        cents_large  = P.gather(dim=1, index=gather_idx_4)                            # [B,K,4]

        # Pairwise dists centroids-to-points for kNN
        # dists: [B,K,Nmax]; inf for invalid padded points
        diff = cents_large.unsqueeze(2) - P.unsqueeze(1)                              # [B,K,N,4]
        dists = diff.square().sum(dim=-1)                                             # [B,K,N]
        dists = dists.masked_fill(~valid_mask.unsqueeze(1), float('inf'))

        k_global = min(self.k_neighbors, Nmax if Nmax > 0 else 1)
        # indices of k nearest neighbors per centroid
        knn_dists, knn_idx = torch.topk(dists, k=k_global, largest=False, dim=2)      # [B,K,k]
        # validity of each neighbor - expand valid_mask to match knn_idx dimensions
        valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, knn_idx.size(1), -1) # [B,K,Nmax]
        knn_valid = valid_mask_expanded.gather(dim=2, index=knn_idx)                   # [B,K,k]

        # Gather neighbor features: [B,K,k,T]
        Fp_expanded = Fp.unsqueeze(1).expand(-1, knn_idx.size(1), -1, -1)             # [B,K,Nmax,T]
        knn_idx_T = knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.token_dim)
        neigh_feats = Fp_expanded.gather(dim=2, index=knn_idx_T)                       # [B,K,k,T]

        # Masked max-pooling across neighbors
        neg_inf = torch.tensor(float('-inf'), device=device, dtype=neigh_feats.dtype)
        masked_neigh = neigh_feats.masked_fill(~knn_valid.unsqueeze(-1), neg_inf)
        pooled = masked_neigh.max(dim=2).values                                       # [B,K,T]
        # Handle edge-case where a centroid has 0 valid neighbors (shouldn't happen if counts>0)
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        # Neighborhood MLP
        tokens_large = self.neighborhood_mlp(pooled)                                   # [B,K,T]

        # Sort large tokens by centroid time (validK controls which K are real)
        ctimes = cents_large[..., 3]
        ctimes_sort = ctimes.masked_fill(~validK, float('inf'))
        order = torch.argsort(ctimes_sort, dim=1)                                      # [B,K]
        order_T  = order.unsqueeze(-1).expand(-1, -1, self.token_dim)
        order_4  = order.unsqueeze(-1).expand(-1, -1, 4)
        tokens_large = tokens_large.gather(dim=1, index=order_T)
        cents_large  = cents_large.gather(dim=1, index=order_4)
        masks_large  = validK.gather(dim=1, index=order)

        # Zero-out padded tail
        tokens_large = tokens_large * masks_large.unsqueeze(-1)
        cents_large  = cents_large  * masks_large.unsqueeze(-1)

        # ---- Choose per-batch branch without Python loops ----
        tokens_out   = tokens_large.clone()
        centroids_out= cents_large.clone()
        masks_out    = masks_large.clone()

        if small_batch.any():
            sb = small_batch
            tokens_out[sb]    = tokens_small[sb]
            centroids_out[sb] = cents_small[sb]
            masks_out[sb]     = masks_small[sb]

        return tokens_out.to(dtype_f), centroids_out.to(dtype_p), masks_out


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

        # Ensure temporal order matches positional encoding expectations.
        if sel_mask.any():
            times = centroids[..., 3]
            inf = torch.finfo(times.dtype).max
            padded_times = torch.where(sel_mask, times, times.new_full((), inf))
            sort_idx = torch.argsort(padded_times, dim=1)
            tokens = torch.gather(tokens, 1, sort_idx.unsqueeze(-1).expand_as(tokens))
            centroids = torch.gather(centroids, 1, sort_idx.unsqueeze(-1).expand_as(centroids))
            sel_mask = torch.gather(sel_mask.long(), 1, sort_idx).bool()

        return tokens, centroids, sel_mask
