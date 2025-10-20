import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional

from torch_fps import farthest_point_sampling_with_knn

class FPSTokenizer(nn.Module):
    """
    GPU-native, vectorized FPS-based tokenizer for point clouds:
      - MLP 1: Per-point feature extraction
      - If num_points <= max_tokens: use all points (no FPS, no kNN pooling)
      - Else: FPS on 4D (x,y,z,t) to select centroids, then kNN pooling
      - MLP 2: Token refinement (always applied for consistent depth)
      - Tokens sorted by time
      - Fully batched and vectorized (no Python loop over batch)
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

        # MLP 1: Per-point feature extraction
        mlp1 = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp1 += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        mlp1 += [nn.Linear(in_dim, token_dim)]
        self.mlp1 = nn.Sequential(*mlp1)

        # MLP 2: Token refinement (always applied)
        self.mlp2 = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim)
        )

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

    # ---------- main forward ----------

    def forward(
        self,
        coords: Tensor,        # [N, 3]
        features: Tensor,      # [N, F]
        batch_ids: Tensor,     # [N]
        times: Tensor,         # [N, 1]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
          coords:    [N, 3] spatial coordinates (x, y, z)
          features:  [N, F]
          batch_ids: [N] batch indices
          times:     [N, 1] time coordinates
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
        points4 = torch.cat([xyz, times], dim=-1)  # [N,4]

        # MLP 1: Per-point feature extraction
        point_feats = self.mlp1(features)  # [N, token_dim]

        # Pack to padded [B, Nmax, *]
        P, Fp, valid_mask, counts = self._pack_flat_to_padded(points4, point_feats, batch_idx)
        B, Nmax, _ = P.shape

        # Identify small vs large batches
        small_batch = counts <= self.max_tokens
        large_batch = ~small_batch

        # Initialize centroid indices and validity masks
        cent_idx = torch.zeros(B, self.max_tokens, device=device, dtype=torch.long)
        validK = torch.zeros(B, self.max_tokens, device=device, dtype=torch.bool)

        # Small batches: use all valid points (no FPS needed)
        if small_batch.any():
            K_eff = min(Nmax, self.max_tokens)
            idx_range = torch.arange(K_eff, device=device)[None, :].expand(small_batch.sum(), -1)
            if K_eff < self.max_tokens:
                pad = torch.zeros(small_batch.sum(), self.max_tokens - K_eff, device=device, dtype=torch.long)
                idx_range = torch.cat([idx_range, pad], dim=1)
            cent_idx[small_batch] = idx_range
            counts_small = counts[small_batch].clamp(max=self.max_tokens)
            validK[small_batch] = torch.arange(self.max_tokens, device=device)[None, :] < counts_small[:, None]

        # Large batches: use fused FPS+kNN
        if large_batch.any():
            k_global = min(self.k_neighbors, Nmax if Nmax > 0 else 1)
            fps_idx, knn_idx = farthest_point_sampling_with_knn(
                P[large_batch],
                valid_mask[large_batch],
                self.max_tokens,
                k_global
            )
            cent_idx[large_batch] = fps_idx
            validK[large_batch] = torch.ones_like(fps_idx, dtype=torch.bool)

        # Gather centroids
        gather_idx_4 = cent_idx.unsqueeze(-1).expand(-1, -1, 4)
        cents = P.gather(dim=1, index=gather_idx_4)  # [B, max_tokens, 4]

        # Get features for each centroid
        token_feats = torch.zeros(B, self.max_tokens, self.token_dim, device=device, dtype=Fp.dtype)

        # Small batches: direct gather (no pooling needed - transformer will mix)
        if small_batch.any():
            gather_idx_T = cent_idx[small_batch].unsqueeze(-1).expand(-1, -1, self.token_dim)
            token_feats[small_batch] = Fp[small_batch].gather(dim=1, index=gather_idx_T)

        # Large batches: kNN pooling using neighbor indices from fused kernel
        if large_batch.any():
            Fp_large = Fp[large_batch]
            valid_large = valid_mask[large_batch]

            # Determine validity of each neighbor
            valid_expanded = valid_large.unsqueeze(1).expand(-1, knn_idx.size(1), -1)
            knn_valid = valid_expanded.gather(dim=2, index=knn_idx)  # [B_large, K, k]

            # Gather neighbor features
            Fp_expanded = Fp_large.unsqueeze(1).expand(-1, knn_idx.size(1), -1, -1)  # [B_large, K, Nmax, T]
            knn_idx_T = knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.token_dim)
            neigh_feats = Fp_expanded.gather(dim=2, index=knn_idx_T)  # [B_large, K, k, T]

            # Max-pool across neighbors
            neg_inf = torch.tensor(float('-inf'), device=device, dtype=neigh_feats.dtype)
            masked_neigh = neigh_feats.masked_fill(~knn_valid.unsqueeze(-1), neg_inf)
            pooled = masked_neigh.max(dim=2).values  # [B_large, K, T]
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

            token_feats[large_batch] = pooled

        # MLP 2: Always apply token refinement (consistent depth)
        token_feats = self.mlp2(token_feats)  # [B, max_tokens, token_dim]

        # No time sorting - let 4D RoPE handle spatial-temporal relationships
        tokens = token_feats
        centroids = cents
        masks = validK

        # Zero out padded positions
        tokens = tokens * masks.unsqueeze(-1)
        centroids = centroids * masks.unsqueeze(-1)

        return tokens.to(dtype_f), centroids.to(dtype_p), masks


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
            nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )

        # Per-point feature MLP: [features || spatial] -> token_dim
        layers: List[nn.Module] = []
        in_dim = feature_dim + 64
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, token_dim)]
        self.mlp = nn.Sequential(*layers)

        # Importance scorer
        self.importance_head = nn.Sequential(
            nn.Linear(token_dim, max(32, token_dim // 2)),
            nn.GELU(),
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
        times: Tensor,         # [N, 1]
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

        # No time sorting - let 4D RoPE handle spatial-temporal relationships
        return tokens, centroids, sel_mask


def _build_mlp(dims: List[int], last_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """
    dims: [in, h1, h2, ..., out]
    Builds an MLP with Linear + GELU blocks (except the final layer). Optionally appends last_activation.
    """
    layers: List[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    if last_activation is not None:
        layers += [last_activation]
    return nn.Sequential(*layers)


@torch.no_grad()
def _counts_from_batch_ids(batch_ids: Tensor) -> Tuple[int, Tensor]:
    """Return (B, counts[B]) for flat lists."""
    if batch_ids.numel() == 0:
        return 0, torch.zeros(0, dtype=torch.long, device=batch_ids.device)
    B = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids.long(), minlength=B)
    return B, counts


def _pack_flat_to_padded_multi(
    points4: Tensor,
    tensors: List[Tensor],
    batch_ids: Tensor,
) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
    """
    Vectorized conversion of flat per-point arrays to padded [B, Nmax, *] without Python loops.

    Returns:
      padded_points4: [B, Nmax, 4]
      padded_tensors: list of [B, Nmax, Dk]
      valid_mask:     [B, Nmax] (bool)
      counts:         [B] number of points per batch
    """
    device = points4.device
    dtype_p = points4.dtype
    B, counts = _counts_from_batch_ids(batch_ids)
    if B == 0:
        return (
            torch.zeros(0, 0, 4, device=device, dtype=dtype_p),
            [torch.zeros(0, 0, t.size(-1), device=device, dtype=t.dtype) for t in tensors],
            torch.zeros(0, 0, device=device, dtype=torch.bool),
            counts,
        )

    N = points4.size(0)
    Nmax = int(counts.max().item())

    sort_idx = torch.argsort(batch_ids.long(), stable=True)
    b_sorted = batch_ids.long()[sort_idx]
    p_sorted = points4[sort_idx]
    t_sorted = [t[sort_idx] for t in tensors]

    offsets = torch.cumsum(counts, dim=0) - counts
    pos_sorted = torch.arange(N, device=device) - offsets[b_sorted]

    padded_points4 = torch.zeros(B, Nmax, 4, device=device, dtype=dtype_p)
    padded_points4[b_sorted, pos_sorted] = p_sorted

    padded_tensors: List[Tensor] = []
    for ts in t_sorted:
        pad = torch.zeros(B, Nmax, ts.size(-1), device=device, dtype=ts.dtype)
        pad[b_sorted, pos_sorted] = ts
        padded_tensors.append(pad)

    valid_mask = (torch.arange(Nmax, device=device)[None, :] < counts[:, None])
    return padded_points4, padded_tensors, valid_mask, counts


class TokenLearnerTokenizer(nn.Module):
    """
    Native TokenLearner for unordered 4D point sets, inspired by Ryoo et al. (2022).

    - Learns S=max_tokens attention weights per point via a shared MLP: α[n, s] ∈ [0,1].
    - Forms tokens by weighted pooling over points:
          z_s = (Σ_n α[n,s] * g(f_n)) / Σ_n α[n,s]
    - Returns soft "centroids" per token by α-weighted average of (x,y,z,t).
    - Tokens are always sorted by centroid time.
    """
    def __init__(
        self,
        feature_dim: int,
        max_tokens: int = 8,
        token_dim: int = 768,
        mlp_layers: List[int] = (256, 512),
        *,
        attn_layers: List[int] = (256, 256, 256),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.token_dim = int(token_dim)
        self.eps = eps

        # Coordinate normalization and spatial encoder
        self.coord_norm = nn.LayerNorm(4)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128)
        )

        mlp_dims = [feature_dim] + list(mlp_layers) + [token_dim]
        self.feat_mlp = _build_mlp(mlp_dims)

        attn_in = feature_dim + 128  # features || spatial_encoded
        self.attn_mlp = _build_mlp([attn_in] + list(attn_layers) + [self.max_tokens])

    def forward(
        self,
        coords: Tensor,        # [N, 3]
        features: Tensor,      # [N, F]
        batch_ids: Tensor,     # [N]
        times: Tensor,         # [N, 1]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = features.device

        if coords.numel() == 0:
            empty_tokens    = torch.zeros((0, self.max_tokens, self.token_dim), device=device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4),              device=device, dtype=coords.dtype)
            empty_masks     = torch.zeros((0, self.max_tokens),                 device=device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        points4 = torch.cat([coords[:, :3], times[:, :1]], dim=-1)

        proj = self.feat_mlp(features)

        spatial_encoded = self.spatial_encoder(self.coord_norm(points4))
        attn_in = torch.cat([features, spatial_encoded], dim=-1)
        attn_logits = self.attn_mlp(attn_in)
        # Normalize over points per token later (after packing to [B,N,S])
        attn = attn_logits

        P, [F, A], valid_mask, counts = _pack_flat_to_padded_multi(points4, [proj, attn], batch_ids)
        mask_N = valid_mask
        # Softmax over points per token: mask padded rows to -inf so they get zero prob
        neg_inf = torch.tensor(float('-inf'), device=device, dtype=A.dtype)
        A = A.masked_fill(~mask_N.unsqueeze(-1), neg_inf)
        A = torch.softmax(A, dim=1)  # normalize across N for each token s
        # Handle batches with no valid points to avoid NaNs (edge case if some batch id has zero points)
        no_valid = (mask_N.sum(dim=1) == 0)
        if no_valid.any():
            A[no_valid] = 0.0

        # Weighted pooling using responsibilities over points
        numer = torch.einsum('bns,bnd->bsd', A, F)
        denom = A.sum(dim=1)
        tokens = numer / (denom.unsqueeze(-1) + self.eps)

        # Soft centroids with the same point-wise normalization (A already sums to 1 over N)
        denom_pos = A.sum(dim=1, keepdim=True) + self.eps
        Wnorm = A / denom_pos
        centroids = torch.einsum('bns,bnd->bsd', Wnorm, P)

        # No time sorting - let 4D RoPE handle spatial-temporal relationships
        masks = (counts > 0).unsqueeze(-1).expand(-1, self.max_tokens)
        return tokens, centroids, masks
