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
    