import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional

from torch_fps import farthest_point_sampling, farthest_point_sampling_with_knn


class FPSTokenizer(nn.Module):
    """
    GPU-native, vectorized FPS-based tokenizer for point clouds:
      - MLP 1: Per-point feature extraction
      - If num_points <= max_tokens: use all points (no FPS, no pooling)
      - Else: FPS on 4D (x,y,z,t) to select centroids, then pool hits per token
        by nearest-centroid (Voronoi) assignment — every hit contributes to
        exactly one token, so no hit is ever dropped — or, in the legacy
        ``assign_mode="knn"``, by k-nearest-neighbor gather around each centroid.
      - Per-token summary scalars (multiplicity, total charge, time spread,
        spatial RMS radius) are appended to the pooled features.
      - MLP 2: Token refinement (always applied for consistent depth)

    The forward performs exactly one host sync (``counts.cpu().tolist()``) and
    uses no data-dependent-shape ops (``nonzero``/boolean indexing): small and
    large events are routed with precomputed index tensors, and only the large
    subset is ever padded (to its own max length), so one bright event no longer
    inflates the whole batch.

    ``metric_time_scale`` weights the time axis in the FPS/assignment metric
    only. Coordinates are in km and time in microseconds (see mmap dataloader);
    at raw units the time axis carries ~99% of the 4D metric variance on real
    events, so selection would cluster nearly purely by arrival time. The
    default 0.3 km/us (light speed) balances space and time; 0.22 is the photon
    group velocity in ice; 1.0 reproduces the legacy raw-unit metric. Returned
    centroids, relative offsets, and summary scalars always stay in raw units —
    the scale changes token membership, never the representation.
    """

    N_EXTRA = 4  # multiplicity, total charge, time spread, RMS radius

    def __init__(self,
                 feature_dim: int,
                 max_tokens: int = 128,
                 token_dim: int = 768,
                 mlp_layers: Optional[List[int]] = None,
                 k_neighbors: int = 16,
                 dropout: float = 0.0,
                 knn_pool: str = "max",
                 rel_pos_hidden: int = 64,
                 charge_weighted_mean: bool = True,
                 charge_col: int = 0,
                 assign_mode: str = "voronoi",
                 metric_time_scale: float = 0.3):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 512, 768]
        if assign_mode not in ("voronoi", "knn"):
            raise ValueError(f"assign_mode must be 'voronoi' or 'knn', got '{assign_mode}'")
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.knn_pool = knn_pool
        self.charge_weighted_mean = charge_weighted_mean
        self.charge_col = charge_col
        self.assign_mode = assign_mode
        self.metric_time_scale = metric_time_scale

        # MLP 1: Per-point feature extraction
        mlp1 = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp1 += [nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = out_dim
        mlp1 += [nn.Linear(in_dim, token_dim)]
        self.mlp1 = nn.Sequential(*mlp1)

        # MLP 2: Token refinement (always applied)
        # Input dim doubles when using max+mean pooling; the per-token summary
        # scalars are appended in both modes.
        pooled_dim = 2 * token_dim if knn_pool == "max_mean" else token_dim
        self.mlp2 = nn.Sequential(
            nn.Linear(pooled_dim + self.N_EXTRA, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim)
        )

        # Relative-geometry encoder: embeds each hit's (dx,dy,dz,dt) offset
        # from its token centroid and adds it to the hit feature before pooling,
        # so a token encodes local cluster shape (otherwise lost in a pure
        # feature max/mean). Runs only on the large-event path.
        self.rel_encoder = nn.Sequential(
            nn.Linear(4, rel_pos_hidden),
            nn.GELU(),
            nn.Linear(rel_pos_hidden, token_dim),
        )

        # Buffer (not a per-forward host tensor: that would cost a pageable
        # H2D copy + stream sync every call).
        self.register_buffer(
            "metric_scale",
            torch.tensor([1.0, 1.0, 1.0, float(metric_time_scale)]),
            persistent=False)

    @staticmethod
    def _route_subset(rows_dev: Tensor, csub: Tensor, starts: Tensor,
                      n_sub: int, device) -> Tuple[Tensor, Tensor, Tensor]:
        """Index tensors routing a subset of events out of batch-sorted flat arrays.

        Inputs are device slices of the single packed routing transfer (no
        per-call H2D syncs, no nonzero):
          seg:   [N_sub] subset-event id per hit
          local: [N_sub] hit position within its event
          src:   [N_sub] row into the batch-sorted flat arrays
        """
        seg = torch.repeat_interleave(
            torch.arange(rows_dev.numel(), device=device), csub, output_size=n_sub)
        local = torch.arange(n_sub, device=device) - (torch.cumsum(csub, 0) - csub)[seg]
        src = starts[seg] + local
        return seg, local, src

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
        K = self.max_tokens
        T = self.token_dim

        if coords.numel() == 0:
            empty_tokens    = torch.zeros((0, K, T), device=device, dtype=dtype_f)
            empty_centroids = torch.zeros((0, K, 4), device=device, dtype=dtype_p)
            empty_masks     = torch.zeros((0, K),    device=device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        batch_idx = batch_ids.long()
        points4 = torch.cat([coords[:, :3], times], dim=-1)   # [N,4] raw units
        point_feats = self.mlp1(features)                     # [N, T]
        q = features[:, self.charge_col]                      # [N] log1p charge

        # The single host sync: one D2H copy of the ids. (bincount on a CUDA
        # tensor would itself sync — it reads max(batch_idx) on the host to
        # size its output — so counting happens on the CPU copy instead.)
        counts_list = torch.bincount(batch_idx.cpu()).tolist()
        B = len(counts_list)

        # Sort hits by batch so each event is a contiguous run.
        sort_idx = torch.argsort(batch_idx, stable=True)
        p_sorted = points4.index_select(0, sort_idx)
        f_sorted = point_feats.index_select(0, sort_idx)
        q_sorted = q.index_select(0, sort_idx)

        # Host-side split into small (<= K hits: every hit is a token) and
        # large (> K: FPS + pooling) events. All routing below comes from these
        # lists — no boolean indexing, no nonzero, no further syncs.
        offsets: List[int] = [0] * B
        acc = 0
        for i, c in enumerate(counts_list):
            offsets[i] = acc
            acc += c
        small_rows = [i for i, c in enumerate(counts_list) if c <= K]
        large_rows = [i for i, c in enumerate(counts_list) if c > K]
        counts_s = [counts_list[i] for i in small_rows]
        counts_l = [counts_list[i] for i in large_rows]
        B_s, B_l = len(small_rows), len(large_rows)
        N_s, N_l = sum(counts_s), sum(counts_l)

        # All routing metadata goes up in ONE pinned non-blocking H2D transfer;
        # per-list torch.tensor(..., device=...) would each be a pageable copy
        # plus a full stream sync.
        routing_host = torch.tensor(
            small_rows + counts_s + [offsets[r] for r in small_rows]
            + large_rows + counts_l + [offsets[r] for r in large_rows],
            dtype=torch.long)
        if device.type == "cuda":
            routing_host = routing_host.pin_memory()
        routing = routing_host.to(device, non_blocking=True)
        rows_s_dev, csub_s, starts_s = routing[:B_s], routing[B_s:2 * B_s], routing[2 * B_s:3 * B_s]
        o = 3 * B_s
        rows_l_dev, csub_l, starts_l_abs = (routing[o:o + B_l], routing[o + B_l:o + 2 * B_l],
                                            routing[o + 2 * B_l:o + 3 * B_l])

        pooled_dim = 2 * T if self.knn_pool == "max_mean" else T
        feat_dtype = point_feats.dtype  # autocast-aware compute dtype

        pre = torch.zeros(B, K, pooled_dim + self.N_EXTRA, device=device, dtype=feat_dtype)
        cents_out = torch.zeros(B, K, 4, device=device, dtype=dtype_p)
        masks = torch.zeros(B, K, device=device, dtype=torch.bool)

        # ---- small events: one hit per token, no padding, no pooling ----
        if B_s > 0:
            seg_s, local_s, src_s = self._route_subset(
                rows_s_dev, csub_s, starts_s, N_s, device)
            f_s = f_sorted.index_select(0, src_s)             # [N_s, T]
            p_s = p_sorted.index_select(0, src_s)             # [N_s, 4]
            q_s = q_sorted.index_select(0, src_s)             # [N_s]
            dest = seg_s * K + local_s                        # unique: local < counts <= K

            g = torch.cat([f_s, f_s], dim=-1) if self.knn_pool == "max_mean" else f_s
            # Single point: multiplicity 1, its own charge, zero spreads.
            zeros_s = torch.zeros_like(q_s)
            ex_s = torch.stack(
                [torch.full_like(q_s, 0.6931471805599453), q_s, zeros_s, zeros_s], dim=-1)
            pre_s = torch.zeros(B_s * K, pooled_dim + self.N_EXTRA,
                                device=device, dtype=feat_dtype)
            pre_s.index_copy_(0, dest, torch.cat([g, ex_s.to(feat_dtype)], dim=-1))
            cents_s = torch.zeros(B_s * K, 4, device=device, dtype=dtype_p)
            cents_s.index_copy_(0, dest, p_s)

            pre.index_copy_(0, rows_s_dev, pre_s.view(B_s, K, -1))
            cents_out.index_copy_(0, rows_s_dev, cents_s.view(B_s, K, 4))
            masks.index_copy_(
                0, rows_s_dev,
                torch.arange(K, device=device)[None, :] < csub_s[:, None])

        # ---- large events: FPS centroids + per-token pooling ----
        if B_l > 0:
            Nmax_l = max(counts_l)
            seg_l, local_l, src_l = self._route_subset(
                rows_l_dev, csub_l, starts_l_abs, N_l, device)
            f_l = f_sorted.index_select(0, src_l)             # [N_l, T]
            p_l = p_sorted.index_select(0, src_l)             # [N_l, 4] raw
            q_l = q_sorted.index_select(0, src_l)             # [N_l]
            n_l = f_l.size(0)

            # Metric copy for FPS/assignment: fp32, time axis scaled. Padding
            # stays zero (finite) — the kernel masks it but must not see NaNs.
            p_l_m = p_l.float() * self.metric_scale           # [N_l, 4]
            P_pad_m = torch.zeros(B_l * Nmax_l, 4, device=device)
            dest_pad = seg_l * Nmax_l + local_l
            P_pad_m.index_copy_(0, dest_pad, p_l_m)
            P_pad_m = P_pad_m.view(B_l, Nmax_l, 4)
            valid_l = torch.arange(Nmax_l, device=device)[None, :] < csub_l[:, None]

            # random_start gates FPS augmentation on train mode: training draws a
            # random seed point per call (augmentation), eval/inference uses a
            # deterministic start so the same event always tokenizes identically.
            # validate=False: counts > K holds by construction of the subset.
            starts_l = torch.cumsum(csub_l, 0) - csub_l       # [B_l] into flat subset

            if self.assign_mode == "voronoi":
                fps_idx = farthest_point_sampling(
                    P_pad_m, valid_l, K,
                    random_start=self.training, validate=False)          # [B_l, K]
                cent_flat = (starts_l[:, None] + fps_idx).reshape(-1)    # [B_l*K]
                cents_raw = p_l.index_select(0, cent_flat)               # [B_l*K, 4]
                cents_m = p_l_m.index_select(0, cent_flat).view(B_l, K, 4)

                # Nearest-centroid assignment in the scaled metric: every hit
                # joins exactly one token (coverage = 1 by construction).
                d = (p_l_m.unsqueeze(1) - cents_m[seg_l]).pow(2).sum(-1)  # [N_l, K]
                assign = d.argmin(dim=1)                                  # [N_l]
                cell = seg_l * K + assign                                 # [N_l]

                rel = p_l - cents_raw.index_select(0, cell)               # raw units
                h = f_l + self.rel_encoder(rel).to(feat_dtype)            # [N_l, T]

                cell_T = cell.unsqueeze(-1).expand(-1, T)
                mx = torch.full((B_l * K, T), float("-inf"),
                                device=device, dtype=h.dtype)
                mx = mx.scatter_reduce(0, cell_T, h, reduce="amax", include_self=True)
                mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

                # fp32 accumulators for the weighted mean and summary scalars.
                w = q_l.float() if self.charge_weighted_mean else torch.ones_like(q_l, dtype=torch.float32)
                wsum = torch.zeros(B_l * K, device=device).scatter_add_(0, cell, w)
                wsafe = wsum.clamp(min=1e-6)

                if self.knn_pool == "max_mean":
                    hw = h.float() * w[:, None]
                    hsum = torch.zeros(B_l * K, T, device=device).scatter_add_(0, cell_T, hw)
                    mean = (hsum / wsafe[:, None]).to(h.dtype)
                    pooled_l = torch.cat([mx, mean], dim=-1)
                else:
                    pooled_l = mx

                ones_h = torch.ones(n_l, device=device)
                n_c = torch.zeros(B_l * K, device=device).scatter_add_(0, cell, ones_h)
                mult = torch.log1p(n_c)
                q_phys = torch.expm1(q_l.float().clamp(min=0))
                Q = torch.log1p(torch.zeros(B_l * K, device=device)
                                .scatter_add_(0, cell, q_phys))
                dt = rel[:, 3].float()
                m1 = torch.zeros(B_l * K, device=device).scatter_add_(0, cell, w * dt) / wsafe
                m2 = torch.zeros(B_l * K, device=device).scatter_add_(0, cell, w * dt * dt) / wsafe
                # clamp INSIDE the sqrt: single-hit cells have exactly-zero
                # variance/radius and sqrt'(0)=inf would NaN the backward.
                dt_std = (m2 - m1 * m1).clamp(min=1e-12).sqrt()
                r2 = torch.zeros(B_l * K, device=device).scatter_add_(
                    0, cell, rel[:, :3].float().pow(2).sum(-1))
                rms = (r2 / n_c.clamp(min=1)).clamp(min=1e-12).sqrt()
                ex_l = torch.stack([mult, Q, dt_std, rms], dim=-1)        # [B_l*K, 4]

                pre_l = torch.cat([pooled_l, ex_l.to(feat_dtype)], dim=-1)
                cents_l_out = cents_raw

            else:  # assign_mode == "knn" (legacy A/B path)
                pre_l, cents_l_out = self._knn_pool_large(
                    P_pad_m, valid_l, p_l, f_l, q_l, starts_l,
                    B_l, Nmax_l, dest_pad, feat_dtype)

            pre.index_copy_(0, rows_l_dev, pre_l.view(B_l, K, -1))
            cents_out.index_copy_(0, rows_l_dev, cents_l_out.view(B_l, K, 4).to(dtype_p))
            masks.index_copy_(
                0, rows_l_dev, torch.ones(B_l, K, device=device, dtype=torch.bool))

        # MLP 2: Always apply token refinement (consistent depth)
        tokens = self.mlp2(pre)                               # [B, K, T]

        # Zero out padded positions
        tokens = tokens * masks.unsqueeze(-1)
        cents_out = cents_out * masks.unsqueeze(-1)

        return tokens.to(dtype_f), cents_out, masks

    def _knn_pool_large(self, P_pad_m: Tensor, valid_l: Tensor, p_l: Tensor,
                        f_l: Tensor, q_l: Tensor, starts_l: Tensor,
                        B_l: int, Nmax_l: int, dest_pad: Tensor,
                        feat_dtype) -> Tuple[Tensor, Tensor]:
        """Legacy fused FPS+kNN pooling on the (already padded) large subset.

        Mirrors the pre-Voronoi implementation exactly — same gather, relative
        encoding, and masked max / charge-weighted mean — with the per-token
        summary scalars computed from the kNN sets so the mlp2 contract matches
        the voronoi path.
        """
        device = p_l.device
        K = self.max_tokens
        T = self.token_dim
        k_global = min(self.k_neighbors, Nmax_l)

        fps_idx, knn_idx = farthest_point_sampling_with_knn(
            P_pad_m, valid_l, K, k_global,
            random_start=self.training, validate=False)       # [B_l,K], [B_l,K,k]

        cent_flat = (starts_l[:, None] + fps_idx).reshape(-1)
        cents_raw = p_l.index_select(0, cent_flat)            # [B_l*K, 4]

        # Padded [features | charge] and geometry for neighbor gathers.
        Faug_pad = torch.zeros(B_l * Nmax_l, T + 1, device=device, dtype=feat_dtype)
        Faug_pad.index_copy_(0, dest_pad, torch.cat(
            [f_l, q_l.unsqueeze(-1).to(feat_dtype)], dim=-1))
        P_pad_raw = torch.zeros(B_l * Nmax_l, 4, device=device, dtype=p_l.dtype)
        P_pad_raw.index_copy_(0, dest_pad, p_l)

        k_local = knn_idx.size(2)
        base = (torch.arange(B_l, device=device, dtype=knn_idx.dtype) * Nmax_l)
        flat_knn = (knn_idx + base.view(-1, 1, 1)).reshape(-1)

        neigh_aug = Faug_pad.index_select(0, flat_knn).reshape(B_l, K, k_local, T + 1)
        neigh_feats = neigh_aug[..., :T]
        knn_valid = valid_l.reshape(-1).index_select(0, flat_knn).reshape(B_l, K, k_local)

        neigh_xyzt = P_pad_raw.index_select(0, flat_knn).reshape(B_l, K, k_local, 4)
        rel = neigh_xyzt - cents_raw.view(B_l, K, 1, 4)
        neigh_feats = neigh_feats + self.rel_encoder(rel).to(neigh_feats.dtype)

        masked = neigh_feats.masked_fill(~knn_valid.unsqueeze(-1), float("-inf"))
        mx = masked.max(dim=2).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

        knn_charge = neigh_aug[..., T]                        # [B_l, K, k]
        if self.charge_weighted_mean:
            w = (knn_charge * knn_valid.to(knn_charge.dtype)).unsqueeze(-1)
        else:
            w = knn_valid.unsqueeze(-1).to(neigh_feats.dtype)
        if self.knn_pool == "max_mean":
            mean = (neigh_feats * w).sum(dim=2) / w.sum(dim=2).clamp(min=1e-6)
            pooled = torch.cat([mx, mean.to(mx.dtype)], dim=-1)
        else:
            pooled = mx

        # Summary scalars over the kNN sets (fp32), matching the voronoi
        # definitions so mlp2 sees a consistent layout across modes.
        vf = knn_valid.float()
        n_c = vf.sum(dim=2)                                   # [B_l, K]
        mult = torch.log1p(n_c)
        Q = torch.log1p((torch.expm1(knn_charge.float().clamp(min=0)) * vf).sum(dim=2))
        wq = (knn_charge.float() * vf) if self.charge_weighted_mean else vf
        wsafe = wq.sum(dim=2).clamp(min=1e-6)
        dt = rel[..., 3].float()
        m1 = (wq * dt).sum(dim=2) / wsafe
        m2 = (wq * dt * dt).sum(dim=2) / wsafe
        # clamp INSIDE the sqrt: sqrt'(0)=inf would NaN the backward on
        # single-neighbor tokens (see voronoi path).
        dt_std = (m2 - m1 * m1).clamp(min=1e-12).sqrt()
        r2 = (rel[..., :3].float().pow(2).sum(-1) * vf).sum(dim=2)
        rms = (r2 / n_c.clamp(min=1)).clamp(min=1e-12).sqrt()
        ex = torch.stack([mult, Q, dt_std, rms], dim=-1)      # [B_l, K, 4]

        pre_l = torch.cat([pooled, ex.to(feat_dtype)], dim=-1).view(B_l * K, -1)
        return pre_l, cents_raw
