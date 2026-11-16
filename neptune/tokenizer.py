import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional

from .fps import farthest_point_sampling_with_assign


class FPSTokenizer(nn.Module):
    """
    GPU-native, vectorized tokenizer for point clouds:
      - MLP 1: per-point feature extraction
      - If num_points <= max_tokens: use all points (no seeding, no pooling)
      - Else: pick max_tokens seed hits in 4D (x,y,z,t), then pool hits per
        token by nearest-seed (Voronoi) assignment — every hit contributes to
        exactly one token, so no hit is ever dropped. Each cell is pooled as
        [max, charge-weighted mean] of its hit features plus an embedding of
        each hit's offset from the cell centroid.
      - Per-token summary scalars (multiplicity, total charge, time spread,
        spatial RMS radius) are appended to the pooled features.
      - MLP 2: token refinement (always applied for consistent depth)

    ``seed_mode`` picks the seeds of large events:
      - ``"charge"`` (default): the max_tokens highest-charge hits. With
        ``lloyd_iters=0`` each token sits exactly on its bright seed hit.
      - ``"fps"``: farthest point sampling (spatial coverage). This, with
        ``lloyd_iters=2``, is the WhaT-1 v1.2 tokenization.

    The forward performs exactly one host sync (``counts.cpu().tolist()``) and
    uses no data-dependent-shape ops (``nonzero``/boolean indexing): small and
    large events are routed with precomputed index tensors, and only the large
    subset is ever padded (to its own max length), so one bright event doesn't
    inflate the whole batch.

    ``metric_time_scale`` weights the time axis in the FPS/assignment metric
    only. Coordinates are in km and time in microseconds (see mmap dataloader);
    at raw units the time axis carries ~99% of the 4D metric variance on real
    events, so selection would cluster nearly purely by arrival time. The
    default 0.3 km/us (light speed) balances space and time; 0.22 is the photon
    group velocity in ice; 1.0 leaves raw units. Returned centroids, relative
    offsets, and summary scalars always stay in raw units — the scale changes
    token membership, never the representation.

    ``lloyd_iters`` refines the seeds with that many charge-weighted Lloyd
    (k-means) iterations in the scaled metric. FPS solves a k-center-style
    objective (equal cell *radius*), which over-compresses dense bright cores
    and wastes tokens on stray periphery hits; Lloyd refinement moves centroids
    toward the charge-weighted vector-quantization optimum. With
    ``lloyd_iters > 0`` centroids become virtual points (charge-weighted cell
    means in raw units over the final membership) rather than hit positions,
    so relative offsets are zero-mean per cell; cells can then be empty
    (all-zero token with multiplicity scalar 0, mask stays True — same contract
    as the coincident-centroid edge case).

    ``anchor_dt_col`` (pulse inputs) names the feature column holding
    log1p(ns since the DOM's first pulse). When set, seeding, assignment and
    centroids use each pulse's DOM first-hit time (all pulses of a DOM share
    one anchor point -> same cell, and token times track the direct-light
    front); relative offsets keep the pulse time. Summary-stats inputs are
    already one point per DOM at its first-hit time and leave this unset.
    """

    N_EXTRA = 4  # multiplicity, total charge, time spread, RMS radius

    def __init__(self,
                 feature_dim: int,
                 max_tokens: int = 128,
                 token_dim: int = 768,
                 mlp_layers: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 rel_pos_hidden: int = 64,
                 charge_col: int = 0,
                 metric_time_scale: float = 0.3,
                 seed_mode: str = "charge",
                 lloyd_iters: int = 0,
                 anchor_dt_col: Optional[int] = None):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 512, 768]
        if seed_mode not in ("charge", "fps"):
            raise ValueError(f"seed_mode must be 'charge' or 'fps', got '{seed_mode}'")
        if lloyd_iters < 0:
            raise ValueError(f"lloyd_iters must be >= 0, got {lloyd_iters}")
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.charge_col = charge_col
        self.metric_time_scale = metric_time_scale
        self.seed_mode = seed_mode
        self.lloyd_iters = lloyd_iters
        self.anchor_dt_col = anchor_dt_col

        # MLP 1: Per-point feature extraction
        mlp1 = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp1 += [nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = out_dim
        mlp1 += [nn.Linear(in_dim, token_dim)]
        self.mlp1 = nn.Sequential(*mlp1)

        # MLP 2: Token refinement (always applied). Input is the [max, mean]
        # pooled features plus the per-token summary scalars.
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * token_dim + self.N_EXTRA, token_dim),
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
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
          coords:    [N, 3] spatial coordinates (x, y, z)
          features:  [N, F]
          batch_ids: [N] batch indices
          times:     [N, 1] time coordinates
          batch_size: optional true batch size B. Without it, B is inferred as
            max(batch_ids)+1, which silently drops trailing zero-hit events
            (their ids never appear) and misaligns the output with the labels.
        Returns (K = min(max_tokens, largest event in the batch), so batches
        of small events are not padded to max_tokens):
          tokens:    [B, K, token_dim]
          centroids: [B, K, 4]   (x,y,z,t)
          masks:     [B, K] bool
        """
        device = coords.device
        dtype_p = coords.dtype
        dtype_f = features.dtype
        K = self.max_tokens
        T = self.token_dim

        if coords.numel() == 0:
            B0 = batch_size or 0
            empty_tokens    = torch.zeros((B0, K, T), device=device, dtype=dtype_f)
            empty_centroids = torch.zeros((B0, K, 4), device=device, dtype=dtype_p)
            empty_masks     = torch.zeros((B0, K),    device=device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        batch_idx = batch_ids.long()
        points4 = torch.cat([coords[:, :3], times], dim=-1)   # [N,4] raw units
        if self.anchor_dt_col is not None:
            dt_dom = torch.expm1(features[:, self.anchor_dt_col].float()) * 1e-3  # ns -> us
            anchor4 = torch.cat([coords[:, :3], times - dt_dom.unsqueeze(-1).to(times.dtype)], dim=-1)
        else:
            anchor4 = points4
        point_feats = self.mlp1(features)                     # [N, T]
        q = features[:, self.charge_col]                      # [N] log1p charge

        # The single host sync: one D2H copy of the ids. (bincount on a CUDA
        # tensor would itself sync — it reads max(batch_idx) on the host to
        # size its output — so counting happens on the CPU copy instead.)
        counts_list = torch.bincount(batch_idx.cpu(), minlength=(batch_size or 0)).tolist()
        B = len(counts_list)
        # Pad only to the largest event: slots past it are masked anyway, but
        # unpacked attention (CPU) would still pay for them.
        K = min(K, max(1, max(counts_list)))

        # Sort hits by batch so each event is a contiguous run.
        sort_idx = torch.argsort(batch_idx, stable=True)
        p_sorted = points4.index_select(0, sort_idx)
        a_sorted = p_sorted if anchor4 is points4 else anchor4.index_select(0, sort_idx)
        f_sorted = point_feats.index_select(0, sort_idx)
        q_sorted = q.index_select(0, sort_idx)

        # Host-side split into small (<= K hits: every hit is a token) and
        # large (> K: seeding + pooling) events. All routing below comes from
        # these lists — no boolean indexing, no nonzero, no further syncs.
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

        pooled_dim = 2 * T                                    # [max, mean]
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

            # Single point: max == mean, multiplicity 1, its own charge, zero spreads.
            zeros_s = torch.zeros_like(q_s)
            ex_s = torch.stack(
                [torch.full_like(q_s, 0.6931471805599453), q_s, zeros_s, zeros_s], dim=-1)
            pre_s = torch.zeros(B_s * K, pooled_dim + self.N_EXTRA,
                                device=device, dtype=feat_dtype)
            pre_s.index_copy_(0, dest, torch.cat([f_s, f_s, ex_s.to(feat_dtype)], dim=-1))
            cents_s = torch.zeros(B_s * K, 4, device=device, dtype=dtype_p)
            cents_s.index_copy_(0, dest, p_s)

            pre.index_copy_(0, rows_s_dev, pre_s.view(B_s, K, -1))
            cents_out.index_copy_(0, rows_s_dev, cents_s.view(B_s, K, 4))
            masks.index_copy_(
                0, rows_s_dev,
                torch.arange(K, device=device)[None, :] < csub_s[:, None])

        # ---- large events: K seeds + per-token (Voronoi cell) pooling ----
        if B_l > 0:
            Nmax_l = max(counts_l)
            seg_l, local_l, src_l = self._route_subset(
                rows_l_dev, csub_l, starts_l_abs, N_l, device)
            f_l = f_sorted.index_select(0, src_l)             # [N_l, T]
            p_l = p_sorted.index_select(0, src_l)             # [N_l, 4] raw
            q_l = q_sorted.index_select(0, src_l)             # [N_l]
            a_l = p_l if a_sorted is p_sorted else a_sorted.index_select(0, src_l)
            n_l = f_l.size(0)

            # Metric copy for seeding/assignment: fp32, time axis scaled. Padding
            # stays zero (finite) — the kernel masks it but must not see NaNs.
            p_l_m = a_l.float() * self.metric_scale           # [N_l, 4]
            P_pad_m = torch.zeros(B_l * Nmax_l, 4, device=device)
            dest_pad = seg_l * Nmax_l + local_l
            P_pad_m.index_copy_(0, dest_pad, p_l_m)
            P_pad_m = P_pad_m.view(B_l, Nmax_l, 4)
            starts_l = torch.cumsum(csub_l, 0) - csub_l       # [B_l] into flat subset

            def _cell_dists(C_m: Tensor) -> Tensor:
                # Non-CUDA fallback: squared metric distances via
                # ||c||^2 - 2 p.c (the ||p||^2 term is constant per hit,
                # hence argmin-invariant, and dropped): one baddbmm
                # materializing [B_l*Nmax_l, K].
                c_sq = C_m.pow(2).sum(-1)                             # [B_l, K]
                d_pad = torch.baddbmm(c_sq.unsqueeze(1), P_pad_m,
                                      C_m.transpose(1, 2), alpha=-2.0)
                return d_pad.reshape(-1, K).index_select(0, dest_pad)  # [N_l, K]

            def _nearest(cm_flat: Tensor) -> Tensor:
                # Streaming Triton argmin on CUDA: no [N_l, K] intermediate.
                # Elsewhere the baddbmm fallback, NaN-guarded to match the
                # kernel (a degenerate Lloyd centroid must never win).
                if device.type == "cuda":
                    from .fps._triton import nearest_assign
                    return nearest_assign(p_l_m, cm_flat, starts_l,
                                          csub_l, Nmax_l, K)
                d = _cell_dists(cm_flat.view(B_l, K, 4))
                return torch.where(d == d, d, float("inf")).argmin(dim=1)

            if self.seed_mode == "charge":
                # The K brightest hits of each event (counts > K by construction
                # of the subset, so the -inf padding is never selected).
                s_pad = torch.full((B_l * Nmax_l,), float("-inf"), device=device)
                s_pad.index_copy_(0, dest_pad, q_l.float())
                seed_idx, assign_pad = s_pad.view(B_l, Nmax_l).topk(K, dim=1).indices, None
            else:
                # Fused FPS + nearest-centroid assignment: the FPS loop already
                # computes every hit-to-centroid distance, so the initial
                # Voronoi assignment is free. Runs fp32 regardless of autocast.
                # random_start gates FPS augmentation on train mode: training
                # draws a random start point per call, eval/inference uses a
                # deterministic start so the same event always tokenizes
                # identically. validate=False: counts > K holds by construction.
                valid_l = torch.arange(Nmax_l, device=device)[None, :] < csub_l[:, None]
                seed_idx, assign_pad = farthest_point_sampling_with_assign(
                    P_pad_m, valid_l, K,
                    random_start=self.training, validate=False,
                    assume_finite=True)               # [B_l, K], [B_l, Nmax_l]
            cent_flat = (starts_l[:, None] + seed_idx).reshape(-1)    # [B_l*K]
            cents_raw = a_l.index_select(0, cent_flat)               # [B_l*K, 4]
            cents_m = p_l_m.index_select(0, cent_flat).view(B_l, K, 4)

            # Every hit joins exactly one token (coverage = 1 by
            # construction); assignment in the scaled metric.
            assign = (_nearest(cents_m.reshape(-1, 4)) if assign_pad is None
                      else assign_pad.reshape(-1).index_select(0, dest_pad))  # [N_l]
            cell = seg_l * K + assign                                 # [N_l]

            q_phys = torch.expm1(q_l.float().clamp(min=0))            # [N_l]

            if self.lloyd_iters > 0:
                # Charge-weighted Lloyd refinement toward the VQ optimum.
                # Empty cells keep their previous centroid; assignment uses
                # standard alternating updates in the scaled metric.
                BK = B_l * K
                cm_flat = cents_m.reshape(BK, 4)
                cell4 = cell.unsqueeze(-1).expand(-1, 4)
                for _ in range(self.lloyd_iters):
                    wsum = torch.zeros(BK, device=device).scatter_add_(0, cell, q_phys)
                    csum = torch.zeros(BK, 4, device=device).scatter_add_(
                        0, cell4, p_l_m * q_phys[:, None])
                    cm_flat = torch.where(
                        (wsum > 0)[:, None],
                        csum / wsum.clamp(min=1e-9)[:, None], cm_flat)
                    assign = _nearest(cm_flat)
                    cell = seg_l * K + assign
                    cell4 = cell.unsqueeze(-1).expand(-1, 4)
                # Raw-unit centroids: charge-weighted cell means over the
                # FINAL membership (zero-mean rel offsets per cell); empty
                # cells fall back to their seed hit position.
                wsum = torch.zeros(BK, device=device).scatter_add_(0, cell, q_phys)
                craw = torch.zeros(BK, 4, device=device).scatter_add_(
                    0, cell4, a_l * q_phys[:, None])
                cents_raw = torch.where(
                    (wsum > 0)[:, None],
                    (craw / wsum.clamp(min=1e-9)[:, None]).to(cents_raw.dtype),
                    cents_raw)

            rel = p_l - cents_raw.index_select(0, cell)               # raw units
            h = f_l + self.rel_encoder(rel).to(feat_dtype)            # [N_l, T]

            cell_T = cell.unsqueeze(-1).expand(-1, T)
            mx = torch.full((B_l * K, T), float("-inf"),
                            device=device, dtype=h.dtype)
            mx = mx.scatter_reduce(0, cell_T, h, reduce="amax", include_self=True)
            mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

            # fp32 accumulators for the charge-weighted mean and summary scalars.
            w = q_l.float()
            wsum = torch.zeros(B_l * K, device=device).scatter_add_(0, cell, w)
            wsafe = wsum.clamp(min=1e-6)
            hsum = torch.zeros(B_l * K, T, device=device).scatter_add_(
                0, cell_T, h.float() * w[:, None])
            mean = (hsum / wsafe[:, None]).to(h.dtype)

            ones_h = torch.ones(n_l, device=device)
            n_c = torch.zeros(B_l * K, device=device).scatter_add_(0, cell, ones_h)
            mult = torch.log1p(n_c)
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

            pre_l = torch.cat([mx, mean, ex_l.to(feat_dtype)], dim=-1)

            pre.index_copy_(0, rows_l_dev, pre_l.view(B_l, K, -1))
            cents_out.index_copy_(0, rows_l_dev, cents_raw.view(B_l, K, 4).to(dtype_p))
            masks.index_copy_(
                0, rows_l_dev, torch.ones(B_l, K, device=device, dtype=torch.bool))

        # MLP 2: Always apply token refinement (consistent depth)
        tokens = self.mlp2(pre)                               # [B, K, T]

        # Zero out padded positions
        tokens = tokens * masks.unsqueeze(-1)
        cents_out = cents_out * masks.unsqueeze(-1)

        return tokens.to(dtype_f), cents_out, masks
