"""Pure-PyTorch FPS backends: any-device fallback and test oracle.

Implements the backend contract shared with the Triton and C++ backends
(see api.py): inputs are contiguous points/mask and a resolved [B] start
index; validation and start resolution have already happened. Semantics
mirror the native kernels exactly:

- validity = mask AND all-finite coordinates,
- distances accumulate in fp32 (fp64 for double inputs),
- ties select the lowest index (torch.argmax/max return the first
  occurrence of the maximum),
- an out-of-range start falls back to index 0; once valid candidates are
  exhausted (K > valid count) the last selection is repeated,
- kNN neighbors are closest-first over all valid points (the centroid and
  already-selected points included); rows with fewer than k valid points
  pad with the centroid index.

Vectorized over the batch; the only Python loop is the inherently
sequential K iterations. No host syncs.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _acc(points: Tensor) -> Tensor:
    return points if points.dtype == torch.float64 else points.float()


def _fps_state(points: Tensor, mask: Tensor, start_idx: Tensor):
    B, N, _ = points.shape
    pts = _acc(points)
    valid = mask & points.isfinite().all(dim=-1)
    inf = float("inf")
    min_d = torch.where(valid, inf, -inf).to(pts.dtype)
    in_range = (start_idx >= 0) & (start_idx < N)
    last = torch.where(in_range, start_idx, torch.zeros_like(start_idx))
    return pts, valid, min_d, last


def fps_reference(points: Tensor, mask: Tensor, start_idx: Tensor, K: int) -> Tensor:
    B, N, _ = points.shape
    idx = torch.empty(B, K, device=points.device, dtype=torch.long)
    pts, valid, min_d, last = _fps_state(points, mask, start_idx)
    rows = torch.arange(B, device=points.device)
    for i in range(K):
        idx[:, i] = last
        min_d[rows, last] = -float("inf")
        if i + 1 == K:
            break
        c = pts[rows, last]
        d = (pts - c[:, None, :]).square().sum(dim=2)
        # kernel-exact update: invalid lanes keep -inf, NaN distances (only
        # reachable via a pathological non-finite start) never overwrite
        min_d = torch.where(valid & (d < min_d), d, min_d)
        vals, nxt = min_d.max(dim=1)
        # exhausted rows (all -inf) repeat the previous selection
        last = torch.where(vals.isneginf(), last, nxt)
    return idx


def assign_reference(points: Tensor, idx: Tensor) -> Tensor:
    """Nearest-centroid assignment to the FPS-selected centroids ``idx``.

    Returns [B, N] int64 in [0, K); values at invalid lanes are unspecified.
    Per-dim accumulation in fp32 matches the Triton kernels' order bitwise
    (no [B,N,K,D] broadcast — the [B,N,K] accumulator keeps memory at the
    same scale as the old baddbmm intermediate). NaN centroid distances
    promote to +inf so a degenerate centroid is never chosen.
    """
    B, N, D = points.shape
    pts = _acc(points)
    cents = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, D))  # [B,K,D]
    d = pts.new_zeros(B, N, idx.size(1))
    for dim in range(D):
        diff = pts[:, :, dim].unsqueeze(-1) - cents[:, :, dim].unsqueeze(1)
        d = d + diff * diff
    d = torch.where(d == d, d, float("inf"))
    return d.argmin(dim=-1)


def fps_knn_reference(points: Tensor, mask: Tensor, start_idx: Tensor,
                      K: int, k_neighbors: int) -> tuple[Tensor, Tensor]:
    B, N, D = points.shape
    centroid_idx = fps_reference(points, mask, start_idx, K)
    pts = _acc(points)
    valid = mask & points.isfinite().all(dim=-1)
    cents = torch.gather(pts, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, D))
    d = (cents.unsqueeze(2) - pts.unsqueeze(1)).square().sum(dim=-1)   # [B, K, N]
    d = torch.where(valid.unsqueeze(1), d, float("inf"))
    # stable sort ⇒ lowest index first among exact ties, like the kernels
    svals, sidx = d.sort(dim=-1, stable=True)
    svals, sidx = svals[..., :k_neighbors], sidx[..., :k_neighbors]
    # slots past the row's valid count carry +inf → pad with the centroid
    return centroid_idx, torch.where(svals.isinf(), centroid_idx.unsqueeze(-1), sidx)
