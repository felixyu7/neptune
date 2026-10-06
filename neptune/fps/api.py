"""Public FPS API: validation, start resolution, and backend dispatch.

Drop-in replacement for torch-fps 0.5 (the front-end below is vendored from
torch_fps/fps.py). Backends:

- CUDA: Triton kernels (_triton.py) — JIT-compiled for the local GPU, so
  there is no compute-capability coupling at install time. float64 inputs
  (via ``precision=torch.float64``) fall back to the reference backend.
- CPU: C++ kernels JIT-compiled on first use (_cpu.py); pure-torch reference
  when no compiler is available.
- Other devices (e.g. MPS): pure-torch reference backend.

All backends produce bitwise-identical indices. With ``validate=False`` a
call performs no host synchronization (the tokenizer's contract). Rows whose
valid count is exhausted are padded by repeating the last selection; the
content of rows with zero valid points is unspecified beyond shape.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor

from . import _cpu, _reference

_warned_cuda_fallback = False


def _dispatch(points_c: Tensor, mask_c: Tensor, start_idx: Tensor,
              K: int, k_neighbors: Optional[int]):
    """Route a resolved call to the best backend for its device/dtype."""
    global _warned_cuda_fallback
    if points_c.is_cuda and points_c.dtype != torch.float64:
        try:
            from . import _triton
        except ImportError as exc:
            if not _warned_cuda_fallback:
                _warned_cuda_fallback = True
                warnings.warn(
                    f"neptune.fps: Triton unavailable on CUDA ({exc}); using "
                    "the slower pure-PyTorch implementation.", RuntimeWarning)
        else:
            if k_neighbors is None:
                return _triton.fps(points_c, mask_c, start_idx, K)
            return _triton.fps_knn(points_c, mask_c, start_idx, K, k_neighbors)
    elif points_c.device.type == "cpu":
        mod = _cpu.get_module()
        if mod is not None:
            if k_neighbors is None:
                return mod.fps_forward(points_c, mask_c, start_idx, K)
            return mod.fps_with_knn_forward(points_c, mask_c, start_idx, K, k_neighbors)
    if k_neighbors is None:
        return _reference.fps_reference(points_c, mask_c, start_idx, K)
    return _reference.fps_knn_reference(points_c, mask_c, start_idx, K, k_neighbors)


def _resolve_start_idx(
    valid: Tensor,
    counts: Optional[Tensor],
    B: int,
    N: int,
    K: int,
    device: torch.device,
    start_idx: Optional[Tensor],
    random_start: bool,
    generator: Optional[torch.Generator],
    validate: bool = True,
) -> Tensor:
    """Produce a validated contiguous [B] long start_idx with minimal GPU syncs.

    ``valid`` marks selectable points: mask-true AND all-finite coordinates.
    With ``validate=False`` the checks (and their host sync) are skipped
    entirely; ``counts`` may then be None on the ``start_idx is None`` path.
    All start resolution is pure-tensor: no host-device sync beyond validation.
    """
    if start_idx is None:
        if validate:
            problems = counts < K
            if bool(problems.any()):
                raise ValueError(
                    f"FPS requires K <= number of valid points. "
                    f"Found batch(es) with K={K} but fewer valid points."
                )
        if not random_start:
            # Deterministic start: first valid index per row (all-invalid rows
            # give 0; the kernel pads those rows anyway).
            return valid.long().argmax(dim=1).contiguous()
        # Random start: masked-random argmax draws uniformly over valid points
        # without multinomial's fp32 prob copy. All-invalid rows argmax to 0,
        # matching the kernel's documented pad behavior.
        scores = torch.rand(B, N, device=device, generator=generator)
        scores = torch.where(valid, scores, float("-inf"))
        return scores.argmax(dim=1).contiguous()

    # User-supplied path. Fuse all validation into a single sync. Bad
    # inputs (K>counts, out-of-range) raise; a start_idx that points to an
    # invalid slot (masked out or non-finite) is silently repaired to the first
    # valid index in that row.
    if start_idx.device != device:
        raise ValueError(
            "start_idx must be on the same device as points (a cross-device "
            "copy here would silently synchronize the host)"
        )
    start_idx = start_idx.to(dtype=torch.long)
    if start_idx.numel() != B:
        raise ValueError("start_idx must have shape [B]")
    start_idx = start_idx.reshape(B)

    if validate:
        out_of_range = (start_idx < 0) | (start_idx >= N)
        insufficient = counts < K
        problems = out_of_range | insufficient
        if bool(problems.any()):
            if bool(insufficient.any()):
                raise ValueError(
                    f"FPS requires K <= number of valid points. "
                    f"Found batch(es) with K={K} but fewer valid points."
                )
            raise ValueError("start_idx values must be within [0, N)")

    # Repair any start index that points to an invalid slot. Pure-tensor: no sync.
    has_valid = counts > 0
    safe_start = start_idx.clamp(0, max(N - 1, 0))
    supplied_valid = valid.gather(1, safe_start.unsqueeze(-1)).squeeze(-1)
    first_valid = torch.argmax(valid.long(), dim=1)
    repaired = torch.where(supplied_valid | ~has_valid, start_idx, first_valid)
    return repaired.contiguous()


def _prepare(points: Tensor, valid_mask: Tensor,
             precision: Optional[torch.dtype]) -> tuple[Tensor, Tensor]:
    device = points.device
    if precision is None:
        if points.dtype != torch.float32:
            points = points.to(dtype=torch.float32)
    else:
        if device.type == 'cpu' and precision == torch.bfloat16:
            raise ValueError(
                "bfloat16 is not supported on CPU (use float16, float32, or float64)")
        if points.dtype != precision:
            points = points.to(dtype=precision)
    if valid_mask.device != device:
        valid_mask = valid_mask.to(device)
    valid_mask = valid_mask.to(dtype=torch.bool)
    return points.contiguous(), valid_mask.contiguous()


def _prepare_and_resolve(points, valid_mask, K, start_idx, random_start,
                         generator, precision, validate, assume_finite):
    """Shared front-end: shape checks, dtype prep, validity, start resolution.

    Returns (points_c, mask_c, resolved_start_idx).
    """
    if points.dim() != 3:
        raise ValueError("points tensor must have shape [B, N, D]")
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask tensor must have shape [B, N]")
    if points.shape[:2] != valid_mask.shape:
        raise ValueError("points and valid_mask must agree on batch & point dims")
    if K < 0:
        raise ValueError("K must be non-negative")

    device = points.device
    points_c, mask_c = _prepare(points, valid_mask, precision)
    B, N, _ = points_c.shape

    # Empty-input edge cases: with K == 0 skip start resolution entirely (the
    # callers early-return an empty result and never use start_idx; argmax over
    # N == 0 would raise). N == 0 with K > 0 can never satisfy K <= valid
    # points, so reject it deterministically even with validate=False.
    if K == 0:
        return points_c, mask_c, torch.zeros(B, dtype=torch.long, device=device)
    if N == 0:
        raise ValueError("FPS with K > 0 requires at least one point (got N=0)")

    # Selectable = mask-true AND all-finite; validation and start repair both
    # count from this same predicate, so validate=True enforces exactly the
    # documented "K <= valid points" precondition. assume_finite skips the
    # [B,N,D] finiteness pass on the caller's guarantee.
    valid = mask_c if assume_finite else (mask_c & points_c.isfinite().all(dim=-1))
    counts = (
        valid.sum(dim=1, dtype=torch.long)
        if (validate or start_idx is not None) else None
    )
    start_idx = _resolve_start_idx(
        valid, counts, B, N, K, device,
        start_idx, random_start, generator, validate,
    )
    return points_c, mask_c, start_idx


def farthest_point_sampling(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tensor:
    """
    Farthest point sampling with Triton (CUDA) / C++ (CPU) acceleration.

    Args:
        points:
            Float tensor with shape `[B, N, D]` (batch, points, features).
        valid_mask:
            Bool tensor with shape `[B, N]`; False marks padded / invalid points.
        K:
            Integer number of samples to draw per batch element.
            Must satisfy `K <= number of valid points` for all batches.
        start_idx:
            Optional `[B]` long tensor providing the first index per batch.
            Must live on the same device as `points`; an index pointing at an
            invalid slot (masked out or non-finite) is repaired to the row's first
            valid index.
        random_start:
            If `True` (default) and `start_idx` is not supplied, draw a random
            first index from valid points (mask-true and all-finite). If `False`
            and `start_idx` is not supplied, the row's first valid index is
            used.
        generator:
            Optional `torch.Generator` used for deterministic random starts.
        precision:
            Optional dtype for internal computations. If None (default), uses
            float32 on all devices for numerical stability. Can override:
            float16, float32, float64 (CPU/GPU) or bfloat16 (GPU only).
        validate:
            If `True` (default), verify `K <= valid count` per batch row (and
            range-check a user-supplied `start_idx`). The check costs one
            host-device sync per call; callers that already guarantee the
            precondition can pass `False` to keep the call fully asynchronous.
            With `validate=False` a violated precondition is NOT diagnosed:
            the kernel pads the output with repeated indices instead of raising.
        assume_finite:
            If `True`, the caller guarantees every mask-true point has finite
            coordinates, and the `[B,N,D]` finiteness pass is skipped (validity
            = mask alone). With non-finite inputs under this flag, a non-finite
            point may be selected as the start (producing NaN downstream)
            instead of being excluded. In-kernel guards are unaffected.

    Returns:
        idx:
            Long tensor `[B, K]` with the selected point indices.
    """
    points_c, mask_c, start_idx = _prepare_and_resolve(
        points, valid_mask, K, start_idx, random_start,
        generator, precision, validate, assume_finite)
    if K == 0:
        return torch.zeros((points_c.shape[0], 0), device=points_c.device, dtype=torch.long)

    return _dispatch(points_c, mask_c, start_idx, K, None)


def farthest_point_sampling_with_knn(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    k_neighbors: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Fused farthest point sampling + k-nearest neighbors.

    Performs FPS and kNN in a single fused kernel: the distances computed
    during FPS are reused to find the k nearest neighbors of each selected
    centroid. Neighbors are sorted closest-first; the centroid itself and
    already-selected points are eligible neighbors, and rows with fewer than
    `k_neighbors` valid points pad with the centroid index. All other
    arguments and semantics match :func:`farthest_point_sampling`
    (`k_neighbors` must satisfy `0 < k_neighbors <= N`).

    Returns:
        centroid_idx:
            Long tensor `[B, K]` with the selected FPS centroid indices.
        neighbor_idx:
            Long tensor `[B, K, k_neighbors]` with the k nearest neighbor
            indices for each centroid, sorted by distance (closest first).
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    points_c, mask_c, start_idx = _prepare_and_resolve(
        points, valid_mask, K, start_idx, random_start,
        generator, precision, validate, assume_finite)
    B, N, _ = points_c.shape

    if K == 0:
        centroid_idx = torch.zeros((B, 0), device=points_c.device, dtype=torch.long)
        neighbor_idx = torch.zeros((B, 0, k_neighbors), device=points_c.device, dtype=torch.long)
        return centroid_idx, neighbor_idx

    if k_neighbors > N:
        raise ValueError(f"k_neighbors ({k_neighbors}) must be <= N ({N})")

    return _dispatch(points_c, mask_c, start_idx, K, k_neighbors)


def farthest_point_sampling_with_assign(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Fused farthest point sampling + nearest-centroid (Voronoi) assignment.

    The FPS loop already computes every point's distance to each selected
    centroid, so the assignment comes at negligible extra cost on the fused
    CUDA path. Ties assign to the lowest centroid index. All arguments and
    semantics match :func:`farthest_point_sampling`.

    Returns:
        idx:
            Long tensor `[B, K]` with the selected centroid indices.
        assign:
            Long tensor `[B, N]` with each point's nearest centroid as a
            position in `[0, K)` (i.e. an index into `idx`, not into `points`).
            Values at invalid lanes (masked out or non-finite) are unspecified.
    """
    points_c, mask_c, start_idx = _prepare_and_resolve(
        points, valid_mask, K, start_idx, random_start,
        generator, precision, validate, assume_finite)
    B, N, D = points_c.shape

    if K == 0:
        return (torch.zeros((B, 0), device=points_c.device, dtype=torch.long),
                torch.zeros((B, N), device=points_c.device, dtype=torch.long))

    if points_c.is_cuda and points_c.dtype != torch.float64:
        try:
            from . import _triton
        except ImportError:
            pass
        else:
            if N <= _triton._single_tile_cap(_triton.SINGLE_TILE_MAX_N, D):
                return _triton.fps_assign(points_c, mask_c, start_idx, K)
            # tiled-N CUDA: tiled FPS + tiled nearest-centroid kernel. The
            # dense reference would materialize [B, N, K] fp32 distances
            # (GBs for pulse-level events with N ~ 1e5).
            if K <= 512:
                idx = _dispatch(points_c, mask_c, start_idx, K, None)
                cents = torch.gather(points_c, 1, idx.unsqueeze(-1).expand(-1, -1, D))
                starts = torch.arange(B, device=points_c.device, dtype=torch.long) * N
                counts = torch.full((B,), N, device=points_c.device, dtype=torch.long)
                assign = _triton.nearest_assign(
                    points_c.reshape(B * N, D), cents.reshape(B * K, D),
                    starts, counts, N, K)
                return idx, assign.view(B, N)
    # composed route: CPU and other devices
    idx = _dispatch(points_c, mask_c, start_idx, K, None)
    return idx, _reference.assign_reference(points_c, idx)
