"""Base class, shared utilities, and grid generation for directional distributions on S²."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._plotting import plot_mollweide

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Mixed-precision helper
# ---------------------------------------------------------------------------

# Low-precision dtypes whose exp/log/erf normalization math is unstable and must
# be computed in fp32 instead.
_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Working dtype for sensitive normalization math: upcast fp16/bf16 to fp32,
    never downcast fp64. Shared by the vMF and angular-Gaussian losses/log-pdfs."""
    return torch.float32 if dtype in _LOW_PRECISION else dtype


# ---------------------------------------------------------------------------
# Reduction helper for *_nll_loss functions
# ---------------------------------------------------------------------------

def _apply_reduction(loss: Tensor, reduction: str) -> Tensor:
    """Apply a PyTorch-style reduction to an elementwise loss tensor.

    Args:
        loss: Per-sample loss tensor.
        reduction: One of ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Reduced (or unreduced) loss tensor.
    """
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(
        f"reduction must be one of 'mean', 'sum', 'none'; got {reduction!r}"
    )


# ---------------------------------------------------------------------------
# Sphere grid
# ---------------------------------------------------------------------------

@dataclass
class SphereGrid:
    """A lat/lon grid of points on the unit sphere S².

    Attributes:
        points: [N, 3] unit vectors (Cartesian coordinates).
        lon: [N] longitude in radians, range (-π, π].
        lat: [N] latitude in radians, range [-π/2, π/2].
        shape: (n_lat, n_lon) for reshaping flat arrays into 2D grids.
    """

    points: Tensor
    lon: Tensor
    lat: Tensor
    shape: tuple


def make_grid(
    n_lat: int = 181, n_lon: int = 360, device: torch.device | None = None
) -> SphereGrid:
    """Generate a uniform lat/lon grid of points on S².

    Args:
        n_lat: Number of latitude bins (pole to pole).
        n_lon: Number of longitude bins.
        device: Torch device for the output tensors.

    Returns:
        SphereGrid with n_lat * n_lon points.
    """
    lat = torch.linspace(-np.pi / 2, np.pi / 2, n_lat, device=device)
    lon = torch.linspace(-np.pi, np.pi, n_lon + 1, device=device)[:-1]

    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.reshape(-1)
    lon_flat = lon_grid.reshape(-1)

    cos_lat = torch.cos(lat_flat)
    x = cos_lat * torch.cos(lon_flat)
    y = cos_lat * torch.sin(lon_flat)
    z = torch.sin(lat_flat)

    points = torch.stack([x, y, z], dim=1)

    return SphereGrid(points=points, lon=lon_flat, lat=lat_flat, shape=(n_lat, n_lon))



def _construct_orthonormal_basis(mu: Tensor) -> Tuple[Tensor, Tensor]:
    """Construct two orthonormal vectors perpendicular to mu using Gram-Schmidt.

    Avoids the singularity when mu is aligned with the x-axis.

    Args:
        mu: [B, 3] mean direction vectors (not necessarily normalized).

    Returns:
        xi1, xi2: [B, 3] orthonormal vectors perpendicular to mu.
    """
    B = mu.shape[0]
    device, dtype = mu.device, mu.dtype
    mu_norm_len = mu.norm(p=2, dim=1, keepdim=True)
    mu_norm = mu / mu_norm_len.clamp_min(1e-12)

    # For zero-mean vectors, pick a deterministic direction so the basis remains valid.
    zero_mu = (mu_norm_len.squeeze(1) <= 1e-12).unsqueeze(1)
    fallback_mu = torch.zeros(B, 3, device=device, dtype=dtype)
    fallback_mu[:, 2] = 1.0
    mu_norm = torch.where(zero_mu, fallback_mu, mu_norm)

    ref1 = torch.zeros(B, 3, device=device, dtype=dtype)
    ref1[:, 0] = 1.0
    ref2 = torch.zeros(B, 3, device=device, dtype=dtype)
    ref2[:, 1] = 1.0

    dot1 = torch.abs((mu_norm * ref1).sum(dim=1))
    use_ref2 = dot1 > 0.9
    ref = torch.where(use_ref2.unsqueeze(1), ref2, ref1)

    xi1 = ref - (ref * mu_norm).sum(dim=1, keepdim=True) * mu_norm
    xi1 = F.normalize(xi1, p=2, dim=1)
    xi2 = torch.cross(mu_norm, xi1, dim=1)
    xi2 = F.normalize(xi2, p=2, dim=1)

    return xi1, xi2


# ---------------------------------------------------------------------------
# Cholesky parameterization utilities (used by GAG / GSPC)
# ---------------------------------------------------------------------------

# Bound on the free log-diagonals so exp() can never overflow to Inf (under low
# precision or extreme concentration). L_ii ∈ [e⁻¹⁵, e¹⁵] is far outside any
# physical regime, so this never bites normal training.
_LOG_DIAG_CLAMP = 15.0


def _build_cholesky(pred: Tensor) -> Tensor:
    """Construct the normalised lower-triangular Cholesky factor L from raw
    network outputs.

    The det(L) = 1 constraint is enforced directly: only two log-diagonal
    entries are taken from the network and the third is fixed by
    log L₃₃ = -(log L₁₁ + log L₂₂), so the diagonal entries multiply to 1.
    This consumes exactly the 2 degrees of freedom the constraint leaves,
    avoiding the translation gauge-freedom that arises when mean-centring
    three log-diagonal outputs (where adding a constant to all three is a
    no-op and the corresponding output direction receives zero gradient).

    Args:
        pred: [B, 8] where pred[:, 3:5] are the first two log-diagonal
              entries (log L₁₁, log L₂₂) and pred[:, 5:8] are the
              off-diagonal entries (L₂₁, L₃₁, L₃₂).

    Returns:
        L: [B, 3, 3] lower-triangular with det(L) = 1 and V⁻¹ = LLᵀ SPD.
    """
    B = pred.shape[0]
    device, dtype = pred.device, pred.dtype

    # Clamp the free log-diagonals so exp() stays finite; det(L) = 1 is
    # preserved since log L₃₃ = -(clamped sum).
    log_diag_12 = pred[:, 3:5].clamp(-_LOG_DIAG_CLAMP, _LOG_DIAG_CLAMP)  # [B, 2]  → log L₁₁, log L₂₂
    off_diag = pred[:, 5:8]      # [B, 3]  → L₂₁, L₃₁, L₃₂

    # Third log-diagonal fixed by det(L) = 1  ⟹  Σ log Lᵢᵢ = 0
    log_diag_3 = -log_diag_12.sum(dim=1, keepdim=True)             # [B, 1]
    diag = torch.exp(torch.cat([log_diag_12, log_diag_3], dim=1))  # [B, 3], product = 1

    # Assemble L  (lower triangular)
    L = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    L[:, 0, 0] = diag[:, 0]
    L[:, 1, 1] = diag[:, 1]
    L[:, 2, 2] = diag[:, 2]
    L[:, 1, 0] = off_diag[:, 0]   # L₂₁
    L[:, 2, 0] = off_diag[:, 1]   # L₃₁
    L[:, 2, 1] = off_diag[:, 2]   # L₃₂

    return L


# ---------------------------------------------------------------------------
# Base distribution class
# ---------------------------------------------------------------------------

class BaseDistribution:
    """Base class for directional distributions on the unit sphere S².

    Subclasses must set :attr:`n_params` and implement
    :attr:`mean_direction` and :meth:`log_pdf`.
    """

    n_params: int

    def __init__(self, pred: Tensor) -> None:
        if pred.shape[-1] != self.n_params:
            raise ValueError(
                f"{type(self).__name__} expects pred with last dim "
                f"{self.n_params}, got {pred.shape[-1]}"
            )
        self._pred = pred

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        raise NotImplementedError

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log-PDF at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        raise NotImplementedError

    def pdf(self, points: Tensor) -> Tensor:
        """Evaluate PDF at points on S²."""
        return self.log_pdf(points).exp()

    def plot_mollweide(
        self,
        idx: int = 0,
        n_lat: int = 181,
        n_lon: int = 360,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot PDF on a Mollweide projection for a single sample.

        Args:
            idx: Batch index to plot.
            n_lat: Number of latitude bins.
            n_lon: Number of longitude bins.
            **kwargs: Forwarded to :func:`plot_mollweide`.

        Returns:
            ``(fig, ax)`` tuple.
        """
        grid = make_grid(n_lat, n_lon, device=self._pred.device)
        with torch.no_grad():
            vals = self.pdf(grid.points)[idx]
        vals_2d = vals.reshape(grid.shape).cpu()
        return plot_mollweide(grid, vals_2d, **kwargs)
