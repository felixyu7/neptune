"""Angular Gaussian family: IAG, ESAG, and GAG distributions on S².

This module collects the three members of the angular Gaussian family,
obtained by projecting a trivariate normal N(mu, V) onto the sphere
via z -> z/||z||.  They form a nested hierarchy:

    IAG (3 params)  <  ESAG (5 params)  <  GAG (8 params)

References
----------
Paine et al. (2018), "An elliptically symmetric angular Gaussian
distribution", Stat Comput 28:689-697.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import (
    BaseDistribution,
    _apply_reduction,
    _compute_dtype,
    _construct_orthonormal_basis,
    _build_cholesky,
    make_grid,
)


_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# ``fast=True`` preset for ESAG/GAG ``.mode()``. Tuned for typical
# (non-adversarially-narrow) basins: a 4° coarse pitch is enough to land in
# the right basin, and 5 refinement levels push the quantization floor to
# ~0.35 arcsec. ~3x faster than the default while staying within 0.07° of the
# default mode at the 99th percentile on real WhaT-1 tracks.
_MODE_FAST_PRESET = dict(n_lat=46, n_lon=90, n_levels=5)


# ---------------------------------------------------------------------------
# AG-family math utility
# ---------------------------------------------------------------------------

def _log_M2(alpha: Tensor) -> Tensor:
    """Compute log(M_2(alpha)) numerically stably.

    M_2(alpha) = (1 + alpha^2) * Phi(alpha) + alpha * phi(alpha)

    where Phi is the standard normal CDF and phi is the standard normal PDF.

    For large negative alpha, direct computation suffers from catastrophic
    cancellation.  The negative-tail branch rewrites ``Phi`` with ``erfcx`` and
    computes the sensitive subtraction in float64:

        M_2(-x) = exp(-x^2/2) *
                  [0.5(1 + x^2) erfcx(x / sqrt(2)) - x / sqrt(2pi)]

    This avoids the slow ``torch.func`` batching fallback for
    ``torch.special.log_ndtr`` while retaining stable tail values.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (4).
    """
    # Direct computation (accurate for moderate alpha)
    alpha_direct = torch.clamp(alpha, min=-3.5)
    log_phi = -0.5 * alpha_direct ** 2 - 0.5 * _LOG_2PI
    phi = torch.exp(log_phi)
    Phi = 0.5 * (1.0 + torch.erf(alpha_direct / _SQRT_2))
    M2_direct = (1.0 + alpha_direct ** 2) * Phi + alpha_direct * phi

    alpha_hi = alpha.to(torch.float64)
    x_hi = torch.clamp(-alpha_hi, min=0.0)
    inner_hi = (
        0.5 * (1.0 + x_hi ** 2) * torch.special.erfcx(x_hi / _SQRT_2)
        - x_hi / _SQRT_2PI
    )
    M2_tail = (
        -0.5 * x_hi ** 2
        + torch.log(torch.clamp(inner_hi, min=1e-300))
    ).to(alpha.dtype)

    # Use direct for alpha >= -3.5, stable form for alpha < -3.5.
    # The direct branch suffers catastrophic cancellation in M2_direct for
    # alpha < ~-3.8 (float32), while the tail branch (computed in float64)
    # is accurate for all alpha < 0.
    return torch.where(
        alpha >= -3.5,
        torch.log(torch.clamp(M2_direct, min=1e-40)),
        M2_tail,
    )


def _ag_log_kernel(Q: Tensor, S: Tensor, T: Tensor) -> Tensor:
    """Angular Gaussian log-density without the ``-log(2pi)`` constant."""
    Q_safe = torch.clamp(Q, min=1e-8)
    return -1.5 * torch.log(Q_safe) - 0.5 * (S - T ** 2) + _log_M2(T)


def _safe_unit_vector(vec: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize vectors, using +z when the direction is undefined."""
    norm = vec.norm(p=2, dim=-1, keepdim=True)
    fallback = torch.zeros_like(vec)
    fallback[..., 2] = 1.0
    return torch.where(norm > eps, vec / norm.clamp_min(eps), fallback)


# ---------------------------------------------------------------------------
# Mode-finding on S^2 (hierarchical grid refinement)
# ---------------------------------------------------------------------------

def _hierarchical_mode_on_sphere(
    log_p_batched,
    params: tuple,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
    n_lat: int = 91,
    n_lon: int = 180,
    n_levels: int = 3,
    patch_size: int = 11,
    shrink: float = 0.25,
    grid_chunk: int = 256,
) -> Tensor:
    """Find S^2 modes by coarse lat/lon argmax + tangent-plane refinement.

    Level 0 is a global ``n_lat x n_lon`` argmax (chunked across the batch to
    bound memory). Each refinement level evaluates a ``patch_size x patch_size``
    tangent-plane grid centred on the current best point with radius shrinking
    by ``shrink``. After ``K`` levels the resolution is
    ``(pi / (n_lat - 1)) * shrink^K / (patch_size - 1)``.

    Args:
        log_p_batched: callable ``(params_slice, cands) -> log_p`` where
            ``cands`` is ``[b, M, 3]`` per-sample candidate directions and the
            result is ``[b, M]``. ``params_slice`` is a tuple of per-sample
            parameters sliced to the current chunk.
        params: tuple of ``[B, ...]`` tensors carrying per-sample parameters.
        B, device, dtype: shape/placement for the working buffer.
        n_lat, n_lon: coarse global grid resolution (default 91 x 180 ≈ 2°).
        n_levels: refinement passes after the global scan.
        patch_size: P; the tangent grid is P x P (recommend odd).
        shrink: per-level radius shrink factor.
        grid_chunk: row chunk for the level-0 eval to bound peak memory.

    Returns:
        ``[B, 3]`` unit-norm MAP directions (detached).
    """
    grid = make_grid(n_lat=n_lat, n_lon=n_lon, device=device).points.to(dtype)
    grid = F.normalize(grid, p=2, dim=1)                            # [G, 3]
    G = grid.shape[0]

    y = torch.empty(B, 3, device=device, dtype=dtype)
    for s in range(0, B, grid_chunk):
        e = min(s + grid_chunk, B)
        ps = tuple(p[s:e] for p in params)
        cands = grid.unsqueeze(0).expand(e - s, G, 3)               # [b, G, 3]
        idx = log_p_batched(ps, cands).argmax(dim=-1)               # [b]
        y[s:e] = grid[idx]

    if n_levels <= 0:
        return y

    P = patch_size
    u = torch.linspace(-1.0, 1.0, P, device=device, dtype=dtype)
    eta_x, eta_y = torch.meshgrid(u, u, indexing="ij")
    eta_uv = torch.stack([eta_x.flatten(), eta_y.flatten()], dim=-1)  # [P*P, 2]
    radius = math.pi / max(n_lat - 1, 1)
    arange_B = torch.arange(B, device=device)

    for _ in range(n_levels):
        e1, e2 = _construct_orthonormal_basis(y)                    # [B, 3]
        eta = radius * eta_uv                                       # [M, 2]
        # tangent[b, m] = eta[m, 0] * e1[b] + eta[m, 1] * e2[b]
        tangent = (eta[None, :, 0:1] * e1[:, None, :]
                   + eta[None, :, 1:2] * e2[:, None, :])            # [B, M, 3]
        theta = tangent.norm(dim=-1, keepdim=True)
        direction = tangent / theta.clamp_min(1e-30)
        cands = (torch.cos(theta) * y[:, None, :]
                 + torch.sin(theta) * direction)                    # [B, M, 3]
        cands = F.normalize(cands, p=2, dim=-1)
        best = log_p_batched(params, cands).argmax(dim=-1)          # [B]
        y = cands[arange_B, best]
        radius *= shrink

    return y


# ---------------------------------------------------------------------------
# Isotropic Angular Gaussian (IAG)
# ---------------------------------------------------------------------------

def iag_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """
    Isotropic Angular Gaussian (IAG) negative log-likelihood loss.

    The IAG distribution is the angular Gaussian with V = I (identity covariance),
    making it rotationally symmetric about the mean direction. It is a special
    case of ESAG with gamma = (0, 0).

    The density is:
        f_IAG(y) = (1/2pi) * exp[0.5 * ((y.mu)^2 - ||mu||^2)] * M_2(y.mu)

    where M_2(alpha) = (1 + alpha^2)Phi(alpha) + alpha*phi(alpha), with phi and Phi being the standard
    normal PDF and CDF respectively.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 3] predicted mean vectors mu. The magnitude ||mu|| controls
              concentration (higher = more peaked), and mu/||mu|| is the mean direction.
        y_true: [B, 3] true unit direction vectors on S^2.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    # The erf/exp kernel is unstable in bf16/fp16; compute it in fp32 with
    # autocast disabled, then cast the loss back (matches the vMF loss).
    orig_dtype = pred.dtype
    dtype = _compute_dtype(orig_dtype)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        mu = pred.to(dtype)                            # [B, 3]
        y = F.normalize(y_true.to(dtype), p=2, dim=1)  # [B, 3]

        mu_norm_sq = (mu ** 2).sum(dim=1)  # ||mu||^2 [B]
        y_dot_mu = (y * mu).sum(dim=1)     # y.mu [B]

        # NLL = log(2pi) - angular-Gaussian log kernel with Q = 1
        nll = _LOG_2PI - _ag_log_kernel(torch.ones_like(y_dot_mu), mu_norm_sq, y_dot_mu)
        loss = _apply_reduction(nll, reduction)
    return loss.to(orig_dtype)


class IAG(BaseDistribution):
    """Isotropic Angular Gaussian distribution on S^2.

    The IAG distribution is rotationally symmetric about the mean direction,
    with concentration controlled by ||mu||.

    The density is:
        f_IAG(y) = (1/2pi) exp[0.5((y.mu)^2 - ||mu||^2)] M_2(y.mu)

    Reference: Paine et al. (2018), Stat Comput 28:689-697.

    Args:
        pred: [B, 3] raw network output (the mean vector mu).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return _safe_unit_vector(self._pred)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B]. Higher = more peaked."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_IAG(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        orig_dtype = self._pred.dtype
        dtype = _compute_dtype(orig_dtype)
        with torch.autocast(device_type=self._pred.device.type, enabled=False):
            mu = self._pred.to(dtype)  # [B, 3]
            points = points.to(dtype)
            mu_norm_sq = (mu ** 2).sum(dim=1)  # [B]

            y_dot_mu = points @ mu.T  # [N, B]
            # log f = -log(2pi) + angular-Gaussian log kernel with Q = 1
            log_p = (
                -_LOG_2PI
                + _ag_log_kernel(torch.ones_like(y_dot_mu), mu_norm_sq[None, :], y_dot_mu)
            )
        return log_p.T.to(orig_dtype)  # [B, N]

    def mode(self) -> Tensor:
        """MAP direction on S^2.

        IAG is rotationally symmetric about mu, so the mode is exactly
        ``mu / ||mu||``.

        Returns:
            [B, 3] unit-norm MAP directions.
        """
        return self.mean_direction.detach()


# ---------------------------------------------------------------------------
# Elliptically Symmetric Angular Gaussian (ESAG)
# ---------------------------------------------------------------------------

def esag_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """
    Elliptically Symmetric Angular Gaussian (ESAG) negative log-likelihood loss.

    The ESAG distribution has ellipse-like contours on the sphere, enabling
    modeling of anisotropic directional uncertainty. It generalizes IAG by
    adding shape parameters gamma = (gamma_1, gamma_2) that control the ellipticity.

    The density is:
        f_ESAG(y) = C_3 / (y'V^{-1}y)^(3/2) * exp[0.5*((y.mu)^2/(y'V^{-1}y) - ||mu||^2)]
                    * M_2(y.mu / sqrt(y'V^{-1}y))

    where V^{-1} is constructed from mu and gamma per Equation (18) of the paper.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3] = mu (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = gamma = (gamma_1, gamma_2) (shape parameters for ellipticity)
              Setting gamma = (0, 0) recovers the IAG distribution.
        y_true: [B, 3] true unit direction vectors on S^2.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    # The erf/exp kernel is unstable in bf16/fp16; compute it in fp32 with
    # autocast disabled, then cast the loss back (matches the vMF loss).
    orig_dtype = pred.dtype
    dtype = _compute_dtype(orig_dtype)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred = pred.to(dtype)
        mu = pred[:, :3]      # [B, 3]
        gamma1 = pred[:, 3]   # [B]
        gamma2 = pred[:, 4]   # [B]

        y = F.normalize(y_true.to(dtype), p=2, dim=1)  # [B, 3]

        # Basic terms
        mu_norm_sq = (mu ** 2).sum(dim=1)  # ||mu||^2 [B]
        y_dot_mu = (y * mu).sum(dim=1)     # y.mu [B]

        # Construct orthonormal basis {xi_1, xi_2} perpendicular to mu
        xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

        # Projections of y onto the basis vectors
        a = (y * xi1).sum(dim=1)  # y.xi_1 [B]
        b = (y * xi2).sum(dim=1)  # y.xi_2 [B]

        # Compute y'V^{-1}y using Equation (18):
        # y'V^{-1}y = 1 + gamma_1(a^2 - b^2) + 2*gamma_2*a*b + (sqrt(1 + gamma_1^2 + gamma_2^2) - 1)(a^2 + b^2)
        gamma_sq = gamma1 ** 2 + gamma2 ** 2
        sqrt_term = torch.sqrt(1.0 + gamma_sq)
        a_sq_plus_b_sq = a ** 2 + b ** 2

        y_Vinv_y = (1.0
                    + gamma1 * (a ** 2 - b ** 2)
                    + 2.0 * gamma2 * a * b
                    + (sqrt_term - 1.0) * a_sq_plus_b_sq)

        # Clamp for numerical stability (V^{-1} is positive definite, so this should be > 0)
        y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)

        # NLL = log(2pi) - angular-Gaussian log kernel
        alpha = y_dot_mu / torch.sqrt(y_Vinv_y)
        nll = _LOG_2PI - _ag_log_kernel(y_Vinv_y, mu_norm_sq, alpha)
        loss = _apply_reduction(nll, reduction)
    return loss.to(orig_dtype)


class ESAG(BaseDistribution):
    """Elliptically Symmetric Angular Gaussian distribution on S^2.

    The ESAG distribution generalizes IAG with ellipse-like contours on the
    sphere, controlled by shape parameters gamma = (gamma_1, gamma_2). Setting gamma = (0, 0)
    recovers the IAG distribution.

    Reference: Paine et al. (2018), Stat Comput 28:689-697.

    Args:
        pred: [B, 5] raw network output where pred[:, :3] is mu and
            pred[:, 3:5] is gamma = (gamma_1, gamma_2).
    """

    n_params = 5

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return _safe_unit_vector(self._pred[:, :3])

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B]. Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def gamma(self) -> Tensor:
        """Ellipticity parameters (gamma_1, gamma_2) [B, 2]."""
        return self._pred[:, 3:5]

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_ESAG(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        orig_dtype = self._pred.dtype
        dtype = _compute_dtype(orig_dtype)
        with torch.autocast(device_type=self._pred.device.type, enabled=False):
            mu = self._pred[:, :3].to(dtype)   # [B, 3]
            gamma1 = self._pred[:, 3].to(dtype) # [B]
            gamma2 = self._pred[:, 4].to(dtype) # [B]
            points = points.to(dtype)

            mu_norm_sq = (mu ** 2).sum(dim=1)  # [B]

            # Construct orthonormal basis perpendicular to mu
            xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

            # y.mu, y.xi_1, y.xi_2 for all (point, sample) pairs
            y_dot_mu = points @ mu.T    # [N, B]
            a = points @ xi1.T          # [N, B]  (y.xi_1)
            b = points @ xi2.T          # [N, B]  (y.xi_2)

            # y'V^{-1}y (Equation 18)
            gamma_sq = gamma1 ** 2 + gamma2 ** 2        # [B]
            sqrt_term = torch.sqrt(1.0 + gamma_sq)      # [B]
            a_sq_plus_b_sq = a ** 2 + b ** 2             # [N, B]

            y_Vinv_y = (1.0
                        + gamma1[None, :] * (a ** 2 - b ** 2)
                        + 2.0 * gamma2[None, :] * a * b
                        + (sqrt_term - 1.0)[None, :] * a_sq_plus_b_sq)

            y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)  # [N, B]

            # log f = -log(2pi) + angular-Gaussian log kernel
            alpha = y_dot_mu / torch.sqrt(y_Vinv_y)
            log_p = -_LOG_2PI + _ag_log_kernel(y_Vinv_y, mu_norm_sq[None, :], alpha)
        return log_p.T.to(orig_dtype)  # [B, N]

    def mode(
        self,
        fast: bool = False,
        n_lat: int = 181,
        n_lon: int = 360,
        n_levels: int = 4,
        patch_size: int = 11,
        shrink: float = 0.25,
        grid_chunk: int = 256,
    ) -> Tensor:
        """MAP direction on S^2 via hierarchical grid refinement.

        A coarse lat/lon argmax locates the basin, then ``n_levels`` of
        tangent-plane refinement bring the estimate to machine precision.
        Each level is a pure ``log_pdf`` evaluation — no autograd, no Hessian.

        Args:
            fast: if ``True``, override ``n_lat``, ``n_lon``, and ``n_levels``
                with a preset tuned for non-adversarial basins (~3x faster,
                quantization floor ~0.35 arcsec). See :data:`_MODE_FAST_PRESET`.
            n_lat, n_lon: coarse global grid resolution (default 181 x 360 ≈ 1°).
            n_levels: refinement passes after the global scan.
            patch_size: P; the per-event refinement grid is P x P.
            shrink: per-level radius shrink factor.
            grid_chunk: row chunk for the level-0 eval to bound peak memory.

        Returns:
            [B, 3] unit-norm MAP directions (detached from the autograd graph).
        """
        if fast:
            n_lat, n_lon, n_levels = (_MODE_FAST_PRESET["n_lat"],
                                      _MODE_FAST_PRESET["n_lon"],
                                      _MODE_FAST_PRESET["n_levels"])
        pred = self._pred.detach()
        mu = pred[:, :3]
        gamma1 = pred[:, 3]
        gamma2 = pred[:, 4]
        xi1, xi2 = _construct_orthonormal_basis(mu)
        mu_norm_sq = (mu ** 2).sum(dim=1)

        def log_p(params, cands):
            mu_b, xi1_b, xi2_b, g1_b, g2_b, mns_b = params
            a = (cands * xi1_b[:, None, :]).sum(dim=-1)
            b = (cands * xi2_b[:, None, :]).sum(dim=-1)
            y_dot_mu = (cands * mu_b[:, None, :]).sum(dim=-1)
            sqrt_term = torch.sqrt(1.0 + g1_b * g1_b + g2_b * g2_b)
            r2 = a * a + b * b
            y_Vinv_y = (1.0
                        + g1_b[:, None] * (a * a - b * b)
                        + 2.0 * g2_b[:, None] * a * b
                        + (sqrt_term - 1.0)[:, None] * r2).clamp_min(1e-8)
            T = y_dot_mu / torch.sqrt(y_Vinv_y)
            return _ag_log_kernel(y_Vinv_y, mns_b[:, None], T)

        return _hierarchical_mode_on_sphere(
            log_p, (mu, xi1, xi2, gamma1, gamma2, mu_norm_sq),
            B=pred.shape[0], device=pred.device, dtype=pred.dtype,
            n_lat=n_lat, n_lon=n_lon, n_levels=n_levels,
            patch_size=patch_size, shrink=shrink, grid_chunk=grid_chunk,
        )


# ---------------------------------------------------------------------------
# General Angular Gaussian (GAG)
# ---------------------------------------------------------------------------

def gag_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """General Angular Gaussian (GAG) negative log-likelihood loss.

    The GAG is the full 8-parameter angular Gaussian on S^2, with density:

        f_AG(y; mu, V) = (2pi)^{-1} |V|^{-1/2} (y^T V^{-1} y)^{-3/2}
                        * exp[1/2((y^T V^{-1} mu)^2 / (y^T V^{-1} y) - mu^T V^{-1} mu)]
                        * M_2(y^T V^{-1} mu / sqrt(y^T V^{-1} y))

    V^{-1} is parameterised as LL^T via a log-Cholesky factor with det(L) = 1,
    so |V| = 1 and the log-determinant term vanishes.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (2).

    Args:
        pred: [B, 8] predictions where:
              - pred[:, :3]  = mu  (mean vector, unconstrained)
              - pred[:, 3:5] = log L_11, log L_22 (log-diagonal of L;
                               log L_33 fixed by det(L) = 1)
              - pred[:, 5:8] = off-diagonal entries (L_21, L_31, L_32)
        y_true: [B, 3] true unit direction vectors on S^2.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    # The erf/exp kernel and Cholesky exp are unstable in bf16/fp16; compute in
    # fp32 with autocast disabled, then cast the loss back (matches the vMF loss).
    orig_dtype = pred.dtype
    dtype = _compute_dtype(orig_dtype)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred = pred.to(dtype)
        mu = pred[:, :3]                           # [B, 3]
        y = F.normalize(y_true.to(dtype), p=2, dim=1)  # [B, 3]
        L = _build_cholesky(pred)                  # [B, 3, 3]

        # Transformed vectors: z_y = L^T y,  z_mu = L^T mu
        # (batch matrix-vector multiply via einsum)
        z_y = torch.einsum('bji,bj->bi', L, y)    # [B, 3]
        z_mu = torch.einsum('bji,bj->bi', L, mu)  # [B, 3]

        Q = (z_y ** 2).sum(dim=1)                  # ||z_y||^2 = y^T V^{-1} y  [B]
        S = (z_mu ** 2).sum(dim=1)                 # ||z_mu||^2 = mu^T V^{-1} mu  [B]
        z_dot = (z_y * z_mu).sum(dim=1)            # z_y.z_mu = y^T V^{-1} mu  [B]

        Q_safe = torch.clamp(Q, min=1e-8)
        T = z_dot / torch.sqrt(Q_safe)             # [B]

        # NLL = log(2pi) - angular-Gaussian log kernel
        nll = _LOG_2PI - _ag_log_kernel(Q_safe, S, T)
        loss = _apply_reduction(nll, reduction)
    return loss.to(orig_dtype)


class GAG(BaseDistribution):
    """General Angular Gaussian distribution on S^2.

    The GAG is the full 8-parameter angular Gaussian, generalising ESAG by
    allowing the covariance eigenvectors to be independent of mu.  This
    enables asymmetric, non-elliptical contours on the sphere.

    V^{-1} is parameterised via a log-Cholesky factor L with det(L) = 1.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (2).

    Args:
        pred: [B, 8] raw network output where pred[:, :3] is mu,
            pred[:, 3:5] is (log L_11, log L_22) (log L_33 fixed by
            det(L) = 1), and pred[:, 5:8] is the off-diagonal
            (L_21, L_31, L_32).
    """

    n_params = 8

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return _safe_unit_vector(self._pred[:, :3])

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def cholesky_factor(self) -> Tensor:
        """Normalised lower-triangular Cholesky factor L [B, 3, 3].

        V^{-1} = LL^T with det(L) = 1.
        """
        return _build_cholesky(self._pred)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_GAG(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        orig_dtype = self._pred.dtype
        dtype = _compute_dtype(orig_dtype)
        with torch.autocast(device_type=self._pred.device.type, enabled=False):
            pred = self._pred.to(dtype)
            mu = pred[:, :3]                             # [B, 3]
            L = _build_cholesky(pred)                    # [B, 3, 3]
            points = points.to(dtype)

            mu_Vinv_mu = (torch.einsum('bji,bj->bi', L, mu) ** 2).sum(dim=1)  # S [B]

            # Transform all grid points through each batch element's L^T:
            # z_y[b, n, i] = L^T[b, i, j] . points[n, j] = L[b, j, i] . points[n, j]
            z_y = torch.einsum('bji,nj->bni', L, points)   # [B, N, 3]

            # Also transform mu: z_mu[b, i] = L^T[b, i, j] . mu[b, j]
            z_mu = torch.einsum('bji,bj->bi', L, mu)       # [B, 3]

            Q = (z_y ** 2).sum(dim=2)                       # [B, N]
            z_dot = torch.einsum('bni,bi->bn', z_y, z_mu)   # [B, N]

            Q_safe = torch.clamp(Q, min=1e-8)
            T = z_dot / torch.sqrt(Q_safe)                  # [B, N]

            # log f = -log(2pi) + angular-Gaussian log kernel
            log_p = -_LOG_2PI + _ag_log_kernel(Q_safe, mu_Vinv_mu[:, None], T)
        return log_p.to(orig_dtype)  # [B, N]

    def mode(
        self,
        fast: bool = False,
        n_lat: int = 181,
        n_lon: int = 360,
        n_levels: int = 4,
        patch_size: int = 11,
        shrink: float = 0.25,
        grid_chunk: int = 256,
    ) -> Tensor:
        """MAP direction on S^2 via hierarchical grid refinement.

        A coarse lat/lon argmax locates the basin, then ``n_levels`` of
        tangent-plane refinement bring the estimate to machine precision.
        Each level is a pure ``log_pdf`` evaluation — no autograd, no Hessian.

        Args:
            fast: if ``True``, override ``n_lat``, ``n_lon``, and ``n_levels``
                with a preset tuned for non-adversarial basins (~3x faster,
                quantization floor ~0.35 arcsec). See :data:`_MODE_FAST_PRESET`.
            n_lat, n_lon: coarse global grid resolution (default 181 x 360 ≈ 1°).
            n_levels: refinement passes after the global scan.
            patch_size: P; the per-event refinement grid is P x P.
            shrink: per-level radius shrink factor.
            grid_chunk: row chunk for the level-0 eval to bound peak memory.

        Returns:
            [B, 3] unit-norm MAP directions (detached from the autograd graph).
        """
        if fast:
            n_lat, n_lon, n_levels = (_MODE_FAST_PRESET["n_lat"],
                                      _MODE_FAST_PRESET["n_lon"],
                                      _MODE_FAST_PRESET["n_levels"])
        pred = self._pred.detach()
        L = _build_cholesky(pred)                                   # [B, 3, 3]
        z_mu = torch.einsum('bji,bj->bi', L, pred[:, :3])           # [B, 3]
        S = (z_mu ** 2).sum(dim=1)                                  # [B]

        def log_p(params, cands):
            L_b, z_mu_b, S_b = params
            # z_y[b, m, i] = L_b.T[b, i, j] . cands[b, m, j] = L_b[b, j, i] . cands[b, m, j]
            z_y = torch.einsum('bji,bmj->bmi', L_b, cands)
            Q = (z_y * z_y).sum(dim=-1).clamp_min(1e-8)
            T = (z_y * z_mu_b[:, None, :]).sum(dim=-1) / torch.sqrt(Q)
            return _ag_log_kernel(Q, S_b[:, None], T)

        return _hierarchical_mode_on_sphere(
            log_p, (L, z_mu, S),
            B=pred.shape[0], device=pred.device, dtype=pred.dtype,
            n_lat=n_lat, n_lon=n_lon, n_levels=n_levels,
            patch_size=patch_size, shrink=shrink, grid_chunk=grid_chunk,
        )
