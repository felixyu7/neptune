"""Projected t family: IPT, EPT, and GPT distributions on S².

This module collects the three members of the projected multivariate t
family, obtained by projecting a trivariate Student-t  t_ν(μ, Σ)  onto
the sphere via  z → z/‖z‖.  They form a nested hierarchy:

    IPT (3 params)  <  EPT (5 params)  <  GPT (8 params)

The degrees-of-freedom parameter ν is a fixed positive integer passed
at construction time (not learned by the network).

Special cases:
    ν = 1   →  projected Cauchy  (SIPC / SESPC / GSPC)
    ν → ∞   →  angular Gaussian  (IAG / ESAG / GAG)

The density on S² is

    f(y; μ, Σ, ν) = C(ν) · |Σ|^{-1/2} · (y^T Σ^{-1} y)^{-3/2}
                     · I(α_eff, β_eff, ν)

where C(ν) = Γ((ν+3)/2) / (Γ(ν/2) · (νπ)^{3/2}),  α_eff and β_eff
are functions of y and Σ^{-1}μ, and  I  is a radial integral evaluated
via a closed-form recurrence relation for integer ν.

Note: for ν > ~200, intermediate powers p^n may lose precision in
float64.  Use the AG family (IAG / ESAG / GAG) for Gaussian-like
behaviour at large ν.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import (
    BaseDistribution,
    _apply_reduction,
    _construct_orthonormal_basis,
    _build_cholesky,
)


# ---------------------------------------------------------------------------
# PT-family math utilities
# ---------------------------------------------------------------------------

def _pt_log_density(A: Tensor, B: Tensor, Gamma_sq: Tensor, nu: int) -> Tensor:
    """Compute the log-density kernel for the projected t distribution.

    Evaluates  log C(ν) − 1.5·log(B) + log I(α, β, ν)  where I is the
    radial integral, using a fast recurrence for integer ν.

    The radial integral is

        I = ∫₀^∞ r² · (ν + r² − 2rα + β)^{−(ν+3)/2} dr

    After substitution u = r − α it decomposes as

        I = J₀^{m−1} + (α² − c)·J₀^m + 2α·J₁^m

    where m = (ν+3)/2, c = β − α² + ν, p = β + ν, and

        J₀^n = ∫_{-α}^∞ (c + u²)^{-n} du
        J₁^m = p^{1−m} / (2(m−1))

    J₀ is computed via recurrence:

        J₀^{n+1} = [α·p^{-n} + (2n−1)·J₀^n] / (2cn)

    with base cases:
        odd  ν (integer m):       J₀^1   = [π/2 + atan(α/√c)] / √c
        even ν (half-integer m):  J₀^{3/2} = (√p + α) / (c√p)

    Args:
        A: [...] y^T Σ^{-1} μ  (or y·μ for isotropic case).
        B: [...] y^T Σ^{-1} y  (1 for isotropic case).
        Gamma_sq: [...] μ^T Σ^{-1} μ  (‖μ‖² for isotropic case).
        nu: positive integer degrees of freedom.

    Returns:
        [...] log-probability density values (same shape as inputs).
    """
    if not (isinstance(nu, int) and nu >= 1):
        raise ValueError(f"nu must be a positive integer, got {nu}")

    m = (nu + 3) / 2.0

    # Upcast to float64 for numerical stability.
    A64 = A.double()
    B64 = B.double().clamp(min=1e-30)
    G64 = Gamma_sq.double()

    sqrt_B = torch.sqrt(B64)
    alpha = A64 / sqrt_B           # effective α
    beta = G64                     # effective β

    # c = β − α² + ν  (≥ ν by Cauchy-Schwarz)
    c = beta - alpha ** 2 + nu
    c = c.clamp(min=nu * 0.5)     # safety clamp

    # p = β + ν  (= α² + c)
    p = beta + nu
    p = p.clamp(min=nu * 0.5)

    log_p = torch.log(p)

    # ---- J₁^m (closed form) ----
    J1_m = torch.exp((1.0 - m) * log_p) / (2.0 * (m - 1.0))

    # ---- J₀ via recurrence ----
    if nu % 2 == 1:
        # Odd ν → integer m → base case J₀^1 = [π/2 + atan(α/√c)] / √c
        sqrt_c = torch.sqrt(c)
        J0 = (np.pi / 2.0 + torch.atan(alpha / sqrt_c)) / sqrt_c
        n_cur = 1.0
    else:
        # Even ν → half-integer m → base case J₀^{3/2} = (√p + α) / (c√p)
        sqrt_p = torch.sqrt(p)
        J0 = (sqrt_p + alpha) / (c * sqrt_p)
        n_cur = 1.5

    # Recur from n_cur up to m.  We need J₀^{m-1} and J₀^m.
    steps = int(round(m - n_cur))   # (ν+1)/2 for odd, ν/2 for even

    J0_m1 = J0   # will be overwritten unless steps <= 1
    for i in range(steps):
        n = n_cur + i
        p_neg_n = torch.exp(-n * log_p)   # p^{-n}, safe via log
        J0_new = (alpha * p_neg_n + (2.0 * n - 1.0) * J0) / (2.0 * c * n)
        if i == steps - 2:
            J0_m1 = J0_new
        J0 = J0_new

    # Handle edge: steps == 1 → J₀^{m-1} is the base case itself
    if steps <= 1:
        J0_m1 = J0 if steps == 0 else J0_m1
        # When steps==1: base is J₀^{m-1}, single step gives J₀^m
        if steps == 1:
            # J0_m1 was not set by the loop (i == steps-2 == -1 never fires)
            # Recompute: base case IS J₀^{m-1}
            sqrt_c = torch.sqrt(c) if nu % 2 == 1 else None
            if nu % 2 == 1:
                J0_m1 = (np.pi / 2.0 + torch.atan(alpha / sqrt_c)) / sqrt_c
            else:
                sqrt_p = torch.sqrt(p)
                J0_m1 = (sqrt_p + alpha) / (c * sqrt_p)

    J0_m = J0

    # ---- Radial integral ----
    I_val = J0_m1 + (alpha ** 2 - c) * J0_m + 2.0 * alpha * J1_m
    log_I = torch.log(I_val.clamp(min=1e-300))

    # ---- Log normalizing constant (scalar, independent of data) ----
    log_C = (math.lgamma(m)
             - math.lgamma(nu / 2.0)
             + (nu / 2.0) * math.log(nu)
             - 1.5 * math.log(math.pi))

    log_density = log_C - 1.5 * torch.log(B64) + log_I
    return log_density.to(A.dtype)


# ---------------------------------------------------------------------------
# Isotropic Projected t (IPT)
# ---------------------------------------------------------------------------

def ipt_nll_loss(
    pred: Tensor, y_true: Tensor, nu: int = 5, reduction: str = "mean"
) -> Tensor:
    """Isotropic Projected t (IPT) negative log-likelihood loss.

    The IPT is the projected t with Σ = I, making it rotationally
    symmetric about the mean direction.  It interpolates between
    SIPC (ν=1) and IAG (ν→∞).

    Args:
        pred: [B, 3] predictions where pred[:, :3] = μ (mean vector,
              magnitude controls concentration).
        y_true: [B, 3] true unit direction vectors on S².
        nu: positive integer degrees of freedom (default 5).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    mu = pred[:, :3]                                  # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)               # [B, 3]

    A = (y * mu).sum(dim=1)                            # [B]
    B = torch.ones_like(A)                             # [B]
    Gamma_sq = (mu ** 2).sum(dim=1)                    # [B]

    return _apply_reduction(-_pt_log_density(A, B, Gamma_sq, nu), reduction)


class IPT(BaseDistribution):
    """Isotropic Projected t distribution on S².

    The IPT is rotationally symmetric about the mean direction, with
    concentration controlled by ‖μ‖ and tail weight controlled by ν.
    Setting ν=1 recovers SIPC; as ν→∞ it approaches IAG.

    Args:
        pred: [B, 3] raw network output where pred[:, :3] is μ.
        nu: positive integer degrees of freedom (default 3).
    """

    n_params = 3

    def __init__(self, pred: Tensor, nu: int = 5) -> None:
        super().__init__(pred)
        if not (isinstance(nu, int) and nu >= 1):
            raise ValueError(f"nu must be a positive integer, got {nu}")
        self._nu = nu

    @property
    def nu(self) -> int:
        """Degrees of freedom ν (fixed integer)."""
        return self._nu

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ‖μ‖ [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_IPT(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                         # [B, 3]
        Gamma_sq = (mu ** 2).sum(dim=1)                # [B]

        A = points @ mu.T                              # [N, B]
        B = torch.ones_like(A)                         # [N, B]
        Gamma_sq_exp = Gamma_sq[None, :].expand_as(A)  # [N, B]

        return _pt_log_density(A, B, Gamma_sq_exp, self._nu).T  # [B, N]


# ---------------------------------------------------------------------------
# Elliptically Symmetric Projected t (EPT)
# ---------------------------------------------------------------------------

def ept_nll_loss(
    pred: Tensor, y_true: Tensor, nu: int = 5, reduction: str = "mean"
) -> Tensor:
    """Elliptically Symmetric Projected t (EPT) negative log-likelihood loss.

    The EPT generalises IPT with ellipse-like contours on the sphere,
    controlled by shape parameters γ = (γ₁, γ₂).  Setting γ = (0, 0)
    recovers the IPT distribution.

    The Σ⁻¹ construction follows Paine et al. (2018), identical to ESAG/SESPC.

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3]  = μ (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = γ = (γ₁, γ₂) (shape parameters for ellipticity)
        y_true: [B, 3] true unit direction vectors on S².
        nu: positive integer degrees of freedom (default 5).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    mu = pred[:, :3]                                    # [B, 3]
    gamma1 = pred[:, 3]                                 # [B]
    gamma2 = pred[:, 4]                                 # [B]

    y = F.normalize(y_true, p=2, dim=1)                 # [B, 3]

    # Since Σμ = μ: A = y·μ, Γ² = ‖μ‖²
    A = (y * mu).sum(dim=1)                              # [B]
    Gamma_sq = (mu ** 2).sum(dim=1)                      # [B]

    # Construct orthonormal basis perpendicular to μ
    xi1, xi2 = _construct_orthonormal_basis(mu)          # [B, 3] each

    # Projections
    a = (y * xi1).sum(dim=1)                             # [B]
    b = (y * xi2).sum(dim=1)                             # [B]

    # B = y^T Σ⁻¹ y  (Paine et al. 2018, Eq. 18)
    gamma_sq = gamma1 ** 2 + gamma2 ** 2
    sqrt_term = torch.sqrt(1.0 + gamma_sq)
    a_sq_plus_b_sq = a ** 2 + b ** 2

    B = (1.0
         + gamma1 * (a ** 2 - b ** 2)
         + 2.0 * gamma2 * a * b
         + (sqrt_term - 1.0) * a_sq_plus_b_sq)

    B = torch.clamp(B, min=1e-8)                        # [B]

    return _apply_reduction(-_pt_log_density(A, B, Gamma_sq, nu), reduction)


class EPT(BaseDistribution):
    """Elliptically Symmetric Projected t distribution on S².

    The EPT generalises IPT with ellipse-like contours on the sphere,
    controlled by shape parameters γ = (γ₁, γ₂).  Setting γ = (0, 0)
    recovers the IPT distribution.

    Args:
        pred: [B, 5] raw network output where pred[:, :3] is μ
            and pred[:, 3:5] is γ = (γ₁, γ₂).
        nu: positive integer degrees of freedom (default 3).
    """

    n_params = 5

    def __init__(self, pred: Tensor, nu: int = 5) -> None:
        super().__init__(pred)
        if not (isinstance(nu, int) and nu >= 1):
            raise ValueError(f"nu must be a positive integer, got {nu}")
        self._nu = nu

    @property
    def nu(self) -> int:
        """Degrees of freedom ν (fixed integer)."""
        return self._nu

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ‖μ‖ [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def gamma(self) -> Tensor:
        """Ellipticity parameters (γ₁, γ₂) [B, 2]."""
        return self._pred[:, 3:5]

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_EPT(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                           # [B, 3]
        gamma1 = self._pred[:, 3]                        # [B]
        gamma2 = self._pred[:, 4]                        # [B]
        Gamma_sq = (mu ** 2).sum(dim=1)                  # [B]

        xi1, xi2 = _construct_orthonormal_basis(mu)      # [B, 3] each

        # [N, B] intermediates
        A = points @ mu.T                                # [N, B]
        a = points @ xi1.T                               # [N, B]
        b = points @ xi2.T                               # [N, B]

        gamma_sq = gamma1 ** 2 + gamma2 ** 2             # [B]
        sqrt_term = torch.sqrt(1.0 + gamma_sq)           # [B]
        a_sq_plus_b_sq = a ** 2 + b ** 2                 # [N, B]

        B = (1.0
             + gamma1[None, :] * (a ** 2 - b ** 2)
             + 2.0 * gamma2[None, :] * a * b
             + (sqrt_term - 1.0)[None, :] * a_sq_plus_b_sq)

        B = torch.clamp(B, min=1e-8)                    # [N, B]

        Gamma_sq_exp = Gamma_sq[None, :].expand_as(A)    # [N, B]

        return _pt_log_density(A, B, Gamma_sq_exp, self._nu).T  # [B, N]


# ---------------------------------------------------------------------------
# General Projected t (GPT)
# ---------------------------------------------------------------------------

def gpt_nll_loss(
    pred: Tensor, y_true: Tensor, nu: int = 5, reduction: str = "mean"
) -> Tensor:
    """General Projected t (GPT) negative log-likelihood loss.

    The GPT is the full 8-parameter projected t on S², with Σ⁻¹
    parameterised as LLᵀ via a log-Cholesky factor with det(L) = 1,
    identical to the GAG/GSPC parameterisation.

    Args:
        pred: [B, 8] predictions where:
              - pred[:, :3]  = μ  (mean vector, unconstrained)
              - pred[:, 3:5] = log L₁₁, log L₂₂ (log-diagonal of L;
                               log L₃₃ fixed by det(L) = 1)
              - pred[:, 5:8] = off-diagonal entries (L₂₁, L₃₁, L₃₂)
        y_true: [B, 3] true unit direction vectors on S².
        nu: positive integer degrees of freedom (default 5).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    mu = pred[:, :3]                                     # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)                  # [B, 3]
    L = _build_cholesky(pred)                            # [B, 3, 3]

    # Transformed vectors: z_y = Lᵀy,  z_μ = Lᵀμ
    z_y = torch.einsum('bji,bj->bi', L, y)              # [B, 3]
    z_mu = torch.einsum('bji,bj->bi', L, mu)            # [B, 3]

    B = (z_y ** 2).sum(dim=1)                            # [B]
    Gamma_sq = (z_mu ** 2).sum(dim=1)                    # [B]
    A = (z_y * z_mu).sum(dim=1)                          # [B]

    return _apply_reduction(-_pt_log_density(A, B, Gamma_sq, nu), reduction)


class GPT(BaseDistribution):
    """General Projected t distribution on S².

    The GPT is the full 8-parameter projected t, generalising EPT
    by allowing the scatter matrix eigenvectors to be independent of μ.
    This enables asymmetric, non-elliptical contours on the sphere.

    Σ⁻¹ is parameterised via a log-Cholesky factor L with det(L) = 1.

    Args:
        pred: [B, 8] raw network output where pred[:, :3] is μ,
            pred[:, 3:5] is (log L₁₁, log L₂₂) (log L₃₃ fixed by
            det(L) = 1), and pred[:, 5:8] is the off-diagonal
            (L₂₁, L₃₁, L₃₂).
        nu: positive integer degrees of freedom (default 3).
    """

    n_params = 8

    def __init__(self, pred: Tensor, nu: int = 5) -> None:
        super().__init__(pred)
        if not (isinstance(nu, int) and nu >= 1):
            raise ValueError(f"nu must be a positive integer, got {nu}")
        self._nu = nu

    @property
    def nu(self) -> int:
        """Degrees of freedom ν (fixed integer)."""
        return self._nu

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ‖μ‖ [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def cholesky_factor(self) -> Tensor:
        """Normalised lower-triangular Cholesky factor L [B, 3, 3].

        Σ⁻¹ = LLᵀ with det(L) = 1.
        """
        return _build_cholesky(self._pred)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_GPT(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                            # [B, 3]
        L = _build_cholesky(self._pred)                   # [B, 3, 3]

        # Transform all grid points and μ through Lᵀ
        z_y = torch.einsum('bji,nj->bni', L, points)     # [B, N, 3]
        z_mu = torch.einsum('bji,bj->bi', L, mu)         # [B, 3]

        B = (z_y ** 2).sum(dim=2)                         # [B, N]
        Gamma_sq = (z_mu ** 2).sum(dim=1)                 # [B]
        A = torch.einsum('bni,bi->bn', z_y, z_mu)        # [B, N]

        Gamma_sq_exp = Gamma_sq[:, None].expand_as(A)     # [B, N]

        return _pt_log_density(A, B, Gamma_sq_exp, self._nu)  # [B, N]
