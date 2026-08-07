"""Power Spherical distribution: loss function and evaluation.

Reference: De Cao & Aziz (2020), "The Power Spherical distribution",
ICML INNF+ 2020 Workshop, arXiv:2006.04437.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _apply_reduction


def _log_normalizer(kappa: Tensor) -> Tensor:
    """Log-normalizer of the Power Spherical distribution on S² (d=3).

    N(κ, 3) = 2^(2+κ) · π / (1+κ)

    log N = (2+κ)·log(2) + log(π) - log(1+κ)
    """
    return (2.0 + kappa) * np.log(2) + np.log(np.pi) - torch.log1p(kappa)


def ps_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """Power Spherical negative log-likelihood loss on S² (d=3).

    The Power Spherical distribution has density:

        f(y | μ, κ) = N(κ)⁻¹ (1 + μ̂ᵀy)^κ

    where μ̂ = μ/‖μ‖ is the mean direction, κ = ‖μ‖ is the concentration,
    and N(κ, 3) = 2^(2+κ) π / (1+κ) is the normalizing constant.

    Unlike von Mises-Fisher, the normalizer involves no Bessel functions
    and is numerically stable for all κ.

    Args:
        pred: [B, 3] predicted mean vectors μ.  Direction = μ/‖μ‖, κ = ‖μ‖.
        y_true: [B, 3] true unit direction vectors on S².
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    y = F.normalize(y_true, p=2, dim=1)
    mu_hat = F.normalize(pred, p=2, dim=1)
    kappa = pred.norm(p=2, dim=1)

    dot = (mu_hat * y).sum(dim=1)  # μ̂·y ∈ [-1, 1]

    # NLL = log N(κ) - κ · log(1 + μ̂·y)
    nll = _log_normalizer(kappa) - kappa * torch.log1p(dot.clamp(min=-1 + 1e-7))

    return _apply_reduction(nll, reduction)


class PowerSpherical(BaseDistribution):
    """Power Spherical distribution on S².

    The Power Spherical distribution has density:

        f(y | μ, κ) = N(κ)⁻¹ (1 + μ̂ᵀy)^κ

    where N(κ, 3) = 2^(2+κ) π / (1+κ).

    Direction and concentration are coupled: the network outputs μ ∈ ℝ³,
    with direction = μ/‖μ‖ and κ = ‖μ‖.

    Compared to vMF, the Power Spherical has polynomial (rather than
    exponential) concentration and a normalizer free of Bessel functions,
    making it numerically stable at all concentrations.

    Reference: De Cao & Aziz (2020), arXiv:2006.04437.

    Args:
        pred: [B, 3] raw network output (the mean vector μ).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred, p=2, dim=1)

    @property
    def kappa(self) -> Tensor:
        """Concentration parameter κ = ‖μ‖ [B]."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f(y | μ, κ) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu_hat = self.mean_direction  # [B, 3]
        kappa = self.kappa            # [B]

        dot = points @ mu_hat.T  # [N, B]

        # log f = -log N(κ) + κ · log(1 + μ̂ᵀy)
        log_norm = _log_normalizer(kappa)  # [B]
        log_p = -log_norm[None, :] + kappa[None, :] * torch.log1p(
            dot.clamp(min=-1 + 1e-7)
        )

        return log_p.T  # [B, N]
