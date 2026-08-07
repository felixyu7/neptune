"""Directional distribution loss functions and evaluation utilities for PyTorch."""

from .vmf import von_mises_fisher_loss, VMF
from .ag import iag_nll_loss, IAG, esag_nll_loss, ESAG, gag_nll_loss, GAG
from .ps import ps_nll_loss, PowerSpherical
from .pc import sipc_nll_loss, SIPC, sespc_nll_loss, SESPC, gspc_nll_loss, GSPC
from .pt import ipt_nll_loss, IPT, ept_nll_loss, EPT, gpt_nll_loss, GPT
from ._base import SphereGrid, make_grid
from ._plotting import plot_mollweide, set_style, COLOR_CYCLE

__all__ = [
    # Loss functions (Angular Gaussian family)
    "von_mises_fisher_loss",
    "iag_nll_loss",
    "esag_nll_loss",
    "gag_nll_loss",
    # Loss functions (Power Spherical family)
    "ps_nll_loss",
    # Loss functions (Spherical Projected Cauchy family)
    "sipc_nll_loss",
    "sespc_nll_loss",
    "gspc_nll_loss",
    # Loss functions (Projected t family)
    "ipt_nll_loss",
    "ept_nll_loss",
    "gpt_nll_loss",
    # Distribution classes (Angular Gaussian family)
    "VMF",
    "IAG",
    "ESAG",
    "GAG",
    # Distribution classes (Power Spherical family)
    "PowerSpherical",
    # Distribution classes (Spherical Projected Cauchy family)
    "SIPC",
    "SESPC",
    "GSPC",
    # Distribution classes (Projected t family)
    "IPT",
    "EPT",
    "GPT",
    # Grid utilities
    "SphereGrid",
    "make_grid",
    # Plotting
    "plot_mollweide",
    "set_style",
    "COLOR_CYCLE",
]
