"""Directional distribution loss functions and evaluation utilities for PyTorch."""

from .vmf import von_mises_fisher_loss, VMF
from .ag import iag_nll_loss, IAG, esag_nll_loss, ESAG, gag_nll_loss, GAG
from ._base import SphereGrid, make_grid
from ._plotting import plot_mollweide, set_style, COLOR_CYCLE

__all__ = [
    # Loss functions (Angular Gaussian family)
    "von_mises_fisher_loss",
    "iag_nll_loss",
    "esag_nll_loss",
    "gag_nll_loss",
    # Distribution classes (Angular Gaussian family)
    "VMF",
    "IAG",
    "ESAG",
    "GAG",
    # Grid utilities
    "SphereGrid",
    "make_grid",
    # Plotting
    "plot_mollweide",
    "set_style",
    "COLOR_CYCLE",
]
