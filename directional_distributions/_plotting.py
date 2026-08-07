"""Mollweide projection plotting for directional distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor

    from ._base import SphereGrid


# ---------------------------------------------------------------------------
# Publication-quality rcParams
# ---------------------------------------------------------------------------

#: Color palette for consistent styling across plots.
COLOR_CYCLE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#e67e22",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # magenta
    "#8c564b",  # brown
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
]

_RCPARAMS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.8,
    "figure.figsize": (7, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def set_style() -> None:
    """Apply publication-quality matplotlib rcParams globally.

    Sets LaTeX rendering with Computer Modern serif fonts, inward ticks on
    all four sides with visible minor ticks, no grid lines, and a white
    background.  Call once at the start of a script or notebook.
    """
    import matplotlib as mpl

    mpl.rcParams.update(_RCPARAMS)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_CYCLE)


# ---------------------------------------------------------------------------
# Mollweide projection
# ---------------------------------------------------------------------------


def plot_mollweide(
    grid: SphereGrid,
    values: Tensor | np.ndarray,
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    title: str | None = None,
    **pcolormesh_kwargs,
) -> tuple[Figure, Axes]:
    """Plot a scalar field on S² using a Mollweide projection.

    Applies publication-quality styling (LaTeX labels, white background, no
    grid lines) consistent with ``set_style``.

    Args:
        grid: SphereGrid from :func:`make_grid`.
        values: [n_lat, n_lon] array of scalar values (e.g. PDF values).
        ax: Optional Matplotlib axes with ``projection='mollweide'``.
            Created automatically if *None*.
        cmap: Colormap name (default ``"viridis"``).
        colorbar: Whether to add a colorbar.
        title: Optional title rendered in LaTeX.
        **pcolormesh_kwargs: Forwarded to ``ax.pcolormesh``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    n_lat, n_lon = grid.shape

    # Apply style within a local context so we don't permanently mutate
    # the user's rcParams when they haven't called set_style() themselves.
    with mpl.rc_context(_RCPARAMS):
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_CYCLE)

        if ax is None:
            fig, ax = plt.subplots(
                subplot_kw={"projection": "mollweide"},
                figsize=(7, 5),
            )
        else:
            fig = ax.figure

        # Convert to numpy
        lon = grid.lon.detach().cpu().numpy().reshape(n_lat, n_lon)
        lat = grid.lat.detach().cpu().numpy().reshape(n_lat, n_lon)
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        vals = np.asarray(values).reshape(n_lat, n_lon)

        mesh = ax.pcolormesh(
            lon, lat, vals, cmap=cmap, shading="auto", **pcolormesh_kwargs
        )

        if colorbar:
            fig.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)

        if title is not None:
            ax.set_title(title, pad=14)

        # Light grid lines for lat/lon reference; hide longitude tick labels
        # since they render inside the Mollweide oval and cause clutter.
        ax.grid(True, linewidth=0.4, alpha=0.4, color="gray")
        ax.set_xticklabels([])

    return fig, ax
