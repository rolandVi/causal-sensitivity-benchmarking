"""Shared plotting style and save helpers for the experiment figures.

Centralizes a publication-quality matplotlib style so every figure looks
consistent and matches the LaTeX paper, and routes figures to the right
output directory. Debug figures go to results/plots only, curated paper
figures additionally go to paper/figures.
"""

import os
from weakref import WeakKeyDictionary

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# Per-axes record of drawn std bands, used to blend overlapping regions.
# Weak keys let closed-figure axes be garbage-collected without leaking.
BAND_REGISTRY = WeakKeyDictionary()

# Output locations (relative to repo root, where the scripts are run from)
PLOT_DIR = os.path.join("results", "plots")
PAPER_DIR = os.path.join("paper", "figures")

# Figure widths in inches for the IJCAI two-column layout. Sized a bit wider
# than the printed column so on-figure text renders smaller (more zoomed out)
# after scaling to \columnwidth in the paper.
COL_WIDTH = 4.0  # single column
TEXT_WIDTH = 6.95  # full text width, spans both columns


def figsize(width=COL_WIDTH, ratio=0.72):
    """Figure size in inches at a given width and height-to-width ratio."""
    return (width, width * ratio)


def apply_paper_style():
    """Apply a serif, publication-quality matplotlib style.

    Uses a Times-like serif with Computer-Modern mathtext to match the LaTeX
    body text without requiring a LaTeX install. Sizes are tuned for figures
    rendered at column width, so text stays legible after scaling in the paper.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "errorbar.capsize": 2,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def tint_color(color, strength=0.35):
    """Opaque pastel tint of color, as a translucent fill of it would look over white.

    Args:
        color: any matplotlib color
        strength: how far the tint sits from white toward color (0 white, 1 color)

    Returns:
        An (r, g, b) array in [0, 1]
    """
    import numpy as np

    rgb = np.asarray(to_rgb(color))
    return strength * rgb + (1.0 - strength) * np.ones(3)


def fill_band(x, lo, hi, color, strength=0.35, zorder=1, ax=None):
    """Fill an opaque tinted band between lo and hi, blending overlaps.

    The band uses an opaque tint instead of a translucent fill, so bands never
    darken where they stack. Where this band overlaps an earlier band on the
    same axes, the shared region is filled with a 50/50 mix of the two tints,
    an equally light in-between colour, so both bands stay readable.

    Args:
        x: x-axis values
        lo: lower edge of the band (array or scalar)
        hi: upper edge of the band (array or scalar)
        color: band color
        strength: tint intensity, how far the tint sits from white toward color
        zorder: draw order for the band (the overlap mix sits just above it)
        ax: axes to draw on (defaults to current axes)
    """
    import numpy as np

    ax = ax or plt.gca()
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.ndim == 0: lo = np.full_like(x, lo)
    if hi.ndim == 0: hi = np.full_like(x, hi)
    tint = tint_color(color, strength)

    # Mix this band with any earlier band on this axes where they overlap,
    # drawn just above both so the shared region reads as its own colour
    prior = BAND_REGISTRY.get(ax, [])
    for px, plo, phi, ptint in prior:
        if len(px) == len(x) and np.allclose(px, x):
            olo, ohi = np.maximum(plo, lo), np.minimum(phi, hi)
            overlap = ohi > olo
            if overlap.any():
                mix = 0.5 * (tint + ptint)
                ax.fill_between(x, olo, ohi, where=overlap.tolist(), color=mix,
                                linewidth=0, zorder=zorder + 0.5, interpolate=True)

    ax.fill_between(x, lo, hi, color=tint, linewidth=0, zorder=zorder)
    prior.append((x, lo, hi, tint))
    BAND_REGISTRY[ax] = prior


def plot_band(x, mean, std, color, label=None, marker=None, linestyle="-", strength=0.35, ax=None):
    """Plot a mean line with a solid +/-1 std band instead of error bars.

    The band is an opaque pastel tint of color, and overlaps with earlier bands
    on the same axes are blended to a light mix rather than darkened (see
    fill_band). For sweeps the default is a continuous line with no markers.

    Args:
        x: x-axis values
        mean: mean curve
        std: standard deviation at each x (band is mean +/- std)
        color: line and band color
        label: legend label for the line
        marker: line marker (None for a continuous line)
        linestyle: line style
        strength: band intensity, how far the tint sits from white toward color
        ax: axes to draw on (defaults to current axes)

    Returns:
        The Line2D for the mean curve
    """
    import numpy as np

    ax = ax or plt.gca()
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    # Band below the line so the mean stays visible across every series
    fill_band(x, mean - std, mean + std, color, strength=strength, zorder=1, ax=ax)
    line, = ax.plot(x, mean, marker=marker, linestyle=linestyle, color=color, label=label, zorder=2)
    return line


def save_figure(fig_name, paper=False):
    """Save the current figure to results/plots, and to paper/figures if paper.

    Args:
        fig_name: File name including extension (e.g. "exp01_quadratic_loo.png")
        paper: If True, also save a copy to paper/figures as a curated figure
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    if paper:
        os.makedirs(PAPER_DIR, exist_ok=True)
        plt.savefig(os.path.join(PAPER_DIR, fig_name))
    plt.close()
