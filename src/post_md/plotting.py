"""Matplotlib plot helpers for the CLI.

All plots are rendered on the headless ``Agg`` backend so the CLI works
on servers without a display. Each function accepts a :class:`PlotStyle`
that lets the user override colors, axis labels, title, legend, figure
size, DPI, font size, and line width — enough for publication-quality
figures without dropping into the Python API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MaxNLocator  # noqa: E402


@dataclass
class PlotStyle:
    """Style overrides applied by the plotting helpers.

    A field set to ``None`` means *use the function's default*.
    """

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    color: str | None = None
    linewidth: float = 1.6
    legend_label: str | None = None
    show_legend: bool = False
    grid: bool = True
    figsize: tuple[float, float] = (7.5, 4.5)
    dpi: int = 150
    font_size: int = 12
    # Axis range overrides — any of the four can be left as None to let
    # matplotlib auto-scale that bound. Half-open ranges work too
    # (e.g. ymin=0, ymax=None pins the floor but lets the top float).
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    # Multi-panel plot accents (used by plot_pca / plot_clusters):
    accent_color: str | None = None
    cmap: str | None = None


def _apply_rc(style: PlotStyle) -> None:
    plt.rcParams.update(
        {
            "font.size": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size - 1,
            "ytick.labelsize": style.font_size - 1,
            "legend.fontsize": style.font_size - 1,
        }
    )


def _save(fig, path: str | Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    output_path: str | Path,
    style: PlotStyle | None = None,
    default_title: str = "",
    default_xlabel: str = "Frame",
    default_ylabel: str = "Value",
) -> None:
    style = style or PlotStyle()
    _apply_rc(style)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(
        x, y,
        linewidth=style.linewidth,
        color=style.color,
        label=style.legend_label,
    )
    # Snug fit to data: removes matplotlib's default ~5% breathing room so
    # the curve starts at the y-axis with no visible gap. User-supplied
    # x/y min-max overrides applied below take precedence.
    ax.margins(x=0, y=0)
    ax.set_xlabel(style.xlabel or default_xlabel)
    ax.set_ylabel(style.ylabel or default_ylabel)
    ax.set_title(style.title or default_title)
    _apply_axis_limits(ax, style)
    # Y-axis: extend down to 0 for non-negative quantities (RMSD/RMSF/Rg)
    # so the "0" tick is always visible as a reference.
    # X-axis: do NOT force to 0 — trajectories that start at e.g. 100 ns
    # (continuation runs) should display their actual time range, not a
    # blank 0–100 ns stretch.
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if style.ymin is None and y_arr.size and float(y_arr.min()) >= 0:
        ax.set_ylim(bottom=0)
    _apply_publication_ticks(ax)
    if x_arr.size:
        _force_boundary_ticks_x(ax, float(x_arr.min()), float(x_arr.max()))
    if style.grid:
        ax.grid(alpha=0.3)
    if style.show_legend or style.legend_label:
        ax.legend(loc="best", frameon=True)
    _save(fig, output_path, style.dpi)


def plot_lines_multi(
    curves: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    output_path: str | Path,
    style: PlotStyle | None = None,
    default_title: str = "",
    default_xlabel: str = "Frame",
    default_ylabel: str = "Value",
    cmap: str | None = None,
    colors: list[str | None] | None = None,
) -> None:
    """Plot several line curves on a single axis with distinct colors + legend.

    Used by multi-system comparison runs (WT vs mutants etc.). Each curve
    is one ``(x, y)`` pair; ``labels`` is its display name in the legend.

    Color resolution per curve, in order: explicit ``colors[i]`` (any
    matplotlib color spec, ``None`` to fall back) → the supplied / styled
    colormap → tab10 default. The fallback keeps multi-system runs
    readable when only some systems have a colour assigned.
    """
    style = style or PlotStyle()
    _apply_rc(style)

    cmap_name = cmap or style.cmap or "tab10"
    palette = plt.get_cmap(cmap_name)

    # If every curve was passed the SAME color (a common consequence of
    # leaving the frontend's per-system colour picker at its default), the
    # plot would look like a single line. Fall back to the colormap so
    # each curve still gets a distinct hue.
    if colors and len(curves) > 1:
        non_null = [c for c in colors if c]
        if len(set(non_null)) <= 1:
            colors = None

    fig, ax = plt.subplots(figsize=style.figsize)
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")
    for i, ((cx, cy), label) in enumerate(zip(curves, labels, strict=False)):
        cx_arr = np.asarray(cx)
        cy_arr = np.asarray(cy)
        if cx_arr.size == 0:
            continue
        per_curve = (colors[i] if (colors is not None and i < len(colors)) else None)
        color = per_curve if per_curve else (
            palette(i % palette.N) if palette.N else None
        )
        ax.plot(
            cx_arr, cy_arr,
            linewidth=style.linewidth,
            color=color,
            label=label,
        )
        x_min = min(x_min, float(cx_arr.min()))
        x_max = max(x_max, float(cx_arr.max()))
        y_min = min(y_min, float(cy_arr.min()))
        y_max = max(y_max, float(cy_arr.max()))

    ax.margins(x=0, y=0)
    ax.set_xlabel(style.xlabel or default_xlabel)
    ax.set_ylabel(style.ylabel or default_ylabel)
    ax.set_title(style.title or default_title)
    _apply_axis_limits(ax, style)
    # Same Y-only-to-zero default as plot_line — see note there.
    if style.ymin is None and y_min != float("inf") and y_min >= 0:
        ax.set_ylim(bottom=0)
    _apply_publication_ticks(ax)
    if x_min != float("inf"):
        _force_boundary_ticks_x(ax, x_min, x_max)
    if style.grid:
        ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True)
    _save(fig, output_path, style.dpi)


def _force_boundary_ticks_x(ax, x_min: float, x_max: float) -> None:
    """Guarantee the x-axis labels the data start AND end, snapped to a
    clean integer / "nice" multiple of the auto tick interval.

    matplotlib's `MaxNLocator` picks tidy round numbers in the middle of
    the range (e.g. 120, 140, 160, 180 for data spanning 100–200) but
    skips the actual data boundaries. We want the user to see the run's
    actual start and end. Snapping the boundary labels to the same step
    size as the auto-picked ticks keeps the formatting consistent —
    100.15 → ``100``, 199.15 → ``200`` — instead of leaking decimals
    into otherwise integer-looking ticks.
    """
    import math

    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        return
    span = x_max - x_min
    if span < 1e-12:
        return

    lo, hi = ax.get_xlim()
    auto = sorted(t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9)

    # Snap step: use the auto tick spacing when matplotlib's locator gave
    # us at least two ticks, otherwise fall back to ~5 % of the data span.
    if len(auto) >= 2:
        step = auto[1] - auto[0]
    else:
        step = span / 5.0
    if step <= 0:
        return

    nice_min = math.floor(x_min / step) * step
    nice_max = math.ceil(x_max / step) * step

    # Drop any auto tick that's within half a step of the new boundaries
    # so the two don't collide / double-label.
    tol = step * 0.5
    auto = [t for t in auto if abs(t - nice_min) > tol and abs(t - nice_max) > tol]
    ticks = sorted({float(nice_min), float(nice_max), *auto})

    # Extend the axis to the snapped boundaries so the labels sit exactly
    # at the spine edges; the gap is tiny (≤ half a step) and invisible.
    ax.set_xlim(min(nice_min, lo), max(nice_max, hi))
    ax.set_xticks(ticks)


def _apply_publication_ticks(ax) -> None:
    """Major ticks include the axis boundaries; minor ticks fill the gaps.

    matplotlib's default ``AutoLocator`` prunes ticks at the axis edges
    when it thinks they'd collide with adjacent axis labels — which is
    why a snug-fit RMSD plot misses the leading "0" on the time axis.
    ``MaxNLocator(prune=None)`` keeps the boundary tick; ``AutoMinorLocator``
    adds the in-between tick marks that match a publication aesthetic.
    """
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(
            MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune=None)
        )
        axis.set_minor_locator(AutoMinorLocator())
    # Inward ticks on the left + bottom axes only — no mirrored ticks
    # on the top / right spines.
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.tick_params(which="major", length=5, width=0.9)
    ax.tick_params(which="minor", length=2.5, width=0.6)


def _apply_axis_limits(ax, style: PlotStyle) -> None:
    """Pin any of the four bounds the user set, leave the rest auto-scaled."""
    if style.xmin is not None or style.xmax is not None:
        cur_lo, cur_hi = ax.get_xlim()
        ax.set_xlim(
            style.xmin if style.xmin is not None else cur_lo,
            style.xmax if style.xmax is not None else cur_hi,
        )
    if style.ymin is not None or style.ymax is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(
            style.ymin if style.ymin is not None else cur_lo,
            style.ymax if style.ymax is not None else cur_hi,
        )


def plot_pca(
    eigenvalues: np.ndarray,
    projections: np.ndarray,
    output_path: str | Path,
    style: PlotStyle | None = None,
) -> None:
    """Scree (bar + cumulative) on the left, PC1-vs-PC2 scatter on the right."""
    style = style or PlotStyle()
    _apply_rc(style)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize)

    n = len(eigenvalues)
    idx = np.arange(1, n + 1)
    total = float(eigenvalues.sum())
    frac = eigenvalues / total if total > 0 else eigenvalues
    cum = np.cumsum(frac)

    bar_color = style.color or "steelblue"
    cum_color = style.accent_color or "darkorange"

    ax1.bar(idx, frac, color=bar_color, alpha=0.85, label="per-PC fraction")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (fraction)")
    ax1.set_title("Scree")
    ax1.set_xticks(idx)
    if style.grid:
        ax1.grid(axis="y", alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(idx, cum, marker="o", color=cum_color, linewidth=style.linewidth, label="cumulative")
    ax1b.set_ylim(0, 1.05)
    ax1b.set_ylabel("Cumulative variance")

    if projections.shape[1] >= 2:
        sc = ax2.scatter(
            projections[:, 0],
            projections[:, 1],
            c=np.arange(len(projections)),
            cmap=style.cmap or "viridis",
            s=18,
            alpha=0.85,
        )
        cbar = fig.colorbar(sc, ax=ax2)
        cbar.set_label("Frame")
        ax2.set_xlabel(style.xlabel or "PC1")
        ax2.set_ylabel(style.ylabel or "PC2")
        ax2.set_title("Projections (PC1 vs PC2)")
    else:
        ax2.plot(
            np.arange(len(projections)),
            projections[:, 0],
            linewidth=style.linewidth,
            color=bar_color,
        )
        ax2.set_xlabel(style.xlabel or "Frame")
        ax2.set_ylabel(style.ylabel or "PC1")
        ax2.set_title("Projection onto PC1")
    if style.grid:
        ax2.grid(alpha=0.3)

    if style.title:
        fig.suptitle(style.title)

    _save(fig, output_path, style.dpi)


def plot_clusters(
    projections: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray | None,
    representative_frames: np.ndarray | None,
    output_path: str | Path,
    style: PlotStyle | None = None,
) -> None:
    """Scatter of cluster assignments in PC space with centers + representatives marked."""
    style = style or PlotStyle()
    _apply_rc(style)

    k = int(labels.max()) + 1 if labels.size else 0
    fig, ax = plt.subplots(figsize=style.figsize)

    cmap = style.cmap or "tab10"
    center_color = style.color or "black"
    rep_color = style.accent_color or "red"

    if projections.shape[1] >= 2:
        ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=labels,
            cmap=cmap,
            s=22,
            alpha=0.85,
        )
        if centers is not None and centers.shape[1] >= 2:
            ax.scatter(
                centers[:, 0], centers[:, 1],
                marker="X", c=center_color, s=160, label="centers",
                edgecolors="white", linewidths=1.0,
            )
        if representative_frames is not None:
            valid = representative_frames[representative_frames >= 0]
            if valid.size:
                ax.scatter(
                    projections[valid, 0],
                    projections[valid, 1],
                    marker="*", c=rep_color, s=260,
                    edgecolors="black", linewidths=1.0,
                    label="representatives",
                )
        ax.set_xlabel(style.xlabel or "PC1")
        ax.set_ylabel(style.ylabel or "PC2")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", frameon=True)
    else:
        ax.scatter(
            np.arange(len(projections)), projections[:, 0],
            c=labels, cmap=cmap, s=22,
        )
        ax.set_xlabel(style.xlabel or "Frame")
        ax.set_ylabel(style.ylabel or "PC1")

    ax.set_title(style.title or f"K-means clusters (k={k})")
    if style.grid:
        ax.grid(alpha=0.3)
    _save(fig, output_path, style.dpi)
