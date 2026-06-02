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
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    accent_color: str | None = None
    cmap: str | None = None
    open_frame: bool = False
    show_average: bool = False


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


def _apply_open_frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_publication_ticks(ax, open_frame: bool = False) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(
            MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune=None)
        )
        axis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.tick_params(which="major", length=5, width=0.9)
    ax.tick_params(which="minor", length=2.5, width=0.6)
    if open_frame:
        _apply_open_frame(ax)


def _apply_axis_limits(ax, style: PlotStyle) -> None:
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


def _nice_step(span: float) -> float:
    """A '1/2/2.5/5 x 10^n' step sized so ``span`` spans roughly five of them.

    Used to round axis limits up/down to human-friendly values (e.g. a top of
    4.87 → 5, 18.3 → 20) instead of leaving the raw padded data extreme.
    """
    import math

    if span <= 0:
        return 1.0
    raw = span / 5.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def _set_robust_ylim(ax, y_arr: np.ndarray, style: PlotStyle) -> None:
    import math

    if y_arr.size == 0:
        return
    p1, p99 = float(np.percentile(y_arr, 1)), float(np.percentile(y_arr, 99))
    span = p99 - p1
    if span < 1e-12:
        return
    pad = span * 0.10
    lo = p1 - pad
    hi = p99 + pad
    if p1 >= 0 and p1 <= max(p99, 0.0) * 0.05:
        lo = 0
    # Round the top up to a nice round number, then add one more step of
    # headroom, so the axis ends on a clean value one tick above the data
    # (e.g. data topping out near 5 → axis to 6). This keeps peaks clear of
    # the frame and never clips them.
    step = _nice_step(span)
    hi = math.ceil(hi / step) * step + step
    if style.ymin is None:
        ax.set_ylim(bottom=lo)
    if style.ymax is None:
        ax.set_ylim(top=hi)


def _force_boundary_ticks_x(ax, x_min: float, x_max: float) -> None:
    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        return
    span = x_max - x_min
    if span < 1e-12:
        return

    # Clamp the axis tight to the real data range — never pad out to the next
    # "nice" round number (a 0–200 ns run must end at 200, not 225).
    ax.set_xlim(x_min, x_max)

    # Keep only the nice round auto ticks that fall inside the data range.
    # We deliberately do NOT pin ticks at the exact data extremes: when a run
    # starts/ends off a round value (e.g. 0.2 → 200.2 ns) that would label the
    # ends "0.2" / "200.2", which is noise. Clean integer ticks read better.
    ticks = [t for t in ax.get_xticks() if x_min - 1e-9 <= t <= x_max + 1e-9]
    if ticks:
        ax.set_xticks(ticks)


def _finalize_axes(ax, x_arr, y_arr, style: PlotStyle) -> None:
    if style.ymin is None and y_arr.size:
        _set_robust_ylim(ax, y_arr, style)
    _apply_publication_ticks(ax, open_frame=style.open_frame)
    if x_arr.size:
        _force_boundary_ticks_x(ax, float(x_arr.min()), float(x_arr.max()))
    _apply_axis_limits(ax, style)
    if style.grid:
        ax.grid(alpha=0.3)


def _draw_avg_bar(ax_bar, avg_labels: list[str], avg_values: list[float],
                  bar_colors: list[str], style: PlotStyle) -> None:
    x_pos = np.arange(len(avg_labels))
    bars = ax_bar.bar(x_pos, avg_values, color=bar_colors, alpha=0.85, width=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(avg_labels, rotation=30, ha="right", fontsize=style.font_size - 1)
    ax_bar.set_ylabel("Average")
    ax_bar.set_title("Average")
    for bar, val in zip(bars, avg_values, strict=False):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=style.font_size - 2)
    if style.grid:
        ax_bar.grid(axis="y", alpha=0.3)
    if style.open_frame:
        _apply_open_frame(ax_bar)


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

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if style.show_average and y_arr.size:
        fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(style.figsize[0] + 2.5, style.figsize[1]),
                                         gridspec_kw={"width_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(figsize=style.figsize)

    ax.plot(
        x_arr, y_arr,
        linewidth=style.linewidth,
        color=style.color,
        label=style.legend_label,
    )
    ax.margins(x=0, y=0)
    ax.set_xlabel(style.xlabel or default_xlabel)
    ax.set_ylabel(style.ylabel or default_ylabel)
    ax.set_title(style.title or default_title)

    _finalize_axes(ax, x_arr, y_arr, style)

    if style.show_legend or style.legend_label:
        ax.legend(loc="best", frameon=not style.open_frame)

    if style.show_average and y_arr.size:
        lbl = style.legend_label or "System"
        c = style.color or "#1f77b4"
        _draw_avg_bar(ax_bar, [lbl], [float(np.mean(y_arr))], [c], style)

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
    style = style or PlotStyle()
    _apply_rc(style)

    cmap_name = cmap or style.cmap or "tab10"
    palette = plt.get_cmap(cmap_name)

    if colors and len(curves) > 1:
        non_null = [c for c in colors if c]
        if len(set(non_null)) <= 1:
            colors = None

    if style.show_average and curves:
        fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(style.figsize[0] + 2.5, style.figsize[1]),
                                         gridspec_kw={"width_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(figsize=style.figsize)

    all_y_parts: list[np.ndarray] = []
    resolved_colors: list[str] = []
    x_min = float("inf")
    x_max = float("-inf")

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
        all_y_parts.append(cy_arr)
        resolved_colors.append(color if isinstance(color, str) else f"C{i}")
        x_min = min(x_min, float(cx_arr.min()))
        x_max = max(x_max, float(cx_arr.max()))

    ax.margins(x=0, y=0)
    ax.set_xlabel(style.xlabel or default_xlabel)
    ax.set_ylabel(style.ylabel or default_ylabel)
    ax.set_title(style.title or default_title)

    all_y = np.concatenate(all_y_parts) if all_y_parts else np.array([])
    x_arr_range = np.array([x_min, x_max]) if x_min != float("inf") else np.array([])
    _finalize_axes(ax, x_arr_range, all_y, style)

    ax.legend(loc="best", frameon=not style.open_frame)

    if style.show_average and all_y_parts:
        avgs = [float(np.mean(yp)) for yp in all_y_parts]
        _draw_avg_bar(ax_bar, list(labels), avgs, resolved_colors, style)

    _save(fig, output_path, style.dpi)


def plot_pca(
    eigenvalues: np.ndarray,
    projections: np.ndarray,
    output_path: str | Path,
    style: PlotStyle | None = None,
) -> None:
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
    if style.open_frame:
        _apply_open_frame(ax1)
    ax1b = ax1.twinx()
    ax1b.plot(idx, cum, marker="o", color=cum_color, linewidth=style.linewidth, label="cumulative")
    ax1b.set_ylim(0, 1.05)
    ax1b.set_ylabel("Cumulative variance")
    if style.open_frame:
        ax1b.spines["top"].set_visible(False)

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
    if style.open_frame:
        _apply_open_frame(ax2)

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
            ax.legend(loc="best", frameon=not style.open_frame)
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
    if style.open_frame:
        _apply_open_frame(ax)
    _save(fig, output_path, style.dpi)
