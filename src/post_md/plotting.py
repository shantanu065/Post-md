"""Matplotlib plot helpers for the CLI.

All plots are rendered on the headless ``Agg`` backend so the CLI works
on servers without a display. Each function accepts a :class:`PlotStyle`
that lets the user override colors, axis labels, title, legend, figure
size, DPI, font size, and line width — enough for publication-quality
figures without dropping into the Python API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    ax.set_xlabel(style.xlabel or default_xlabel)
    ax.set_ylabel(style.ylabel or default_ylabel)
    ax.set_title(style.title or default_title)
    if style.grid:
        ax.grid(alpha=0.3)
    if style.show_legend or style.legend_label:
        ax.legend(loc="best", frameon=True)
    _save(fig, output_path, style.dpi)


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
