"""Post_MD command-line interface (Typer-based).

Default output format is ``.dat`` (whitespace-separated, ``#``-prefixed
header — cpptraj-style). ``.xvg`` (GROMACS xmgrace) and ``.csv`` are also
supported; the format is selected from the output extension.

Every analysis command writes its data file *and* a matching PNG plot.
The plot can be customised with ``--title``, ``--xlabel``, ``--ylabel``,
``--color``, ``--legend-label``, ``--figsize``, ``--dpi``, ``--font-size``,
``--linewidth``, ``--no-grid``, ``--no-legend``, ``--no-plot``. Line plots
also accept ``--dt`` (ps per frame) and ``--time-unit {ps,ns,fs}`` to put
real time on the x-axis instead of frame number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

app = typer.Typer(
    help="Post_MD — rapid MD trajectory analysis (PCA + clustering).",
    add_completion=False,
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Loaders / generic helpers
# ---------------------------------------------------------------------------


def _load(topology: str, trajectory: str):
    from post_md.core.universe import Universe

    return Universe.load(topology, trajectory)


def _maybe_mass_weights(group) -> np.ndarray | None:
    masses = group.masses
    return masses if float(masses.sum()) > 0 else None


# ---------------------------------------------------------------------------
# Table writers — dispatched by output extension
# ---------------------------------------------------------------------------


def _write_dat(path: str, headers: list[str], data: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# " + "  ".join(headers) + "\n")
        for row in data:
            parts = [f"{row[0]:14.6f}"]
            parts.extend(f"{v:14.6f}" for v in row[1:])
            f.write(" ".join(parts) + "\n")


def _write_xvg(path: str, headers: list[str], data: np.ndarray, *, title: str = "") -> None:
    with open(path, "w") as f:
        f.write("# Created by Post_MD\n")
        f.write(f'@    title "{title or (headers[1] + " vs " + headers[0])}"\n')
        f.write(f'@    xaxis  label "{headers[0]}"\n')
        f.write(f'@    yaxis  label "{headers[1]}"\n')
        f.write("@TYPE xy\n")
        for row in data:
            cells = "  ".join(f"{v:14.6f}" for v in row)
            f.write(cells + "\n")


def _write_csv(path: str, headers: list[str], data: np.ndarray) -> None:
    fmt = ["%.6g"] * data.shape[1]
    np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="", fmt=fmt)


def _write_table(
    path: str,
    headers: list[str],
    data: np.ndarray,
    *,
    title: str = "",
) -> None:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        _write_csv(path, headers, data)
    elif ext == ".xvg":
        _write_xvg(path, headers, data, title=title)
    else:  # .dat, .txt, anything else
        _write_dat(path, headers, data)


def _plot_path(output: str) -> str:
    return str(Path(output).with_suffix(".png"))


# ---------------------------------------------------------------------------
# Plot-style helpers
# ---------------------------------------------------------------------------


def _parse_figsize(spec: str | None) -> tuple[float, float] | None:
    if not spec:
        return None
    for sep in ("x", "X", ","):
        if sep in spec:
            a, b = spec.split(sep, 1)
            return (float(a.strip()), float(b.strip()))
    raise typer.BadParameter(f"--figsize must look like '8x5', got {spec!r}")


def _build_style(
    *,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    color: str | None,
    accent_color: str | None,
    cmap: str | None,
    linewidth: float,
    legend_label: str | None,
    show_legend: bool,
    grid: bool,
    figsize: str | None,
    dpi: int,
    font_size: int,
    default_figsize: tuple[float, float] = (7.5, 4.5),
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    from post_md.plotting import PlotStyle

    size = _parse_figsize(figsize) or default_figsize
    return PlotStyle(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        accent_color=accent_color,
        cmap=cmap,
        linewidth=linewidth,
        legend_label=legend_label,
        show_legend=show_legend,
        grid=grid,
        figsize=size,
        dpi=dpi,
        font_size=font_size,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


# ---------------------------------------------------------------------------
# Time-axis derivation
# ---------------------------------------------------------------------------

_TIME_UNIT_FACTORS = {"ps": 1.0, "ns": 1e-3, "fs": 1e3, "us": 1e-6}


def _derive_time_axis(
    times_ps: np.ndarray,
    dt: float | None,
    time_unit: str,
) -> tuple[np.ndarray, str]:
    """Return (x_array, xlabel). dt is ps/frame; --time-unit names the display unit."""
    n = len(times_ps)
    if dt is not None:
        x_ps = np.arange(n, dtype=np.float64) * float(dt)
        return x_ps * _TIME_UNIT_FACTORS[time_unit], f"Time ({time_unit})"

    # Heuristic: if all times are zero or look like sequential frame indices,
    # the trajectory format didn't carry real time stamps — fall back to frame index.
    sequential = (
        n >= 2
        and np.array_equal(times_ps, np.arange(n, dtype=times_ps.dtype))
    )
    all_zero = bool(np.all(times_ps == 0.0))
    if sequential or all_zero:
        return np.arange(n, dtype=np.float64), "Frame"
    return times_ps * _TIME_UNIT_FACTORS[time_unit], f"Time ({time_unit})"


# ---------------------------------------------------------------------------
# PDB writer for cluster representatives
# ---------------------------------------------------------------------------


def _write_pdb_models(
    path: str,
    universe,
    frame_indices,
    atom_indices,
) -> None:
    top = universe.topology
    atom_indices = [int(a) for a in atom_indices]
    with open(path, "w") as out:
        for k, fi in enumerate(frame_indices):
            fi = int(fi)
            if fi < 0:
                continue
            out.write(f"MODEL     {k + 1:>4}\n")
            frame = universe.trajectory.read_frame(fi)
            for serial, ai in enumerate(atom_indices, start=1):
                name = str(top.atom_names[ai])
                res_name = str(top.residue_names[ai])
                res_id = int(top.residue_ids[ai])
                el = str(top.elements[ai])
                x, y, z = frame.coordinates[ai]
                out.write(
                    f"ATOM  {serial:5d} {name:<4s} {res_name:<3s}  {res_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n"
                )
            out.write("ENDMDL\n")
        out.write("END\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


PLOT_PANEL = "Plot customisation"


@app.command()
def info(
    topology: str = typer.Option(..., "-p", "--topology"),
    trajectory: str = typer.Option(..., "-t", "--trajectory"),
) -> None:
    """Print topology + trajectory metadata."""
    u = _load(topology, trajectory)
    typer.echo(f"Universe: {u}")
    typer.echo(f"  atoms:    {u.topology.n_atoms}")
    typer.echo(f"  residues: {u.topology.n_residues}")
    typer.echo(f"  bonds:    {len(u.topology.bonds)}")
    typer.echo(f"  frames:   {u.trajectory.n_frames}")


@app.command()
def rmsd(
    topology: str = typer.Option(..., "-p", "--topology"),
    trajectory: str = typer.Option(..., "-t", "--trajectory"),
    selection: str = typer.Option("name CA", "-s", "--selection"),
    reference: int = typer.Option(0, "--reference"),
    output: str = typer.Option("rmsd.dat", "-o", "--output", help="Output data file (.dat / .xvg / .csv chosen by extension)."),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    # ---- plot customisation ----
    dt: float | None = typer.Option(None, "--dt", help="Time step per frame in ps; switches x-axis to time.", rich_help_panel=PLOT_PANEL),
    time_unit: str = typer.Option("ps", "--time-unit", help="ps | ns | fs | us — display unit for the time axis.", rich_help_panel=PLOT_PANEL),
    title: str | None = typer.Option(None, "--title", rich_help_panel=PLOT_PANEL),
    xlabel: str | None = typer.Option(None, "--xlabel", rich_help_panel=PLOT_PANEL),
    ylabel: str | None = typer.Option(None, "--ylabel", rich_help_panel=PLOT_PANEL),
    color: str | None = typer.Option(None, "--color", help="Line color (matplotlib name or hex).", rich_help_panel=PLOT_PANEL),
    linewidth: float = typer.Option(1.6, "--linewidth", rich_help_panel=PLOT_PANEL),
    legend_label: str | None = typer.Option(None, "--legend-label", rich_help_panel=PLOT_PANEL),
    no_legend: bool = typer.Option(False, "--no-legend", rich_help_panel=PLOT_PANEL),
    no_grid: bool = typer.Option(False, "--no-grid", rich_help_panel=PLOT_PANEL),
    figsize: str | None = typer.Option(None, "--figsize", help="e.g. '8x5'", rich_help_panel=PLOT_PANEL),
    dpi: int = typer.Option(150, "--dpi", rich_help_panel=PLOT_PANEL),
    font_size: int = typer.Option(12, "--font-size", rich_help_panel=PLOT_PANEL),
    xmin: float | None = typer.Option(None, "--xmin", help="Pin x-axis lower bound (auto if blank).", rich_help_panel=PLOT_PANEL),
    xmax: float | None = typer.Option(None, "--xmax", help="Pin x-axis upper bound (auto if blank).", rich_help_panel=PLOT_PANEL),
    ymin: float | None = typer.Option(None, "--ymin", help="Pin y-axis lower bound (auto if blank).", rich_help_panel=PLOT_PANEL),
    ymax: float | None = typer.Option(None, "--ymax", help="Pin y-axis upper bound (auto if blank).", rich_help_panel=PLOT_PANEL),
) -> None:
    """Compute per-frame RMSD (Å) vs a reference frame after Kabsch alignment."""
    from post_md.analysis.rmsd import rmsd_trajectory

    if time_unit not in _TIME_UNIT_FACTORS:
        raise typer.BadParameter(f"--time-unit must be one of {list(_TIME_UNIT_FACTORS)}")

    u = _load(topology, trajectory)
    ag = u.select_atoms(selection)
    coords, times_ps = ag.coordinates_and_times()
    ref = coords[reference]
    values = rmsd_trajectory(coords, ref, weights=_maybe_mass_weights(ag))

    x, x_label_default = _derive_time_axis(times_ps, dt, time_unit)
    data = np.column_stack([x, values])
    _write_table(output, [x_label_default, "RMSD (A)"], data, title="RMSD")
    typer.echo(f"Wrote {output} ({len(values)} rows)")

    if plot:
        from post_md.plotting import plot_line

        style = _build_style(
            title=title, xlabel=xlabel, ylabel=ylabel,
            color=color, accent_color=None, cmap=None,
            linewidth=linewidth,
            legend_label=legend_label, show_legend=not no_legend and legend_label is not None,
            grid=not no_grid, figsize=figsize, dpi=dpi, font_size=font_size,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        )
        png = _plot_path(output)
        plot_line(
            x, values, png, style=style,
            default_xlabel=x_label_default,
            default_ylabel="RMSD (Å)",
            default_title="RMSD over time",
        )
        typer.echo(f"Wrote {png}")


@app.command()
def rmsf(
    topology: str = typer.Option(..., "-p", "--topology"),
    trajectory: str = typer.Option(..., "-t", "--trajectory"),
    selection: str = typer.Option("protein", "-s", "--selection"),
    output: str = typer.Option("rmsf.dat", "-o", "--output"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    per_atom: bool = typer.Option(
        False, "--per-atom",
        help="Emit raw per-atom RMSF instead of the mass-weighted per-residue mean.",
    ),
    # ---- plot customisation ----
    title: str | None = typer.Option(None, "--title", rich_help_panel=PLOT_PANEL),
    xlabel: str | None = typer.Option(None, "--xlabel", rich_help_panel=PLOT_PANEL),
    ylabel: str | None = typer.Option(None, "--ylabel", rich_help_panel=PLOT_PANEL),
    color: str | None = typer.Option(None, "--color", rich_help_panel=PLOT_PANEL),
    linewidth: float = typer.Option(1.6, "--linewidth", rich_help_panel=PLOT_PANEL),
    legend_label: str | None = typer.Option(None, "--legend-label", rich_help_panel=PLOT_PANEL),
    no_legend: bool = typer.Option(False, "--no-legend", rich_help_panel=PLOT_PANEL),
    no_grid: bool = typer.Option(False, "--no-grid", rich_help_panel=PLOT_PANEL),
    figsize: str | None = typer.Option(None, "--figsize", rich_help_panel=PLOT_PANEL),
    dpi: int = typer.Option(150, "--dpi", rich_help_panel=PLOT_PANEL),
    font_size: int = typer.Option(12, "--font-size", rich_help_panel=PLOT_PANEL),
) -> None:
    """Compute RMSF (Å). Default: one value per residue (mass-weighted mean
    of its atoms), matching cpptraj ``byres`` / GROMACS ``gmx rmsf -res``.
    Use ``--per-atom`` to emit the underlying per-atom array instead.
    """
    from post_md.analysis.rmsf import rmsf as rmsf_fn

    u = _load(topology, trajectory)
    ag = u.select_atoms(selection)
    coords = ag.coordinates()
    atom_values = rmsf_fn(coords, weights=_maybe_mass_weights(ag))

    if per_atom:
        x_vals = ag.indices.astype(np.float64)
        values = atom_values
        x_label = "Atom index"
        data_header = "Atom index"
        title_default = f"RMSF per atom  |  {selection!r}"
    else:
        residue_ids = ag.residue_ids
        unique_resids, first_idx = np.unique(residue_ids, return_index=True)
        unique_resids = unique_resids[np.argsort(first_idx)]
        masses = ag.masses.astype(np.float64)
        use_masses = float(masses.sum()) > 0
        values = np.empty(unique_resids.size, dtype=np.float64)
        for k, rid in enumerate(unique_resids):
            mask = residue_ids == rid
            vals = atom_values[mask]
            if use_masses:
                w = masses[mask]
                w_sum = float(w.sum())
                values[k] = float((vals * w).sum() / w_sum) if w_sum > 0 else float(vals.mean())
            else:
                values[k] = float(vals.mean())
        x_vals = unique_resids.astype(np.float64)
        x_label = "Residue"
        data_header = "Residue"
        title_default = f"RMSF per residue  |  {selection!r}"

    data = np.column_stack([x_vals, values])
    _write_table(output, [data_header, "RMSF (A)"], data, title="RMSF")
    typer.echo(f"Wrote {output} ({len(values)} rows)")

    if plot:
        from post_md.plotting import plot_line

        style = _build_style(
            title=title, xlabel=xlabel, ylabel=ylabel,
            color=color, accent_color=None, cmap=None,
            linewidth=linewidth,
            legend_label=legend_label, show_legend=not no_legend and legend_label is not None,
            grid=not no_grid, figsize=figsize, dpi=dpi, font_size=font_size,
        )
        png = _plot_path(output)
        plot_line(
            x_vals, values, png, style=style,
            default_xlabel=x_label,
            default_ylabel="RMSF (Å)",
            default_title=title_default,
        )
        typer.echo(f"Wrote {png}")


@app.command()
def rg(
    topology: str = typer.Option(..., "-p", "--topology"),
    trajectory: str = typer.Option(..., "-t", "--trajectory"),
    selection: str = typer.Option("protein", "-s", "--selection"),
    output: str = typer.Option("rg.dat", "-o", "--output"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    # ---- plot customisation ----
    dt: float | None = typer.Option(None, "--dt", help="Time step per frame in ps.", rich_help_panel=PLOT_PANEL),
    time_unit: str = typer.Option("ps", "--time-unit", rich_help_panel=PLOT_PANEL),
    title: str | None = typer.Option(None, "--title", rich_help_panel=PLOT_PANEL),
    xlabel: str | None = typer.Option(None, "--xlabel", rich_help_panel=PLOT_PANEL),
    ylabel: str | None = typer.Option(None, "--ylabel", rich_help_panel=PLOT_PANEL),
    color: str | None = typer.Option(None, "--color", rich_help_panel=PLOT_PANEL),
    linewidth: float = typer.Option(1.6, "--linewidth", rich_help_panel=PLOT_PANEL),
    legend_label: str | None = typer.Option(None, "--legend-label", rich_help_panel=PLOT_PANEL),
    no_legend: bool = typer.Option(False, "--no-legend", rich_help_panel=PLOT_PANEL),
    no_grid: bool = typer.Option(False, "--no-grid", rich_help_panel=PLOT_PANEL),
    figsize: str | None = typer.Option(None, "--figsize", rich_help_panel=PLOT_PANEL),
    dpi: int = typer.Option(150, "--dpi", rich_help_panel=PLOT_PANEL),
    font_size: int = typer.Option(12, "--font-size", rich_help_panel=PLOT_PANEL),
) -> None:
    """Compute radius of gyration (Å) per frame."""
    from post_md.analysis.rg import radius_of_gyration

    if time_unit not in _TIME_UNIT_FACTORS:
        raise typer.BadParameter(f"--time-unit must be one of {list(_TIME_UNIT_FACTORS)}")

    u = _load(topology, trajectory)
    ag = u.select_atoms(selection)
    coords, times_ps = ag.coordinates_and_times()
    masses = ag.masses
    if float(masses.sum()) == 0:
        masses = np.ones_like(masses)
    values = radius_of_gyration(coords, masses)

    x, x_label_default = _derive_time_axis(times_ps, dt, time_unit)
    data = np.column_stack([x, values])
    _write_table(output, [x_label_default, "Rg (A)"], data, title="Radius of gyration")
    typer.echo(f"Wrote {output} ({len(values)} rows)")

    if plot:
        from post_md.plotting import plot_line

        style = _build_style(
            title=title, xlabel=xlabel, ylabel=ylabel,
            color=color, accent_color=None, cmap=None,
            linewidth=linewidth,
            legend_label=legend_label, show_legend=not no_legend and legend_label is not None,
            grid=not no_grid, figsize=figsize, dpi=dpi, font_size=font_size,
        )
        png = _plot_path(output)
        plot_line(
            x, values, png, style=style,
            default_xlabel=x_label_default,
            default_ylabel="Rg (Å)",
            default_title=f"Radius of gyration  |  {selection!r}",
        )
        typer.echo(f"Wrote {png}")


@app.command()
def pca(
    topology: str = typer.Option(..., "-p", "--topology"),
    trajectory: str = typer.Option(..., "-t", "--trajectory"),
    selection: str = typer.Option("name CA", "-s", "--selection"),
    n_components: int = typer.Option(10, "--n-components"),
    output: str = typer.Option("pca.npz", "-o", "--output"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    # ---- plot customisation ----
    title: str | None = typer.Option(None, "--title", rich_help_panel=PLOT_PANEL),
    xlabel: str | None = typer.Option(None, "--xlabel", rich_help_panel=PLOT_PANEL),
    ylabel: str | None = typer.Option(None, "--ylabel", rich_help_panel=PLOT_PANEL),
    color: str | None = typer.Option(None, "--color", help="Scree bar color.", rich_help_panel=PLOT_PANEL),
    accent_color: str | None = typer.Option(None, "--accent-color", help="Cumulative-line color.", rich_help_panel=PLOT_PANEL),
    cmap: str | None = typer.Option(None, "--cmap", help="Colormap for the PC1/PC2 scatter.", rich_help_panel=PLOT_PANEL),
    linewidth: float = typer.Option(1.6, "--linewidth", rich_help_panel=PLOT_PANEL),
    no_grid: bool = typer.Option(False, "--no-grid", rich_help_panel=PLOT_PANEL),
    figsize: str | None = typer.Option(None, "--figsize", rich_help_panel=PLOT_PANEL),
    dpi: int = typer.Option(150, "--dpi", rich_help_panel=PLOT_PANEL),
    font_size: int = typer.Option(12, "--font-size", rich_help_panel=PLOT_PANEL),
) -> None:
    """Cartesian PCA on a selection; save modes, projections, and selection metadata."""
    from post_md.analysis.pca import pca_cartesian

    u = _load(topology, trajectory)
    ag = u.select_atoms(selection)
    coords = ag.coordinates()
    result = pca_cartesian(
        coords, n_components=n_components, weights=_maybe_mass_weights(ag)
    )
    np.savez_compressed(
        output,
        mean=result.mean,
        eigenvalues=result.eigenvalues,
        components=result.components,
        projections=result.projections,
        selection_indices=ag.indices,
        atom_names=ag.names,
    )
    cum = np.cumsum(result.scree())
    typer.echo(f"Wrote {output}")
    typer.echo(f"Cumulative variance explained: {cum}")

    if plot:
        from post_md.plotting import plot_pca

        style = _build_style(
            title=title, xlabel=xlabel, ylabel=ylabel,
            color=color, accent_color=accent_color, cmap=cmap,
            linewidth=linewidth,
            legend_label=None, show_legend=False,
            grid=not no_grid, figsize=figsize, dpi=dpi, font_size=font_size,
            default_figsize=(12.0, 4.5),
        )
        png = _plot_path(output)
        plot_pca(result.eigenvalues, result.projections, png, style=style)
        typer.echo(f"Wrote {png}")


@app.command()
def cluster(
    pca: str = typer.Option(..., "--pca", help="Path to pca.npz produced by `post-md pca`."),
    k: int = typer.Option(5, "--k"),
    n_pcs: int = typer.Option(10, "--n-pcs"),
    output: str = typer.Option("clusters.dat", "-o", "--output"),
    rep_pdb: str | None = typer.Option(None, "--rep-pdb"),
    topology: str | None = typer.Option(None, "-p", "--topology"),
    trajectory: str | None = typer.Option(None, "-t", "--trajectory"),
    seed: int = typer.Option(0, "--seed"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    # ---- plot customisation ----
    title: str | None = typer.Option(None, "--title", rich_help_panel=PLOT_PANEL),
    xlabel: str | None = typer.Option(None, "--xlabel", rich_help_panel=PLOT_PANEL),
    ylabel: str | None = typer.Option(None, "--ylabel", rich_help_panel=PLOT_PANEL),
    color: str | None = typer.Option(None, "--color", help="Color of center markers.", rich_help_panel=PLOT_PANEL),
    accent_color: str | None = typer.Option(None, "--accent-color", help="Color of representative markers.", rich_help_panel=PLOT_PANEL),
    cmap: str | None = typer.Option(None, "--cmap", help="Categorical colormap (default tab10).", rich_help_panel=PLOT_PANEL),
    no_grid: bool = typer.Option(False, "--no-grid", rich_help_panel=PLOT_PANEL),
    figsize: str | None = typer.Option(None, "--figsize", rich_help_panel=PLOT_PANEL),
    dpi: int = typer.Option(150, "--dpi", rich_help_panel=PLOT_PANEL),
    font_size: int = typer.Option(12, "--font-size", rich_help_panel=PLOT_PANEL),
) -> None:
    """K-means cluster on PCA projections; optionally write representative frames as PDB."""
    from post_md.analysis.clustering import kmeans_cluster

    if not Path(pca).exists():
        raise typer.BadParameter(f"PCA file not found: {pca}")
    data = np.load(pca)
    projections = data["projections"]
    n_pcs_use = min(n_pcs, projections.shape[1])
    result = kmeans_cluster(projections[:, :n_pcs_use], k=k, seed=seed)

    table = np.column_stack(
        [np.arange(len(result.labels)).astype(np.float64), result.labels.astype(np.float64)]
    )
    _write_table(output, ["Frame", "Cluster"], table, title="K-means clusters")
    typer.echo(f"Wrote {output} (k={k}, n_pcs={n_pcs_use})")

    if rep_pdb is not None:
        if topology is None or trajectory is None:
            raise typer.BadParameter("--rep-pdb requires --topology and --trajectory")
        u = _load(topology, trajectory)
        _write_pdb_models(rep_pdb, u, result.representative_frames, data["selection_indices"])
        typer.echo(f"Wrote {rep_pdb} (k representative frames)")

    if plot:
        from post_md.plotting import plot_clusters

        style = _build_style(
            title=title, xlabel=xlabel, ylabel=ylabel,
            color=color, accent_color=accent_color, cmap=cmap,
            linewidth=1.0,
            legend_label=None, show_legend=False,
            grid=not no_grid, figsize=figsize, dpi=dpi, font_size=font_size,
            default_figsize=(7.5, 6.0),
        )
        png = _plot_path(output)
        plot_clusters(
            projections=projections,
            labels=result.labels,
            centers=result.centers,
            representative_frames=result.representative_frames,
            output_path=png,
            style=style,
        )
        typer.echo(f"Wrote {png}")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address. 0.0.0.0 for LAN access."),
    port: int = typer.Option(8000, "--port"),
    workdir: str = typer.Option("./post_md_web", "--workdir", help="Where uploads + outputs live."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
) -> None:
    """Launch the local web UI (browser-based form for all analyses)."""
    try:
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter(
            "Web extras not installed. Run: pip install -e \".[web]\""
        ) from exc

    from post_md.web.app import create_app

    fastapi_app = create_app(workdir=workdir)
    typer.echo(f"Post_MD web UI starting at http://{host}:{port}")
    typer.echo(f"Workdir: {Path(workdir).resolve()}")
    typer.echo("Press CTRL+C to stop.")
    import uvicorn  # noqa: E402

    # httptools + uvloop are bundled with `uvicorn[standard]` and give a
    # measurable throughput boost on large streaming uploads vs the pure-
    # Python defaults. timeout_keep_alive is bumped so a long-running
    # chunked upload doesn't lose its keepalive between chunks.
    server_kwargs = {
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": "info",
        "timeout_keep_alive": 300,
    }
    try:
        import httptools  # noqa: F401
        server_kwargs["http"] = "httptools"
    except ImportError:
        pass
    try:
        import uvloop  # noqa: F401
        server_kwargs["loop"] = "uvloop"
    except ImportError:
        pass

    uvicorn.run(fastapi_app, **server_kwargs)


if __name__ == "__main__":
    app()
