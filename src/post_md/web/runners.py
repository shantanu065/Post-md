"""Analysis runners called by the web routes.

Each runner takes a topology + trajectory path, a working directory, and a
plain ``dict`` of options (string-keyed, all values JSON-serialisable). They
mirror the CLI command bodies so the web UI exposes the same surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from post_md.core.universe import Universe
from post_md.plotting import PlotStyle

_TIME_UNIT_FACTORS = {"ps": 1.0, "ns": 1e-3, "fs": 1e3, "us": 1e-6}


def _maybe_mass_weights(group):
    masses = group.masses
    return masses if float(masses.sum()) > 0 else None


def _derive_time_axis(times_ps: np.ndarray, dt, time_unit: str):
    if time_unit not in _TIME_UNIT_FACTORS:
        time_unit = "ps"
    n = len(times_ps)
    if dt is not None and float(dt) > 0:
        x_ps = np.arange(n, dtype=np.float64) * float(dt)
        return x_ps * _TIME_UNIT_FACTORS[time_unit], f"Time ({time_unit})"
    sequential = (
        n >= 2 and np.array_equal(times_ps, np.arange(n, dtype=times_ps.dtype))
    )
    all_zero = bool(np.all(times_ps == 0.0))
    if sequential or all_zero:
        return np.arange(n, dtype=np.float64), "Frame"
    return times_ps * _TIME_UNIT_FACTORS[time_unit], f"Time ({time_unit})"


def _write_table(path: Path, headers: list[str], data: np.ndarray, *, title: str = "") -> None:
    ext = path.suffix.lower()
    if ext == ".csv":
        fmt = ["%.6g"] * data.shape[1]
        np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="", fmt=fmt)
    elif ext == ".xvg":
        with open(path, "w") as f:
            f.write("# Created by Post_MD\n")
            f.write(f'@    title "{title or (headers[1] + " vs " + headers[0])}"\n')
            f.write(f'@    xaxis  label "{headers[0]}"\n')
            f.write(f'@    yaxis  label "{headers[1]}"\n')
            f.write("@TYPE xy\n")
            for row in data:
                f.write("  ".join(f"{v:14.6f}" for v in row) + "\n")
    else:  # .dat / .txt / anything else
        with open(path, "w") as f:
            f.write("# " + "  ".join(headers) + "\n")
            for row in data:
                f.write(" ".join(f"{v:14.6f}" for v in row) + "\n")


def _opt(opts: dict[str, Any], key: str, default=None):
    v = opts.get(key)
    if v is None or v == "":
        return default
    return v


def _parse_figsize(spec: str | None, default=(7.5, 4.5)) -> tuple[float, float]:
    if not spec:
        return default
    for sep in ("x", "X", ","):
        if sep in spec:
            a, b = spec.split(sep, 1)
            try:
                return (float(a.strip()), float(b.strip()))
            except ValueError:
                return default
    return default


def _build_style(opts: dict, default_figsize=(7.5, 4.5)) -> PlotStyle:
    return PlotStyle(
        title=_opt(opts, "title"),
        xlabel=_opt(opts, "xlabel"),
        ylabel=_opt(opts, "ylabel"),
        color=_opt(opts, "color"),
        accent_color=_opt(opts, "accent_color"),
        cmap=_opt(opts, "cmap"),
        linewidth=float(_opt(opts, "linewidth", 1.6)),
        legend_label=_opt(opts, "legend_label"),
        show_legend=bool(_opt(opts, "legend_label")) and not bool(opts.get("no_legend")),
        grid=not bool(opts.get("no_grid")),
        figsize=_parse_figsize(_opt(opts, "figsize"), default_figsize),
        dpi=int(_opt(opts, "dpi", 150)),
        font_size=int(_opt(opts, "font_size", 12)),
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_info(topology: Path, trajectory: Path) -> dict[str, Any]:
    u = Universe.load(str(topology), str(trajectory))
    return {
        "n_atoms": int(u.topology.n_atoms),
        "n_residues": int(u.topology.n_residues),
        "n_bonds": int(len(u.topology.bonds)),
        "n_frames": int(u.trajectory.n_frames),
    }


def run_rmsd(topology: Path, trajectory: Path, workdir: Path, opts: dict) -> dict:
    from post_md.analysis.rmsd import rmsd_trajectory
    from post_md.plotting import plot_line

    u = Universe.load(str(topology), str(trajectory))
    ag = u.select_atoms(_opt(opts, "selection", "name CA"))
    coords, times_ps = ag.coordinates_and_times()
    reference = int(_opt(opts, "reference", 0))
    ref = coords[reference]
    values = rmsd_trajectory(coords, ref, weights=_maybe_mass_weights(ag))

    x, x_label = _derive_time_axis(times_ps, _opt(opts, "dt"), _opt(opts, "time_unit", "ps"))
    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")
    data_path = workdir / f"rmsd.{ext}"
    plot_path = workdir / "rmsd.png"

    table = np.column_stack([x, values])
    _write_table(data_path, [x_label, "RMSD (A)"], table, title="RMSD")

    if not opts.get("no_plot"):
        style = _build_style(opts)
        plot_line(
            x, values, plot_path, style=style,
            default_xlabel=x_label, default_ylabel="RMSD (Å)",
            default_title=f"RMSD vs frame {reference}",
        )
    return {"data": data_path.name, "plot": plot_path.name if not opts.get("no_plot") else None,
            "rows": int(len(values))}


def run_rmsf(topology: Path, trajectory: Path, workdir: Path, opts: dict) -> dict:
    from post_md.analysis.rmsf import rmsf as rmsf_fn
    from post_md.plotting import plot_line

    u = Universe.load(str(topology), str(trajectory))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords = ag.coordinates()
    values = rmsf_fn(coords, weights=_maybe_mass_weights(ag))

    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")
    data_path = workdir / f"rmsf.{ext}"
    plot_path = workdir / "rmsf.png"
    table = np.column_stack([ag.indices.astype(np.float64), values])
    _write_table(data_path, ["Atom index", "RMSF (A)"], table, title="RMSF")

    if not opts.get("no_plot"):
        style = _build_style(opts)
        plot_line(
            ag.indices, values, plot_path, style=style,
            default_xlabel="Atom index", default_ylabel="RMSF (Å)",
            default_title="RMSF",
        )
    return {"data": data_path.name, "plot": plot_path.name if not opts.get("no_plot") else None,
            "rows": int(len(values))}


def run_rg(topology: Path, trajectory: Path, workdir: Path, opts: dict) -> dict:
    from post_md.analysis.rg import radius_of_gyration
    from post_md.plotting import plot_line

    u = Universe.load(str(topology), str(trajectory))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords, times_ps = ag.coordinates_and_times()
    masses = ag.masses
    if float(masses.sum()) == 0:
        masses = np.ones_like(masses)
    values = radius_of_gyration(coords, masses)

    x, x_label = _derive_time_axis(times_ps, _opt(opts, "dt"), _opt(opts, "time_unit", "ps"))
    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")
    data_path = workdir / f"rg.{ext}"
    plot_path = workdir / "rg.png"
    table = np.column_stack([x, values])
    _write_table(data_path, [x_label, "Rg (A)"], table, title="Radius of gyration")

    if not opts.get("no_plot"):
        style = _build_style(opts)
        plot_line(
            x, values, plot_path, style=style,
            default_xlabel=x_label, default_ylabel="Rg (Å)",
            default_title="Radius of gyration",
        )
    return {"data": data_path.name, "plot": plot_path.name if not opts.get("no_plot") else None,
            "rows": int(len(values))}


def run_pca(topology: Path, trajectory: Path, workdir: Path, opts: dict) -> dict:
    from post_md.analysis.pca import pca_cartesian
    from post_md.plotting import plot_pca

    u = Universe.load(str(topology), str(trajectory))
    ag = u.select_atoms(_opt(opts, "selection", "name CA"))
    coords = ag.coordinates()
    n_components = int(_opt(opts, "n_components", 10))
    result = pca_cartesian(coords, n_components=n_components,
                           weights=_maybe_mass_weights(ag))

    data_path = workdir / "pca.npz"
    np.savez_compressed(
        data_path,
        mean=result.mean,
        eigenvalues=result.eigenvalues,
        components=result.components,
        projections=result.projections,
        selection_indices=ag.indices,
        atom_names=ag.names,
    )
    plot_path = workdir / "pca.png"
    if not opts.get("no_plot"):
        style = _build_style(opts, default_figsize=(12.0, 4.5))
        plot_pca(result.eigenvalues, result.projections, plot_path, style=style)

    return {
        "data": data_path.name,
        "plot": plot_path.name if not opts.get("no_plot") else None,
        "scree": [float(v) for v in result.scree()],
        "cumulative": [float(v) for v in np.cumsum(result.scree())],
    }


def run_cluster(topology: Path, trajectory: Path, workdir: Path, opts: dict) -> dict:
    from post_md.analysis.clustering import kmeans_cluster
    from post_md.plotting import plot_clusters

    pca_path = workdir / "pca.npz"
    if not pca_path.exists():
        raise FileNotFoundError("pca.npz not found — run PCA first.")
    data = np.load(pca_path)
    projections = data["projections"]
    n_pcs = min(int(_opt(opts, "n_pcs", 10)), projections.shape[1])
    k = int(_opt(opts, "k", 5))
    seed = int(_opt(opts, "seed", 0))
    result = kmeans_cluster(projections[:, :n_pcs], k=k, seed=seed)

    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")
    data_path = workdir / f"clusters.{ext}"
    table = np.column_stack(
        [np.arange(len(result.labels)).astype(np.float64), result.labels.astype(np.float64)]
    )
    _write_table(data_path, ["Frame", "Cluster"], table, title="K-means clusters")

    rep_pdb_name = None
    if opts.get("rep_pdb"):
        from post_md.cli.main import _write_pdb_models

        u = Universe.load(str(topology), str(trajectory))
        rep_path = workdir / "reps.pdb"
        _write_pdb_models(str(rep_path), u, result.representative_frames, data["selection_indices"])
        rep_pdb_name = rep_path.name

    plot_path = workdir / "clusters.png"
    if not opts.get("no_plot"):
        style = _build_style(opts, default_figsize=(7.5, 6.0))
        plot_clusters(
            projections=projections, labels=result.labels,
            centers=result.centers, representative_frames=result.representative_frames,
            output_path=plot_path, style=style,
        )

    return {
        "data": data_path.name,
        "plot": plot_path.name if not opts.get("no_plot") else None,
        "rep_pdb": rep_pdb_name,
        "k": k,
        "n_pcs": n_pcs,
    }


RUNNERS = {
    "rmsd": run_rmsd,
    "rmsf": run_rmsf,
    "rg": run_rg,
    "pca": run_pca,
    "cluster": run_cluster,
}
