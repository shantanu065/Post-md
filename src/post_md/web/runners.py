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


def _axis_limit(opts: dict, key: str) -> float | None:
    v = opts.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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
        xmin=_axis_limit(opts, "xmin"),
        xmax=_axis_limit(opts, "xmax"),
        ymin=_axis_limit(opts, "ymin"),
        ymax=_axis_limit(opts, "ymax"),
        open_frame=bool(opts.get("open_frame")),
        show_average=bool(opts.get("show_average")),
        running_avg=bool(opts.get("running_avg")),
        running_avg_window=int(_opt(opts, "running_avg_window", 0) or 0),
    )


def _parse_regions(spec: str | None) -> list[tuple[str, float, float, str | None]]:
    """Parse ``"Antigen:1-120:#1f77b4; Nanobody:121-250"`` into
    ``[(name, lo, hi, color|None), ...]``. Colour is optional. Entries that
    don't carry a valid ``lo-hi`` span are skipped."""
    if not spec:
        return []
    out: list[tuple[str, float, float, str | None]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) < 2:
            continue
        name = parts[0]
        rng = parts[1]
        color = parts[2] if len(parts) >= 3 and parts[2] else None
        if "-" not in rng:
            continue
        a, b = rng.split("-", 1)
        try:
            lo, hi = float(a), float(b)
        except ValueError:
            continue
        out.append((name, min(lo, hi), max(lo, hi), color))
    return out


def _parse_ranges(spec: str | None) -> list[tuple[float, float]]:
    """Parse ``"26-35,50-58,97-110"`` into ``[(lo, hi), ...]``."""
    if not spec:
        return []
    out: list[tuple[float, float]] = []
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok or "-" not in tok:
            continue
        a, b = tok.split("-", 1)
        try:
            lo, hi = float(a), float(b)
        except ValueError:
            continue
        out.append((min(lo, hi), max(lo, hi)))
    return out


# ---------------------------------------------------------------------------
# Plot from an existing data file (.dat / .xvg / .csv / .txt)
# ---------------------------------------------------------------------------


def parse_xy_table(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a 2-D numeric table out of a .dat/.xvg/.csv/.txt file body.

    Comment / metadata lines (those starting with ``# @ & ; %``, used by
    Post_MD, GROMACS .xvg, gnuplot, etc.) are skipped. Commas are treated as
    whitespace so CSV works too. The first column is taken as X and the second
    as Y; a single-column file falls back to X = row index.
    """
    rows: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#@&;%":
            continue
        try:
            vals = [float(v) for v in s.replace(",", " ").split()]
        except ValueError:
            continue  # stray non-numeric row (e.g. a column header) — skip
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError("No numeric data rows found in the file.")
    ncol = min(len(r) for r in rows)
    arr = np.array([r[:ncol] for r in rows], dtype=float)
    if arr.shape[1] == 1:
        return np.arange(arr.shape[0], dtype=float), arr[:, 0]
    return arr[:, 0], arr[:, 1]


def plot_from_data(
    datasets: list[tuple[str, str]],
    output_path: Path,
    opts: dict[str, Any],
    colors: list[str | None] | None = None,
) -> dict[str, Any]:
    """Render a line graph from already-collected data files.

    ``datasets`` is a list of ``(file_text, label)`` pairs — one curve each,
    overlaid on a single axes. Reuses the same :func:`_build_style` options
    (title, axis labels, open frame, grid, colour, …) as the live analyses,
    so an uploaded .dat plots identically to a freshly-computed one.
    """
    from post_md.plotting import plot_line, plot_lines_multi

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    for text, label in datasets:
        x, y = parse_xy_table(text)
        curves.append((x, y))
        labels.append(label)

    style = _build_style(opts)
    style.show_legend = len(curves) > 1 or bool(style.legend_label)

    # MM-GBSA mode: a dedicated plotter (per-residue ΔG bars or per-frame
    # binding-energy line). Uses the first uploaded file only.
    if str(_opt(opts, "plot_type", "") or "").lower() == "mmgbsa":
        from post_md.plotting import plot_mmgbsa
        if colors and colors[0]:
            style.color = colors[0]
        plot_mmgbsa(
            curves[0][0], curves[0][1], output_path, style=style,
            mode=str(_opt(opts, "mmgbsa_mode", "auto") or "auto"),
            default_title=labels[0] or "MM-GBSA",
        )
        return {"plot": output_path.name}

    if len(curves) == 1:
        if colors and colors[0]:
            style.color = colors[0]
        plot_line(
            curves[0][0], curves[0][1], output_path, style=style,
            default_xlabel="X", default_ylabel="Y",
            default_title=labels[0],
        )
    else:
        plot_lines_multi(
            curves, labels, output_path, style=style,
            default_xlabel="X", default_ylabel="Y",
            colors=colors,
        )
    return {"plot": output_path.name}


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
            default_title="RMSD over time",
        )
    return {"data": data_path.name, "plot": plot_path.name if not opts.get("no_plot") else None,
            "rows": int(len(values))}


def run_rmsf(topology: Path, trajectory: Path, workdir: Path, opts: dict) -> dict:
    from post_md.analysis.rmsf import rmsf as rmsf_fn
    from post_md.plotting import plot_line

    u = Universe.load(str(topology), str(trajectory))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords = ag.coordinates()
    atom_values = rmsf_fn(coords, weights=_maybe_mass_weights(ag))

    # Default convention (cpptraj `byres`, GROMACS gmx rmsf -res, etc.):
    # one RMSF value per residue, computed as the mass-weighted mean of
    # the per-atom fluctuations belonging to that residue. The user can
    # opt out via `per_atom=true` if they want raw per-atom data.
    per_atom = bool(opts.get("per_atom"))
    if per_atom:
        x_vals = ag.indices.astype(np.float64)
        values = atom_values
        x_label = "Atom index"
        data_header = "Atom index"
        title_default = "RMSF (per atom)"
    else:
        residue_ids = ag.residue_ids
        # Preserve the order residues first appear in the selection so the
        # x-axis flows naturally even for non-contiguous selections.
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
        title_default = "RMSF (per residue)"

    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")
    data_path = workdir / f"rmsf.{ext}"
    plot_path = workdir / "rmsf.png"
    table = np.column_stack([x_vals, values])
    _write_table(data_path, [data_header, "RMSF (A)"], table, title="RMSF")

    if not opts.get("no_plot"):
        style = _build_style(opts)
        plot_line(
            x_vals, values, plot_path, style=style,
            default_xlabel=x_label, default_ylabel="RMSF (Å)",
            default_title=title_default,
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


# ---------------------------------------------------------------------------
# Batch runner — one HTTP call, many analyses × many systems
# ---------------------------------------------------------------------------


def _slugify(label: str, default: str = "system") -> str:
    """Safe filename component derived from a user label."""
    import re
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label or default)).strip("_")
    return s or default


def _compute_rmsd(top: Path, traj: Path, opts: dict) -> tuple[np.ndarray, np.ndarray, str]:
    from post_md.analysis.rmsd import rmsd_trajectory

    u = Universe.load(str(top), str(traj))
    ag = u.select_atoms(_opt(opts, "selection", "name CA"))
    coords, times_ps = ag.coordinates_and_times()
    ref = coords[int(_opt(opts, "reference", 0))]
    values = rmsd_trajectory(coords, ref, weights=_maybe_mass_weights(ag))
    x, x_label = _derive_time_axis(times_ps, _opt(opts, "dt"), _opt(opts, "time_unit", "ps"))
    return x, values, x_label


def _compute_rmsf(top: Path, traj: Path, opts: dict) -> tuple[np.ndarray, np.ndarray, str]:
    from post_md.analysis.rmsf import rmsf as rmsf_fn

    u = Universe.load(str(top), str(traj))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords = ag.coordinates()
    atom_values = rmsf_fn(coords, weights=_maybe_mass_weights(ag))

    if bool(opts.get("per_atom")):
        return ag.indices.astype(np.float64), atom_values, "Atom index"

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
    return unique_resids.astype(np.float64), values, "Residue"


def _compute_rg(top: Path, traj: Path, opts: dict) -> tuple[np.ndarray, np.ndarray, str]:
    from post_md.analysis.rg import radius_of_gyration

    u = Universe.load(str(top), str(traj))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords, times_ps = ag.coordinates_and_times()
    masses = ag.masses
    if float(masses.sum()) == 0:
        masses = np.ones_like(masses)
    values = radius_of_gyration(coords, masses)
    x, x_label = _derive_time_axis(times_ps, _opt(opts, "dt"), _opt(opts, "time_unit", "ps"))
    return x, values, x_label


def _compute_sasa(top: Path, traj: Path, opts: dict) -> tuple[np.ndarray, np.ndarray, str]:
    from post_md.analysis.sasa import sasa_trajectory, vdw_radii_for

    u = Universe.load(str(top), str(traj))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords, times_ps = ag.coordinates_and_times()
    radii = vdw_radii_for(ag.elements)
    probe = float(_opt(opts, "probe_radius", 1.4))
    n_pts = int(_opt(opts, "n_sphere_points", 96))
    values = sasa_trajectory(coords, radii, probe_radius=probe, n_sphere_points=n_pts)
    x, x_label = _derive_time_axis(times_ps, _opt(opts, "dt"), _opt(opts, "time_unit", "ps"))
    return x, values, x_label


def _compute_hbond(top: Path, traj: Path, opts: dict) -> tuple[np.ndarray, np.ndarray, str]:
    from post_md.analysis.hbond import hbond_count_trajectory

    u = Universe.load(str(top), str(traj))
    ag = u.select_atoms(_opt(opts, "selection", "protein"))
    coords, times_ps = ag.coordinates_and_times()
    d_a = float(_opt(opts, "d_a_cutoff", 3.5))
    angle = float(_opt(opts, "angle_cutoff", 150.0))
    counts = hbond_count_trajectory(
        coords, ag.elements,
        d_a_cutoff=d_a, angle_cutoff_deg=angle,
    ).astype(np.float64)
    x, x_label = _derive_time_axis(times_ps, _opt(opts, "dt"), _opt(opts, "time_unit", "ps"))
    return x, counts, x_label


_OVERLAY_SPECS = {
    "rmsd":  dict(compute=_compute_rmsd, default_x="Time", ylabel="RMSD (Å)",
                  default_title="RMSD over time"),
    "rmsf":  dict(compute=_compute_rmsf, default_x="Residue", ylabel="RMSF (Å)",
                  default_title="RMSF"),
    "rg":    dict(compute=_compute_rg,   default_x="Time", ylabel="Rg (Å)",
                  default_title="Radius of gyration"),
    "sasa":  dict(compute=_compute_sasa, default_x="Time", ylabel="SASA (Å²)",
                  default_title="SASA over time"),
    "hbond": dict(compute=_compute_hbond, default_x="Time", ylabel="H-bonds",
                  default_title="Hydrogen-bond count over time"),
}


def _run_overlay(name: str, systems: list[dict], workdir: Path, opts: dict) -> dict:
    """Compute ``name`` on each system, write per-system data + one overlay plot.

    Single-system case still writes ``<name>.dat`` / ``<name>.png`` so the
    workspace file list keeps the old recognisable names. Multi-system
    case writes ``<name>_<label>.dat`` per system and one combined
    ``<name>.png`` with a legend.
    """
    from post_md.plotting import plot_line, plot_lines_multi

    spec = _OVERLAY_SPECS[name]
    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    per_system_colors: list[str | None] = []
    per_system: list[dict[str, Any]] = []
    x_label_default = spec["default_x"]

    for sys_ in systems:
        label = sys_["label"]
        top = Path(sys_["topology_path"])
        traj = Path(sys_["trajectory_path"])
        x, y, x_label = spec["compute"](top, traj, opts)
        x_label_default = x_label  # last writer wins, all systems share label
        slug = _slugify(label)
        data_name = f"{name}_{slug}.{ext}" if len(systems) > 1 else f"{name}.{ext}"
        data_path = workdir / data_name
        table = np.column_stack([x.astype(np.float64), y.astype(np.float64)])
        _write_table(data_path, [x_label, spec["ylabel"].split(" (")[0]], table, title=name.upper())
        curves.append((x, y))
        labels.append(label)
        per_system_colors.append(sys_.get("color") or None)
        per_system.append({"label": label, "data": data_name, "rows": int(len(y))})

    if opts.get("no_plot"):
        return {"systems": per_system, "plot": None}

    plot_path = workdir / f"{name}.png"
    style = _build_style(opts)
    # Antibody-aware RMSF: colour antigen / antibody / nanobody regions and
    # shade CDR loops. Only meaningful for a single system's residue axis.
    rmsf_regions = _parse_regions(_opt(opts, "regions")) if name == "rmsf" else []
    rmsf_cdrs = _parse_ranges(_opt(opts, "cdr")) if name == "rmsf" else []
    if name == "rmsf" and len(systems) == 1 and (rmsf_regions or rmsf_cdrs):
        from post_md.plotting import plot_rmsf_regions
        plot_rmsf_regions(
            curves[0][0], curves[0][1], plot_path, style=style,
            regions=rmsf_regions, cdr_ranges=rmsf_cdrs,
            default_xlabel=x_label_default, default_ylabel=spec["ylabel"],
            default_title=spec["default_title"],
        )
    elif len(systems) == 1:
        # Honour the per-system colour even for single-curve plots so the
        # legend pill colour matches the line.
        if per_system_colors[0]:
            style.color = per_system_colors[0]
        plot_line(
            curves[0][0], curves[0][1], plot_path, style=style,
            default_xlabel=x_label_default,
            default_ylabel=spec["ylabel"],
            default_title=spec["default_title"],
        )
    else:
        plot_lines_multi(
            curves, labels, plot_path, style=style,
            default_xlabel=x_label_default,
            default_ylabel=spec["ylabel"],
            default_title=spec["default_title"],
            colors=per_system_colors,
        )
    return {"systems": per_system, "plot": plot_path.name}


def _run_pca_per_system(systems: list[dict], workdir: Path, opts: dict) -> dict:
    from post_md.analysis.pca import pca_cartesian
    from post_md.plotting import plot_pca

    n_components = int(_opt(opts, "n_components", 10))
    per_system: list[dict[str, Any]] = []
    for sys_ in systems:
        label = sys_["label"]
        slug = _slugify(label)
        u = Universe.load(sys_["topology_path"], sys_["trajectory_path"])
        ag = u.select_atoms(_opt(opts, "selection", "name CA"))
        coords = ag.coordinates()
        result = pca_cartesian(coords, n_components=n_components,
                               weights=_maybe_mass_weights(ag))
        data_name = "pca.npz" if len(systems) == 1 else f"pca_{slug}.npz"
        data_path = workdir / data_name
        np.savez_compressed(
            data_path,
            mean=result.mean, eigenvalues=result.eigenvalues,
            components=result.components, projections=result.projections,
            selection_indices=ag.indices, atom_names=ag.names,
        )
        plot_name = "pca.png" if len(systems) == 1 else f"pca_{slug}.png"
        plot_path: str | None = None
        if not opts.get("no_plot"):
            style = _build_style(opts, default_figsize=(12.0, 4.5))
            plot_pca(result.eigenvalues, result.projections, workdir / plot_name, style=style)
            plot_path = plot_name
        per_system.append({
            "label": label,
            "data": data_name,
            "plot": plot_path,
            "scree": [float(v) for v in result.scree()],
            "cumulative": [float(v) for v in np.cumsum(result.scree())],
        })
    return {"systems": per_system}


def _run_cluster_per_system(systems: list[dict], workdir: Path, opts: dict) -> dict:
    from post_md.analysis.clustering import kmeans_cluster
    from post_md.plotting import plot_clusters

    n_pcs_req = int(_opt(opts, "n_pcs", 10))
    k = int(_opt(opts, "k", 5))
    seed = int(_opt(opts, "seed", 0))
    ext = str(_opt(opts, "output_ext", "dat")).lstrip(".")
    per_system: list[dict[str, Any]] = []

    for sys_ in systems:
        label = sys_["label"]
        slug = _slugify(label)
        pca_name = "pca.npz" if len(systems) == 1 else f"pca_{slug}.npz"
        pca_path = workdir / pca_name
        if not pca_path.exists():
            raise FileNotFoundError(
                f"{pca_name} missing — tick PCA in the same batch for {label!r} first."
            )
        data = np.load(pca_path)
        projections = data["projections"]
        n_pcs = min(n_pcs_req, projections.shape[1])
        result = kmeans_cluster(projections[:, :n_pcs], k=k, seed=seed)

        data_name = f"clusters.{ext}" if len(systems) == 1 else f"clusters_{slug}.{ext}"
        data_path = workdir / data_name
        table = np.column_stack([
            np.arange(len(result.labels)).astype(np.float64),
            result.labels.astype(np.float64),
        ])
        _write_table(data_path, ["Frame", "Cluster"], table, title="K-means clusters")

        rep_pdb_name = None
        if opts.get("rep_pdb"):
            from post_md.cli.main import _write_pdb_models
            u = Universe.load(sys_["topology_path"], sys_["trajectory_path"])
            rep_name = "reps.pdb" if len(systems) == 1 else f"reps_{slug}.pdb"
            _write_pdb_models(
                str(workdir / rep_name), u,
                result.representative_frames, data["selection_indices"],
            )
            rep_pdb_name = rep_name

        plot_name = "clusters.png" if len(systems) == 1 else f"clusters_{slug}.png"
        plot_path: str | None = None
        if not opts.get("no_plot"):
            style = _build_style(opts, default_figsize=(7.5, 6.0))
            plot_clusters(
                projections=projections, labels=result.labels,
                centers=result.centers,
                representative_frames=result.representative_frames,
                output_path=workdir / plot_name, style=style,
            )
            plot_path = plot_name
        per_system.append({
            "label": label,
            "data": data_name,
            "plot": plot_path,
            "rep_pdb": rep_pdb_name,
            "k": k,
            "n_pcs": n_pcs,
        })
    return {"systems": per_system}


def run_batch(systems: list[dict], analyses: dict[str, dict], workdir: Path) -> dict[str, Any]:
    """Orchestrate every checked analysis across every selected system.

    Returns one entry per analysis; overlay analyses (rmsd/rmsf/rg) carry
    a single combined plot when len(systems) > 1, per-system analyses
    (pca/cluster) carry one plot per system. Cluster is always run after
    pca within the same batch so its dependency on pca.npz is satisfied.
    """
    out: dict[str, Any] = {}
    # Strict order: overlay analyses first, then PCA, then Cluster (so it
    # can read the pca*.npz files we just wrote).
    for name in ("rmsd", "rmsf", "rg", "sasa", "hbond"):
        if name in analyses:
            out[name] = _run_overlay(name, systems, workdir, analyses[name] or {})
    if "pca" in analyses:
        out["pca"] = _run_pca_per_system(systems, workdir, analyses["pca"] or {})
    if "cluster" in analyses:
        out["cluster"] = _run_cluster_per_system(systems, workdir, analyses["cluster"] or {})
    return out


RUNNERS = {
    "rmsd": run_rmsd,
    "rmsf": run_rmsf,
    "rg": run_rg,
    "pca": run_pca,
    "cluster": run_cluster,
}
