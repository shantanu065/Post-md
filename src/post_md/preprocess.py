"""Trajectory preprocessing — strip waters / ions before analysis.

The bottleneck on a solvated 100k+ atom trajectory is not the analysis
math, it's the I/O: every frame the underlying NetCDF reader has to
seek across mostly-uninteresting water bytes. Stripping the solvent
once produces a much smaller file (typically 10-15x shrinkage), which
every subsequent RMSD / RMSF / PCA / cluster run reads in a fraction
of the time.

This module is pure-Python — no mdtraj, no MDAnalysis, no other MD
library. It uses the existing post_md NetCDF reader, the post_md
selection engine, ``scipy.io.netcdf_file`` for writing the output
trajectory, and our own :mod:`post_md.io.amber.prmtop_writer` for the
output topology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from post_md.core.universe import Universe
from post_md.io.amber.prmtop_writer import write_minimal_prmtop


# Common solvent / counterion residue labels seen across AMBER, GROMACS,
# CHARMM force fields. Matching is case-insensitive and trimmed.
SOLVENT_RESNAMES = frozenset({
    "WAT", "HOH", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "TP3", "TP4",
    "SPC", "SPCE", "T3P", "T4P", "T5P", "OPC", "OPC3", "OPC4",
})
ION_RESNAMES = frozenset({
    # AMBER / common
    "NA", "CL", "K", "MG", "ZN", "CA", "FE", "CU", "MN", "CO",
    "NA+", "CL-", "K+", "MG2+", "CA2+", "ZN2+", "FE2+", "FE3+",
    "Na+", "Cl-", "K+", "Mg2+", "Ca2+", "Zn2+",
    "RB", "CS", "BR", "I",
    # CHARMM
    "POT", "SOD", "CLA", "CES", "RUB", "CIO",
})
STRIP_DEFAULT = SOLVENT_RESNAMES | ION_RESNAMES


def _atoms_to_keep(universe: Universe, strip_resnames: set[str]) -> np.ndarray:
    """Return the indices of atoms NOT in any stripped residue."""
    strip_upper = {r.strip().upper() for r in strip_resnames}
    res_names = np.char.upper(np.char.strip(universe.topology.residue_names))
    keep = ~np.isin(res_names, list(strip_upper))
    return np.nonzero(keep)[0].astype(np.int64)


def prepare_trajectory(
    topology_path: str | Path,
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    anchor_selection: str = "protein",
    strip_resnames: set[str] | None = None,
    output_basename: str | None = None,
    chunk_frames: int = 256,
    autoimage: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path, dict]:
    """Single-pass autoimage + strip — the runner's default pre-flight.

    Combined and *narrowed*: the slab read pulls only the atoms that
    survive ``strip_resnames`` (typically 5–10% of a solvated system),
    and box matrices are read in one batched mmap stride instead of
    200k separate disk seeks. On a 200k-frame, 120k-atom AMBER NetCDF
    the difference is ~10× wall-time and ~20× less RAM per chunk.

    Per frame:
      1. Read the kept atoms (anchor + ligand + ions kept) — waters &
         everything in ``strip_resnames`` is skipped.
      2. Re-stitch the anchor (subset of kept) across PBC using its own
         coordinates; re-centre kept coords so the anchor centroid sits
         at the box centre.
      3. Append directly to the output NetCDF — no intermediate full-size
         write, no post-hoc subset.

    Returns ``(new_topology_path, new_trajectory_path, summary)``.
    """
    from scipy.io import netcdf_file
    from post_md.core.selection import select
    from post_md.imaging import (
        autoimage_frame,
        filter_molecules_to_kept,
        find_molecules,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_basename is None:
        output_basename = Path(trajectory_path).stem + "-prepared"

    universe = Universe.load(str(topology_path), str(trajectory_path))
    strip_set = strip_resnames if strip_resnames is not None else STRIP_DEFAULT
    keep_idx = _atoms_to_keep(universe, strip_set)
    n_total = universe.topology.n_atoms
    n_keep = int(keep_idx.size)
    n_frames = int(universe.trajectory.n_frames)
    if n_keep == 0:
        raise ValueError(
            f"No atoms left after stripping {sorted(strip_set)}; refusing to write empty topology."
        )

    anchor_indices_full = select(universe.topology, anchor_selection)
    if anchor_indices_full.size == 0:
        anchor_indices_full = keep_idx

    # Build anchor mask in the *kept* index space (position within keep_idx).
    keep_pos = {int(a): i for i, a in enumerate(keep_idx.tolist())}
    anchor_pos = [keep_pos[int(a)] for a in anchor_indices_full.tolist() if int(a) in keep_pos]
    if not anchor_pos:
        # Anchor was outside the kept set (e.g. user passed `resname WAT`).
        # Fall back to "anchor = everything kept" so we still re-stitch
        # the kept atoms across PBC.
        anchor_in_kept_idx = np.arange(n_keep, dtype=np.int64)
    else:
        anchor_in_kept_idx = np.asarray(sorted(anchor_pos), dtype=np.int64)
    anchor_mask_in_kept = np.zeros(n_keep, dtype=bool)
    anchor_mask_in_kept[anchor_in_kept_idx] = True

    # cpptraj-style multi-molecule pre-compute: connected components of
    # the full-system bond graph, then mapped to kept-atom positions.
    # Empty bond list (some prmtops omit them) falls back to "one big
    # molecule", which is exactly the legacy index-walk behaviour.
    molecules_full = find_molecules(universe.topology.bonds, n_total)
    molecules_kept = filter_molecules_to_kept(molecules_full, keep_idx)
    if not molecules_kept:
        molecules_kept = [np.arange(n_keep, dtype=np.int64)]

    # 1. Write topology with only kept atoms.
    top_out = output_dir / f"{output_basename}.prmtop"
    write_minimal_prmtop(top_out, universe.topology, indices=keep_idx)

    traj_out = output_dir / f"{output_basename}.nc"
    if traj_out.exists():
        traj_out.unlink()

    co_slab = getattr(universe.trajectory, "coordinates_slab", None)
    t_slab = getattr(universe.trajectory, "times_slab", None)
    b_slab = getattr(universe.trajectory, "boxes_slab", None)
    has_slab = callable(co_slab) and callable(t_slab)
    has_box_slab = callable(b_slab)

    n_no_box = 0
    with netcdf_file(str(traj_out), "w", version=2) as nc:
        nc.Conventions = b"AMBER"
        nc.ConventionVersion = b"1.0"
        nc.program = b"post_md"
        nc.programVersion = b"0.1"
        nc.title = b"prepared_by_post_md"

        nc.createDimension("frame", None)
        nc.createDimension("spatial", 3)
        nc.createDimension("atom", n_keep)
        nc.createDimension("cell_spatial", 3)
        nc.createDimension("cell_angular", 3)

        v_c = nc.createVariable("coordinates", "f", ("frame", "atom", "spatial"))
        v_c.units = b"angstrom"
        v_t = nc.createVariable("time", "f", ("frame",))
        v_t.units = b"picosecond"
        v_l = nc.createVariable("cell_lengths", "f", ("frame", "cell_spatial"))
        v_l.units = b"angstrom"
        v_a = nc.createVariable("cell_angles", "f", ("frame", "cell_angular"))
        v_a.units = b"degree"

        from post_md.utils import raise_if_cancelled

        write_cursor = 0
        for start in range(0, n_frames, chunk_frames):
            raise_if_cancelled()
            stop = min(start + chunk_frames, n_frames)
            n_chunk = stop - start

            if has_slab:
                coords_kept = co_slab(start, stop, selection=keep_idx)  # (n_chunk, n_keep, 3)
                times = t_slab(start, stop).astype(np.float32, copy=False)
            else:
                coords_kept = np.empty((n_chunk, n_keep, 3), dtype=np.float32)
                times = np.empty(n_chunk, dtype=np.float32)
                for i in range(start, stop):
                    f = universe.trajectory.read_frame(i)
                    coords_kept[i - start] = f.coordinates[keep_idx]
                    times[i - start] = float(f.time)

            if has_box_slab:
                boxes, has_box = b_slab(start, stop)
            else:
                # Fall back: per-frame reads. Used by non-AMBER readers.
                boxes = np.zeros((n_chunk, 3, 3), dtype=np.float32)
                has_box = np.zeros(n_chunk, dtype=bool)
                for j in range(n_chunk):
                    frame = universe.trajectory.read_frame(start + j)
                    if frame.box is not None:
                        boxes[j] = frame.box
                        has_box[j] = True

            # Vectorised re-stitch + centre, one frame at a time over the
            # kept atoms only (typically 5-10% of n_total).
            if autoimage:
                for j in range(n_chunk):
                    if not has_box[j]:
                        n_no_box += 1
                        continue
                    # Per-molecule autoimage (cpptraj-style): each chain
                    # or ligand is re-stitched independently, then rigid-
                    # body imaged to the closest periodic image of the
                    # primary anchor molecule's centroid.
                    coords_kept[j] = autoimage_frame(
                        coords_kept[j], boxes[j],
                        anchor_mask=anchor_mask_in_kept,
                        molecules=molecules_kept,
                        center_anchor=True,
                    )
            else:
                # User opted out of autoimage — strip only. Frames without
                # a box are still tallied for the summary so the UI can
                # surface "frames without PBC info" if relevant.
                for j in range(n_chunk):
                    if not has_box[j]:
                        n_no_box += 1

            # Recover cell_lengths / cell_angles for the output NetCDF.
            box_lengths = np.zeros((n_chunk, 3), dtype=np.float32)
            box_angles = np.full((n_chunk, 3), 90.0, dtype=np.float32)
            for j in range(n_chunk):
                if not has_box[j]:
                    continue
                box_arr = boxes[j].astype(np.float64)
                box_lengths[j] = np.linalg.norm(box_arr, axis=1)
                if not np.allclose(box_arr, np.diag(np.diag(box_arr)), atol=1e-4):
                    a, b, c = box_arr[0], box_arr[1], box_arr[2]
                    def _ang(u, v):
                        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
                        if nu == 0 or nv == 0:
                            return 90.0
                        cos = float(np.dot(u, v) / (nu * nv))
                        return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))
                    box_angles[j] = (_ang(b, c), _ang(a, c), _ang(a, b))

            v_c[write_cursor : write_cursor + n_chunk] = coords_kept
            v_t[write_cursor : write_cursor + n_chunk] = times
            v_l[write_cursor : write_cursor + n_chunk] = box_lengths
            v_a[write_cursor : write_cursor + n_chunk] = box_angles
            write_cursor += n_chunk
            if progress is not None:
                progress(write_cursor, n_frames)

    summary = {
        "input_atoms": int(n_total),
        "output_atoms": int(n_keep),
        "stripped_atoms": int(n_total - n_keep),
        "n_frames": int(n_frames),
        "anchor_atoms": int(anchor_indices_full.size),
        "frames_without_box": int(n_no_box),
        "input_trajectory_bytes": int(Path(trajectory_path).stat().st_size),
        "output_trajectory_bytes": int(traj_out.stat().st_size),
        "output_topology": str(top_out),
        "output_trajectory": str(traj_out),
        "autoimage": bool(autoimage),
    }
    return top_out, traj_out, summary


# Filename marker used to detect "this system was already prepared in a
# previous Run analysis click" — saves an expensive re-prep on subsequent
# runs unless the user explicitly resets.
PREPARED_MARKER = "-prepared_"


def strip_solvent(
    topology_path: str | Path,
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    strip_resnames: set[str] | None = None,
    output_basename: str | None = None,
    chunk_frames: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path, dict]:
    """Strip waters + ions from a trajectory + topology pair.

    Streams the trajectory in chunks of ``chunk_frames`` so that even a
    100 GB+ source file uses bounded RAM. ``progress(done, total)`` is
    called once per chunk (good for a status bar / log line).

    Returns ``(new_topology_path, new_trajectory_path, summary)``.
    ``summary`` carries before/after atom counts + bytes for the UI.
    """
    from scipy.io import netcdf_file  # SciPy is already a dep, not MD-specific

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_basename is None:
        output_basename = Path(trajectory_path).stem + "-stripped"

    universe = Universe.load(str(topology_path), str(trajectory_path))
    strip_set = strip_resnames if strip_resnames is not None else STRIP_DEFAULT
    keep_idx = _atoms_to_keep(universe, strip_set)
    n_keep = int(keep_idx.size)
    n_total = universe.topology.n_atoms
    if n_keep == 0:
        raise ValueError(
            f"No atoms left after stripping {sorted(strip_set)}. "
            "Refusing to write an empty topology."
        )
    if n_keep == n_total:
        # Nothing matched the strip set — still produce a copy, but tell
        # the caller so they can decide whether to surface a warning.
        pass

    # 1. Topology — slim prmtop with the kept atoms only.
    top_out = output_dir / f"{output_basename}.prmtop"
    write_minimal_prmtop(top_out, universe.topology, indices=keep_idx)

    # 2. Trajectory — stream frames in chunks, write to AMBER NetCDF.
    traj_out = output_dir / f"{output_basename}.nc"
    n_frames = int(universe.trajectory.n_frames)

    # Detect whether the reader exposes a fast slab/times path. Falls back
    # to the per-frame iterator if not (e.g., XTC / TRR).
    co_slab = getattr(universe.trajectory, "coordinates_slab", None)
    t_slab = getattr(universe.trajectory, "times_slab", None)
    has_slab = callable(co_slab) and callable(t_slab)

    if traj_out.exists():
        traj_out.unlink()
    with netcdf_file(str(traj_out), "w", version=2) as nc:
        nc.Conventions = b"AMBER"
        nc.ConventionVersion = b"1.0"
        nc.program = b"post_md"
        nc.programVersion = b"0.1"
        nc.title = b"stripped_by_post_md"

        nc.createDimension("frame", None)         # unlimited (record)
        nc.createDimension("spatial", 3)
        nc.createDimension("atom", n_keep)
        nc.createDimension("cell_spatial", 3)
        nc.createDimension("cell_angular", 3)

        v_c = nc.createVariable("coordinates", "f", ("frame", "atom", "spatial"))
        v_c.units = b"angstrom"
        v_t = nc.createVariable("time", "f", ("frame",))
        v_t.units = b"picosecond"
        v_l = nc.createVariable("cell_lengths", "f", ("frame", "cell_spatial"))
        v_l.units = b"angstrom"
        v_a = nc.createVariable("cell_angles", "f", ("frame", "cell_angular"))
        v_a.units = b"degree"

        write_cursor = 0
        for start in range(0, n_frames, chunk_frames):
            stop = min(start + chunk_frames, n_frames)
            if has_slab:
                # Fast path — strided slab read narrows to keep_idx in one shot.
                coords = co_slab(start, stop, selection=keep_idx)
                times = t_slab(start, stop)
                # cell_lengths / cell_angles aren't (yet) on a slab API;
                # fall back to per-frame iter for those.
                box_lengths = np.zeros((stop - start, 3), dtype=np.float32)
                box_angles = np.full((stop - start, 3), 90.0, dtype=np.float32)
                for i in range(start, stop):
                    f = universe.trajectory.read_frame(i)
                    if f.box is not None:
                        b = np.asarray(f.box, dtype=np.float32)
                        # Triclinic 3x3 → lengths + angles for round-trip.
                        lengths = np.linalg.norm(b, axis=1)
                        box_lengths[i - start] = lengths
                        if not np.allclose(b, np.diag(np.diag(b)), atol=1e-4):
                            # Recover angles from the triclinic basis.
                            a, bv, cv = b[0], b[1], b[2]
                            def _ang(u, v):
                                nu = np.linalg.norm(u); nv = np.linalg.norm(v)
                                if nu == 0 or nv == 0:
                                    return 90.0
                                cos = float(np.dot(u, v) / (nu * nv))
                                cos = max(-1.0, min(1.0, cos))
                                return float(np.degrees(np.arccos(cos)))
                            box_angles[i - start] = (_ang(bv, cv), _ang(a, cv), _ang(a, bv))
            else:
                coords = np.empty((stop - start, n_keep, 3), dtype=np.float32)
                times = np.empty(stop - start, dtype=np.float32)
                box_lengths = np.zeros((stop - start, 3), dtype=np.float32)
                box_angles = np.full((stop - start, 3), 90.0, dtype=np.float32)
                for i in range(start, stop):
                    f = universe.trajectory.read_frame(i)
                    coords[i - start] = f.coordinates[keep_idx]
                    times[i - start] = float(f.time)
                    if f.box is not None:
                        b = np.asarray(f.box, dtype=np.float32)
                        box_lengths[i - start] = np.linalg.norm(b, axis=1)

            n_chunk = stop - start
            v_c[write_cursor : write_cursor + n_chunk] = coords
            v_t[write_cursor : write_cursor + n_chunk] = times.astype(np.float32)
            v_l[write_cursor : write_cursor + n_chunk] = box_lengths
            v_a[write_cursor : write_cursor + n_chunk] = box_angles
            write_cursor += n_chunk
            if progress is not None:
                progress(write_cursor, n_frames)

    summary = {
        "input_atoms": int(n_total),
        "output_atoms": int(n_keep),
        "stripped_atoms": int(n_total - n_keep),
        "n_frames": int(n_frames),
        "input_trajectory_bytes": int(Path(trajectory_path).stat().st_size),
        "output_trajectory_bytes": int(traj_out.stat().st_size),
        "output_topology": str(top_out),
        "output_trajectory": str(traj_out),
    }
    return top_out, traj_out, summary
