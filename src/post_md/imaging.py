"""Per-frame re-imaging for PBC-wrapped trajectories — pure-Python.

When a periodic-boundary MD trajectory is written without imagemoles /
nojump post-processing, individual molecules occasionally hop a box
length between consecutive frames as atoms wrap through the PBC. For
the protein this shows up as a sudden RMSD spike when a chain "splits"
across the box; for waters it makes the centroid leap around. Both
break Kabsch alignment.

This module emulates the parts of ``cpptraj autoimage`` we need without
calling cpptraj or any MD library: pure NumPy maths, our own NetCDF
reader, ``scipy.io.netcdf_file`` for writing the output trajectory.

Algorithm per frame, parameterised by an *anchor* atom set (default:
the protein):

1. Re-stitch the anchor — wrap each anchor atom to the closest periodic
   image of the first anchor atom. This makes the anchor contiguous
   even if it spans the box edge.
2. Translate the whole frame so the anchor centroid sits at the box
   centre. Kabsch alignment is translation-invariant so this is purely
   cosmetic for RMSD; it makes the file usable in viewers and lines up
   the periodic images for step 3.
3. For each non-anchor residue, translate its rigid body by an integer
   combination of box vectors so its centroid is in the primary image
   around the anchor centre. This stops waters / ions wandering off
   over the run and brings dimer partners back together.

Works for orthorhombic and general triclinic boxes (any 3×3 box matrix
whose columns are the cell vectors).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from post_md.core.selection import select
from post_md.core.universe import Universe


def _wrap_to_closest_image(deltas: np.ndarray, inv_box: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Given ``deltas`` (..., 3) in Cartesian and inverse / forward box
    matrices, return the deltas wrapped to ``[-0.5, 0.5]`` fractional
    coords (closest image)."""
    frac = deltas @ inv_box.T
    frac -= np.round(frac)
    return frac @ box.T


def find_molecules(bonds: np.ndarray, n_atoms: int) -> list[np.ndarray]:
    """Connected components of the bond graph — one entry per molecule.

    cpptraj's autoimage groups atoms by molecule (every covalently-bonded
    cluster) and images each one as an independent rigid unit. We use
    ``scipy.sparse.csgraph.connected_components`` to recover the same
    partition from the prmtop's ``BONDS_*`` sections.

    Returns a list of ``int64`` arrays of atom indices. Each array is
    sorted ascending so a downstream cumsum-walk follows topology order
    (which mirrors the bonded path on AMBER-laid-out trajectories).
    """
    if n_atoms <= 0:
        return []
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2) if bonds is not None else np.empty((0, 2), dtype=np.int64)
    if bonds.size == 0:
        # No bond info ⇒ one big "molecule" (legacy fallback).
        return [np.arange(n_atoms, dtype=np.int64)]

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([bonds[:, 0], bonds[:, 1]])
    cols = np.concatenate([bonds[:, 1], bonds[:, 0]])
    data = np.ones(rows.size, dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n_atoms, n_atoms))
    _, labels = connected_components(graph, directed=False)
    n_components = int(labels.max()) + 1 if labels.size else 0
    return [np.sort(np.where(labels == k)[0].astype(np.int64)) for k in range(n_components)]


def filter_molecules_to_kept(
    molecules_full: list[np.ndarray],
    keep_idx: np.ndarray,
) -> list[np.ndarray]:
    """Re-express full-system molecules in the kept-atom index space.

    For each molecule in the full topology, drop atoms that aren't in
    ``keep_idx`` and map the survivors to their position within
    ``keep_idx``. Molecules that lose every atom are removed entirely.
    """
    keep_pos = {int(a): i for i, a in enumerate(keep_idx.tolist())}
    result: list[np.ndarray] = []
    for mol in molecules_full:
        kept = [keep_pos[int(a)] for a in mol.tolist() if int(a) in keep_pos]
        if kept:
            result.append(np.sort(np.asarray(kept, dtype=np.int64)))
    return result


def _restitch_inplace(
    coords: np.ndarray,
    mol_indices: np.ndarray,
    box: np.ndarray,
    inv_box: np.ndarray,
) -> None:
    """In-place: walk ``mol_indices`` in order, wrapping each atom to the
    closest periodic image of the previous one. Vectorised via cumsum."""
    if mol_indices.size <= 1:
        return
    positions = coords[mol_indices].astype(np.float64)
    deltas = np.diff(positions, axis=0)
    frac = deltas @ inv_box.T
    frac -= np.round(frac)
    positions[1:] = positions[0] + np.cumsum(frac @ box.T, axis=0)
    coords[mol_indices] = positions.astype(coords.dtype, copy=False)


def _bond_walk_restitch(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Re-stitch a polymer by walking atoms in topology order.

    For each consecutive pair, the second atom is wrapped to the closest
    periodic image of the first. Because bonded atoms in a prmtop are
    typically 1-2 Å apart (always << box/2), the wrap can never pick
    the wrong image — unlike "wrap everything to atom 0" which fails on
    large proteins whose ends straddle PBC.

    Implementation is vectorised: ``cumsum`` of wrapped Δ along axis 0
    gives every atom's new position in one pass. Multi-chain inputs
    still work as long as consecutive atoms across the chain break are
    closer than ``box/2`` in the source frame (true for any reasonable
    AMBER/GROMACS layout)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n <= 1:
        return coords.copy()
    box64 = np.asarray(box, dtype=np.float64)
    inv_box = np.linalg.inv(box64)

    deltas = np.diff(coords, axis=0)
    frac = deltas @ inv_box.T
    frac -= np.round(frac)
    wrapped_deltas = frac @ box64.T

    out = np.empty_like(coords)
    out[0] = coords[0]
    out[1:] = coords[0] + np.cumsum(wrapped_deltas, axis=0)
    return out


def autoimage_frame(
    coords: np.ndarray,
    box: np.ndarray | None,
    anchor_mask: np.ndarray,
    residue_ids: np.ndarray | None = None,
    *,
    molecules: list[np.ndarray] | None = None,
    center_anchor: bool = True,
) -> np.ndarray:
    """Per-frame multi-molecule autoimage — cpptraj-style algorithm.

    Steps:
      1. **Re-stitch every molecule.** Each molecule's atoms are walked
         in index order; consecutive atoms are wrapped to closest image
         of the previous one. Bonded atoms are always ~1.5 Å apart so
         the wrap never picks the wrong image. Vectorised via cumsum.
      2. **Pick the primary anchor molecule** — the largest molecule
         that contains at least one anchor atom. cpptraj picks the
         "first solute" by default; largest-molecule is the same idea
         applied to whatever the user labelled as anchor.
      3. **Re-centre the primary molecule** so its anchor-atom centroid
         sits at the box centre (when ``center_anchor=True``).
      4. **Rigid-body image the rest.** Every other molecule is
         translated as a unit by an integer combination of box vectors
         that brings its centroid into the primary image around the box
         centre — protein chains, peptide partners, ions, waters all
         get the same treatment, exactly like cpptraj.

    ``residue_ids`` is accepted (for back-compat with older callers) but
    is only consulted when ``molecules`` is None: in that case the input
    is treated as one big molecule (no bond info available).
    """
    if box is None:
        return np.asarray(coords, dtype=np.float32)

    coords = np.asarray(coords, dtype=np.float64).copy()
    box64 = np.asarray(box, dtype=np.float64)
    inv_box = np.linalg.inv(box64)
    n_atoms = coords.shape[0]

    if molecules is None or not molecules:
        molecules = [np.arange(n_atoms, dtype=np.int64)]

    # 1. Re-stitch every molecule.
    for mol in molecules:
        _restitch_inplace(coords, mol, box64, inv_box)

    # 2. Identify primary anchor molecule (largest one overlapping anchor_mask).
    anchor_idx = np.nonzero(anchor_mask)[0]
    if anchor_idx.size == 0:
        return coords.astype(np.float32)
    primary_id = -1
    primary_size = -1
    for k, mol in enumerate(molecules):
        if mol.size > primary_size and np.any(anchor_mask[mol]):
            primary_id = k
            primary_size = mol.size
    if primary_id < 0:
        return coords.astype(np.float32)

    # 3. Re-centre the primary molecule so its anchor atoms (only the
    # anchor subset, not the whole molecule) sit at the box centre.
    primary_mol = molecules[primary_id]
    anchor_in_primary = primary_mol[anchor_mask[primary_mol]]
    anchor_centre = coords[anchor_in_primary].mean(axis=0)
    box_centre = 0.5 * box64.sum(axis=0)
    if center_anchor:
        shift_primary = box_centre - anchor_centre
        coords[primary_mol] += shift_primary
        anchor_centre = box_centre

    # 4. Rigid-body image every other molecule to closest periodic image
    # of the (now box-centred) anchor centroid.
    for k, mol in enumerate(molecules):
        if k == primary_id:
            continue
        mol_centre = coords[mol].mean(axis=0)
        delta = mol_centre - anchor_centre
        frac = inv_box @ delta
        shift_int = np.round(frac)
        if shift_int.any():
            cart_shift = shift_int @ box64.T
            coords[mol] -= cart_shift

    return coords.astype(np.float32)


def autoimage_trajectory(
    topology_path: str | Path,
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    anchor_selection: str = "protein",
    output_basename: str | None = None,
    chunk_frames: int = 256,
    center_anchor: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path, dict]:
    """Re-image every frame of a trajectory and write to a new NetCDF.

    Returns ``(topology_path, new_trajectory_path, summary)``. The topology
    is unchanged (no atom subset, no renumbering) so the original prmtop
    works with the new trajectory. ``summary`` reports the anchor count
    and any frames that had no box (and were copied through unchanged).
    """
    from scipy.io import netcdf_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_basename is None:
        output_basename = Path(trajectory_path).stem + "-autoimaged"

    u = Universe.load(str(topology_path), str(trajectory_path))
    n_atoms = int(u.topology.n_atoms)
    n_frames = int(u.trajectory.n_frames)

    anchor_indices = select(u.topology, anchor_selection)
    if anchor_indices.size == 0:
        raise ValueError(
            f"Anchor selection {anchor_selection!r} matched 0 atoms — "
            "pick something like 'protein' or a residue range."
        )
    anchor_mask = np.zeros(n_atoms, dtype=bool)
    anchor_mask[anchor_indices] = True
    residue_ids = np.asarray(u.topology.residue_ids, dtype=np.int64)
    # Build the bond-graph molecules once. Empty bond list ⇒ one big
    # molecule (legacy fallback inside autoimage_frame).
    molecules = find_molecules(u.topology.bonds, n_atoms)

    traj_out = output_dir / f"{output_basename}.nc"
    if traj_out.exists():
        traj_out.unlink()

    co_slab = getattr(u.trajectory, "coordinates_slab", None)
    t_slab = getattr(u.trajectory, "times_slab", None)
    has_slab = callable(co_slab) and callable(t_slab)

    n_no_box = 0
    with netcdf_file(str(traj_out), "w", version=2) as nc:
        nc.Conventions = b"AMBER"
        nc.ConventionVersion = b"1.0"
        nc.program = b"post_md"
        nc.programVersion = b"0.1"
        nc.title = b"autoimaged_by_post_md"

        nc.createDimension("frame", None)
        nc.createDimension("spatial", 3)
        nc.createDimension("atom", n_atoms)
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
            # Pull the chunk's coordinates in one shot for AMBER NetCDF; fall back
            # to per-frame reads for non-slab readers (XTC / TRR).
            if has_slab:
                coords_chunk = co_slab(start, stop, selection=None)
                times_chunk = t_slab(start, stop)
            else:
                coords_chunk = np.empty((stop - start, n_atoms, 3), dtype=np.float32)
                times_chunk = np.empty(stop - start, dtype=np.float64)
                for i in range(start, stop):
                    f = u.trajectory.read_frame(i)
                    coords_chunk[i - start] = f.coordinates
                    times_chunk[i - start] = float(f.time)

            box_lengths = np.zeros((stop - start, 3), dtype=np.float32)
            box_angles = np.full((stop - start, 3), 90.0, dtype=np.float32)
            for j in range(stop - start):
                frame = u.trajectory.read_frame(start + j)
                box = frame.box
                if box is None:
                    n_no_box += 1
                    # Frame has no PBC info — leave coords untouched. Box vars
                    # default to (0,0,0) lengths so the absence is honest.
                    continue
                coords_chunk[j] = autoimage_frame(
                    coords_chunk[j], box, anchor_mask, residue_ids,
                    molecules=molecules,
                    center_anchor=center_anchor,
                )
                box_arr = np.asarray(box, dtype=np.float64)
                box_lengths[j] = np.linalg.norm(box_arr, axis=1)
                if not np.allclose(box_arr, np.diag(np.diag(box_arr)), atol=1e-4):
                    a, b, c = box_arr[0], box_arr[1], box_arr[2]
                    def _ang(u_, v_):
                        nu = np.linalg.norm(u_)
                        nv = np.linalg.norm(v_)
                        if nu == 0 or nv == 0:
                            return 90.0
                        cos = float(np.dot(u_, v_) / (nu * nv))
                        return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))
                    box_angles[j] = (_ang(b, c), _ang(a, c), _ang(a, b))

            n_chunk = stop - start
            v_c[write_cursor : write_cursor + n_chunk] = coords_chunk
            v_t[write_cursor : write_cursor + n_chunk] = times_chunk.astype(np.float32)
            v_l[write_cursor : write_cursor + n_chunk] = box_lengths
            v_a[write_cursor : write_cursor + n_chunk] = box_angles
            write_cursor += n_chunk
            if progress is not None:
                progress(write_cursor, n_frames)

    summary = {
        "n_atoms": int(n_atoms),
        "n_frames": int(n_frames),
        "anchor_atoms": int(anchor_mask.sum()),
        "anchor_selection": anchor_selection,
        "frames_without_box": int(n_no_box),
        "input_trajectory_bytes": int(Path(trajectory_path).stat().st_size),
        "output_trajectory_bytes": int(traj_out.stat().st_size),
        "output_trajectory": str(traj_out),
    }
    return Path(topology_path), traj_out, summary
