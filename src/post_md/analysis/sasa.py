"""Solvent-accessible surface area (SASA) — pure-Python Shrake-Rupley.

Shrake & Rupley (1973): cover each atom's sphere (vdW radius + probe
radius) with quasi-uniform test points; a test point is "exposed" if no
*other* atom's sphere contains it. Per-atom SASA is then the fraction
of exposed points times the sphere area. Sum across atoms gives the
total SASA.

We use a Fibonacci-lattice point distribution (more isotropic than
spherical-coords sampling) and a scipy ``cKDTree`` neighbour search so
the per-atom point check is O(n_points × n_local_neighbours) instead of
O(n_points × n_atoms).
"""

from __future__ import annotations

import numpy as np


# Element → van der Waals radius (Å). Bondi-style values; matches what
# Shrake & Rupley used + the AMBER mbondi set. Unknown elements get a
# conservative 1.7 Å (carbon).
_VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Na": 2.27, "Mg": 1.73, "K": 2.75, "Ca": 2.31, "Zn": 1.39, "Fe": 1.94,
}


def vdw_radii_for(elements: np.ndarray) -> np.ndarray:
    """Map an array of element symbols to their vdW radii (Å)."""
    return np.fromiter(
        (_VDW_RADII.get(str(e).strip(), 1.70) for e in elements),
        dtype=np.float64,
        count=len(elements),
    )


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Quasi-uniform ``n`` points on the unit sphere (Fibonacci lattice)."""
    idx = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * idx
    sin_phi = np.sin(phi)
    return np.stack([sin_phi * np.cos(theta), sin_phi * np.sin(theta), np.cos(phi)], axis=-1)


def shrake_rupley(
    coords: np.ndarray,
    radii: np.ndarray,
    *,
    probe_radius: float = 1.4,
    n_sphere_points: int = 96,
) -> np.ndarray:
    """Per-atom SASA (Å²) for a single frame.

    coords: (n_atoms, 3) Cartesian positions in Å.
    radii:  (n_atoms,) vdW radii of each atom in Å.
    probe_radius: solvent probe (1.4 Å = water). Set to 0 for raw vdW SA.
    n_sphere_points: more points = smoother estimate, more cost. 96 is a
        good speed/accuracy balance; 256 is publication-grade.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if coords.shape[0] != radii.shape[0]:
        raise ValueError("coords and radii lengths disagree")

    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return np.zeros(0, dtype=np.float64)

    expanded = radii + probe_radius                # sphere radius for each atom
    sphere = _fibonacci_sphere(n_sphere_points)    # (n_points, 3) unit vectors

    # Neighbour-search radius is generous: 2 × (max expanded radius) covers
    # any atom whose sphere could possibly overlap atom i's sphere.
    max_exp = float(expanded.max())
    tree = cKDTree(coords)

    per_atom_area = 4.0 * np.pi * expanded * expanded   # 4πR² per atom
    point_weight = per_atom_area / n_sphere_points
    sasa = np.zeros(n_atoms, dtype=np.float64)

    for i in range(n_atoms):
        pts = coords[i] + sphere * expanded[i]      # (n_points, 3)
        # Candidates whose own sphere reaches atom i's surface.
        nbrs = tree.query_ball_point(coords[i], expanded[i] + max_exp)
        nbr_filtered = [j for j in nbrs if j != i]
        if not nbr_filtered:
            sasa[i] = per_atom_area[i]
            continue
        nbr = np.asarray(nbr_filtered, dtype=np.int64)
        nbr_coords = coords[nbr]
        nbr_exp_sq = expanded[nbr] ** 2

        # For each test point, distance² to every neighbour.
        # Broadcast: (n_points, n_nbr) — memory is O(n_points × n_nbr_local)
        # which is small (~96 × ~30 typical = 3000 doubles).
        diffs = pts[:, None, :] - nbr_coords[None, :, :]
        d2 = np.einsum("pnd,pnd->pn", diffs, diffs)
        buried = (d2 < nbr_exp_sq[None, :]).any(axis=1)
        sasa[i] = point_weight[i] * (~buried).sum()

    return sasa


def sasa_trajectory(
    coords: np.ndarray,
    radii: np.ndarray,
    *,
    probe_radius: float = 1.4,
    n_sphere_points: int = 96,
) -> np.ndarray:
    """Total SASA per frame (Å²). coords: (n_frames, n_atoms, 3)."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must be (n_frames, n_atoms, 3)")
    out = np.empty(coords.shape[0], dtype=np.float64)
    for f in range(coords.shape[0]):
        out[f] = shrake_rupley(
            coords[f], radii,
            probe_radius=probe_radius, n_sphere_points=n_sphere_points,
        ).sum()
    return out
