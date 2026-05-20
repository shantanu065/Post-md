"""Hydrogen-bond detection — pure-Python geometric criteria.

Standard geometric definition: an h-bond exists when a donor heavy atom
(N / O bonded to H) and an acceptor heavy atom (N / O) satisfy

    distance(donor_heavy, acceptor)         < d_a_cutoff   (default 3.5 Å)
    angle(donor_heavy − H − acceptor)       > angle_cutoff (default 150°)

Equivalent in cosine form: ``cos(D-H-A) < cos(angle_cutoff)`` — for
angles near 180° (linear h-bond) the cosine is close to ``-1``.

Donors and acceptors are identified from the trajectory itself rather
than from a force-field bond list, because the slim prmtops we emit
during preprocessing don't carry bonds. The first frame's heavy ↔ H
neighbour table (≤ 1.3 Å separation) is used as the donor list for
every subsequent frame — that's correct as long as covalent bonds
don't rearrange during the simulation.
"""

from __future__ import annotations

import numpy as np


_HBOND_DONOR_ACCEPTOR_ELEMENTS = ("N", "O")


def find_donors_acceptors(
    coords: np.ndarray,
    elements: np.ndarray,
    *,
    xh_cutoff: float = 1.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify donor heavy / donor H / acceptor index arrays.

    Returns three int arrays:
      donor_heavy : (n_donors,)  heavy atom index of each D-H pair
      donor_h     : (n_donors,)  H atom index of each D-H pair
      acceptor    : (n_acc,)     N / O indices (all heavy atoms qualify)

    Multiple H's on one heavy atom yield multiple entries in the donor
    arrays (e.g. lysine -NH3 produces 3 rows).
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    elements = np.asarray(elements)
    if coords.shape[0] != elements.shape[0]:
        raise ValueError("coords and elements lengths disagree")

    heavy_mask = np.isin(elements, _HBOND_DONOR_ACCEPTOR_ELEMENTS)
    h_mask = (elements == "H")
    heavy_idx = np.nonzero(heavy_mask)[0]
    h_idx = np.nonzero(h_mask)[0]

    if heavy_idx.size == 0 or h_idx.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            heavy_idx.astype(np.int64),
        )

    tree_h = cKDTree(coords[h_idx])
    donor_heavy: list[int] = []
    donor_h: list[int] = []
    for hi, heavy in enumerate(heavy_idx):
        # H atoms within covalent-bond distance of this heavy atom.
        cands = tree_h.query_ball_point(coords[heavy], xh_cutoff)
        for k in cands:
            donor_heavy.append(int(heavy))
            donor_h.append(int(h_idx[k]))

    return (
        np.asarray(donor_heavy, dtype=np.int64),
        np.asarray(donor_h, dtype=np.int64),
        heavy_idx.astype(np.int64),
    )


def hbonds_in_frame(
    coords: np.ndarray,
    donor_heavy: np.ndarray,
    donor_h: np.ndarray,
    acceptor: np.ndarray,
    *,
    d_a_cutoff: float = 3.5,
    angle_cutoff_deg: float = 150.0,
) -> int:
    """Count h-bonds in a single frame using D-A distance + D-H-A angle."""
    if donor_heavy.size == 0 or acceptor.size == 0:
        return 0
    from scipy.spatial.distance import cdist

    donor_pos = coords[donor_heavy]
    h_pos = coords[donor_h]
    acc_pos = coords[acceptor]

    # Distance filter first — narrows down candidate (D, A) pairs.
    d_a = cdist(donor_pos, acc_pos)
    close = (d_a > 1e-6) & (d_a < d_a_cutoff)
    # Same atom isn't a candidate.
    same = donor_heavy[:, None] == acceptor[None, :]
    close &= ~same

    if not close.any():
        return 0

    i_d, j_a = np.nonzero(close)
    # D-H-A angle: vectors from H to D and H to A. Linear h-bond -> 180° -> cos ≈ -1.
    vec_hd = donor_pos[i_d] - h_pos[i_d]
    vec_ha = acc_pos[j_a] - h_pos[i_d]
    norms = np.linalg.norm(vec_hd, axis=1) * np.linalg.norm(vec_ha, axis=1)
    norms = np.maximum(norms, 1e-12)
    cos_dha = np.einsum("ij,ij->i", vec_hd, vec_ha) / norms
    cos_cut = float(np.cos(np.radians(angle_cutoff_deg)))
    return int((cos_dha < cos_cut).sum())


def hbond_count_trajectory(
    coords: np.ndarray,
    elements: np.ndarray,
    *,
    d_a_cutoff: float = 3.5,
    angle_cutoff_deg: float = 150.0,
    xh_cutoff: float = 1.3,
) -> np.ndarray:
    """H-bond count per frame across the trajectory. Returns (n_frames,)."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must be (n_frames, n_atoms, 3)")

    donor_heavy, donor_h, acceptor = find_donors_acceptors(
        coords[0], elements, xh_cutoff=xh_cutoff,
    )
    counts = np.empty(coords.shape[0], dtype=np.int64)
    for f in range(coords.shape[0]):
        counts[f] = hbonds_in_frame(
            coords[f], donor_heavy, donor_h, acceptor,
            d_a_cutoff=d_a_cutoff, angle_cutoff_deg=angle_cutoff_deg,
        )
    return counts
