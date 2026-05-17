"""Per-frame RMSD vs a reference structure."""

from __future__ import annotations

import numpy as np

from post_md.analysis.alignment import kabsch
from post_md.analysis.alignment import rmsd as _rmsd_pointwise


def rmsd_trajectory(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
    align: bool = True,
) -> np.ndarray:
    """Compute per-frame RMSD vs ``reference``.

    coords: (n_frames, n_atoms, 3). reference: (n_atoms, 3).
    Returns (n_frames,) RMSD values (Å).
    """
    coords = np.asarray(coords)
    reference = np.asarray(reference)
    n_frames = coords.shape[0]
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        if align:
            _, _, aligned = kabsch(coords[i], reference, weights=weights)
            out[i] = _rmsd_pointwise(aligned, reference, weights=weights)
        else:
            out[i] = _rmsd_pointwise(coords[i], reference, weights=weights)
    return out
