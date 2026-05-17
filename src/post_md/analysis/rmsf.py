"""Per-atom RMS fluctuation about the mean structure."""

from __future__ import annotations

import numpy as np

from post_md.analysis.alignment import kabsch


def rmsf(
    coords: np.ndarray,
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """coords: (n_frames, n_atoms, 3). Returns (n_atoms,) RMSF in Å.

    Aligns every frame to ``reference`` (or to the trajectory mean if ``None``),
    then computes per-atom standard deviation about the post-alignment mean.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_frames = coords.shape[0]

    if reference is None:
        reference = coords.mean(axis=0)
    else:
        reference = np.asarray(reference, dtype=np.float64)

    aligned = np.empty_like(coords)
    for i in range(n_frames):
        _, _, a = kabsch(coords[i], reference, weights=weights)
        aligned[i] = a

    mean = aligned.mean(axis=0)
    diff = aligned - mean
    return np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))
