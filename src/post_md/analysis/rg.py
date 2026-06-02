"""Radius of gyration per frame."""

from __future__ import annotations

import numpy as np


def radius_of_gyration(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """coords: (n_frames, n_atoms, 3). masses: (n_atoms,). Returns (n_frames,) Rg in Å."""
    coords = np.asarray(coords, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if not np.any(masses > 0):
        masses = np.ones_like(masses)
    total = masses.sum()
    com = (masses[:, None] * coords).sum(axis=1) / total  # (n_frames, 3)
    diff = coords - com[:, None, :]
    r2 = np.sum(diff * diff, axis=2)  # (n_frames, n_atoms)
    return np.sqrt((masses * r2).sum(axis=1) / total)
