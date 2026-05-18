"""Per-frame RMSD vs a reference structure.

Uses the batched Theobald-QCP routine when alignment is enabled: every
frame's optimal rotation is reduced to one batched eigendecomposition,
which is dramatically faster than per-frame SVD-based Kabsch.
"""

from __future__ import annotations

import numpy as np

from post_md.analysis.alignment import qcp_rmsd_batch
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

    With ``align=True`` (default) each frame is optimally superposed on
    ``reference`` before measuring RMSD, using the QCP batched solver.
    With ``align=False`` the raw pointwise RMSD is returned per frame.
    """
    coords = np.asarray(coords)
    reference = np.asarray(reference)

    if align:
        return qcp_rmsd_batch(coords, reference, weights=weights)

    # No-alignment path: pointwise per-frame, fully vectorised.
    coords64 = coords.astype(np.float64, copy=False)
    ref64 = reference.astype(np.float64, copy=False)
    diff = coords64 - ref64
    sq = np.einsum("fai,fai->fa", diff, diff)
    if weights is None:
        return np.sqrt(sq.mean(axis=1))
    w = np.asarray(weights, dtype=np.float64)
    return np.sqrt((sq * w).sum(axis=1) / w.sum())


# Keep the legacy single-frame helper available for callers that want it.
__all__ = ["rmsd_trajectory", "_rmsd_pointwise"]
