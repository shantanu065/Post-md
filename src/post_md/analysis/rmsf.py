"""Per-atom RMS fluctuation about the mean structure.

Uses the batched QCP aligner in chunks plus a Welford-style running
accumulator, so the analysis no longer materialises a full
``(n_frames, n_atoms, 3)`` aligned-coordinates array. Peak memory is
``O(chunk × n_atoms × 3)``, not ``O(n_frames × n_atoms × 3)``.
"""

from __future__ import annotations

import numpy as np

from post_md.analysis.alignment import qcp_align_batch

# How many frames to align in one batched QCP call. Larger = better LAPACK
# throughput, more transient memory. 1024 keeps the working set near a few
# tens of MB for typical protein systems while still amortising overhead.
_RMSF_BATCH = 1024


def rmsf(
    coords: np.ndarray,
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """coords: (n_frames, n_atoms, 3). Returns (n_atoms,) RMSF in Å.

    Aligns every frame to ``reference`` (or to the trajectory mean if
    ``None``) using the batched Theobald-QCP rotor, then accumulates a
    Welford-style sum of squared deviations about the post-alignment
    mean — single pass, no full aligned-trajectory allocation.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_frames, n_atoms, _ = coords.shape

    if reference is None:
        reference = coords.mean(axis=0)
    else:
        reference = np.asarray(reference, dtype=np.float64)

    # Welford accumulators: per-atom running mean and sum of squared deviations.
    # Shape (n_atoms, 3); xyz components are tracked independently and only
    # summed at the very end to give the radial fluctuation.
    mean = np.zeros((n_atoms, 3), dtype=np.float64)
    m2 = np.zeros((n_atoms, 3), dtype=np.float64)
    seen = 0

    for start in range(0, n_frames, _RMSF_BATCH):
        end = min(start + _RMSF_BATCH, n_frames)
        aligned = qcp_align_batch(coords[start:end], reference, weights=weights)
        for f in range(aligned.shape[0]):
            seen += 1
            delta = aligned[f] - mean
            mean += delta / seen
            delta2 = aligned[f] - mean
            m2 += delta * delta2

    # RMSF = sqrt( mean over frames of |r_i - <r_i>|^2 )
    #       = sqrt( sum_xyz M2_xyz / n_frames )
    if seen == 0:
        return np.zeros(n_atoms, dtype=np.float64)
    return np.sqrt(m2.sum(axis=1) / seen)
