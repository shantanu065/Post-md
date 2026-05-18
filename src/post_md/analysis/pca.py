"""Cartesian PCA on aligned trajectory coordinates.

Two algorithmic upgrades vs. a textbook implementation:

* Per-frame Kabsch SVD alignment is replaced with the batched Theobald
  QCP routine — one LAPACK call for every frame's rotation matrix.
* The principal-component decomposition is computed with a truncated /
  randomized SVD whenever the user asks for fewer components than
  ``min(n_frames, 3·n_atoms)``. Full SVD on the centred trajectory
  matrix is ``O(min(m,n)^2 · max(m,n))`` which is intractable for long
  trajectories; the truncated path is ``O(m·n·k)`` and converges in
  seconds even on large systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from post_md.analysis.alignment import qcp_align_batch

# Threshold: when the requested k is below this share of min(m,n) the
# truncated SVD path is unambiguously faster than full SVD. Above it we
# fall back to numpy's full SVD because randomized SVD has overhead and
# truncated solvers behave poorly near the maximum rank.
_TRUNCATED_RATIO = 0.5

# Batch size for the QCP aligner — same trade-off as RMSF.
_PCA_ALIGN_BATCH = 1024


@dataclass
class PCAResult:
    mean: np.ndarray             # (n_atoms, 3)
    eigenvalues: np.ndarray      # (n_components,)
    components: np.ndarray       # (n_components, n_atoms, 3)
    projections: np.ndarray      # (n_frames, n_components)

    def scree(self) -> np.ndarray:
        """Fractional variance explained per component."""
        total = self.eigenvalues.sum()
        return self.eigenvalues / total if total > 0 else self.eigenvalues * 0.0

    def extreme_frames(self, pc: int, amplitude: float = 2.0, n: int = 11) -> np.ndarray:
        """Frames sampled from −amplitude·σ to +amplitude·σ along principal component `pc`."""
        sigma = float(np.sqrt(max(self.eigenvalues[pc], 0.0)))
        t = np.linspace(-amplitude * sigma, amplitude * sigma, n)
        mode = self.components[pc]
        return self.mean[None, :, :] + t[:, None, None] * mode[None, :, :]


def _truncated_svd(Xc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top-``k`` SVD components of centred matrix ``Xc``.

    Tries ``scipy.sparse.linalg.svds`` first (ARPACK / LOBPCG), which
    operates on ``Xc`` as a dense linear operator and is much faster
    than full SVD when ``k`` is small. Falls back to numpy's full SVD
    if scipy is unavailable or refuses to converge — that fallback is
    always correct, just slower.

    Returns ``(U, s, Vt)`` with singular values in **descending** order.
    """
    try:
        from scipy.sparse.linalg import svds
    except ImportError:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k]

    try:
        # svds returns ascending singular values; reverse them.
        U, s, Vt = svds(Xc, k=k, which="LM")
    except Exception:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k]

    order = np.argsort(s)[::-1]
    return U[:, order], s[order], Vt[order]


def pca_cartesian(
    coords: np.ndarray,
    n_components: int = 10,
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> PCAResult:
    """Compute Cartesian PCA on a trajectory.

    All frames are aligned to ``reference`` (or to frame 0 if ``None``) via
    the batched QCP rotor before computing the covariance. Returns a
    :class:`PCAResult` with the top-``n_components`` principal modes.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_frames, n_atoms, _ = coords.shape

    if reference is None:
        ref = coords[0]
    else:
        ref = np.asarray(reference, dtype=np.float64)

    # Align in chunks to keep peak memory bounded by _PCA_ALIGN_BATCH.
    aligned = np.empty_like(coords)
    for start in range(0, n_frames, _PCA_ALIGN_BATCH):
        end = min(start + _PCA_ALIGN_BATCH, n_frames)
        aligned[start:end] = qcp_align_batch(coords[start:end], ref, weights=weights)

    X = aligned.reshape(n_frames, -1)
    mean = X.mean(axis=0)
    Xc = X - mean

    max_k = min(Xc.shape)
    k = min(int(n_components), max_k)

    # scipy.sparse.linalg.svds also imposes k < min(m, n); when the user
    # asks for nearly all components we go straight to full SVD.
    if k >= max_k - 1 or k / max_k > _TRUNCATED_RATIO:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        s = s[:k]
        Vt = Vt[:k]
        U = U[:, :k]
    else:
        U, s, Vt = _truncated_svd(Xc, k)

    eigenvalues = (s ** 2) / max(n_frames - 1, 1)
    components = Vt.reshape(k, n_atoms, 3)
    projections = U * s

    return PCAResult(
        mean=mean.reshape(n_atoms, 3),
        eigenvalues=eigenvalues,
        components=components,
        projections=projections,
    )
