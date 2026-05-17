"""Cartesian PCA on aligned trajectory coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from post_md.analysis.alignment import kabsch


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


def pca_cartesian(
    coords: np.ndarray,
    n_components: int = 10,
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> PCAResult:
    """Compute Cartesian PCA on a trajectory.

    All frames are aligned to ``reference`` (or to frame 0 if ``None``) via
    Kabsch before computing the covariance. Returns a :class:`PCAResult`.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_frames, n_atoms, _ = coords.shape

    if reference is None:
        ref = coords[0]
    else:
        ref = np.asarray(reference, dtype=np.float64)

    aligned = np.empty_like(coords)
    for i in range(n_frames):
        _, _, a = kabsch(coords[i], ref, weights=weights)
        aligned[i] = a

    X = aligned.reshape(n_frames, -1)
    mean = X.mean(axis=0)
    Xc = X - mean

    max_k = min(Xc.shape)
    k = min(int(n_components), max_k)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    s = s[:k]
    Vt = Vt[:k]

    eigenvalues = (s ** 2) / max(n_frames - 1, 1)
    components = Vt.reshape(k, n_atoms, 3)
    projections = U[:, :k] * s

    return PCAResult(
        mean=mean.reshape(n_atoms, 3),
        eigenvalues=eigenvalues,
        components=components,
        projections=projections,
    )
