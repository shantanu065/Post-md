"""K-means clustering on PC-space projections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2


@dataclass
class ClusteringResult:
    labels: np.ndarray                    # (n_frames,)
    centers: np.ndarray                   # (k, n_features)
    representative_frames: np.ndarray     # (k,) medoid frame index per cluster


def kmeans_cluster(
    projections: np.ndarray, k: int = 5, seed: int = 0
) -> ClusteringResult:
    """Run k-means on projections; representative frame per cluster is the medoid."""
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim == 1:
        proj = proj[:, None]
    centers, labels = kmeans2(proj, k=int(k), seed=int(seed), minit="++")
    centers = np.asarray(centers, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    reps = np.full(int(k), -1, dtype=np.int64)
    for c in range(int(k)):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        d = np.sum((proj[members] - centers[c]) ** 2, axis=1)
        reps[c] = int(members[int(np.argmin(d))])

    return ClusteringResult(labels=labels, centers=centers, representative_frames=reps)
