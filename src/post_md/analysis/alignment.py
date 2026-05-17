"""Kabsch superposition + unweighted RMSD helper."""

from __future__ import annotations

import numpy as np


def kabsch(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, t, aligned) mapping `mobile` onto `reference`.

    Both inputs are (n, 3). With `weights`, the alignment is mass-weighted.
    `aligned` = R @ (mobile - mobile_com) + reference_com.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and reference must be (N, 3) and same shape")

    if weights is None:
        w = np.ones(mobile.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive")

    mc = (w[:, None] * mobile).sum(axis=0) / w_sum
    rc = (w[:, None] * reference).sum(axis=0) / w_sum

    P = mobile - mc
    Q = reference - rc

    H = (w[:, None] * P).T @ Q
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3)
    D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = rc - R @ mc
    aligned = P @ R.T + rc
    return R, t, aligned


def rmsd(
    a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Pointwise RMSD between aligned coordinate sets (no alignment performed)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    sq = np.sum(diff * diff, axis=1)
    if weights is None:
        return float(np.sqrt(np.mean(sq)))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt((sq * w).sum() / w.sum()))
