"""Superposition algorithms — Kabsch (SVD), Theobald QCP (batched), helpers.

The Theobald QCP variant is the workhorse for per-frame RMSD / alignment:
it folds the optimal rotation into a 4×4 symmetric eigenproblem so the whole
trajectory can be reduced to a single :func:`numpy.linalg.eigh` call instead
of one SVD per frame. The Kabsch SVD path is kept for the few callers that
need a 3×3 rotation matrix (and for the pre-existing tests).

References:
- Theobald, D. L. (2005). Rapid calculation of RMSDs using a quaternion-
  based characteristic polynomial. Acta Cryst. A 61, 478–480.
- Liu, P.; Agrafiotis, D. K.; Theobald, D. L. (2010). Fast determination of
  the optimal rotational matrix for macromolecular superpositions.
  J. Comput. Chem. 31, 1561–1563.
"""

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


# ---------------------------------------------------------------------------
# Theobald QCP — batched RMSD and rotation matrices
# ---------------------------------------------------------------------------


def _qcp_k_matrix(M: np.ndarray) -> np.ndarray:
    """Build the 4x4 Theobald K matrix from the cross-covariance ``M`` (..., 3, 3).

    The eigenvalue spectrum of K depends only on M; the largest eigenvalue
    gives the optimal squared cosine-sum used by the QCP RMSD formula, and
    its eigenvector is the optimal unit quaternion.
    """
    sxx, sxy, sxz = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    syx, syy, syz = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    szx, szy, szz = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]

    out = np.empty(M.shape[:-2] + (4, 4), dtype=M.dtype)
    out[..., 0, 0] =  sxx + syy + szz
    out[..., 0, 1] =  syz - szy
    out[..., 0, 2] =  szx - sxz
    out[..., 0, 3] =  sxy - syx
    out[..., 1, 0] = out[..., 0, 1]
    out[..., 1, 1] =  sxx - syy - szz
    out[..., 1, 2] =  sxy + syx
    out[..., 1, 3] =  szx + sxz
    out[..., 2, 0] = out[..., 0, 2]
    out[..., 2, 1] = out[..., 1, 2]
    out[..., 2, 2] = -sxx + syy - szz
    out[..., 2, 3] =  syz + szy
    out[..., 3, 0] = out[..., 0, 3]
    out[..., 3, 1] = out[..., 1, 3]
    out[..., 3, 2] = out[..., 2, 3]
    out[..., 3, 3] = -sxx - syy + szz
    return out


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a (..., 4) unit quaternion (w, x, y, z) to a (..., 3, 3) rotation."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _prepare_qcp_inputs(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Centre coords/reference and pre-compute weights.

    Returns:
      centered_coords (n_frames, n_atoms, 3) float64
      centered_ref    (n_atoms, 3)           float64
      coord_coms      (n_frames, 3)          float64  — for re-translating
      w               (n_atoms,) or None     float64
      w_sum           scalar
    """
    coords = np.asarray(coords, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must be (n_frames, n_atoms, 3)")
    if reference.shape != coords.shape[1:]:
        raise ValueError("reference must match (n_atoms, 3) of coords")

    if weights is None:
        w = None
        w_sum = float(coords.shape[1])
        coms = coords.mean(axis=1)                       # (n_frames, 3)
        ref_com = reference.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (coords.shape[1],):
            raise ValueError("weights must have shape (n_atoms,)")
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("Sum of weights must be positive")
        wn = w / w_sum
        coms = np.einsum("a,fai->fi", wn, coords)       # (n_frames, 3)
        ref_com = wn @ reference

    centered_coords = coords - coms[:, None, :]
    centered_ref = reference - ref_com
    return centered_coords, centered_ref, coms, w, w_sum


def qcp_rmsd_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Batched optimal-superposition RMSD for every frame in ``coords``.

    coords: (n_frames, n_atoms, 3). reference: (n_atoms, 3). Returns
    (n_frames,) RMSDs after the rotation that minimises each frame's
    distance to the reference. No per-frame Python loop — the whole
    batch is reduced to one ``eigvalsh`` call on a (n_frames, 4, 4)
    stack, which calls LAPACK once.
    """
    cP, cQ, _coms, w, w_sum = _prepare_qcp_inputs(coords, reference, weights)
    n_atoms = cP.shape[1]

    if w is None:
        # M[f] = cP[f].T @ cQ
        M = np.einsum("fai,aj->fij", cP, cQ)
        gA = np.einsum("fai,fai->f", cP, cP)
        gB = float(np.einsum("ai,ai->", cQ, cQ))
        norm = float(n_atoms)
    else:
        wcP = cP * w[None, :, None]
        M = np.einsum("fai,aj->fij", wcP, cQ)
        gA = np.einsum("a,fai,fai->f", w, cP, cP)
        gB = float(np.einsum("a,ai,ai->", w, cQ, cQ))
        norm = w_sum

    K = _qcp_k_matrix(M)
    # eigvalsh returns ascending — largest is the last column.
    eigs = np.linalg.eigvalsh(K)
    lam = eigs[..., -1]
    msd = (gA + gB - 2.0 * lam) / norm
    np.clip(msd, 0.0, None, out=msd)            # guard tiny negatives from round-off
    return np.sqrt(msd)


def qcp_align_batch(
    coords: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``coords`` rotated frame-by-frame onto ``reference``.

    Translation is removed and the optimal rotation (Theobald QCP) is
    applied. The translation back is to ``reference``'s centroid, so the
    output sits in the reference frame (same convention as Kabsch).
    """
    cP, cQ, _coms, w, _w_sum = _prepare_qcp_inputs(coords, reference, weights)

    if w is None:
        M = np.einsum("fai,aj->fij", cP, cQ)
    else:
        wcP = cP * w[None, :, None]
        M = np.einsum("fai,aj->fij", wcP, cQ)

    K = _qcp_k_matrix(M)
    eigvals, eigvecs = np.linalg.eigh(K)
    # Largest eigenvalue is the last column.
    q = eigvecs[..., -1]                              # (n_frames, 4)
    # Make the quaternion canonical (w >= 0) for numerical determinism.
    q = q * np.where(q[..., 0:1] < 0, -1.0, 1.0)
    R = _quat_to_rotmat(q)                            # (n_frames, 3, 3)

    # cP and reference are centred; rotate cP and translate back to reference centroid.
    rotated = np.einsum("fij,faj->fai", R, cP)
    # reference centroid in original frame:
    if weights is None:
        ref_com = np.asarray(reference, dtype=np.float64).mean(axis=0)
    else:
        wn = np.asarray(weights, dtype=np.float64)
        wn = wn / wn.sum()
        ref_com = wn @ np.asarray(reference, dtype=np.float64)
    rotated += ref_com
    return rotated
