"""Kabsch superposition + RMSD tests."""

from __future__ import annotations

import numpy as np
import pytest

from post_md.analysis.alignment import kabsch, rmsd


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(20, 3))
    R_true = _random_rotation(rng)
    t_true = rng.normal(size=3)
    mobile = (R_true @ ref.T).T + t_true

    R, t, aligned = kabsch(mobile, ref)
    assert aligned.shape == ref.shape
    assert np.allclose(aligned, ref, atol=1e-10)
    assert rmsd(aligned, ref) < 1e-10


def test_kabsch_weighted_alignment_does_not_increase_rmsd():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(15, 3))
    mobile = ref + rng.normal(scale=0.1, size=ref.shape)
    weights = rng.uniform(0.5, 2.0, size=15)

    _, _, aligned = kabsch(mobile, ref, weights=weights)
    assert rmsd(aligned, ref, weights=weights) <= rmsd(mobile, ref, weights=weights) + 1e-12


def test_kabsch_rejects_shape_mismatch():
    a = np.zeros((10, 3))
    b = np.zeros((9, 3))
    with pytest.raises(ValueError):
        kabsch(a, b)
