"""Smoke tests for RMSD / RMSF / Rg / PCA / clustering."""

from __future__ import annotations

import numpy as np

from post_md.analysis.clustering import kmeans_cluster
from post_md.analysis.pca import pca_cartesian
from post_md.analysis.rg import radius_of_gyration
from post_md.analysis.rmsd import rmsd_trajectory
from post_md.analysis.rmsf import rmsf


def test_rmsd_zero_for_self(small_universe):
    coords = small_universe.trajectory.coordinates()
    values = rmsd_trajectory(coords, coords[0])
    assert values[0] < 1e-6
    assert np.all(values >= 0)


def test_rmsf_nonnegative(small_universe):
    coords = small_universe.trajectory.coordinates()
    values = rmsf(coords)
    assert values.shape == (coords.shape[1],)
    assert np.all(values >= 0)


def test_rg_constant_for_static_structure():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 3))
    coords = np.broadcast_to(base, (5, 10, 3)).copy()
    masses = np.ones(10)
    values = radius_of_gyration(coords, masses)
    assert np.allclose(values, values[0])


def test_pca_recovers_anisotropic_variance():
    """An internal collective mode along x should dominate PC1.

    Pure rigid translations are removed by Kabsch alignment, so the test
    must drive *relative* motion: atom 0 anchored, atom 1 oscillates along x.
    """
    rng = np.random.default_rng(0)
    n_frames, n_atoms = 200, 5
    base = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0],
         [9.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
    )
    coords = np.broadcast_to(base, (n_frames, n_atoms, 3)).copy()
    drive = rng.normal(scale=1.0, size=n_frames)
    coords[:, 1, 0] += drive                          # internal mode on atom 1
    coords += rng.normal(scale=0.02, size=coords.shape)

    result = pca_cartesian(coords, n_components=3)
    scree = result.scree()
    assert scree[0] > 0.9                              # most variance on PC1
    assert result.projections.shape == (n_frames, 3)


def test_kmeans_recovers_two_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-5, size=(50, 2))
    b = rng.normal(loc=+5, size=(50, 2))
    proj = np.vstack([a, b])

    result = kmeans_cluster(proj, k=2, seed=0)
    # Labels for first 50 should be uniform, for second 50 uniform (but cluster id arbitrary)
    first = result.labels[:50]
    second = result.labels[50:]
    assert (first == first[0]).all()
    assert (second == second[0]).all()
    assert first[0] != second[0]
    assert result.representative_frames.shape == (2,)
    assert (result.representative_frames >= 0).all()
