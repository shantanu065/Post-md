"""Shared pytest fixtures: synthetic topology + trajectory for unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from post_md.core.topology import Topology
from post_md.core.trajectory import Frame, Trajectory


class _InMemoryTrajectory(Trajectory):
    def __init__(self, coords: np.ndarray):
        # coords: (n_frames, n_atoms, 3)
        self._coords = np.asarray(coords, dtype=np.float32)

    @property
    def n_atoms(self) -> int:
        return self._coords.shape[1]

    @property
    def n_frames(self) -> int:
        return self._coords.shape[0]

    def read_frame(self, index: int) -> Frame:
        return Frame(
            index=index, coordinates=self._coords[index].copy(), box=None, time=float(index)
        )


@pytest.fixture
def small_topology() -> Topology:
    """A 6-atom 'protein': 2 residues of (N, CA, C) backbone."""
    return Topology(
        atom_names=np.array(["N", "CA", "C", "N", "CA", "C"], dtype="U4"),
        elements=np.array(["N", "C", "C", "N", "C", "C"], dtype="U2"),
        residue_ids=np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
        residue_names=np.array(["ALA", "ALA", "ALA", "GLY", "GLY", "GLY"], dtype="U4"),
        masses=np.array([14.0, 12.0, 12.0, 14.0, 12.0, 12.0], dtype=np.float32),
        charges=np.zeros(6, dtype=np.float32),
        bonds=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32),
    )


@pytest.fixture
def small_trajectory():
    rng = np.random.default_rng(42)
    base = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0],
         [4.5, 0.0, 0.0], [6.0, 0.0, 0.0], [7.5, 0.0, 0.0]], dtype=np.float32,
    )
    coords = np.stack(
        [base + rng.normal(scale=0.05, size=base.shape).astype(np.float32) for _ in range(20)]
    )
    return _InMemoryTrajectory(coords)


@pytest.fixture
def small_universe(small_topology, small_trajectory):
    from post_md.core.universe import Universe

    return Universe(small_topology, small_trajectory)
