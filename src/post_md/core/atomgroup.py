"""AtomGroup: a view over a subset of atoms in a Universe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from post_md.core.universe import Universe


class AtomGroup:
    def __init__(self, universe: Universe, indices: np.ndarray):
        self.universe = universe
        self.indices = np.asarray(indices, dtype=np.int64)

    @property
    def n_atoms(self) -> int:
        return int(self.indices.size)

    def __len__(self) -> int:
        return self.n_atoms

    @property
    def names(self) -> np.ndarray:
        return self.universe.topology.atom_names[self.indices]

    @property
    def elements(self) -> np.ndarray:
        return self.universe.topology.elements[self.indices]

    @property
    def residue_ids(self) -> np.ndarray:
        return self.universe.topology.residue_ids[self.indices]

    @property
    def residue_names(self) -> np.ndarray:
        return self.universe.topology.residue_names[self.indices]

    @property
    def masses(self) -> np.ndarray:
        return self.universe.topology.masses[self.indices]

    @property
    def charges(self) -> np.ndarray:
        return self.universe.topology.charges[self.indices]

    def coordinates(self) -> np.ndarray:
        """Return (n_frames, n_atoms, 3) for atoms in this group."""
        return self.universe.trajectory.coordinates(selection=self.indices)

    def coordinates_and_times(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (coords (n_frames, n_atoms, 3), times (n_frames,) in ps)."""
        return self.universe.trajectory.coordinates_and_times(selection=self.indices)

    def __repr__(self) -> str:
        return f"<AtomGroup n_atoms={self.n_atoms}>"
