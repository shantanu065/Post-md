"""Topology: per-atom metadata (names, residues, masses, charges, bonds)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Topology:
    atom_names: np.ndarray       # ('U4',) shape (n_atoms,)
    elements: np.ndarray         # ('U2',) shape (n_atoms,)
    residue_ids: np.ndarray      # int32, shape (n_atoms,) — 1-indexed
    residue_names: np.ndarray    # ('U4',) shape (n_atoms,)
    masses: np.ndarray           # float32, shape (n_atoms,) in amu
    charges: np.ndarray          # float32, shape (n_atoms,) in electron units
    bonds: np.ndarray            # int32, shape (n_bonds, 2) — 0-indexed atom pairs

    @property
    def n_atoms(self) -> int:
        return int(self.atom_names.shape[0])

    @property
    def n_residues(self) -> int:
        return int(np.unique(self.residue_ids).size)

    def __len__(self) -> int:
        return self.n_atoms

    def __repr__(self) -> str:
        return (
            f"<Topology n_atoms={self.n_atoms} "
            f"n_residues={self.n_residues} n_bonds={len(self.bonds)}>"
        )
