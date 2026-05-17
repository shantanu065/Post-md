"""Minimal PDB topology reader (ATOM/HETATM records only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from post_md.core.topology import Topology


def read_pdb_topology(path: str | Path) -> Topology:
    atom_names: list[str] = []
    res_names: list[str] = []
    res_ids: list[int] = []
    elements: list[str] = []

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_names.append(line[12:16].strip())
        res_names.append(line[17:20].strip())
        try:
            res_ids.append(int(line[22:26]))
        except ValueError:
            res_ids.append(0)
        el = line[76:78].strip() if len(line) >= 78 else ""
        if not el:
            el = atom_names[-1][0] if atom_names[-1] else "X"
        elements.append(el)

    n = len(atom_names)
    return Topology(
        atom_names=np.array(atom_names, dtype="U4"),
        elements=np.array(elements, dtype="U2"),
        residue_ids=np.array(res_ids, dtype=np.int32),
        residue_names=np.array(res_names, dtype="U4"),
        masses=np.zeros(n, dtype=np.float32),
        charges=np.zeros(n, dtype=np.float32),
        bonds=np.empty((0, 2), dtype=np.int32),
    )
