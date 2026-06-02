"""GROMACS .gro format — single- or multi-frame ASCII coordinates (nm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from post_md.core.topology import Topology
from post_md.core.trajectory import Frame, Trajectory
from post_md.core.units import NM_TO_ANGSTROM


def _guess_element(name: str) -> str:
    s = name.strip().upper()
    if not s:
        return "X"
    head = s[1:] if s[0].isdigit() and len(s) > 1 else s
    return head[0] if head else "X"


def _parse_frame(lines: list[str], start: int):
    """Parse one frame starting at lines[start]. Returns frame_tuple, end_index."""
    title = lines[start].rstrip()
    n_atoms = int(lines[start + 1].strip())
    atom_lines = lines[start + 2 : start + 2 + n_atoms]
    if len(atom_lines) < n_atoms:
        raise ValueError(f"GRO frame truncated at line {start}")
    box_line = lines[start + 2 + n_atoms]

    res_ids = np.empty(n_atoms, dtype=np.int32)
    res_names = np.empty(n_atoms, dtype="U5")
    atom_names = np.empty(n_atoms, dtype="U5")
    coords = np.empty((n_atoms, 3), dtype=np.float32)

    for i, line in enumerate(atom_lines):
        res_ids[i] = int(line[0:5])
        res_names[i] = line[5:10].strip()
        atom_names[i] = line[10:15].strip()
        coords[i, 0] = float(line[20:28]) * NM_TO_ANGSTROM
        coords[i, 1] = float(line[28:36]) * NM_TO_ANGSTROM
        coords[i, 2] = float(line[36:44]) * NM_TO_ANGSTROM

    parts = box_line.split()
    vals = [float(p) * NM_TO_ANGSTROM for p in parts]
    if len(vals) == 3:
        box: np.ndarray | None = np.diag(vals).astype(np.float32)
    elif len(vals) == 9:
        # GROMACS triclinic layout: v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        xx, yy, zz, xy, xz, yx, yz, zx, zy = vals
        box = np.array(
            [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]], dtype=np.float32
        )
    elif not vals:
        box = None
    else:
        raise ValueError(f"Unexpected GRO box line: {box_line!r}")

    end = start + 3 + n_atoms
    return (atom_names, res_names, res_ids, coords, box, title), end


def read_gro_topology(path: str | Path) -> Topology:
    """Use a single-frame .gro file as topology source. Coordinates are discarded."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    (atom_names, res_names, res_ids, _coords, _box, _title), _end = _parse_frame(lines, 0)
    n_atoms = atom_names.size
    elements = np.array([_guess_element(n) for n in atom_names], dtype="U2")
    return Topology(
        atom_names=atom_names.astype("U4"),
        elements=elements,
        residue_ids=res_ids,
        residue_names=res_names.astype("U4"),
        masses=np.zeros(n_atoms, dtype=np.float32),
        charges=np.zeros(n_atoms, dtype=np.float32),
        bonds=np.empty((0, 2), dtype=np.int32),
    )


class GroTrajectory(Trajectory):
    """Multi-frame .gro reader (uncommon but valid)."""

    def __init__(self, path: str | Path, n_atoms: int):
        self.path = Path(path)
        self._n_atoms = int(n_atoms)
        self._frames: list[Frame] = []
        self._load()

    @classmethod
    def open(cls, path: str | Path, n_atoms: int) -> GroTrajectory:
        return cls(path, n_atoms)

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def read_frame(self, index: int) -> Frame:
        if not (0 <= index < self.n_frames):
            raise IndexError(index)
        return self._frames[index]

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        i = 0
        while i + 2 < len(lines):
            if not lines[i].strip() and not lines[i + 1].strip():
                break
            (atom_names, _, _, coords, box, _), j = _parse_frame(lines, i)
            if coords.shape[0] != self._n_atoms:
                raise ValueError(
                    f"GRO frame at line {i} has {coords.shape[0]} atoms; expected {self._n_atoms}"
                )
            idx = len(self._frames)
            self._frames.append(
                Frame(index=idx, coordinates=coords, box=box, time=float(idx))
            )
            i = j
