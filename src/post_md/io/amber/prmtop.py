"""AMBER prmtop/parm7 topology parser (FORMAT-aware, pure Python).

Spec: https://ambermd.org/FileFormats.php#topology
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from post_md.core.topology import Topology

# Atomic-number → element. Biological + common ions; unknowns → 'X'.
_ELEMENT_TABLE: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 35: "Br", 53: "I",
}

# AMBER charges are stored multiplied by this constant; divide to recover electron units.
_AMBER_CHARGE_UNIT = 18.2223

_FORMAT_RE = re.compile(r"\(\s*(\d+)?\s*([aAEeFfIi])\s*(\d+)\s*(?:\.\s*\d+)?\s*\)")


def _parse_format(fmt: str) -> tuple[int, str, int]:
    """Parse '(20a4)' → (count, type_char, width)."""
    m = _FORMAT_RE.search(fmt)
    if not m:
        raise ValueError(f"Cannot parse FORMAT: {fmt!r}")
    count = int(m.group(1) or 1)
    type_ = m.group(2).lower()
    width = int(m.group(3))
    return count, type_, width


def _parse_section(lines: list[str], type_: str, width: int) -> list:
    out: list = []
    for line in lines:
        line = line.rstrip("\r\n")
        for off in range(0, len(line), width):
            chunk = line[off : off + width]
            if not chunk.strip():
                continue
            if type_ == "a":
                out.append(chunk.strip())
            elif type_ == "i":
                out.append(int(chunk))
            elif type_ in ("e", "f"):
                out.append(float(chunk))
            else:
                raise ValueError(f"Unsupported FORMAT type {type_!r}")
    return out


def _split_sections(text: str) -> dict[str, tuple[str, list[str]]]:
    sections: dict[str, tuple[str, list[str]]] = {}
    current_flag: str | None = None
    current_format: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith(("%VERSION", "%COMMENT")):
            continue
        if line.startswith("%FLAG"):
            if current_flag is not None and current_format is not None:
                sections[current_flag] = (current_format, buffer)
            current_flag = line[5:].strip()
            current_format = None
            buffer = []
        elif line.startswith("%FORMAT"):
            current_format = line.strip()
        else:
            buffer.append(line)
    if current_flag is not None and current_format is not None:
        sections[current_flag] = (current_format, buffer)
    return sections


def _guess_atomic_number(name: str) -> int:
    s = name.strip().upper()
    if not s:
        return 0
    head = s[1:] if s[0].isdigit() and len(s) > 1 else s
    table = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16}
    return table.get(head[0], 0)


def read_prmtop(path: str | Path) -> Topology:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections = _split_sections(text)

    def parse_section(flag: str):
        if flag not in sections:
            return None
        fmt, lines = sections[flag]
        _, type_, width = _parse_format(fmt)
        return _parse_section(lines, type_, width)

    pointers = parse_section("POINTERS")
    if pointers is None:
        raise ValueError(f"prmtop missing POINTERS section: {path}")
    n_atoms = int(pointers[0])
    n_residues = int(pointers[11])

    atom_names_list = parse_section("ATOM_NAME") or [f"X{i}" for i in range(n_atoms)]
    masses_list = parse_section("MASS") or [0.0] * n_atoms
    charges_list = parse_section("CHARGE") or [0.0] * n_atoms
    atomic_numbers = parse_section("ATOMIC_NUMBER")
    if atomic_numbers is None:
        atomic_numbers = [_guess_atomic_number(n) for n in atom_names_list]

    residue_label = parse_section("RESIDUE_LABEL") or [""] * n_residues
    residue_pointer = parse_section("RESIDUE_POINTER") or [1] * n_residues

    bonds_inc_h = parse_section("BONDS_INC_HYDROGEN") or []
    bonds_no_h = parse_section("BONDS_WITHOUT_HYDROGEN") or []

    if len(atom_names_list) != n_atoms:
        raise ValueError(
            f"ATOM_NAME has {len(atom_names_list)} entries, expected {n_atoms}"
        )

    atom_names = np.array(atom_names_list, dtype="U4")
    masses = np.array(masses_list, dtype=np.float32)
    charges = (np.array(charges_list, dtype=np.float32) / _AMBER_CHARGE_UNIT).astype(np.float32)
    elements = np.array([_ELEMENT_TABLE.get(int(z), "X") for z in atomic_numbers], dtype="U2")

    residue_ids = np.empty(n_atoms, dtype=np.int32)
    residue_names = np.empty(n_atoms, dtype="U4")
    starts = list(int(p) for p in residue_pointer) + [n_atoms + 1]
    for r in range(n_residues):
        a0 = starts[r] - 1
        a1 = starts[r + 1] - 1
        residue_ids[a0:a1] = r + 1
        residue_names[a0:a1] = residue_label[r]

    bond_pairs: list[tuple[int, int]] = []
    for entries in (bonds_inc_h, bonds_no_h):
        for k in range(0, len(entries), 3):
            i = int(entries[k]) // 3
            j = int(entries[k + 1]) // 3
            bond_pairs.append((i, j))
    bonds_arr = (
        np.asarray(bond_pairs, dtype=np.int32)
        if bond_pairs
        else np.empty((0, 2), dtype=np.int32)
    )

    return Topology(
        atom_names=atom_names,
        elements=elements,
        residue_ids=residue_ids,
        residue_names=residue_names,
        masses=masses,
        charges=charges,
        bonds=bonds_arr,
    )
