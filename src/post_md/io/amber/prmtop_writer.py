"""Minimal AMBER prmtop writer — pure Python, no force-field info.

The full prmtop format carries everything an MD engine needs (bonds,
angles, dihedrals, Lennard-Jones, ...). Post_MD analyses only read the
per-atom + per-residue metadata, so this writer emits a slim prmtop
that's enough for ``post_md.io.amber.prmtop.read_prmtop`` to round-trip
a stripped topology. The output is loadable by post_md itself but is
NOT a valid AMBER MD-engine prmtop — that's intentional, the file is a
post-processing artefact, not a re-simulation input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from post_md.core.topology import Topology

# Must match the divisor used by the reader so charges round-trip.
_AMBER_CHARGE_UNIT = 18.2223

# Reverse map: element symbol → atomic number (for the elements we know
# how to read back via _ELEMENT_TABLE in prmtop.py).
_Z_FROM_ELEMENT = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Br": 35, "I": 53,
}


def _format_block(values, width: int, per_line: int, kind: str) -> list[str]:
    """Slice ``values`` into Fortran-style fixed-width lines."""
    out: list[str] = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        if kind == "a":
            out.append("".join(str(v)[:width].ljust(width) for v in chunk))
        elif kind == "i":
            out.append("".join(f"{int(v):{width}d}" for v in chunk))
        elif kind == "e":
            # Scientific 'E' notation with `width` columns, 8 digits after point.
            out.append("".join(f"{float(v):{width}.8E}" for v in chunk))
        else:
            raise ValueError(kind)
    if not out:  # empty section must still have an empty data line
        out.append("")
    return out


def write_minimal_prmtop(
    out_path: str | Path,
    topology: Topology,
    indices: np.ndarray | None = None,
) -> Path:
    """Write a slim prmtop for ``topology`` (or its subset ``indices``).

    Sections emitted: TITLE, POINTERS, ATOM_NAME, CHARGE, ATOMIC_NUMBER,
    MASS, RESIDUE_LABEL, RESIDUE_POINTER. Everything else (LJ, bonded
    terms, ...) is omitted because post_md analyses don't read it.

    The new residue numbering is dense: ``unique(residue_ids[indices])``
    is renumbered 1..K, and RESIDUE_POINTER reflects atom positions in
    the subset. Charges are re-scaled by 18.2223 so the reader recovers
    the original electron-unit values.
    """
    out_path = Path(out_path)

    if indices is None:
        idx = np.arange(topology.n_atoms, dtype=np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)

    atom_names = topology.atom_names[idx]
    masses = topology.masses[idx].astype(np.float64)
    charges = (topology.charges[idx].astype(np.float64) * _AMBER_CHARGE_UNIT)
    elements = topology.elements[idx]
    residue_ids_old = topology.residue_ids[idx]
    residue_names_full = topology.residue_names[idx]

    n_atoms = int(idx.size)

    # Map atomic numbers from elements.
    z_arr = np.array(
        [_Z_FROM_ELEMENT.get(str(e).strip(), 0) for e in elements],
        dtype=np.int32,
    )

    # Renumber residues densely while preserving order of first appearance.
    # AMBER prmtop requires atoms of the same residue to be contiguous,
    # which is already guaranteed by post_md's own selection semantics
    # (selections preserve original atom order).
    unique_resids, inverse = np.unique(residue_ids_old, return_inverse=True)
    # Reorder unique_resids by first appearance.
    first_pos = np.empty_like(unique_resids)
    seen: set[int] = set()
    order: list[int] = []
    for r in residue_ids_old.tolist():
        if r not in seen:
            seen.add(r)
            order.append(r)
    ordered_resids = np.asarray(order, dtype=residue_ids_old.dtype)
    rank = {int(r): i for i, r in enumerate(ordered_resids.tolist())}

    # Residue labels in new order — pick first label seen per residue.
    new_residue_labels: list[str] = []
    for r in ordered_resids.tolist():
        first_idx = int(np.argmax(residue_ids_old == r))
        new_residue_labels.append(str(residue_names_full[first_idx]))

    # RESIDUE_POINTER: 1-indexed atom number where each residue starts.
    residue_pointer: list[int] = []
    last_rank = -1
    for i, r in enumerate(residue_ids_old.tolist()):
        rk = rank[int(r)]
        if rk != last_rank:
            residue_pointer.append(i + 1)
            last_rank = rk

    n_residues = len(residue_pointer)

    # POINTERS array — 32 ints, only the few post_md cares about are real.
    pointers = [0] * 32
    pointers[0] = n_atoms
    pointers[1] = 1               # NTYPES (dummy)
    pointers[11] = n_residues
    nmxrs = max(
        residue_pointer[i + 1] - residue_pointer[i] if i + 1 < n_residues
        else n_atoms + 1 - residue_pointer[i]
        for i in range(n_residues)
    ) if n_residues else 1
    pointers[28] = nmxrs

    lines: list[str] = []
    lines.append("%VERSION  VERSION_STAMP = V0001.000  DATE = post_md")
    lines += ["%FLAG TITLE", "%FORMAT(20a4)", "stripped_by_post_md".ljust(80)]
    lines += ["%FLAG POINTERS", "%FORMAT(10I8)"]
    lines += _format_block(pointers, width=8, per_line=10, kind="i")
    lines += ["%FLAG ATOM_NAME", "%FORMAT(20a4)"]
    lines += _format_block(atom_names, width=4, per_line=20, kind="a")
    lines += ["%FLAG CHARGE", "%FORMAT(5E16.8)"]
    lines += _format_block(charges, width=16, per_line=5, kind="e")
    lines += ["%FLAG ATOMIC_NUMBER", "%FORMAT(10I8)"]
    lines += _format_block(z_arr.tolist(), width=8, per_line=10, kind="i")
    lines += ["%FLAG MASS", "%FORMAT(5E16.8)"]
    lines += _format_block(masses, width=16, per_line=5, kind="e")
    lines += ["%FLAG RESIDUE_LABEL", "%FORMAT(20a4)"]
    lines += _format_block(new_residue_labels, width=4, per_line=20, kind="a")
    lines += ["%FLAG RESIDUE_POINTER", "%FORMAT(10I8)"]
    lines += _format_block(residue_pointer, width=8, per_line=10, kind="i")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
