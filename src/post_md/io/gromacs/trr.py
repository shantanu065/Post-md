"""GROMACS .trr (uncompressed) trajectory reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from post_md.core.trajectory import Frame, Trajectory
from post_md.core.units import NM_TO_ANGSTROM
from post_md.io.gromacs._xdr import (
    read_double,
    read_float,
    read_int,
    read_xdr_string,
)

GROMACS_MAGIC = 1993


def _read_header(f):
    """Read one TRR frame header. Returns the header dict or None at clean EOF."""
    try:
        magic = read_int(f)
    except EOFError:
        return None
    if magic != GROMACS_MAGIC:
        raise ValueError(f"Bad TRR magic at offset {f.tell() - 4}: {magic}")
    _version = read_xdr_string(f)
    ir_size = read_int(f)
    e_size = read_int(f)
    box_size = read_int(f)
    vir_size = read_int(f)
    pres_size = read_int(f)
    top_size = read_int(f)
    sym_size = read_int(f)
    x_size = read_int(f)
    v_size = read_int(f)
    f_size = read_int(f)
    natoms = read_int(f)
    step = read_int(f)
    nre = read_int(f)

    if x_size and natoms:
        sizeof_real = x_size // (3 * natoms)
    elif v_size and natoms:
        sizeof_real = v_size // (3 * natoms)
    elif f_size and natoms:
        sizeof_real = f_size // (3 * natoms)
    elif box_size:
        sizeof_real = box_size // 9
    else:
        sizeof_real = 4
    if sizeof_real not in (4, 8):
        raise ValueError(f"Unexpected TRR sizeof_real={sizeof_real}")

    time = read_float(f) if sizeof_real == 4 else read_double(f)
    lam = read_float(f) if sizeof_real == 4 else read_double(f)
    return dict(
        ir_size=ir_size, e_size=e_size, box_size=box_size, vir_size=vir_size,
        pres_size=pres_size, top_size=top_size, sym_size=sym_size,
        x_size=x_size, v_size=v_size, f_size=f_size,
        natoms=natoms, step=step, nre=nre, time=time, lam=lam,
        sizeof_real=sizeof_real,
    )


class TrrTrajectory(Trajectory):
    def __init__(self, path: str | Path, n_atoms: int):
        self.path = Path(path)
        self._n_atoms_hint = int(n_atoms)
        self._frame_offsets: list[int] = []
        self._natoms: int = 0
        self._index()

    @classmethod
    def open(cls, path: str | Path, n_atoms: int) -> TrrTrajectory:
        return cls(path, n_atoms)

    @property
    def n_atoms(self) -> int:
        return self._natoms

    @property
    def n_frames(self) -> int:
        return len(self._frame_offsets)

    def _index(self) -> None:
        with self.path.open("rb") as f:
            while True:
                start = f.tell()
                h = _read_header(f)
                if h is None:
                    break
                total = (
                    h["box_size"] + h["vir_size"] + h["pres_size"]
                    + h["x_size"] + h["v_size"] + h["f_size"]
                )
                f.seek(total, 1)
                self._frame_offsets.append(start)
                if not self._natoms:
                    self._natoms = h["natoms"]

    def read_frame(self, index: int) -> Frame:
        if not (0 <= index < self.n_frames):
            raise IndexError(index)
        with self.path.open("rb") as f:
            f.seek(self._frame_offsets[index])
            h = _read_header(f)
            if h is None:
                raise OSError(f"failed reading frame {index}")

            box: np.ndarray | None = None
            if h["box_size"]:
                bb = f.read(h["box_size"])
                dt = ">f4" if h["sizeof_real"] == 4 else ">f8"
                box = (
                    np.frombuffer(bb, dtype=dt).reshape(3, 3).astype(np.float32)
                    * NM_TO_ANGSTROM
                )

            f.seek(h["vir_size"] + h["pres_size"], 1)

            if not h["x_size"]:
                raise ValueError(f"TRR frame {index} has no coordinates")
            xb = f.read(h["x_size"])
            dt = ">f4" if h["sizeof_real"] == 4 else ">f8"
            coords = (
                np.frombuffer(xb, dtype=dt).reshape(h["natoms"], 3).astype(np.float32)
                * NM_TO_ANGSTROM
            )
        return Frame(index=index, coordinates=coords.copy(), box=box, time=float(h["time"]))
