"""Minimal NetCDF3-classic reader for AMBER trajectory files.

Implements just enough of the NetCDF3 spec
(https://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html)
to extract coordinates, time, and unit-cell variables from AMBER files.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from post_md.core.trajectory import Frame, Trajectory

# NetCDF type tags
NC_DIMENSION = 10
NC_VARIABLE = 11
NC_ATTRIBUTE = 12

# (struct format char, byte width, numpy dtype) keyed by nc_type
_NC_TYPES = {
    1: ("b", 1, np.int8),
    2: ("c", 1, np.uint8),
    3: ("h", 2, np.int16),
    4: ("i", 4, np.int32),
    5: ("f", 4, np.float32),
    6: ("d", 8, np.float64),
}


def _pad4(n: int) -> int:
    return (4 - (n % 4)) % 4


def _i32(buf: memoryview, off: int) -> tuple[int, int]:
    return struct.unpack_from(">i", buf, off)[0], off + 4


def _i64(buf: memoryview, off: int) -> tuple[int, int]:
    return struct.unpack_from(">q", buf, off)[0], off + 8


def _read_name(buf: memoryview, off: int) -> tuple[str, int]:
    nelems, off = _i32(buf, off)
    raw = bytes(buf[off : off + nelems])
    off += nelems + _pad4(nelems)
    return raw.decode("ascii", errors="replace"), off


def _read_att_values(buf: memoryview, off: int, nc_type: int, nelems: int):
    fmt_char, size, _ = _NC_TYPES[nc_type]
    nbytes = size * nelems
    if nc_type == 2:
        val = bytes(buf[off : off + nelems]).decode("ascii", errors="replace")
        off += nelems + _pad4(nelems)
        return val, off
    vals = struct.unpack_from(f">{nelems}{fmt_char}", buf, off)
    off += nbytes + _pad4(nbytes)
    return vals, off


def _read_att_list(buf: memoryview, off: int) -> tuple[dict, int]:
    tag, off = _i32(buf, off)
    n, off = _i32(buf, off)
    if tag == 0 and n == 0:
        return {}, off
    if tag != NC_ATTRIBUTE:
        raise ValueError(f"Expected ATTRIBUTE tag, got {tag}")
    atts: dict = {}
    for _ in range(n):
        name, off = _read_name(buf, off)
        nc_type, off = _i32(buf, off)
        nv, off = _i32(buf, off)
        val, off = _read_att_values(buf, off, nc_type, nv)
        atts[name] = val
    return atts, off


@dataclass
class _NCDim:
    name: str
    size: int  # 0 means unlimited (record dim)


@dataclass
class _NCVar:
    name: str
    dim_ids: list[int]
    atts: dict
    nc_type: int
    vsize: int
    begin: int
    is_record: bool = False
    shape: tuple[int, ...] = field(default_factory=tuple)


def _parse_header(buf: memoryview, large_offsets: bool):
    off = 4  # skip magic
    numrecs, off = _i32(buf, off)

    # dim_list
    tag, off = _i32(buf, off)
    n, off = _i32(buf, off)
    dims: list[_NCDim] = []
    if tag == NC_DIMENSION:
        for _ in range(n):
            name, off = _read_name(buf, off)
            size, off = _i32(buf, off)
            dims.append(_NCDim(name, size))
    elif not (tag == 0 and n == 0):
        raise ValueError(f"Expected DIMENSION tag, got {tag}")

    # global attributes (discarded except for the existence check)
    _, off = _read_att_list(buf, off)

    # var_list
    tag, off = _i32(buf, off)
    n, off = _i32(buf, off)
    vars_: list[_NCVar] = []
    if tag == NC_VARIABLE:
        for _ in range(n):
            name, off = _read_name(buf, off)
            ndims, off = _i32(buf, off)
            dim_ids = []
            for _ in range(ndims):
                d, off = _i32(buf, off)
                dim_ids.append(d)
            atts, off = _read_att_list(buf, off)
            nc_type, off = _i32(buf, off)
            vsize, off = _i32(buf, off)
            if large_offsets:
                begin, off = _i64(buf, off)
            else:
                begin, off = _i32(buf, off)
            v = _NCVar(
                name=name, dim_ids=dim_ids, atts=atts,
                nc_type=nc_type, vsize=vsize, begin=begin,
            )
            v.is_record = bool(dim_ids) and dims[dim_ids[0]].size == 0
            v.shape = tuple(dims[d].size for d in dim_ids)
            vars_.append(v)
    elif not (tag == 0 and n == 0):
        raise ValueError(f"Expected VARIABLE tag, got {tag}")

    return numrecs, dims, vars_


def _triclinic_box(lengths, angles) -> np.ndarray:
    a, b, c = (float(x) for x in lengths)
    alpha, beta, gamma = (float(x) for x in np.deg2rad(angles))
    out = np.zeros((3, 3), dtype=np.float64)
    out[0, 0] = a
    out[1, 0] = b * np.cos(gamma)
    out[1, 1] = b * np.sin(gamma)
    out[2, 0] = c * np.cos(beta)
    out[2, 1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    sq = c * c - out[2, 0] ** 2 - out[2, 1] ** 2
    out[2, 2] = np.sqrt(max(sq, 0.0))
    return out


class AmberNetCDFTrajectory(Trajectory):
    def __init__(self, path: str | Path, n_atoms_expected: int):
        self.path = Path(path)
        self._buf = self.path.read_bytes()
        magic = self._buf[:4]
        if magic == b"CDF\x01":
            large = False
        elif magic == b"CDF\x02":
            large = True
        else:
            raise ValueError(f"Not a NetCDF3 classic file: {self.path}")

        mv = memoryview(self._buf)
        numrecs, dims, vars_ = _parse_header(mv, large)
        self._dims = {d.name: d for d in dims}
        self._vars = {v.name: v for v in vars_}

        atom_dim = self._dims.get("atom")
        if atom_dim is None:
            raise ValueError(f"NetCDF missing 'atom' dimension: {self.path}")
        if atom_dim.size != n_atoms_expected:
            raise ValueError(
                f"NetCDF has {atom_dim.size} atoms, topology has {n_atoms_expected}"
            )
        self._n_atoms = atom_dim.size

        rec_vars = [v for v in vars_ if v.is_record]
        self._rec_size = sum(v.vsize for v in rec_vars)
        if "coordinates" not in self._vars:
            raise ValueError("NetCDF file missing 'coordinates' variable")

        # numrecs may be -1 in streaming mode — compute from file size.
        if numrecs < 0 and self._rec_size > 0:
            rec_start = min(v.begin for v in rec_vars)
            numrecs = (len(self._buf) - rec_start) // self._rec_size
        self._numrecs = int(numrecs)

    @classmethod
    def open(cls, path: str | Path, n_atoms: int) -> AmberNetCDFTrajectory:
        return cls(path, n_atoms)

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return self._numrecs

    def _read_record(self, name: str, frame: int, count: int) -> np.ndarray:
        v = self._vars[name]
        fmt_char, _, _ = _NC_TYPES[v.nc_type]
        offset = v.begin + frame * self._rec_size
        return np.frombuffer(
            self._buf, dtype=np.dtype(">" + fmt_char), count=count, offset=offset
        )

    def read_frame(self, index: int) -> Frame:
        if not (0 <= index < self.n_frames):
            raise IndexError(index)

        coords = self._read_record("coordinates", index, self._n_atoms * 3)
        coords = np.asarray(coords, dtype=np.float32).reshape(self._n_atoms, 3).copy()

        time = 0.0
        if "time" in self._vars and self._vars["time"].is_record:
            t = self._read_record("time", index, 1)
            time = float(t[0])

        box = None
        if "cell_lengths" in self._vars and self._vars["cell_lengths"].is_record:
            cl = self._read_record("cell_lengths", index, 3).astype(np.float64)
            angles = np.array([90.0, 90.0, 90.0], dtype=np.float64)
            if "cell_angles" in self._vars and self._vars["cell_angles"].is_record:
                angles = self._read_record("cell_angles", index, 3).astype(np.float64)
            if np.allclose(angles, 90.0, atol=0.5):
                box = np.diag(cl).astype(np.float32)
            else:
                box = _triclinic_box(cl, angles).astype(np.float32)

        return Frame(index=index, coordinates=coords, box=box, time=time)
