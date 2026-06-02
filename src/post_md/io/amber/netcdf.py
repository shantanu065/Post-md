"""Minimal NetCDF3-classic reader for AMBER trajectory files.

Implements just enough of the NetCDF3 spec
(https://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html)
to extract coordinates, time, and unit-cell variables from AMBER files.
"""

from __future__ import annotations

import mmap
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
        # Memory-map the file rather than reading it into RAM. AMBER
        # production trajectories regularly hit hundreds of GB and won't
        # fit in physical memory, but mmap costs only virtual address
        # space and lets the kernel page bytes in on demand. The mapping
        # is held on the instance and stays alive as long as any derived
        # array view (np.frombuffer / stride_tricks) is referenced.
        self._file = open(self.path, "rb")
        try:
            self._buf = mmap.mmap(
                self._file.fileno(), 0, access=mmap.ACCESS_READ,
            )
        except (ValueError, OSError):
            self._file.close()
            raise

        magic = bytes(self._buf[:4])
        if magic == b"CDF\x01":
            large = False
        elif magic == b"CDF\x02":
            large = True
        else:
            self._buf.close()
            self._file.close()
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

    # ------------------------------------------------------------------
    # Batched / slab reads
    # ------------------------------------------------------------------

    # Cap the peak working set while materialising the strided byte-swap
    # batch by batch. 512 MiB is comfortable on every modern workstation
    # but keeps a 100k-atom system streaming smoothly.
    _SLAB_CHUNK_BYTES = 512 * 1024 * 1024

    def coordinates_slab(
        self,
        start: int = 0,
        stop: int | None = None,
        selection: np.ndarray | slice | None = None,
    ) -> np.ndarray:
        """Return coordinates for a contiguous range of frames in one shot.

        NetCDF3 record layout interleaves coordinates with time / cell
        variables, so the per-frame coordinate slabs are not bytewise
        contiguous. They are *evenly spaced* though — exactly ``_rec_size``
        bytes apart — so we construct a strided view over the file buffer
        and byte-swap into a native-endian buffer one chunk at a time.

        Selection is applied per chunk, which is the difference between
        2 GB and 200 GB of peak memory on a 100k-atom, 200k-frame system:
        we never materialise the full-trajectory float32 cube.

        Returns a contiguous ``(n_frames_in_range, n_sel_atoms, 3)`` float32
        array in native byte order.
        """
        if stop is None:
            stop = self.n_frames
        start = max(0, int(start))
        stop = min(self.n_frames, int(stop))

        # Resolve selection up front so we know the output's n_sel.
        if selection is None:
            sel_arr: np.ndarray | None = None
            sel_slice: slice | None = None
            n_sel = self._n_atoms
        elif isinstance(selection, slice):
            sel_slice = selection
            sel_arr = None
            n_sel = len(range(*selection.indices(self._n_atoms)))
        else:
            sel_arr = np.asarray(selection, dtype=np.int64)
            sel_slice = None
            n_sel = int(sel_arr.size)

        if stop <= start:
            return np.empty((0, n_sel, 3), dtype=np.float32)

        coord_var = self._vars["coordinates"]
        if coord_var.nc_type != 5:  # NC_FLOAT
            # Defensive: AMBER NetCDF coordinates are always float32 in
            # practice. Fall back to per-frame read if a producer ever
            # writes them otherwise.
            return np.stack(
                [self.read_frame(i).coordinates for i in range(start, stop)],
                axis=0,
            )

        rec_size = self._rec_size
        # How many frames worth of raw bytes fit in the chunk budget?
        bytes_per_full_frame = self._n_atoms * 12  # 3 * float32 per atom
        frames_per_chunk = max(1, self._SLAB_CHUNK_BYTES // max(bytes_per_full_frame, 1))

        from numpy.lib.stride_tricks import as_strided
        out = np.empty((stop - start, n_sel, 3), dtype=np.float32)

        # Hot read path for analyses that load the whole trajectory
        # (rmsd / rmsf / rg / pca). Long Prod.nc files spend most of
        # their wall-time in this loop, so:
        #   1. raise_if_cancelled() at each chunk so Stop can interrupt;
        #   2. ThreadPoolExecutor parallelises the byteswap + memcpy
        #      across cores. numpy releases the GIL during astype() so
        #      threads actually scale up to ~memory-bandwidth limit.
        from post_md.utils import default_workers, raise_if_cancelled

        buf_size = len(self._buf)

        def _process_chunk(chunk_start: int) -> None:
            raise_if_cancelled()
            cs = chunk_start
            ce = min(cs + frames_per_chunk, stop)
            n_chunk = ce - cs
            base_offset = coord_var.begin + cs * rec_size
            needed = (n_chunk - 1) * rec_size + self._n_atoms * 12
            out_cursor = cs - start
            if base_offset + needed > buf_size:
                # Truncated file — fall back to per-frame reads for the tail.
                for i in range(cs, ce):
                    f = self.read_frame(i)
                    coords_i = (
                        f.coordinates if sel_slice is None and sel_arr is None
                        else f.coordinates[sel_slice] if sel_slice is not None
                        else f.coordinates[sel_arr]
                    )
                    out[out_cursor + (i - cs)] = coords_i
                return

            raw = np.frombuffer(
                self._buf, dtype=np.uint8, offset=base_offset, count=needed,
            )
            as_be = raw.view(">f4")
            strided = as_strided(
                as_be,
                shape=(n_chunk, self._n_atoms, 3),
                strides=(rec_size, 12, 4),
                writeable=False,
            )
            if sel_slice is None and sel_arr is None:
                out[out_cursor : out_cursor + n_chunk] = (
                    strided.astype(np.float32, copy=True)
                )
            elif sel_slice is not None:
                out[out_cursor : out_cursor + n_chunk] = (
                    strided[:, sel_slice, :].astype(np.float32, copy=True)
                )
            else:
                out[out_cursor : out_cursor + n_chunk] = (
                    strided[:, sel_arr, :].astype(np.float32, copy=True)
                )

        chunk_starts = list(range(start, stop, frames_per_chunk))
        # Thread parallelism for the load. Cap at min(workers, 32, n_chunks):
        # past ~16-32 threads we're bandwidth-bound on NFS / local disk
        # and extra threads just contend for the mmap.
        n_threads = min(default_workers(), 32, len(chunk_starts))
        if n_threads <= 1 or len(chunk_starts) <= 1:
            for cs in chunk_starts:
                _process_chunk(cs)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                # ex.map raises the first exception from any worker, which
                # is what we want for cancellation propagation.
                list(ex.map(_process_chunk, chunk_starts))

        return out

    def boxes_slab(self, start: int = 0, stop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame unit-cell matrices for the range [start, stop).

        Returns ``(boxes, has_box)`` where:
          * ``boxes`` is ``(n_frames, 3, 3)`` float32 (zeros for frames
            without a cell record).
          * ``has_box`` is ``(n_frames,)`` bool — False means "no PBC
            info on this frame, callers should skip imaging".

        Uses the same strided-view trick as ``coordinates_slab`` so the
        whole range is read in one mmap byte-swap instead of per-frame
        disk seeks. That's the difference between a few seconds and tens
        of minutes on a 200k-frame trajectory.
        """
        if stop is None:
            stop = self.n_frames
        start = max(0, int(start))
        stop = min(self.n_frames, int(stop))
        n = stop - start
        if n <= 0:
            return np.empty((0, 3, 3), dtype=np.float32), np.empty(0, dtype=bool)

        from numpy.lib.stride_tricks import as_strided
        rec_size = self._rec_size
        boxes = np.zeros((n, 3, 3), dtype=np.float32)
        has_box = np.zeros(n, dtype=bool)

        cl_var = self._vars.get("cell_lengths")
        if cl_var is None or not cl_var.is_record or cl_var.nc_type != 5:
            return boxes, has_box

        buf_size = len(self._buf)

        def _read_three_floats(var, start_frame: int, n_frames: int) -> np.ndarray:
            base = var.begin + start_frame * rec_size
            needed = (n_frames - 1) * rec_size + 12
            if base + needed > buf_size:
                # File truncated mid-cell record — fall back to safe zeros.
                return np.zeros((n_frames, 3), dtype=np.float64)
            raw = np.frombuffer(self._buf, dtype=np.uint8, offset=base, count=needed)
            as_be = raw.view(">f4")
            strided = as_strided(
                as_be,
                shape=(n_frames, 3),
                strides=(rec_size, 4),
                writeable=False,
            )
            return strided.astype(np.float64, copy=True)

        cell_lengths = _read_three_floats(cl_var, start, n)

        ca_var = self._vars.get("cell_angles")
        if ca_var is not None and ca_var.is_record and ca_var.nc_type == 5:
            cell_angles = _read_three_floats(ca_var, start, n)
        else:
            cell_angles = np.full((n, 3), 90.0, dtype=np.float64)

        # A frame's "has box" is true if its lengths look real (positive).
        valid = (cell_lengths > 0).all(axis=1)
        has_box[:] = valid

        # Orthorhombic fast path — most AMBER simulations use truncated
        # octahedron or rectangular cells, both of which keep all angles
        # near 90° on the principal axes.
        ortho = valid & np.all(np.abs(cell_angles - 90.0) < 0.5, axis=1)
        if ortho.any():
            ortho_idx = np.where(ortho)[0]
            for k in ortho_idx:
                boxes[k] = np.diag(cell_lengths[k]).astype(np.float32)

        # Triclinic frames — rebuild the cell matrix one frame at a time.
        tri = valid & ~ortho
        for k in np.where(tri)[0]:
            boxes[k] = _triclinic_box(cell_lengths[k], cell_angles[k]).astype(np.float32)

        return boxes, has_box

    def times_slab(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Per-frame times (ps) for the range [start, stop). float64."""
        if stop is None:
            stop = self.n_frames
        start = max(0, int(start))
        stop = min(self.n_frames, int(stop))
        if stop <= start:
            return np.empty(0, dtype=np.float64)

        if "time" not in self._vars or not self._vars["time"].is_record:
            return np.zeros(stop - start, dtype=np.float64)

        time_var = self._vars["time"]
        if time_var.nc_type != 5:  # NC_FLOAT
            return np.fromiter(
                (float(self._read_record("time", i, 1)[0]) for i in range(start, stop)),
                dtype=np.float64,
                count=stop - start,
            )

        from numpy.lib.stride_tricks import as_strided
        rec_size = self._rec_size
        base_offset = time_var.begin + start * rec_size
        raw = np.frombuffer(self._buf, dtype=np.uint8, offset=base_offset)
        needed = (stop - start - 1) * rec_size + 4
        if needed > raw.size:
            return np.fromiter(
                (float(self._read_record("time", i, 1)[0]) for i in range(start, stop)),
                dtype=np.float64,
                count=stop - start,
            )
        as_be = raw[:needed].view(">f4")
        strided = as_strided(
            as_be,
            shape=(stop - start,),
            strides=(rec_size,),
            writeable=False,
        )
        return strided.astype(np.float64, copy=True)
