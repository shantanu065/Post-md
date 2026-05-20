"""GROMACS .xtc compressed trajectory reader (pure-Python).

Ported from the GROMACS xdrfile reference C implementation (xdr3dfcoord).
The compression is a custom 3D quantizer with smallnum run-length encoding;
pure-Python is functionally correct but ~10-50× slower than the C version.
A native accelerator may be added later as `post_md._xtc_fast`.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from post_md.core.trajectory import Frame, Trajectory
from post_md.core.units import NM_TO_ANGSTROM

XTC_MAGIC = 1995

# Lookup table reproduced verbatim from GROMACS xdrfile_xtc.c.
MAGICINTS: list[int] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645,
    812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501,
    8192, 10321, 13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536,
    82570, 104031, 131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021, 4194304, 5284491, 6658042,
    8388607, 10568983, 13316085, 16777216,
]
FIRSTIDX = 9
LASTIDX = len(MAGICINTS) - 1


def _sizeofint(size: int) -> int:
    nbits = 0
    n = 1
    while size >= n and nbits < 32:
        nbits += 1
        n <<= 1
    return nbits


def _sizeofints(sizes: list[int]) -> int:
    """Total bits needed to pack ``sizes`` ints together (mixed-radix)."""
    bytes_arr = [1, 0, 0, 0, 0]
    num_of_bytes = 1
    for s in sizes:
        tmp = 0
        for j in range(num_of_bytes):
            tmp = bytes_arr[j] * s + tmp
            bytes_arr[j] = tmp & 0xFF
            tmp >>= 8
        while tmp:
            bytes_arr[num_of_bytes] = tmp & 0xFF
            tmp >>= 8
            num_of_bytes += 1
    num_of_bits = 0
    num = 1
    nb = num_of_bytes - 1
    while bytes_arr[nb] >= num:
        num_of_bits += 1
        num *= 2
    return num_of_bits + nb * 8


class _BitReader:
    """Streaming bit reader matching GROMACS receivebits semantics."""

    __slots__ = ("buf", "cnt", "lastbits", "lastbyte")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.cnt = 0
        self.lastbits = 0
        self.lastbyte = 0

    def receive(self, nbits: int) -> int:
        mask = (1 << nbits) - 1
        num = 0
        # 32-bit window for lastbyte to mirror C unsigned int semantics
        while nbits >= 8:
            self.lastbyte = ((self.lastbyte << 8) & 0xFFFFFFFF) | self.buf[self.cnt]
            self.cnt += 1
            num |= (self.lastbyte >> self.lastbits) << (nbits - 8)
            nbits -= 8
        if nbits > 0:
            if self.lastbits < nbits:
                self.lastbits += 8
                self.lastbyte = ((self.lastbyte << 8) & 0xFFFFFFFF) | self.buf[self.cnt]
                self.cnt += 1
            self.lastbits -= nbits
            num |= (self.lastbyte >> self.lastbits) & ((1 << nbits) - 1)
        return num & mask

    def receive_ints(self, n_ints: int, n_bits: int, sizes: list[int]) -> list[int]:
        bytes_arr: list[int] = []
        while n_bits > 8:
            bytes_arr.append(self.receive(8))
            n_bits -= 8
        if n_bits > 0:
            bytes_arr.append(self.receive(n_bits))

        nums = [0] * n_ints
        for i in range(n_ints - 1, 0, -1):
            num = 0
            for j in range(len(bytes_arr) - 1, -1, -1):
                num = (num << 8) | bytes_arr[j]
                p = num // sizes[i]
                bytes_arr[j] = p
                num -= p * sizes[i]
            nums[i] = num
        nums[0] = 0
        for j in range(min(4, len(bytes_arr))):
            nums[0] |= bytes_arr[j] << (8 * j)
        return nums


def _read_header(f) -> tuple[int, int, int, float, np.ndarray] | None:
    """Read XTC frame header. Returns (magic, natoms, step, time, box) or None at EOF."""
    hdr = f.read(52)
    if len(hdr) < 52:
        return None
    magic, natoms, step = struct.unpack(">iii", hdr[:12])
    if magic != XTC_MAGIC:
        raise ValueError(f"Bad XTC magic: {magic}")
    time = struct.unpack(">f", hdr[12:16])[0]
    box = np.frombuffer(hdr[16:52], dtype=">f4").reshape(3, 3).astype(np.float32)
    return magic, natoms, step, time, box


def _decompress(f, natoms: int) -> np.ndarray:
    cmp_hdr = f.read(40)
    if len(cmp_hdr) < 40:
        raise OSError("truncated XTC compressed header")
    natoms_again = struct.unpack(">i", cmp_hdr[0:4])[0]
    precision = struct.unpack(">f", cmp_hdr[4:8])[0]
    minint = list(struct.unpack(">iii", cmp_hdr[8:20]))
    maxint = list(struct.unpack(">iii", cmp_hdr[20:32]))
    smallidx = struct.unpack(">i", cmp_hdr[32:36])[0]
    nbytes = struct.unpack(">i", cmp_hdr[36:40])[0]

    if natoms_again != natoms:
        raise ValueError(
            f"XTC natoms inconsistency in compressed block: {natoms} vs {natoms_again}"
        )

    pad = (4 - (nbytes % 4)) % 4
    buf = f.read(nbytes + pad)
    if len(buf) < nbytes:
        raise OSError("truncated XTC compressed buffer")

    sizeint = [
        maxint[0] - minint[0] + 1,
        maxint[1] - minint[1] + 1,
        maxint[2] - minint[2] + 1,
    ]
    if max(sizeint) > 0xFFFFFF:
        bitsizeint = [_sizeofint(s) for s in sizeint]
        bitsize = 0
    else:
        bitsizeint = [0, 0, 0]
        bitsize = _sizeofints(sizeint)

    smallidx = max(min(smallidx, LASTIDX), 0)
    smaller_idx = max(FIRSTIDX, smallidx - 1)
    smaller = MAGICINTS[smaller_idx] // 2
    smallnum = MAGICINTS[smallidx] // 2
    sizesmall = [MAGICINTS[smallidx]] * 3

    inv_prec = 1.0 / precision
    reader = _BitReader(buf)

    coords = np.empty((natoms, 3), dtype=np.float32)
    prev = [0, 0, 0]
    i = 0
    lfp = 0

    while i < natoms:
        if bitsize == 0:
            thiscoord = [
                reader.receive(bitsizeint[0]),
                reader.receive(bitsizeint[1]),
                reader.receive(bitsizeint[2]),
            ]
        else:
            thiscoord = reader.receive_ints(3, bitsize, sizeint)
        i += 1
        thiscoord[0] += minint[0]
        thiscoord[1] += minint[1]
        thiscoord[2] += minint[2]

        prev[0], prev[1], prev[2] = thiscoord[0], thiscoord[1], thiscoord[2]

        flag = reader.receive(1)
        is_smaller = 0
        run = 0
        if flag == 1:
            run = reader.receive(5)
            is_smaller = run % 3
            run -= is_smaller
            is_smaller -= 1

        if run > 0:
            for k in range(0, run, 3):
                thiscoord = reader.receive_ints(3, smallidx, sizesmall)
                i += 1
                thiscoord[0] += prev[0] - smallnum
                thiscoord[1] += prev[1] - smallnum
                thiscoord[2] += prev[2] - smallnum
                if k == 0:
                    # swap thiscoord ↔ prev
                    thiscoord[0], prev[0] = prev[0], thiscoord[0]
                    thiscoord[1], prev[1] = prev[1], thiscoord[1]
                    thiscoord[2], prev[2] = prev[2], thiscoord[2]
                    coords[lfp, 0] = prev[0] * inv_prec
                    coords[lfp, 1] = prev[1] * inv_prec
                    coords[lfp, 2] = prev[2] * inv_prec
                    lfp += 1
                else:
                    prev[0], prev[1], prev[2] = thiscoord[0], thiscoord[1], thiscoord[2]
                coords[lfp, 0] = thiscoord[0] * inv_prec
                coords[lfp, 1] = thiscoord[1] * inv_prec
                coords[lfp, 2] = thiscoord[2] * inv_prec
                lfp += 1
        else:
            coords[lfp, 0] = thiscoord[0] * inv_prec
            coords[lfp, 1] = thiscoord[1] * inv_prec
            coords[lfp, 2] = thiscoord[2] * inv_prec
            lfp += 1

        smallidx += is_smaller
        if is_smaller < 0:
            smallnum = smaller
            if smallidx > FIRSTIDX:
                smaller = MAGICINTS[smallidx - 1] // 2
            else:
                smaller = 0
        elif is_smaller > 0:
            smaller = smallnum
            smallnum = MAGICINTS[smallidx] // 2
        sizesmall = [MAGICINTS[smallidx]] * 3

    return coords


class XtcTrajectory(Trajectory):
    def __init__(self, path: str | Path, n_atoms: int):
        self.path = Path(path)
        self._n_atoms = int(n_atoms)
        self._frame_offsets: list[int] = []
        self._index()

    @classmethod
    def open(cls, path: str | Path, n_atoms: int) -> XtcTrajectory:
        return cls(path, n_atoms)

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

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
                _, natoms, _, _, _ = h
                if natoms <= 9:
                    f.seek(12 * natoms, 1)
                else:
                    cmp_hdr = f.read(40)
                    if len(cmp_hdr) < 40:
                        break
                    nbytes = struct.unpack(">i", cmp_hdr[36:40])[0]
                    pad = (4 - (nbytes % 4)) % 4
                    f.seek(nbytes + pad, 1)
                self._frame_offsets.append(start)

    def read_frame(self, index: int) -> Frame:
        if not (0 <= index < self.n_frames):
            raise IndexError(index)
        with self.path.open("rb") as f:
            f.seek(self._frame_offsets[index])
            h = _read_header(f)
            if h is None:
                raise OSError(f"failed reading frame {index}")
            _, natoms, _, time, box = h
            if natoms != self._n_atoms:
                raise ValueError(
                    f"XTC frame natoms mismatch: {natoms} vs {self._n_atoms}"
                )
            if natoms <= 9:
                raw = f.read(12 * natoms)
                coords = (
                    np.frombuffer(raw, dtype=">f4")
                    .reshape(natoms, 3)
                    .astype(np.float32)
                )
            else:
                coords = _decompress(f, natoms)
        coords = coords * NM_TO_ANGSTROM
        box = box * NM_TO_ANGSTROM
        return Frame(index=index, coordinates=coords.copy(), box=box, time=float(time))
