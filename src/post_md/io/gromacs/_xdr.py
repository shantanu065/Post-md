"""XDR (RFC 4506) primitives — big-endian, 4-byte aligned."""

from __future__ import annotations

import struct
from typing import BinaryIO


def read_int(f: BinaryIO) -> int:
    b = f.read(4)
    if len(b) < 4:
        raise EOFError("unexpected EOF reading XDR int")
    return struct.unpack(">i", b)[0]


def read_uint(f: BinaryIO) -> int:
    b = f.read(4)
    if len(b) < 4:
        raise EOFError("unexpected EOF reading XDR uint")
    return struct.unpack(">I", b)[0]


def read_float(f: BinaryIO) -> float:
    b = f.read(4)
    if len(b) < 4:
        raise EOFError("unexpected EOF reading XDR float")
    return struct.unpack(">f", b)[0]


def read_double(f: BinaryIO) -> float:
    b = f.read(8)
    if len(b) < 8:
        raise EOFError("unexpected EOF reading XDR double")
    return struct.unpack(">d", b)[0]


def read_opaque(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) < n:
        raise EOFError(f"unexpected EOF reading XDR opaque ({len(b)}/{n})")
    pad = (4 - (n % 4)) % 4
    if pad:
        f.read(pad)
    return b


def read_xdr_string(f: BinaryIO) -> str:
    """XDR string (length prefix + bytes + 4-byte padding). GROMACS-style strings
    include a trailing NUL byte; we strip it."""
    slen = read_int(f)
    raw = read_opaque(f, slen)
    return raw.rstrip(b"\x00").decode("ascii", errors="replace")
