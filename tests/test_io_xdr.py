"""XDR primitive unit tests — no GROMACS file needed."""

from __future__ import annotations

import io
import struct

from post_md.io.gromacs._xdr import (
    read_double,
    read_float,
    read_int,
    read_opaque,
    read_xdr_string,
)


def test_read_int_and_float():
    buf = io.BytesIO(struct.pack(">if", 1993, 3.14))
    assert read_int(buf) == 1993
    assert abs(read_float(buf) - 3.14) < 1e-6


def test_read_double():
    buf = io.BytesIO(struct.pack(">d", 2.71828))
    assert abs(read_double(buf) - 2.71828) < 1e-12


def test_read_opaque_pads_to_4_bytes():
    payload = b"abc"  # 3 bytes → 1 byte of padding
    buf = io.BytesIO(payload + b"\x00" + b"END!")
    assert read_opaque(buf, 3) == b"abc"
    assert buf.read(4) == b"END!"


def test_read_xdr_string_strips_null():
    name = b"GMX_trn_file\x00"  # length 13 incl NUL
    raw = struct.pack(">i", len(name)) + name + b"\x00" * 3  # pad to 16
    buf = io.BytesIO(raw)
    assert read_xdr_string(buf) == "GMX_trn_file"
