"""XTC decoder internals — regressions for large-system compression.

These exercise the pure-Python xdr3dfcoord port without needing a real
.xtc fixture. Both bugs they guard against only surface on big, wide
systems (hundreds of thousands of atoms), which is exactly where the
toolkit's GROMACS support is most useful.
"""

from __future__ import annotations

from post_md.io.gromacs.xtc import _sizeofints


def test_sizeofints_handles_large_systems():
    """Regression: the mixed-radix scratch buffer used to be 5 bytes, which
    overflowed (IndexError) on wide boxes whose per-axis size needs >5 bytes
    of product space — e.g. a ~15.7 nm box at 1000x precision (size 15696).
    """
    # 15696**3 ~= 3.87e12 needs 42 bits / 6 bytes to pack.
    assert _sizeofints([15696, 15696, 15696]) == 42


def test_sizeofints_small_values():
    """Small sizes stay well within the buffer and still count correctly."""
    # 8**3 = 512 = 2**9 -> the value 512 needs 10 bits to represent.
    assert _sizeofints([8, 8, 8]) == 10
    # 256**3 = 2**24 -> 25 bits.
    assert _sizeofints([256, 256, 256]) == 25
