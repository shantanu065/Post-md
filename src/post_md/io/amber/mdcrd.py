"""AMBER ASCII trajectory (.mdcrd / .crd) reader.

Layout: title line, then 10F8.3 packed coordinates, optionally followed by an
8.3-formatted box line of 3 (orthogonal) floats per frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from post_md.core.trajectory import Frame, Trajectory


class AmberMdcrdTrajectory(Trajectory):
    def __init__(self, path: str | Path, n_atoms: int):
        self.path = Path(path)
        self._n_atoms = int(n_atoms)
        self._frames: list[Frame] = []
        self._load()

    @classmethod
    def open(cls, path: str | Path, n_atoms: int) -> AmberMdcrdTrajectory:
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
        with self.path.open("r") as f:
            f.readline()  # title
            data: list[float] = []
            for line in f:
                line = line.rstrip("\r\n")
                for i in range(0, len(line), 8):
                    chunk = line[i : i + 8]
                    if chunk.strip():
                        try:
                            data.append(float(chunk))
                        except ValueError:
                            continue

        n_coords = self._n_atoms * 3
        total = len(data)
        if n_coords == 0:
            return

        if total % (n_coords + 3) == 0:
            stride = n_coords + 3
            has_box = True
        elif total % n_coords == 0:
            stride = n_coords
            has_box = False
        else:
            # Fall back: assume no box, drop trailing floats
            stride = n_coords
            has_box = False

        for k in range(0, total - stride + 1, stride):
            chunk = data[k : k + stride]
            coords = np.asarray(chunk[:n_coords], dtype=np.float32).reshape(self._n_atoms, 3)
            box = None
            if has_box:
                box = np.diag(chunk[n_coords : n_coords + 3]).astype(np.float32)
            idx = len(self._frames)
            self._frames.append(
                Frame(index=idx, coordinates=coords, box=box, time=float(idx))
            )
