"""Trajectory abstract base class + Frame dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    coordinates: np.ndarray             # (n_atoms, 3) float32, Å
    box: np.ndarray | None              # (3, 3) float32 or None — unit cell in Å
    time: float                         # ps


class Trajectory(ABC):
    @property
    @abstractmethod
    def n_atoms(self) -> int: ...

    @property
    @abstractmethod
    def n_frames(self) -> int: ...

    @abstractmethod
    def read_frame(self, index: int) -> Frame: ...

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n_frames):
            yield self.read_frame(i)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self.n_frames
            return self.read_frame(i)
        if isinstance(key, slice):
            return SlicedTrajectory(self, list(range(*key.indices(self.n_frames))))
        if isinstance(key, (list, tuple, np.ndarray)):
            return SlicedTrajectory(self, [int(i) for i in key])
        raise TypeError(f"Cannot index trajectory with {type(key).__name__}")

    def coordinates(self, selection: np.ndarray | None = None) -> np.ndarray:
        """Stack coordinates for `selection` across all frames → (n_frames, n_sel, 3)."""
        coords, _ = self.coordinates_and_times(selection)
        return coords

    def coordinates_and_times(
        self, selection: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Like :meth:`coordinates`, but also returns per-frame times (ps).

        Fast path: if the underlying reader exposes ``coordinates_slab`` /
        ``times_slab`` (e.g. AMBER NetCDF), the whole trajectory is pulled
        in one C-level batched read instead of a per-frame Python loop.
        """
        slab = getattr(self, "coordinates_slab", None)
        tslab = getattr(self, "times_slab", None)
        if callable(slab) and callable(tslab):
            coords = slab(0, self.n_frames, selection)
            times = tslab(0, self.n_frames)
            return coords, times

        if selection is None:
            sel: slice | np.ndarray = slice(None)
            n_sel = self.n_atoms
        else:
            sel = np.asarray(selection, dtype=np.int64)
            n_sel = sel.size
        out = np.empty((self.n_frames, n_sel, 3), dtype=np.float32)
        times = np.empty(self.n_frames, dtype=np.float64)
        for i, frame in enumerate(self):
            out[i] = frame.coordinates[sel]
            times[i] = frame.time
        return out, times

    def times(self) -> np.ndarray:
        """Return per-frame time in ps. (n_frames,) float64."""
        tslab = getattr(self, "times_slab", None)
        if callable(tslab):
            return tslab(0, self.n_frames)
        out = np.empty(self.n_frames, dtype=np.float64)
        for i, frame in enumerate(self):
            out[i] = frame.time
        return out


class SlicedTrajectory(Trajectory):
    """View over a parent trajectory restricted to a list of frame indices."""

    def __init__(self, parent: Trajectory, indices: list[int]):
        self._parent = parent
        self._indices = indices

    @property
    def n_atoms(self) -> int:
        return self._parent.n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._indices)

    def read_frame(self, index: int) -> Frame:
        if not (0 <= index < self.n_frames):
            raise IndexError(index)
        f = self._parent.read_frame(self._indices[index])
        return Frame(index=index, coordinates=f.coordinates, box=f.box, time=f.time)
