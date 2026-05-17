"""I/O dispatch — looks up readers by file extension."""

from __future__ import annotations

import os
from typing import Callable

from post_md.core.topology import Topology
from post_md.core.trajectory import Trajectory

_TOPOLOGY_READERS: dict[str, Callable[[str], Topology]] = {}
_TRAJECTORY_READERS: dict[str, Callable[[str, int], Trajectory]] = {}


def _register_default_readers() -> None:
    if _TOPOLOGY_READERS:
        return
    from post_md.io.amber.mdcrd import AmberMdcrdTrajectory
    from post_md.io.amber.netcdf import AmberNetCDFTrajectory
    from post_md.io.amber.prmtop import read_prmtop
    from post_md.io.gromacs.gro import GroTrajectory, read_gro_topology
    from post_md.io.gromacs.trr import TrrTrajectory
    from post_md.io.gromacs.xtc import XtcTrajectory
    from post_md.io.pdb import read_pdb_topology

    _TOPOLOGY_READERS.update({
        ".prmtop": read_prmtop,
        ".parm7": read_prmtop,
        ".gro": read_gro_topology,
        ".pdb": read_pdb_topology,
    })
    _TRAJECTORY_READERS.update({
        ".nc": AmberNetCDFTrajectory.open,
        ".ncdf": AmberNetCDFTrajectory.open,
        ".mdcrd": AmberMdcrdTrajectory.open,
        ".crd": AmberMdcrdTrajectory.open,
        ".trr": TrrTrajectory.open,
        ".xtc": XtcTrajectory.open,
        ".gro": GroTrajectory.open,
    })


def open_topology(path: str) -> Topology:
    _register_default_readers()
    ext = os.path.splitext(path)[1].lower()
    reader = _TOPOLOGY_READERS.get(ext)
    if reader is None:
        raise ValueError(f"No topology reader registered for extension {ext!r}")
    return reader(path)


def open_trajectory(path: str, n_atoms: int) -> Trajectory:
    _register_default_readers()
    ext = os.path.splitext(path)[1].lower()
    reader = _TRAJECTORY_READERS.get(ext)
    if reader is None:
        raise ValueError(f"No trajectory reader registered for extension {ext!r}")
    return reader(path, n_atoms)


__all__ = ["open_topology", "open_trajectory"]
