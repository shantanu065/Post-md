"""Universe = Topology + Trajectory facade. The main user entry point."""

from __future__ import annotations

from post_md.core.atomgroup import AtomGroup
from post_md.core.selection import select
from post_md.core.topology import Topology
from post_md.core.trajectory import Trajectory


class Universe:
    def __init__(self, topology: Topology, trajectory: Trajectory):
        if trajectory.n_atoms != topology.n_atoms:
            raise ValueError(
                f"Topology has {topology.n_atoms} atoms but trajectory has "
                f"{trajectory.n_atoms}"
            )
        self.topology = topology
        self.trajectory = trajectory

    @classmethod
    def load(cls, topology_path: str, trajectory_path: str) -> Universe:
        from post_md.io import open_topology, open_trajectory

        top = open_topology(topology_path)
        traj = open_trajectory(trajectory_path, top.n_atoms)
        return cls(top, traj)

    def select_atoms(self, query: str) -> AtomGroup:
        indices = select(self.topology, query)
        return AtomGroup(self, indices)

    def __repr__(self) -> str:
        return (
            f"<Universe n_atoms={self.topology.n_atoms} "
            f"n_residues={self.topology.n_residues} "
            f"n_frames={self.trajectory.n_frames}>"
        )
