"""Generate a synthetic 6-atom dipeptide trajectory in GROMACS .gro format.

Run::

    python scripts/make_demo_trajectory.py [output.gro]

The output file works as both topology and trajectory for the `post-md` CLI,
so the README walkthrough only needs this one file.

The system is a fake ALA–GLY backbone (N, CA, C × 2 residues) animated with
a slow collective hinge plus thermal noise — enough internal motion that
PCA recovers a dominant PC1 and k-means finds clear basins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_frames(n_frames: int = 60, seed: int = 0) -> np.ndarray:
    """Return (n_frames, 6, 3) coordinates in nm."""
    rng = np.random.default_rng(seed)
    # Base layout: stretched along x at 0.15 nm spacing.
    base = np.array(
        [
            [0.00, 0.00, 0.00],  # res1 N
            [0.15, 0.00, 0.00],  # res1 CA
            [0.30, 0.00, 0.00],  # res1 C
            [0.45, 0.00, 0.00],  # res2 N
            [0.60, 0.00, 0.00],  # res2 CA
            [0.75, 0.00, 0.00],  # res2 C
        ],
        dtype=np.float64,
    )
    frames = np.broadcast_to(base, (n_frames, 6, 3)).copy()
    # Slow hinge: residue-2 backbone arcs up and back over the run.
    t = np.linspace(0, 2 * np.pi, n_frames)
    frames[:, 3:, 1] += 0.15 * np.sin(t)[:, None]
    frames[:, 3:, 2] += 0.05 * np.cos(t)[:, None]
    # Thermal noise.
    frames += rng.normal(scale=0.005, size=frames.shape)
    return frames.astype(np.float64)


ATOM_RECORDS = [
    (1, "ALA", "N"),
    (1, "ALA", "CA"),
    (1, "ALA", "C"),
    (2, "GLY", "N"),
    (2, "GLY", "CA"),
    (2, "GLY", "C"),
]


def write_gro(path: Path, frames: np.ndarray) -> None:
    box = np.array([2.0, 2.0, 2.0])  # 2 nm cubic box
    with path.open("w") as f:
        for i, coords in enumerate(frames):
            f.write(f"Post_MD demo dipeptide, frame {i}\n")
            f.write(f"{len(ATOM_RECORDS):5d}\n")
            for atom_idx, ((res_id, res_name, atom_name), xyz) in enumerate(
                zip(ATOM_RECORDS, coords, strict=True), start=1
            ):
                x, y, z = xyz
                f.write(
                    f"{res_id:5d}{res_name:<5s}{atom_name:>5s}{atom_idx:5d}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
                )
            f.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output", nargs="?", default="demo.gro", type=Path)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = build_frames(n_frames=args.n_frames, seed=args.seed)
    write_gro(args.output, frames)
    print(f"Wrote {args.output} ({args.n_frames} frames, 6 atoms).")


if __name__ == "__main__":
    main()
