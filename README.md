# Post_MD

Open-source toolkit for rapid analysis of molecular dynamics trajectories.

- **Pure-Python** — no MDAnalysis / MDTraj dependency
- **AMBER** and **GROMACS** trajectory support
- **RMSD, RMSF, Rg, SASA, H-bond, PCA, k-means clustering**
- **Multi-system comparison** — overlay WT vs mutants on one plot
- **CLI** and **browser UI**

## Install

```bash
pip install -e "."          # CLI
pip install -e ".[web]"     # CLI + browser UI
```

Python 3.10+ required.

## Quick start

```bash
# see all commands
post-md --help

# run analyses
post-md info -p system.prmtop -t traj.nc
post-md rmsd -p system.prmtop -t traj.nc -s "name CA" -o rmsd.dat
post-md rmsf -p system.prmtop -t traj.nc -s "protein" -o rmsf.dat
post-md rg   -p system.prmtop -t traj.nc -o rg.dat
post-md sasa -p system.prmtop -t traj.nc -o sasa.dat
post-md hbond -p system.prmtop -t traj.nc -o hbond.dat
post-md pca  -p system.prmtop -t traj.nc -s "name CA" -o pca.npz
post-md cluster --pca pca.npz --k 5 -o clusters.dat -p system.prmtop -t traj.nc
```

Each command writes a data file + PNG plot.

## Web UI

```bash
post-md web
```

Opens at http://127.0.0.1:8000. Upload trajectory files, choose analyses, run, and view results with live progress.

## Supported formats

| Format | Topology | Trajectory |
| --- | :-: | :-: |
| AMBER `.prmtop` / `.parm7` | yes | — |
| AMBER NetCDF `.nc` | — | yes |
| AMBER ASCII `.mdcrd` | — | yes |
| GROMACS `.gro` | yes | yes |
| GROMACS `.trr` | — | yes |
| GROMACS `.xtc` | — | yes |
| PDB `.pdb` | yes | — |

## License

MIT
