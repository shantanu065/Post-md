# Post_MD

Open-source toolkit for rapid analysis of molecular dynamics trajectories.

- **Pure-Python** — trajectory parsers written from scratch, no heavyweight dependencies
- **AMBER** and **GROMACS** trajectory support
- **RMSD, RMSF, Rg, SASA, H-bond, PCA, k-means clustering**
- **Multi-system comparison** — overlay WT vs mutants on one plot
- **Antibody-aware RMSF** — colour antigen vs antibody/nanobody and mark CDR loops
- **MMGBSA plotting** — per-residue ΔG hotspots or per-frame binding energy
- **CLI** and **browser UI**

## Install

Python 3.10+ required.

```bash
git clone https://github.com/shantanu065/Post-md.git
cd Post-md

# create and activate a virtual environment named "post-md" (recommended)
python -m venv post-md
source post-md/bin/activate      
# Windows: post-md\Scripts\activate

# install
pip install -e "."          # CLI
pip install -e ".[web]"     # CLI + browser UI
```

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

## Contributing

This project is published as-is and is **not accepting external contributions**
(issues and pull requests are closed). You are welcome to clone and use it under
the terms of the license below.

## License

Released under the [MIT License](LICENSE). See [NOTICE](NOTICE) for third-party
trademark acknowledgements.
