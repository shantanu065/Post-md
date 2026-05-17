# Post_MD

Open-source toolkit for rapid analysis of molecular dynamics trajectories.

Post_MD provides:

- A **pure-Python** trajectory and topology library — no MDAnalysis / MDTraj / netCDF4 dependency. Every parser is written from the public spec.
- First-class support for **AMBER** (`.prmtop` + NetCDF `.nc` + `.mdcrd`) and **GROMACS** (`.gro` + `.xtc` + `.trr`).
- Standard analyses out of the box: RMSD, RMSF, Rg, **Cartesian PCA**, and **k-means structural clustering** in PC space.
- A scriptable CLI (`post-md …`) covering the full pipeline end-to-end. Default output format is **`.dat`** (cpptraj-style); GROMACS **`.xvg`** and **`.csv`** are picked automatically from the extension. Every analysis command also writes a matching PNG plot with publication-quality controls (title, axes, colors, legend, time-axis, DPI, font size).
- A **browser UI** (`post-md web`) that exposes every CLI option as a form field — for users who'd rather click than type.
- A self-hosted Docker web app (planned for v2) with an embedded NGL / Mol\* viewer for representative cluster frames and PC modes.

**Status:** pre-alpha. API and CLI flags may change before 1.0.

---

## Requirements

- Python **3.10 – 3.13**
- NumPy, SciPy, Typer, Rich (installed automatically)
- Windows / macOS / Linux

---

## 5-minute first run

This walkthrough takes you from a fresh clone to a clustered PCA pipeline with no external data downloads. Every command is copy-pasteable.

### Step 1 — Get the code and create a venv

```powershell
# Windows PowerShell
git clone https://github.com/post-md/post-md.git
cd post-md
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
git clone https://github.com/post-md/post-md.git
cd post-md
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install the package

```bash
pip install -e .
```

Verify the `post-md` console script is on your PATH:

```bash
post-md --help
```

You should see six commands listed: `info`, `rmsd`, `rmsf`, `rg`, `pca`, `cluster`.

### Step 3 — Generate a tiny demo trajectory

The repo ships with a generator that builds a 60-frame synthetic dipeptide in GROMACS `.gro` format. No external downloads required.

```bash
python scripts/make_demo_trajectory.py demo.gro
```

```
Wrote demo.gro (60 frames, 6 atoms).
```

The same file is both topology and trajectory, since `.gro` carries both.

### Step 4 — Inspect the system

```bash
post-md info -p demo.gro -t demo.gro
```

Expected output:

```
Universe: <Universe n_atoms=6 n_residues=2 n_frames=60>
  atoms:    6
  residues: 2
  bonds:    0
  frames:   60
```

### Step 5 — Per-frame RMSD vs frame 0

```bash
post-md rmsd -p demo.gro -t demo.gro -s "name CA" --reference 0 -o rmsd.dat
```

Output:

```
Wrote rmsd.dat (60 rows)
Wrote rmsd.png
```

Produces **two** files:

- `rmsd.dat` — cpptraj-style whitespace table (`# Frame  RMSD (A)` header, then two columns). Row 0 is ~0 (the reference); the rest fluctuate.
- `rmsd.png` — line plot of RMSD vs frame, ready for a report or notebook.

Want a different table format? Just change the extension — Post_MD picks the writer from it:

- `rmsd.dat` → cpptraj-style (whitespace, `#` header) — **default**
- `rmsd.xvg` → GROMACS xmgrace (`@` metadata + xy data) — drops straight into `xmgrace` or `gmx analyze`
- `rmsd.csv` → comma-separated, with a header row

Every analysis command (`rmsd`, `rmsf`, `rg`, `pca`, `cluster`) follows the same convention: a data file plus a matching PNG. Add `--no-plot` to skip the image.

### Step 6 — Cartesian PCA

```bash
post-md pca -p demo.gro -t demo.gro -s "all" --n-components 3 -o pca.npz
```

Expected (numbers will vary slightly with NumPy version):

```
Wrote pca.npz
Cumulative variance explained: [0.866 0.938 1.000]
Wrote pca.png
```

PC1 alone captures ~87% of the internal motion — exactly the slow hinge baked into the demo generator. `pca.png` is a two-panel figure with the scree plot on the left and a PC1-vs-PC2 scatter (frame-coloured) on the right.

`pca.npz` contains:

| key | shape | meaning |
| --- | --- | --- |
| `mean` | `(n_atoms, 3)` | mean structure |
| `eigenvalues` | `(n_components,)` | variance per PC |
| `components` | `(n_components, n_atoms, 3)` | eigenvectors reshaped |
| `projections` | `(n_frames, n_components)` | trajectory in PC space |
| `selection_indices` | `(n_atoms,)` | atom indices included |
| `atom_names` | `(n_atoms,)` | atom names |

### Step 7 — Cluster the PCA projections

```bash
post-md cluster --pca pca.npz --k 3 --n-pcs 3 -o clusters.dat \
                --rep-pdb reps.pdb -p demo.gro -t demo.gro
```

You now have:

- `clusters.dat` — two-column `frame  cluster` table per frame (cpptraj-style).
- `clusters.png` — PC1-vs-PC2 scatter colored by cluster, with k-means centers (✕) and representative medoids (★) marked.
- `reps.pdb` — a multi-MODEL PDB containing one representative (medoid) frame per cluster.

You can re-cluster with a different `--k` without re-running PCA — projections are reused from `pca.npz`.

### Step 8 — Inspect the representatives

Open `reps.pdb` in any molecular viewer:

- **PyMOL:** `pymol reps.pdb`
- **VMD:** `vmd reps.pdb`
- **NGL Viewer / Mol\*:** drag-and-drop into the web UI

Each `MODEL` is one cluster representative.

You're done. Replace `demo.gro` with your own `.prmtop` + `.nc` (AMBER) or `.gro` + `.xtc` (GROMACS) and the same commands work unchanged.

---

## CLI reference

All commands take a **topology** (`-p`) and a **trajectory** (`-t`); format is detected by extension.

Every analysis command (everything except `info`) writes a data file *and* a PNG plot next to it (same stem, `.png`). Pass `--no-plot` to skip the image.

| Command | What it does | Key flags | Default data file | Plot |
| --- | --- | --- | --- | --- |
| `info` | Print n_atoms, n_residues, n_bonds, n_frames | — | (stdout) | — |
| `rmsd` | Per-frame RMSD vs a reference frame (Kabsch-aligned) | `-s`, `--reference N`, `-o`, `--dt`, `--time-unit` | `rmsd.dat` | line plot |
| `rmsf` | Per-atom RMS fluctuation about the mean structure | `-s`, `-o` | `rmsf.dat` | line plot |
| `rg` | Per-frame radius of gyration | `-s`, `-o`, `--dt`, `--time-unit` | `rg.dat` | line plot |
| `pca` | Cartesian PCA → modes + projections | `-s`, `--n-components`, `-o` | `pca.npz` | scree + projection scatter |
| `cluster` | K-means on PCA projections, optional representative PDB | `--pca`, `--k`, `--n-pcs`, `--rep-pdb`, `-p`, `-t`, `--seed` | `clusters.dat` | cluster scatter |

### Plot customisation flags

Every analysis command supports a **Plot customisation** option group (visible via `post-md <cmd> --help`):

| Flag | Applies to | What it does |
| --- | --- | --- |
| `--title TEXT` | all plots | Override the figure title |
| `--xlabel TEXT` / `--ylabel TEXT` | all plots | Override axis labels |
| `--color TEXT` | all plots | Line color (rmsd/rmsf/rg), scree bar color (pca), center-marker color (cluster). Accepts matplotlib name or hex (`red`, `#1f77b4`). |
| `--accent-color TEXT` | pca, cluster | Cumulative-variance line (pca), representative-marker color (cluster) |
| `--cmap TEXT` | pca, cluster | Colormap for scatters (default `viridis` for pca, `tab10` for cluster) |
| `--linewidth FLOAT` | line plots | Default 1.6 |
| `--legend-label TEXT` / `--no-legend` | line plots | Add a labelled legend; omit to hide |
| `--no-grid` | all plots | Disable the grid |
| `--figsize WxH` | all plots | e.g. `8x5` (inches) |
| `--dpi N` | all plots | Default 150 |
| `--font-size N` | all plots | Default 12 |
| `--dt FLOAT` | rmsd, rg | Time step per frame in **ps** — switches x-axis to real time |
| `--time-unit {ps,ns,fs,us}` | rmsd, rg | Display unit when `--dt` is set (default `ps`) |

When `--dt` isn't given, Post_MD uses the frame times stored inside the trajectory (AMBER NetCDF, GROMACS TRR/XTC carry real times in their headers); ASCII formats (mdcrd, gro) fall back to frame index automatically.

### Example: a publication-style RMSD figure

```bash
post-md rmsd -p system.prmtop -t traj.nc -s "backbone" -o rmsd.xvg \
             --dt 2.0 --time-unit ns \
             --title "Backbone RMSD" \
             --ylabel "RMSD (Å)" \
             --color "#1f77b4" --linewidth 2.0 \
             --legend-label "WT system" \
             --figsize 9x4.5 --dpi 300 --font-size 13
```

Output:

- `rmsd.xvg` — GROMACS-format table (loadable in xmgrace, gmx analyze, or `numpy.loadtxt`)
- `rmsd.png` — 300-DPI figure: blue line, Time-in-ns x-axis, legend, large fonts — ready to drop into a manuscript.

### Atom-selection mini-DSL

```
name CA                       # atom names (space-separated list)
resname ALA TRP               # residue names
resid 1-50 65-90              # residue ranges (1-indexed)
index 0-99                    # atom indices (0-indexed)
protein                       # all standard amino acid residues
backbone                      # N CA C O HA H on protein residues
all                           # everything

# Boolean combinators
(name CA or name C) and resname ALA
not (resid 50-60)
```

Geometric selections (`around 5.0 protein`) are deferred to v2.

---

## Python API

The CLI is a thin wrapper over the library:

```python
from post_md import Universe
from post_md.analysis.pca import pca_cartesian
from post_md.analysis.clustering import kmeans_cluster

u = Universe.load("demo.gro", "demo.gro")
print(u)                                          # n_atoms, n_residues, n_frames

ca = u.select_atoms("name CA")                    # AtomGroup view
coords = ca.coordinates()                         # (n_frames, n_atoms, 3) in Å

pca = pca_cartesian(coords, n_components=3)
print(pca.scree())                                # variance explained per PC

clusters = kmeans_cluster(pca.projections, k=3)
print(clusters.labels.shape, clusters.representative_frames)
```

Lower-level access bypassing `Universe.load`:

```python
from post_md.io import open_topology, open_trajectory

top = open_topology("system.prmtop")
traj = open_trajectory("traj.nc", top.n_atoms)
for frame in traj[:100]:                           # slice or iterate
    print(frame.index, frame.time, frame.coordinates.shape)
```

---

## Supported file formats

| Format | Topology | Trajectory | Notes |
| --- | :-: | :-: | --- |
| AMBER `.prmtop` / `.parm7` | ✅ | — | FORTRAN-FORMAT-aware parser |
| AMBER NetCDF `.nc` / `.ncdf` | — | ✅ | Own NetCDF3 classic + 64-bit-offset reader |
| AMBER ASCII `.mdcrd` / `.crd` | — | ✅ | Box auto-detected from total float count |
| GROMACS `.gro` | ✅ | ✅ | Single- or multi-frame; nm → Å on read |
| GROMACS `.trr` | — | ✅ | float32 / float64 auto-detected |
| GROMACS `.xtc` | — | ✅ | Full xdr3dfcoord decoder (see perf note below) |
| PDB `.pdb` | ✅ | — | ATOM / HETATM topology only |

> **XTC performance.** Our pure-Python XTC decoder is functionally correct but **10–50× slower than the C reference** because of the bit-level smallnum decoder. For large production trajectories this is the bottleneck. A native accelerator (Cython or Rust via PyO3) is planned as `post_md._xtc_fast` once the v1 API is stable; the pure-Python decoder will remain as the always-available fallback.

---

## Web UI (no CLI required)

Post_MD ships with a local browser UI that exposes **every** CLI option as a form field — useful when you'd rather click than type. It's a single-user local server (FastAPI + vanilla JS, no Docker, no auth, no queue); the multi-user Docker version is the v2 milestone.

### Install + launch

```bash
pip install -e ".[web]"     # one-time: pulls fastapi, uvicorn, jinja2, python-multipart
post-md web                  # launches http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000> in any browser. Useful flags:

```bash
post-md web --host 0.0.0.0 --port 8080 --workdir ./my_session
```

`--workdir` is where uploaded trajectories and generated outputs live; pass a fresh directory to start a clean session.

### Workflow

1. **Upload** a topology + trajectory (drag-and-drop or file picker). All formats from the CLI table are accepted; `.gro` works as both topology and trajectory in one file.
2. **Pick an analysis tab:** Info / RMSD / RMSF / Rg / PCA / Cluster.
3. **Fill in options.** Every CLI flag has a matching form field, grouped:
   - Required options (selection, reference frame, k, etc.) up front.
   - A collapsible **"Plot customisation"** section with title, axis labels, line color, accent color, colormap, line width, legend label, figure size, DPI, font size, grid toggle.
   - Output format dropdown: `.dat` / `.xvg` / `.csv`.
   - `--dt` and `--time-unit` fields on RMSD and Rg for time-axis plots.
4. **Click "Run".** The plot appears inline; the data file and PDB (for clustering) are listed under "Workspace files" for download.
5. **Re-cluster** with a different `k` without re-running PCA — the saved `pca.npz` is reused automatically.

### REST API

The web app is just a thin shell over a JSON API; it's documented at <http://127.0.0.1:8000/api/docs> when the server is running. Useful endpoints if you want to script against it:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | List the topology / trajectory / outputs currently in the workdir |
| `POST` | `/api/upload` | Multipart form: `topology=@file`, `trajectory=@file` |
| `POST` | `/api/info` | Same as `post-md info` against the uploaded files |
| `POST` | `/api/run/{analysis}` | JSON body of options; `analysis` ∈ `rmsd / rmsf / rg / pca / cluster` |
| `GET` | `/api/result/{filename}` | Download any output file (data, plot, PDB) |
| `POST` | `/api/reset` | Wipe the workdir |

Example: drive an analysis from `curl`:

```bash
curl -F "topology=@system.prmtop" -F "trajectory=@traj.nc" \
     http://127.0.0.1:8000/api/upload

curl -X POST -H "Content-Type: application/json" \
     -d '{"selection":"backbone","reference":0,"output_ext":"xvg",
          "dt":2.0,"time_unit":"ns","color":"#1f77b4","dpi":300}' \
     http://127.0.0.1:8000/api/run/rmsd
```

### Caveats

- The server is **single-user**: uploading a new topology / trajectory replaces the previous one. Don't expose it on a multi-tenant host.
- No long-running job queue — each request runs synchronously in the request handler. Fine for the demo and small trajectories; large XTC files will block the worker thread until they finish.
- For multi-user hosted deployments, wait for v2 (FastAPI + Redis + RQ worker + Docker, with sessions and quotas).

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

27 unit tests cover the selection grammar, Kabsch alignment, RMSD / RMSF / Rg, PCA, k-means, the XDR primitives, and CLI command registration — none require external trajectory files.

End-to-end I/O tests against real `.prmtop` + `.nc` and `.gro` + `.xtc` fixtures are still **TODO** — drop small (≤ 50 KB) reference trajectories into `tests/data/` and they'll be exercised automatically.

Lint + type-check:

```bash
ruff check .
ruff format .
mypy src/post_md
```

---

## Project layout

```
src/post_md/
├── core/        # Topology, Trajectory, Universe, AtomGroup, selection DSL
├── io/          # Format readers
│   ├── amber/   # .prmtop  .nc  .mdcrd
│   ├── gromacs/ # .gro  .trr  .xtc  (+ XDR primitives)
│   └── pdb.py
├── analysis/    # alignment, rmsd, rmsf, rg, pca, clustering
└── cli/         # Typer entry point
scripts/
└── make_demo_trajectory.py    # generates demo.gro for the walkthrough
tests/
```

---

## Roadmap

- **v1** *(current)* — Library + CLI + local browser UI (`post-md web`) for AMBER and GROMACS, AGPL + commercial dual licensing.
- **v1.5** — Native XTC accelerator extension; geometric selection grammar; trajectory writers; embedded NGL / Mol\* viewer in the web UI.
- **v2** — Multi-user Docker deployment (FastAPI + Redis + worker + sessions + quotas) for hosted PCA / clustering workflows.

---

## License

Post_MD is **dual licensed**:

- **AGPL-3.0-or-later** for academic, research, and open-source use. See [LICENSE](LICENSE).
- A separate **commercial license** is available for proprietary and closed-source use. See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

Contributors must agree to the CLA terms in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Contact

Issues / questions: open a GitHub issue.
Commercial-license inquiries: `shantanukumar065@gmail.com`.
