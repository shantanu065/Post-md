"""FastAPI app for the Post_MD local web UI.

Single-user, no auth, no queue. All uploads + outputs live in one workdir
(``./post_md_web`` by default). Uploading a new topology / trajectory
replaces the previous one. Suitable for local desktop use; the v2 plan
adds sessions + a worker pool for the multi-user Docker deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from post_md.web.runners import RUNNERS, run_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Trajectory/topology files accepted by the readers.
_TOPOLOGY_EXTS = {".prmtop", ".parm7", ".gro", ".pdb"}
_TRAJECTORY_EXTS = {".nc", ".ncdf", ".mdcrd", ".crd", ".trr", ".xtc", ".gro"}


def _safe_workdir_path(workdir: Path, name: str) -> Path:
    """Resolve `name` under workdir and reject any traversal."""
    candidate = (workdir / name).resolve()
    if workdir not in candidate.parents and candidate != workdir:
        raise HTTPException(status_code=400, detail="path escapes workdir")
    return candidate


def _state(workdir: Path) -> dict[str, Any]:
    """Snapshot of the workdir: which topology/trajectory are loaded + what outputs exist."""
    topology = None
    trajectory = None
    for entry in workdir.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in _TOPOLOGY_EXTS and topology is None:
            topology = entry.name
        if ext in _TRAJECTORY_EXTS:
            trajectory = entry.name
    outputs = sorted(
        p.name for p in workdir.iterdir()
        if p.is_file() and p.suffix.lower() in {".dat", ".csv", ".xvg", ".npz", ".png", ".pdb"}
    )
    return {
        "topology": topology,
        "trajectory": trajectory,
        "outputs": outputs,
    }


def create_app(workdir: str | Path) -> FastAPI:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Post_MD", docs_url="/api/docs", openapi_url="/api/openapi.json")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return _state(workdir)

    @app.post("/api/upload")
    async def upload(
        topology: UploadFile | None = File(None),
        trajectory: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        saved: list[str] = []
        for f in (topology, trajectory):
            if f is None:
                continue
            name = Path(f.filename or "").name
            if not name:
                continue
            ext = Path(name).suffix.lower()
            if ext not in _TOPOLOGY_EXTS | _TRAJECTORY_EXTS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported extension {ext!r}. Allowed: "
                           f"{sorted(_TOPOLOGY_EXTS | _TRAJECTORY_EXTS)}",
                )
            # Clear any previous file with the same role (topology vs trajectory)
            target_set = _TOPOLOGY_EXTS if ext in _TOPOLOGY_EXTS and ext not in _TRAJECTORY_EXTS else None
            if ext == ".gro":
                # .gro is both — treat per-upload (clear other .gro too)
                target_set = {".gro"}
            elif ext in _TOPOLOGY_EXTS:
                target_set = _TOPOLOGY_EXTS
            else:
                target_set = _TRAJECTORY_EXTS
            for existing in list(workdir.iterdir()):
                if existing.is_file() and existing.suffix.lower() in target_set:
                    try:
                        existing.unlink()
                    except OSError:
                        pass
            dest = _safe_workdir_path(workdir, name)
            with dest.open("wb") as out:
                while True:
                    chunk = await f.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            saved.append(dest.name)
        return {"saved": saved, "state": _state(workdir)}

    @app.post("/api/info")
    async def info_endpoint() -> dict[str, Any]:
        st = _state(workdir)
        if not st["topology"] or not st["trajectory"]:
            raise HTTPException(status_code=400, detail="Upload a topology and trajectory first.")
        try:
            return run_info(workdir / st["topology"], workdir / st["trajectory"])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/run/{analysis}")
    async def run_analysis(analysis: str, payload: dict[str, Any]) -> dict[str, Any]:
        if analysis not in RUNNERS:
            raise HTTPException(status_code=404, detail=f"Unknown analysis {analysis!r}")
        st = _state(workdir)
        if analysis != "cluster":
            if not st["topology"] or not st["trajectory"]:
                raise HTTPException(status_code=400, detail="Upload a topology and trajectory first.")
        else:
            if not (workdir / "pca.npz").exists():
                raise HTTPException(status_code=400, detail="Run PCA before clustering.")
            if payload.get("rep_pdb") and (not st["topology"] or not st["trajectory"]):
                raise HTTPException(status_code=400, detail="Representative PDB needs topology + trajectory uploaded.")
        try:
            top_path = workdir / (st["topology"] or "")
            traj_path = workdir / (st["trajectory"] or "")
            result = RUNNERS[analysis](top_path, traj_path, workdir, payload or {})
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
        return {"result": result, "state": _state(workdir)}

    @app.get("/api/result/{filename}")
    async def get_result(filename: str) -> FileResponse:
        path = _safe_workdir_path(workdir, filename)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        media_type = None
        if path.suffix.lower() == ".png":
            media_type = "image/png"
        return FileResponse(str(path), media_type=media_type, filename=path.name)

    @app.post("/api/reset")
    async def reset() -> dict[str, Any]:
        for entry in workdir.iterdir():
            if entry.is_file():
                try:
                    entry.unlink()
                except OSError:
                    pass
        return {"state": _state(workdir)}

    return app
