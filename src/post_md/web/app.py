"""FastAPI app for the Post_MD local web UI.

Single-user, no auth, no queue. Files can be provided in two ways:

1. **Upload** — streamed via ``POST /api/upload``, saved into the workdir.
2. **Reference an existing file on disk** — ``POST /api/use-path`` records
   the absolute path in ``.post_md_state.json`` and the analysis runners
   load directly from there. Best for multi-GB / TB trajectories that you
   don't want to copy.

Both modes feed the same state shape, so the rest of the API is unchanged.
The v2 plan adds sessions + a worker pool for the multi-user Docker
deployment.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from post_md.web.runners import RUNNERS, run_info

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_TOPOLOGY_EXTS = {".prmtop", ".parm7", ".gro", ".pdb"}
_TRAJECTORY_EXTS = {".nc", ".ncdf", ".mdcrd", ".crd", ".trr", ".xtc", ".gro"}
_STATE_FILENAME = ".post_md_state.json"
_OUTPUT_EXTS = {".dat", ".csv", ".xvg", ".npz", ".png", ".pdb"}

# Disk write buffer for streaming uploads. Larger = fewer syscalls = faster
# throughput, at the cost of more memory while a chunk is being absorbed.
_UPLOAD_WRITE_BUFFER = 8 * 1024 * 1024  # 8 MiB
# Flush-to-disk threshold inside the async chunk loop. Chunks are accumulated
# in a bytearray and handed off to a worker thread in big batches, so the
# event loop is not blocked once per ~16 KiB HTTP chunk.
_UPLOAD_FLUSH_THRESHOLD = 4 * 1024 * 1024  # 4 MiB


def _safe_workdir_path(workdir: Path, name: str) -> Path:
    """Resolve `name` under workdir and reject any traversal."""
    candidate = (workdir / name).resolve()
    if workdir not in candidate.parents and candidate != workdir:
        raise HTTPException(status_code=400, detail="path escapes workdir")
    return candidate


def _state_file(workdir: Path) -> Path:
    return workdir / _STATE_FILENAME


def _read_refs(workdir: Path) -> dict[str, str]:
    """Read .post_md_state.json. Returns {} on missing/corrupt."""
    sf = _state_file(workdir)
    if not sf.exists():
        return {}
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: str(v) for k, v in data.items() if k in ("topology", "trajectory") and v}


def _write_refs(workdir: Path, refs: dict[str, str]) -> None:
    """Persist (or delete) .post_md_state.json."""
    sf = _state_file(workdir)
    if refs:
        sf.write_text(json.dumps(refs, indent=2), encoding="utf-8")
    elif sf.exists():
        try:
            sf.unlink()
        except OSError:
            pass


def _state(workdir: Path) -> dict[str, Any]:
    """Combined view of uploaded files + referenced paths.

    Returns a dict with:
      - topology, trajectory          → display names (basenames)
      - topology_path, trajectory_path → absolute paths (where the runner reads from)
      - topology_source, trajectory_source → "upload" or "path"
      - outputs                       → list of result filenames in workdir
    """
    topology = trajectory = None
    topology_path = trajectory_path = None
    topology_source = trajectory_source = None

    # 1. Fallback: scan workdir for any uploaded topology/trajectory.
    for entry in workdir.iterdir():
        if not entry.is_file() or entry.name == _STATE_FILENAME:
            continue
        ext = entry.suffix.lower()
        if ext in _TOPOLOGY_EXTS and topology is None:
            topology = entry.name
            topology_path = str(entry)
            topology_source = "upload"
        if ext in _TRAJECTORY_EXTS:
            trajectory = entry.name
            trajectory_path = str(entry)
            trajectory_source = "upload"

    # 2. Override with explicit path references, when valid.
    refs = _read_refs(workdir)
    for role in ("topology", "trajectory"):
        raw = refs.get(role)
        if not raw:
            continue
        p = Path(raw)
        if not p.exists() or not p.is_file():
            # Stale reference — skip but keep the entry around so the user
            # can see it failed (we surface as `_stale` flag for clarity).
            continue
        if role == "topology":
            topology = p.name
            topology_path = str(p)
            topology_source = "path"
        else:
            trajectory = p.name
            trajectory_path = str(p)
            trajectory_source = "path"

    outputs = sorted(
        p.name for p in workdir.iterdir()
        if p.is_file()
        and p.name != _STATE_FILENAME
        and p.suffix.lower() in _OUTPUT_EXTS
    )

    return {
        "topology": topology,
        "trajectory": trajectory,
        "topology_path": topology_path,
        "trajectory_path": trajectory_path,
        "topology_source": topology_source,
        "trajectory_source": trajectory_source,
        "outputs": outputs,
    }


def _clear_uploaded_role(workdir: Path, role: str, new_ext: str | None = None) -> None:
    """Delete any uploaded files belonging to `role` from workdir.

    If `new_ext == '.gro'`, only other `.gro` files are removed (since
    `.gro` overlaps topology + trajectory roles).
    """
    if new_ext == ".gro":
        target_set = {".gro"}
    else:
        target_set = _TOPOLOGY_EXTS if role == "topology" else _TRAJECTORY_EXTS
    for existing in list(workdir.iterdir()):
        if (
            existing.is_file()
            and existing.name != _STATE_FILENAME
            and existing.suffix.lower() in target_set
        ):
            try:
                existing.unlink()
            except OSError:
                pass


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

    def _validate_upload_params(role: str | None, filename_raw: str | None) -> tuple[str, str]:
        if role not in ("topology", "trajectory") or not filename_raw:
            raise HTTPException(
                400,
                "Missing 'role' (topology|trajectory) or 'filename' query parameter.",
            )
        name = Path(filename_raw).name
        if not name:
            raise HTTPException(400, "Empty filename.")
        ext = Path(name).suffix.lower()
        allowed = _TOPOLOGY_EXTS if role == "topology" else _TRAJECTORY_EXTS
        if ext not in allowed:
            raise HTTPException(
                415,
                f"Unsupported {role} extension {ext!r}. Allowed: {sorted(allowed)}",
            )
        return name, ext

    @app.post("/api/upload")
    async def upload(request: Request) -> dict[str, Any]:
        """Single-request streaming upload (legacy / small files).

        Body is raw file bytes; `role` and `filename` come in via query
        string. Disk writes are batched and run in a worker thread so the
        event loop stays responsive even for multi-GB transfers. Larger
        files should prefer the resumable chunked endpoints below.
        """
        role = request.query_params.get("role")
        filename_raw = request.query_params.get("filename")
        name, ext = _validate_upload_params(role, filename_raw)

        _clear_uploaded_role(workdir, role, new_ext=ext)

        refs = _read_refs(workdir)
        if role in refs:
            refs.pop(role)
            _write_refs(workdir, refs)

        dest = _safe_workdir_path(workdir, name)
        bytes_written = 0
        loop = asyncio.get_running_loop()
        with dest.open("wb", buffering=_UPLOAD_WRITE_BUFFER) as out:
            buf = bytearray()
            async for chunk in request.stream():
                if not chunk:
                    continue
                buf.extend(chunk)
                bytes_written += len(chunk)
                if len(buf) >= _UPLOAD_FLUSH_THRESHOLD:
                    await loop.run_in_executor(None, out.write, bytes(buf))
                    buf.clear()
            if buf:
                await loop.run_in_executor(None, out.write, bytes(buf))

        return {
            "saved": dest.name,
            "bytes": bytes_written,
            "state": _state(workdir),
        }

    @app.get("/api/browse")
    async def browse(path: str | None = None, role: str | None = None) -> dict[str, Any]:
        """List a server-side directory for the on-disk file picker.

        Returns sub-directories plus files whose extension matches the
        given `role` (topology/trajectory). When `role` is omitted, all
        files are returned. When `path` is omitted, browsing starts at
        the workdir's parent (a sensible jump-off for the common case
        where the user runs the server from their data directory).

        This walks the host filesystem — fine for the default 127.0.0.1
        bind, but the caveat that applies to `/api/use-path` applies
        here too: don't expose this server to an untrusted network.
        """
        if not path:
            start = workdir.parent if workdir.parent != workdir else workdir
        else:
            start = Path(path).expanduser()
        try:
            start = start.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(400, f"Not found: {path}") from exc
        if not start.is_dir():
            raise HTTPException(400, f"Not a directory: {start}")

        if role == "topology":
            allowed: set[str] | None = _TOPOLOGY_EXTS
        elif role == "trajectory":
            allowed = _TRAJECTORY_EXTS
        elif role in (None, "", "any"):
            allowed = None
        else:
            raise HTTPException(400, f"Invalid role: {role!r}")

        dirs: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        try:
            entries = sorted(start.iterdir(), key=lambda p: p.name.lower())
        except PermissionError as exc:
            raise HTTPException(403, f"Permission denied: {start}") from exc

        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry)})
                elif entry.is_file():
                    ext = entry.suffix.lower()
                    if allowed is None or ext in allowed:
                        files.append({
                            "name": entry.name,
                            "path": str(entry),
                            "size": entry.stat().st_size,
                            "ext": ext,
                        })
            except (PermissionError, OSError):
                # Skip unreadable entries rather than failing the whole listing.
                continue

        parent = str(start.parent) if start.parent != start else None
        return {
            "path": str(start),
            "parent": parent,
            "dirs": dirs,
            "files": files,
            "role": role,
        }

    @app.get("/api/upload-status")
    async def upload_status(role: str, filename: str) -> dict[str, Any]:
        """How many bytes of `<role>/<filename>` are already on disk.

        Client uses this on resume: if its localStorage offset matches
        the server's `bytes` and the user picked the same file, the
        next chunk picks up where we left off.
        """
        name, _ = _validate_upload_params(role, filename)
        dest = _safe_workdir_path(workdir, name)
        if not dest.exists() or not dest.is_file():
            return {"role": role, "filename": name, "bytes": 0}
        return {"role": role, "filename": name, "bytes": dest.stat().st_size}

    @app.post("/api/upload-chunk")
    async def upload_chunk(request: Request) -> dict[str, Any]:
        """Append one chunk of a resumable upload.

        Query params:
          role     — 'topology' | 'trajectory'
          filename — final basename
          offset   — byte offset this chunk starts at (must equal current file size)
          total    — total file size in bytes (informational; used for the
                     final-chunk check and the response payload)
          final    — '1' on the last chunk; triggers state-finalisation
                     (clear stale path-refs, return _state)

        Sequential semantics: `offset == file.size`. This keeps resume
        trivially correct — the client just queries `/api/upload-status`
        and continues. Out-of-order chunks return 409.
        """
        qp = request.query_params
        role = qp.get("role")
        filename_raw = qp.get("filename")
        name, ext = _validate_upload_params(role, filename_raw)

        try:
            offset = int(qp.get("offset", ""))
            total = int(qp.get("total", ""))
        except ValueError as exc:
            raise HTTPException(400, "offset/total must be integers") from exc
        if offset < 0 or total < 0 or offset > total:
            raise HTTPException(400, "invalid offset/total")
        final_flag = qp.get("final") in ("1", "true", "True")

        dest = _safe_workdir_path(workdir, name)

        # First chunk of a fresh upload — wipe out any previous file in this
        # role, drop conflicting path-refs, and start a new file at offset 0.
        if offset == 0:
            _clear_uploaded_role(workdir, role, new_ext=ext)
            refs = _read_refs(workdir)
            if role in refs:
                refs.pop(role)
                _write_refs(workdir, refs)
            # Recreate as an empty file so the size check below is consistent.
            dest.write_bytes(b"")

        # Enforce sequential append.
        current = dest.stat().st_size if dest.exists() else 0
        if current != offset:
            raise HTTPException(
                status_code=409,
                detail=f"offset mismatch: server has {current} bytes, client sent {offset}",
            )

        bytes_written = 0
        loop = asyncio.get_running_loop()
        with dest.open("ab", buffering=_UPLOAD_WRITE_BUFFER) as out:
            buf = bytearray()
            async for chunk in request.stream():
                if not chunk:
                    continue
                buf.extend(chunk)
                bytes_written += len(chunk)
                if len(buf) >= _UPLOAD_FLUSH_THRESHOLD:
                    await loop.run_in_executor(None, out.write, bytes(buf))
                    buf.clear()
            if buf:
                await loop.run_in_executor(None, out.write, bytes(buf))

        new_size = dest.stat().st_size
        if new_size > total:
            raise HTTPException(
                status_code=400,
                detail=f"upload overflow: file is {new_size} bytes, declared total {total}",
            )
        if final_flag and new_size != total:
            # Client said "this is the last chunk" but the byte count doesn't
            # add up. Treat as malformed — caller's view of file size is wrong,
            # or they skipped a chunk. Better to fail loudly than mark the
            # upload finished and have downstream analyses read a truncated file.
            raise HTTPException(
                status_code=400,
                detail=f"final chunk size mismatch: server has {new_size} bytes, declared total {total}",
            )

        complete = new_size == total
        return {
            "role": role,
            "filename": name,
            "received": new_size,
            "total": total,
            "chunk_bytes": bytes_written,
            "complete": complete,
            "state": _state(workdir) if complete else None,
        }

    @app.post("/api/use-path")
    async def use_path(payload: dict[str, Any]) -> dict[str, Any]:
        """Register an existing file on disk as topology/trajectory.

        No copy, no symlink — the analysis runners read directly from
        the given absolute path. Designed for very large trajectories
        (multi-GB / TB) where uploading is impractical.
        """
        role = payload.get("role")
        raw_path = payload.get("path", "")
        if role not in ("topology", "trajectory"):
            raise HTTPException(400, "'role' must be 'topology' or 'trajectory'")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise HTTPException(400, "'path' is required")

        p = Path(raw_path.strip()).expanduser()
        try:
            p = p.resolve(strict=True)
        except (OSError, FileNotFoundError):
            raise HTTPException(400, f"File not found: {raw_path}") from None
        if not p.is_file():
            raise HTTPException(400, f"Not a regular file: {p}")
        ext = p.suffix.lower()
        allowed = _TOPOLOGY_EXTS if role == "topology" else _TRAJECTORY_EXTS
        if ext not in allowed:
            raise HTTPException(
                415,
                f"Unsupported {role} extension {ext!r}. Allowed: {sorted(allowed)}",
            )

        # Switching to a referenced path supersedes any uploaded file for this role.
        _clear_uploaded_role(workdir, role, new_ext=ext)

        refs = _read_refs(workdir)
        refs[role] = str(p)
        _write_refs(workdir, refs)

        return {
            "role": role,
            "path": str(p),
            "bytes": p.stat().st_size,
            "state": _state(workdir),
        }

    @app.post("/api/info")
    async def info_endpoint() -> dict[str, Any]:
        st = _state(workdir)
        if not st["topology_path"] or not st["trajectory_path"]:
            raise HTTPException(
                status_code=400,
                detail="Provide a topology and trajectory first (upload or use-path).",
            )
        try:
            return run_info(Path(st["topology_path"]), Path(st["trajectory_path"]))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/run/{analysis}")
    async def run_analysis(analysis: str, payload: dict[str, Any]) -> dict[str, Any]:
        if analysis not in RUNNERS:
            raise HTTPException(status_code=404, detail=f"Unknown analysis {analysis!r}")
        st = _state(workdir)
        if analysis != "cluster":
            if not st["topology_path"] or not st["trajectory_path"]:
                raise HTTPException(
                    status_code=400,
                    detail="Provide a topology and trajectory first (upload or use-path).",
                )
        else:
            if not (workdir / "pca.npz").exists():
                raise HTTPException(status_code=400, detail="Run PCA before clustering.")
            if payload.get("rep_pdb") and (
                not st["topology_path"] or not st["trajectory_path"]
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Representative PDB needs topology + trajectory.",
                )
        try:
            top_path = Path(st["topology_path"]) if st["topology_path"] else workdir
            traj_path = Path(st["trajectory_path"]) if st["trajectory_path"] else workdir
            result = RUNNERS[analysis](top_path, traj_path, workdir, payload or {})
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
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
