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
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from post_md.web.runners import RUNNERS, run_batch, run_info

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


def _read_systems(workdir: Path) -> list[dict[str, str]]:
    """Read the list of registered systems from .post_md_state.json.

    Auto-migrates the legacy single-system format
    ``{"topology": "...", "trajectory": "..."}`` to a one-entry list:
    ``[{"label": "System 1", "topology": "...", "trajectory": "..."}]``.
    """
    sf = _state_file(workdir)
    if not sf.exists():
        return []
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    # New schema
    raw_systems = data.get("systems") if isinstance(data, dict) else None
    if isinstance(raw_systems, list):
        out: list[dict[str, str]] = []
        for i, s in enumerate(raw_systems):
            if not isinstance(s, dict):
                continue
            label = str(s.get("label") or f"System {i + 1}")
            top = str(s.get("topology") or "")
            traj = str(s.get("trajectory") or "")
            color = str(s.get("color") or "").strip()
            if not top and not traj:
                continue
            entry: dict[str, Any] = {"label": label, "topology": top, "trajectory": traj}
            if color:
                entry["color"] = color
            orig_top = str(s.get("original_topology") or "")
            orig_traj = str(s.get("original_trajectory") or "")
            if orig_top:
                entry["original_topology"] = orig_top
            if orig_traj:
                entry["original_trajectory"] = orig_traj
            if "autoimage_used" in s and s["autoimage_used"] is not None:
                entry["autoimage_used"] = bool(s["autoimage_used"])
            out.append(entry)
        return out

    # Legacy single-system schema
    if isinstance(data, dict) and ("topology" in data or "trajectory" in data):
        top = str(data.get("topology") or "")
        traj = str(data.get("trajectory") or "")
        if top or traj:
            return [{"label": "System 1", "topology": top, "trajectory": traj}]
    return []


def _write_systems(workdir: Path, systems: list[dict[str, str]]) -> None:
    """Persist (or delete) .post_md_state.json from a systems list."""
    sf = _state_file(workdir)
    cleaned: list[dict[str, str]] = []
    for s in systems:
        top = str(s.get("topology", "") or "").strip()
        traj = str(s.get("trajectory", "") or "").strip()
        label = str(s.get("label", "") or "").strip()
        color = str(s.get("color", "") or "").strip()
        original_top = str(s.get("original_topology", "") or "").strip()
        original_traj = str(s.get("original_trajectory", "") or "").strip()
        if not top and not traj:
            continue
        entry: dict[str, Any] = {
            "label": label or f"System {len(cleaned) + 1}",
            "topology": top,
            "trajectory": traj,
        }
        if color:
            entry["color"] = color
        if original_top:
            entry["original_topology"] = original_top
        if original_traj:
            entry["original_trajectory"] = original_traj
        if "autoimage_used" in s and s["autoimage_used"] is not None:
            entry["autoimage_used"] = bool(s["autoimage_used"])
        cleaned.append(entry)
    if cleaned:
        sf.write_text(json.dumps({"systems": cleaned}, indent=2), encoding="utf-8")
    elif sf.exists():
        try:
            sf.unlink()
        except OSError:
            pass


def _system_view(s: dict[str, str], index: int) -> dict[str, Any]:
    """Resolve a single registered system into the shape the UI consumes.

    Adds basenames and `*_stale` fields so the front-end can show the
    user a useful badge when a previously-registered path has gone
    missing on disk.
    """
    out: dict[str, Any] = {
        "index": index,
        "label": s.get("label") or f"System {index + 1}",
        "color": s.get("color") or "",
        "topology_path": None,
        "trajectory_path": None,
        "topology": None,
        "trajectory": None,
        "topology_stale": None,
        "trajectory_stale": None,
    }
    for role in ("topology", "trajectory"):
        raw = s.get(role) or ""
        if not raw:
            continue
        p = Path(raw)
        if not p.exists() or not p.is_file():
            out[f"{role}_stale"] = raw
            continue
        out[role] = p.name
        out[f"{role}_path"] = str(p)
    return out


def _state(workdir: Path) -> dict[str, Any]:
    """View consumed by the front-end: list of resolved systems + outputs.

    Output filenames are scanned with no filtering on which system /
    analysis produced them — the workspace files panel lists everything
    the runners wrote so the user can grab raw data + plots.
    """
    systems_raw = _read_systems(workdir)
    systems = [_system_view(s, i) for i, s in enumerate(systems_raw)]

    outputs = sorted(
        p.name for p in workdir.iterdir()
        if p.is_file()
        and p.name != _STATE_FILENAME
        and p.suffix.lower() in _OUTPUT_EXTS
    )

    return {"systems": systems, "outputs": outputs}


def _register_to_system_zero(workdir: Path, role: str, dest: Path) -> None:
    """Set ``systems[0].<role>`` to ``dest`` (creating system 0 if needed).

    Used by the upload endpoints so a freshly-uploaded file is immediately
    visible in the new multi-system state shape without the user having
    to call /api/systems separately.
    """
    systems = _read_systems(workdir)
    if not systems:
        systems = [{"label": "System 1", "topology": "", "trajectory": ""}]
    systems[0][role] = str(dest)
    _write_systems(workdir, systems)


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

    # Long-running preprocessing (strip / autoimage) state — updated from
    # the executor thread, polled by the client.
    #
    # ``_progress`` maps (system_index, op) → {done, total, ts, finished}.
    # ``_locks[system_index]`` is a per-system threading.Lock acquired for
    # the lifetime of any preprocessing op on that system; a 409 is
    # returned if a second op is requested while one is already running.
    _progress: dict[tuple[int, str], dict[str, Any]] = {}
    _progress_lock = threading.Lock()
    _system_locks: dict[int, threading.Lock] = {}

    def _system_lock(idx: int) -> threading.Lock:
        # Single global mutex acquire to safely create per-system locks.
        with _progress_lock:
            lk = _system_locks.get(idx)
            if lk is None:
                lk = threading.Lock()
                _system_locks[idx] = lk
            return lk

    def _set_progress(idx: int, op: str, done: int, total: int, *, finished: bool = False) -> None:
        with _progress_lock:
            _progress[(idx, op)] = {
                "system": idx, "op": op,
                "done": int(done), "total": int(total),
                "ts": time.time(),
                "finished": finished,
            }

    def _get_progress(idx: int) -> dict[str, Any]:
        with _progress_lock:
            return {
                op: dict(_progress[(idx, op)])
                for (i, op) in list(_progress.keys()) if i == idx
            }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/favicon.ico")
    async def favicon() -> Any:
        # Browser auto-requests this on every page load. We don't ship
        # a favicon, so return an empty 204 to keep the access log clean
        # instead of the 404 noise the user sees on every refresh.
        from fastapi.responses import Response
        return Response(status_code=204)

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
        _register_to_system_zero(workdir, role, dest)
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
        # If the caller pointed at a file (common: they typed the file path
        # into the text input and hit Browse), open its parent directory
        # instead of erroring out.
        if start.is_file():
            start = start.parent
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
        # role and start a new file at offset 0.
        if offset == 0:
            _clear_uploaded_role(workdir, role, new_ext=ext)
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
        if complete:
            _register_to_system_zero(workdir, role, dest)
        return {
            "role": role,
            "filename": name,
            "received": new_size,
            "total": total,
            "chunk_bytes": bytes_written,
            "complete": complete,
            "state": _state(workdir) if complete else None,
        }

    def _validated_file_path(raw_path: str, role: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise HTTPException(400, f"'{role}' path is required")
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
        return p

    @app.post("/api/systems")
    async def set_systems(payload: dict[str, Any]) -> dict[str, Any]:
        """Replace the registered systems list in one shot.

        Payload: ``{"systems": [{"label": "WT", "topology": "/p/wt.prmtop",
        "trajectory": "/p/wt.nc"}, ...]}``. Each entry is validated (file
        exists, extension allowed) before the new list is persisted, so a
        bad row never leaves the state half-updated.
        """
        raw_systems = payload.get("systems")
        if not isinstance(raw_systems, list):
            raise HTTPException(400, "'systems' must be a list")
        validated: list[dict[str, str]] = []
        for i, s in enumerate(raw_systems):
            if not isinstance(s, dict):
                raise HTTPException(400, f"systems[{i}] must be an object")
            label = str(s.get("label", "") or "").strip() or f"System {i + 1}"
            top_raw = s.get("topology", "")
            traj_raw = s.get("trajectory", "")
            if not top_raw and not traj_raw:
                continue
            top_path = _validated_file_path(top_raw, "topology")
            traj_path = _validated_file_path(traj_raw, "trajectory")
            # Snapshot the originals on first save so future re-preps
            # (e.g. when the user toggles Autoimage) start from the
            # user-supplied files, not from a previously-cached output.
            entry: dict[str, Any] = {
                "label": label,
                "topology": str(top_path),
                "trajectory": str(traj_path),
                "original_topology": str(top_path),
                "original_trajectory": str(traj_path),
            }
            color = str(s.get("color", "") or "").strip()
            if color:
                entry["color"] = color
            validated.append(entry)
        _write_systems(workdir, validated)
        return {"state": _state(workdir)}

    @app.post("/api/use-path")
    async def use_path(payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy single-system path register — kept for backward compat.

        Updates / creates the first system in the list with the given
        role's path. The new multi-system UI uses ``/api/systems``
        instead, but the old upload-chunk completion flow still calls
        this for the first system.
        """
        role = payload.get("role")
        if role not in ("topology", "trajectory"):
            raise HTTPException(400, "'role' must be 'topology' or 'trajectory'")
        p = _validated_file_path(payload.get("path", ""), role)
        _clear_uploaded_role(workdir, role, new_ext=p.suffix.lower())

        systems = _read_systems(workdir)
        if not systems:
            systems = [{"label": "System 1", "topology": "", "trajectory": ""}]
        systems[0][role] = str(p)
        _write_systems(workdir, systems)
        return {"role": role, "path": str(p), "bytes": p.stat().st_size, "state": _state(workdir)}

    @app.post("/api/info")
    async def info_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        st = _state(workdir)
        systems = st["systems"]
        if not systems:
            raise HTTPException(400, "Register at least one system first.")
        idx = 0
        if payload and "system" in payload:
            try:
                idx = int(payload["system"])
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "'system' must be an integer index.") from exc
        if idx < 0 or idx >= len(systems):
            raise HTTPException(400, f"system index {idx} out of range")
        sys_ = systems[idx]
        if not sys_["topology_path"] or not sys_["trajectory_path"]:
            raise HTTPException(400, f"System {sys_['label']!r} is missing topology/trajectory.")
        try:
            return run_info(Path(sys_["topology_path"]), Path(sys_["trajectory_path"]))
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/preprocess")
    async def preprocess_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        """Strip waters / ions from one system, producing a slim
        topology + trajectory pair on disk. The system entry is then
        rewritten in state to point at the stripped files, so the
        Run-batch flow immediately picks them up — no manual swap.

        Long-running for big trajectories; uses ``run_in_executor`` so
        the FastAPI event loop stays responsive.
        """
        st = _state(workdir)
        systems = _read_systems(workdir)
        if not systems:
            raise HTTPException(400, "Register a system first.")
        try:
            idx = int(payload.get("system", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "'system' must be an integer.") from exc
        if idx < 0 or idx >= len(systems):
            raise HTTPException(400, f"system index {idx} out of range")
        sys_ = systems[idx]
        top = sys_.get("topology") or ""
        traj = sys_.get("trajectory") or ""
        if not top or not traj or not Path(top).is_file() or not Path(traj).is_file():
            raise HTTPException(400, "System is missing valid topology + trajectory files.")

        # Optional custom strip set (UI may pass a comma-separated string)
        extra_strip = payload.get("strip_resnames")
        if isinstance(extra_strip, str):
            extra_set = {tok.strip() for tok in extra_strip.split(",") if tok.strip()}
        elif isinstance(extra_strip, list):
            extra_set = {str(tok).strip() for tok in extra_strip if str(tok).strip()}
        else:
            extra_set = None

        from post_md.preprocess import STRIP_DEFAULT, strip_solvent
        strip_set = (extra_set if extra_set else STRIP_DEFAULT)

        lk = _system_lock(idx)
        if not lk.acquire(blocking=False):
            raise HTTPException(
                409, f"system {idx} already has a preprocessing op in flight."
            )
        try:
            _set_progress(idx, "strip", 0, 0)
            loop = asyncio.get_running_loop()

            def _cb(done, total):
                _set_progress(idx, "strip", done, total)

            try:
                top_out, traj_out, summary = await loop.run_in_executor(
                    None,
                    lambda: strip_solvent(
                        top, traj, workdir,
                        strip_resnames=set(strip_set),
                        output_basename=f"{Path(traj).stem}-stripped_{idx}",
                        progress=_cb,
                    ),
                )
            except FileNotFoundError as exc:
                raise HTTPException(400, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

            systems[idx]["topology"] = str(top_out)
            systems[idx]["trajectory"] = str(traj_out)
            _write_systems(workdir, systems)
            _set_progress(idx, "strip", summary["n_frames"], summary["n_frames"], finished=True)
            return {"summary": summary, "state": _state(workdir)}
        finally:
            lk.release()

    @app.post("/api/autoimage")
    async def autoimage_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        """Re-image a system's trajectory in place: wrap the anchor
        contiguous, recentre at the box, wrap solvent / ions to closest
        periodic image. Mirrors /api/preprocess but doesn't strip atoms,
        so the topology is unchanged."""
        systems = _read_systems(workdir)
        if not systems:
            raise HTTPException(400, "Register a system first.")
        try:
            idx = int(payload.get("system", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "'system' must be an integer.") from exc
        if idx < 0 or idx >= len(systems):
            raise HTTPException(400, f"system index {idx} out of range")
        sys_ = systems[idx]
        top = sys_.get("topology") or ""
        traj = sys_.get("trajectory") or ""
        if not top or not traj or not Path(top).is_file() or not Path(traj).is_file():
            raise HTTPException(400, "System is missing valid topology + trajectory files.")

        anchor = payload.get("anchor_selection") or "protein"
        if not isinstance(anchor, str) or not anchor.strip():
            anchor = "protein"

        from post_md.imaging import autoimage_trajectory

        lk = _system_lock(idx)
        if not lk.acquire(blocking=False):
            raise HTTPException(
                409, f"system {idx} already has a preprocessing op in flight."
            )
        try:
            _set_progress(idx, "autoimage", 0, 0)
            loop = asyncio.get_running_loop()

            def _cb(done, total):
                _set_progress(idx, "autoimage", done, total)

            try:
                _, traj_out, summary = await loop.run_in_executor(
                    None,
                    lambda: autoimage_trajectory(
                        top, traj, workdir,
                        anchor_selection=anchor.strip(),
                        output_basename=f"{Path(traj).stem}-autoimaged_{idx}",
                        progress=_cb,
                    ),
                )
            except FileNotFoundError as exc:
                raise HTTPException(400, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

            systems[idx]["trajectory"] = str(traj_out)
            _write_systems(workdir, systems)
            _set_progress(idx, "autoimage", summary["n_frames"], summary["n_frames"], finished=True)
            return {"summary": summary, "state": _state(workdir)}
        finally:
            lk.release()

    @app.get("/api/preprocess-progress")
    async def preprocess_progress(system: int) -> dict[str, Any]:
        """Snapshot of every running preprocessing op on a system.

        Client polls this every couple of seconds while a Strip or
        Autoimage button is in flight to show a percentage / frame
        counter. Keys are operation names (``strip`` / ``autoimage``)
        whose values carry ``done``, ``total``, ``finished``, and a
        wall-clock timestamp ``ts`` so the client can detect stalls.
        """
        return _get_progress(int(system))

    @app.post("/api/run-batch")
    async def run_batch_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        """Run several analyses across one or more registered systems.

        Payload shape::

            {
              "analyses": {
                "rmsd":  {...params...},
                "rmsf":  {...params...},
                "rg":    {...params...},
                "pca":   {...params...},
                "cluster": {...params...}
              },
              "systems": [0, 1, 2]      # optional — defaults to all
            }

        For RMSD/RMSF/Rg, a combined overlay plot is produced when more
        than one system is selected (one curve per system, tab10 colors,
        legend). For PCA / Cluster, one independent plot is produced per
        system because cross-system comparison is rarely meaningful.
        """
        st = _state(workdir)
        all_systems = st["systems"]
        if not all_systems:
            raise HTTPException(400, "Register at least one system first.")

        analyses = payload.get("analyses") or {}
        if not isinstance(analyses, dict) or not analyses:
            raise HTTPException(400, "'analyses' must be a non-empty object.")
        _allowed = set(RUNNERS) | {"cluster", "sasa", "hbond"}
        for name in analyses:
            if name not in _allowed:
                raise HTTPException(400, f"Unknown analysis: {name!r}")

        sys_indices = payload.get("systems")
        if sys_indices is None:
            sys_indices = list(range(len(all_systems)))
        if not isinstance(sys_indices, list) or not sys_indices:
            raise HTTPException(400, "'systems' must be a non-empty list of indices.")
        selected: list[dict[str, Any]] = []
        for raw_idx in sys_indices:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"system index {raw_idx!r} not an integer") from exc
            if idx < 0 or idx >= len(all_systems):
                raise HTTPException(400, f"system index {idx} out of range")
            sys_ = all_systems[idx]
            if not sys_["topology_path"] or not sys_["trajectory_path"]:
                raise HTTPException(400, f"System {sys_['label']!r} missing topology/trajectory.")
            selected.append(sys_)

        # Pre-flight: silently autoimage + strip each selected system once,
        # unless the registered trajectory already points at a previously
        # prepared file *with the same autoimage flag*. Toggling autoimage
        # invalidates the cache and triggers a re-prep.
        from post_md.preprocess import PREPARED_MARKER, prepare_trajectory

        do_autoimage = bool(payload.get("autoimage", True))

        all_systems_full = _read_systems(workdir)
        loop = asyncio.get_running_loop()

        resolved_for_run: list[dict[str, Any]] = []
        for sys_view in selected:
            idx = int(sys_view.get("index", 0))
            sys_state = all_systems_full[idx]
            # Prep always reads from the user-supplied originals (if
            # recorded). Falling back to current path lets older state
            # files keep working — the next Save will record originals.
            src_top = sys_state.get("original_topology") or sys_state.get("topology") or ""
            src_traj = sys_state.get("original_trajectory") or sys_state.get("trajectory") or ""

            # The two prep variants live at deterministic paths. Looking
            # them up directly is cheaper than re-running prep just
            # because the user toggled the flag back to a value they
            # already prepared once.
            src_stem = Path(src_traj).stem
            tag = "ai" if do_autoimage else "noai"
            expected_traj = workdir / f"{src_stem}{PREPARED_MARKER}{idx}-{tag}.nc"
            expected_top = workdir / f"{src_stem}{PREPARED_MARKER}{idx}-{tag}.prmtop"
            if expected_traj.is_file() and expected_top.is_file():
                # Cache hit on this variant — pivot the state to it.
                all_systems_full[idx]["topology"] = str(expected_top)
                all_systems_full[idx]["trajectory"] = str(expected_traj)
                all_systems_full[idx]["autoimage_used"] = do_autoimage
                _write_systems(workdir, all_systems_full)
                resolved = dict(sys_view)
                resolved["topology_path"] = str(expected_top)
                resolved["trajectory_path"] = str(expected_traj)
                resolved["topology"] = expected_top.name
                resolved["trajectory"] = expected_traj.name
                resolved_for_run.append(resolved)
                continue

            lk = _system_lock(idx)
            if not lk.acquire(blocking=False):
                raise HTTPException(
                    409, f"system {idx} already has a preparation in flight."
                )
            try:
                _set_progress(idx, "prepare", 0, 0)

                def _cb(done, total, _idx=idx):
                    _set_progress(_idx, "prepare", done, total)

                # tag + src_stem already resolved above for the cache
                # lookup. They're reused as the prep output basename so
                # the prepared file lands at the exact path the next
                # cache check will look for.
                try:
                    top_out, traj_out, _summary = await loop.run_in_executor(
                        None,
                        lambda: prepare_trajectory(
                            src_top, src_traj, workdir,
                            output_basename=f"{src_stem}{PREPARED_MARKER}{idx}-{tag}",
                            autoimage=do_autoimage,
                            progress=_cb,
                        ),
                    )
                except Exception as exc:
                    _set_progress(idx, "prepare", 0, 0, finished=True)
                    raise HTTPException(
                        500, f"prepare failed for system {idx}: {type(exc).__name__}: {exc}",
                    ) from exc

                all_systems_full[idx]["topology"] = str(top_out)
                all_systems_full[idx]["trajectory"] = str(traj_out)
                all_systems_full[idx]["autoimage_used"] = bool(do_autoimage)
                _write_systems(workdir, all_systems_full)
                _set_progress(idx, "prepare",
                              _summary["n_frames"], _summary["n_frames"], finished=True)

                resolved = dict(sys_view)
                resolved["topology_path"] = str(top_out)
                resolved["trajectory_path"] = str(traj_out)
                resolved["topology"] = top_out.name
                resolved["trajectory"] = traj_out.name
                resolved_for_run.append(resolved)
            finally:
                lk.release()

        try:
            result = run_batch(resolved_for_run, analyses, workdir)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
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
