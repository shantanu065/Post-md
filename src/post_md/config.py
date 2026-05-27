"""Cross-platform persistent settings for the Post_MD web app.

A small JSON file living alongside ``.post_md_state.json`` in the
workdir. Avoids shell-env-variable gymnastics so the same configuration
works on Windows, macOS, and Linux without touching ``.bashrc`` /
``.zshrc`` / PowerShell profiles.

Currently stored:
  * ``workers`` — int or None. When set, applied to ``POST_MD_WORKERS``
    at server start so every parallel analysis honours it without
    needing to re-thread the value through every call site.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_FILENAME = ".post_md_config.json"


def config_path(workdir: str | Path) -> Path:
    return Path(workdir) / _CONFIG_FILENAME


def load_config(workdir: str | Path) -> dict[str, Any]:
    """Read the workdir's config file. Returns ``{}`` on missing/corrupt."""
    p = config_path(workdir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(workdir: str | Path, config: dict[str, Any]) -> None:
    """Persist ``config`` (atomic write via temp file)."""
    p = config_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    tmp.replace(p)


def apply_to_env(config: dict[str, Any]) -> None:
    """Push relevant config keys into ``os.environ`` so the rest of the
    codebase keeps reading them via env var without per-call plumbing."""
    workers = config.get("workers")
    if workers is None:
        # Empty value means "auto" — clear any prior override so the 80%
        # default kicks back in.
        os.environ.pop("POST_MD_WORKERS", None)
    else:
        try:
            n = int(workers)
            if n > 0:
                os.environ["POST_MD_WORKERS"] = str(n)
        except (TypeError, ValueError):
            os.environ.pop("POST_MD_WORKERS", None)
