"""Utility helpers shared across post_md modules."""

from __future__ import annotations

import os
import threading

# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------
#
# Long-running analyses (prep, SASA, H-bond, ...) check ``is_cancelled()``
# at chunk boundaries; multiprocessing pools are registered while alive
# and terminated immediately by ``request_cancellation()``. The web UI
# exposes a Stop button that calls ``/api/cancel-run``, which in turn
# invokes ``request_cancellation()``.

_cancel_event = threading.Event()
_active_pools: list = []
_pools_lock = threading.Lock()


class Cancelled(RuntimeError):
    """Raised inside prep / analysis code when the user clicks Stop."""


def reset_cancellation() -> None:
    """Clear cancellation state at the start of a fresh run."""
    _cancel_event.clear()
    with _pools_lock:
        _active_pools.clear()


def request_cancellation() -> None:
    """Signal every long-running task to stop ASAP."""
    _cancel_event.set()
    with _pools_lock:
        pools = list(_active_pools)
        _active_pools.clear()
    for pool in pools:
        try:
            pool.terminate()
        except Exception:
            pass


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def raise_if_cancelled() -> None:
    """Lightweight helper for cancellation check points."""
    if _cancel_event.is_set():
        raise Cancelled("Cancelled by user")


def register_pool(pool) -> None:
    """Register a multiprocessing.Pool so a cancellation request can
    terminate it. Pair every register with an :func:`unregister_pool`
    in a ``finally`` block."""
    with _pools_lock:
        _active_pools.append(pool)


def unregister_pool(pool) -> None:
    with _pools_lock:
        try:
            _active_pools.remove(pool)
        except ValueError:
            pass


def default_workers() -> int:
    """Number of worker processes to use for embarrassingly-parallel analyses.

    Resolution order:
      1. ``POST_MD_WORKERS`` env var (any positive integer) — explicit override.
      2. 80% of the logical CPU count reported by the OS.
      3. 1 if nothing else is determinable.

    Clamped to ``[1, 512]`` as a safety rail against typos / runaway
    settings; the real ceiling is the host's logical core count. On a
    typical workstation (4-32 cores) the cap never bites; on big servers
    (64-128 cores) it lets users go beyond the conservative 64-default
    when they explicitly want to.
    """
    env = os.environ.get("POST_MD_WORKERS", "").strip()
    if env:
        try:
            n = int(env)
            if n > 0:
                return min(max(1, n), 512)
        except ValueError:
            pass
    cores = os.cpu_count() or 1
    return min(max(1, int(cores * 0.8)), 512)
