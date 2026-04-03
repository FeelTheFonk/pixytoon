"""Centralized VRAM management utilities.

Single source of truth for GPU memory cleanup, monitoring, and safe model
offloading.  Every module that touches VRAM should use these helpers
instead of ad-hoc gc.collect() / empty_cache() patterns.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger("sddj.vram")

_last_gc: float = 0.0
_gc_lock = threading.Lock()
_GC_COOLDOWN: float = 2.0  # Minimum seconds between gc.collect() to avoid 50-200ms stalls
_gc_skip_count: int = 0  # Diagnostic counter: number of throttled (skipped) GC calls
_MB = 1024 * 1024  # Bytes per megabyte


def vram_cleanup(force: bool = False) -> None:
    """GC-collect then free CUDA cache.

    Both gc.collect() and torch.cuda.empty_cache() are throttled together
    to at most once per _GC_COOLDOWN seconds.  empty_cache() without
    gc.collect() is counterproductive: it releases the CUDA cache but
    Python objects holding tensor references aren't collected, so the
    cache is immediately repopulated.  Running empty_cache() alone also
    costs 5-50ms and forces future allocations to re-fragment.

    Pass force=True for genuine cleanup scenarios (model unload, OOM recovery).
    """
    global _last_gc, _gc_skip_count
    now = time.monotonic()
    with _gc_lock:
        if force or (now - _last_gc) > _GC_COOLDOWN:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _last_gc = time.monotonic()
        else:
            _gc_skip_count += 1


def get_gc_skip_count() -> int:
    """Return the number of GC calls skipped due to cooldown throttling (diagnostic)."""
    return _gc_skip_count


def get_vram_info() -> tuple[float, float, float]:
    """Return (used_mb, free_mb, total_mb).  (0, 0, 0) when no CUDA."""
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)
    free, total = torch.cuda.mem_get_info()
    free_mb = free / _MB
    total_mb = total / _MB
    used_mb = total_mb - free_mb
    return (round(used_mb, 1), round(free_mb, 1), round(total_mb, 1))


def vram_log(label: str) -> None:
    """Log a VRAM snapshot with the given label."""
    used, free, total = get_vram_info()
    if total > 0:
        log.info("VRAM [%s]: %.0f MB used / %.0f MB free / %.0f MB total",
                 label, used, free, total)


def move_to_cpu(module: Optional[nn.Module]) -> None:
    """Safely move a module to CPU, freeing VRAM immediately.

    Handles None and non-Module objects gracefully (no-op).
    """
    if module is not None and isinstance(module, nn.Module):
        try:
            module.to("cpu")
        except Exception:
            log.debug("move_to_cpu failed (non-critical): %s", type(module).__name__)


def check_vram_budget(required_mb: float, min_free_mb: float = 512.0) -> bool:
    """Return True if enough free VRAM for the requested allocation."""
    _, free, _ = get_vram_info()
    if free <= 0:
        return True  # no CUDA — assume OK (CPU mode)
    return free >= (required_mb + min_free_mb)
