"""FreeU v2 application — single source of truth for all pipelines."""

from __future__ import annotations

import logging

from .config import settings

log = logging.getLogger("sddj.freeu")


def apply_freeu(pipe) -> None:
    """Apply FreeU v2 to any diffusers pipeline if enabled in settings."""
    if not settings.enable_freeu:
        return
    # Sentinel attribute: _freeu_applied is monkey-patched onto the pipeline
    # instance to prevent redundant enable_freeu() calls across reuse cycles.
    # getattr with default is used because diffusers pipelines do not define
    # this attribute natively — it exists only after our first successful apply.
    if getattr(pipe, "_freeu_applied", False):
        return
    try:
        pipe.enable_freeu(
            s1=settings.freeu_s1,
            s2=settings.freeu_s2,
            b1=settings.freeu_b1,
            b2=settings.freeu_b2,
        )
        pipe._freeu_applied = True
    except Exception as e:
        log.warning("FreeU v2 unavailable for %s: %s", type(pipe).__name__, e)
