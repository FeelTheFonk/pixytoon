"""FreeU v2 application — single source of truth for all pipelines."""

from __future__ import annotations

import logging

from .config import settings

log = logging.getLogger("sddj.freeu")


def apply_freeu(pipe) -> None:
    """Apply FreeU v2 to any diffusers pipeline if enabled in settings."""
    if not settings.enable_freeu:
        return
    pipe.enable_freeu(
        s1=settings.freeu_s1,
        s2=settings.freeu_s2,
        b1=settings.freeu_b1,
        b2=settings.freeu_b2,
    )
