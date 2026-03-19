"""Background removal wrapper around rembg.

Loaded lazily to avoid holding VRAM/RAM when unused.
Can run on CPU to keep GPU free for diffusion.
"""

from __future__ import annotations

from PIL import Image

from .config import settings

__all__ = ["remove_bg", "unload"]

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session
        providers = ["CPUExecutionProvider"] if settings.rembg_on_cpu else None
        _session = new_session(model_name=settings.rembg_model, providers=providers)
    return _session


def remove_bg(img: Image.Image) -> Image.Image:
    from rembg import remove
    session = _get_session()
    return remove(img, session=session)


def unload():
    """Free the rembg session from memory."""
    global _session
    _session = None
