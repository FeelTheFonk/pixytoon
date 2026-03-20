"""Background removal wrapper around rembg.

Loaded lazily to avoid holding VRAM/RAM when unused.
Can run on CPU to keep GPU free for diffusion.
"""

from __future__ import annotations

import threading

from PIL import Image

from .config import settings

__all__ = ["remove_bg", "unload"]

# Verify onnxruntime is available — rembg requires it at runtime
try:
    import onnxruntime  # noqa: F401
except ImportError as _e:
    import logging as _logging
    _logging.getLogger("pixytoon.rembg").error(
        "onnxruntime is not installed — rembg background removal will fail. "
        "Install with: pip install onnxruntime"
    )

_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    if _session is not None:
        return _session
    with _session_lock:
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
    with _session_lock:
        _session = None
