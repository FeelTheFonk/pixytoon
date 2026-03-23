"""Background removal wrapper around rembg.

Loaded lazily to avoid holding VRAM/RAM when unused.
Can run on CPU to keep GPU free for diffusion.
"""

from __future__ import annotations

import threading

from PIL import Image

from .config import settings

__all__ = ["remove_bg", "unload"]

_onnx_available = True
try:
    import onnxruntime  # noqa: F401
except ImportError:
    _onnx_available = False

_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            from rembg import new_session
            providers = ["CPUExecutionProvider"] if settings.rembg_on_cpu else None
            _session = new_session(model_name=settings.rembg_model, providers=providers)
        return _session


def remove_bg(img: Image.Image) -> Image.Image:
    if not _onnx_available:
        raise RuntimeError("Background removal unavailable: onnxruntime not installed")
    from rembg import remove
    session = _get_session()
    return remove(img, session=session)


def unload():
    """Free the rembg session from memory."""
    global _session
    import gc
    with _session_lock:
        _session = None
    gc.collect()
