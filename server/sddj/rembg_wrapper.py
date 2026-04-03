"""Background removal wrapper around rembg.

Loaded lazily to avoid holding VRAM/RAM when unused.
Can run on CPU to keep GPU free for diffusion.
"""

from __future__ import annotations

import threading

from PIL import Image

from .config import settings

__all__ = ["remove_bg", "unload"]

_session = None
_session_lock = threading.Lock()
_onnx_checked: bool | None = None


def _is_onnx_available() -> bool:
    """Lazy check for onnxruntime — deferred so the import cost is only paid when needed."""
    global _onnx_checked
    if _onnx_checked is None:
        try:
            import onnxruntime  # noqa: F401
            _onnx_checked = True
        except ImportError:
            _onnx_checked = False
    return _onnx_checked


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            from rembg import new_session
            providers = None
            if not settings.rembg_on_cpu:
                try:
                    import onnxruntime
                    available = onnxruntime.get_available_providers()
                    ranked = []
                    if "TensorrtExecutionProvider" in available:
                        ranked.append("TensorrtExecutionProvider")
                    if "CUDAExecutionProvider" in available:
                        ranked.append("CUDAExecutionProvider")
                    if ranked:
                        providers = ranked
                except Exception:
                    pass
            if not providers:
                providers = ["CPUExecutionProvider"]
            _session = new_session(model_name=settings.rembg_model, providers=providers)
        return _session


def remove_bg(img: Image.Image) -> Image.Image:
    if not _is_onnx_available():
        raise RuntimeError("Background removal unavailable: onnxruntime not installed")
    from rembg import remove
    session = _get_session()
    return remove(img, session=session)


def unload():
    """Free the rembg session from memory."""
    global _session
    with _session_lock:
        _session = None
        if not settings.rembg_on_cpu:
            import torch
            if torch.cuda.is_available():
                from .vram_utils import vram_cleanup
                vram_cleanup()
