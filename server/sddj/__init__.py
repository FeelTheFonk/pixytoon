"""SDDj — Local SOTA generation and animation server for Aseprite."""

import os as _os
import warnings as _warnings
from pathlib import Path as _Path

# Force offline mode at the earliest possible point: never fetch from
# HuggingFace Hub at runtime.  Models must be pre-cached via
# scripts/download_models.py or placed locally.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("DO_NOT_TRACK", "1")

# ── Persist compile caches & CUDA allocator config ───────────
# Set before torch is imported so Triton/Inductor pick up the paths.
_SERVER_ROOT = _Path(__file__).resolve().parent.parent
_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_SERVER_ROOT / ".cache" / "inductor"))
_os.environ.setdefault("TRITON_CACHE_DIR", str(_SERVER_ROOT / ".cache" / "triton"))
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Suppress known harmless library warnings ──────────────────
# Centralized here (earliest import point) for a clean console from boot.
# All filters target third-party noise — our own code should never warn.

# diffusers: safety checker / feature extractor / LoRA / FreeInit
_warnings.filterwarnings("ignore", message=".*safety checker.*")
_warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*")
_warnings.filterwarnings("ignore", message=".*No LoRA keys associated.*")
_warnings.filterwarnings("ignore", message=".*FreeInitMixin.*")

# transformers: attention implementation fallback notice
_warnings.filterwarnings("ignore", message=".*_attn_implementation.*")

# torch: CUDA / compile / Triton noise
_warnings.filterwarnings("ignore", message=".*expandable_segments.*")
_warnings.filterwarnings("ignore", message=".*ComplexHalf support is experimental.*")
_warnings.filterwarnings("ignore", message=".*Torchinductor does not support code generation for complex.*")
_warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")
_warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
_warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

# PEFT: adapter name warnings during fuse/unfuse
_warnings.filterwarnings("ignore", message=".*already in list of adapters.*")

# audioread (librosa transitive dep): imports deprecated aifc/sunau on Python 3.13
_warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioread")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module="standard_aifc")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module="standard_sunau")

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("sddj-server")
except Exception:
    __version__ = "dev"

