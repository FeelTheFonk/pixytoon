"""SDDj — Local SOTA generation and animation server for Aseprite."""

import os as _os

# Force offline mode at the earliest possible point: never fetch from
# HuggingFace Hub at runtime.  Models must be pre-cached via
# scripts/download_models.py or placed locally.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("DO_NOT_TRACK", "1")

__version__ = "0.9.3"
