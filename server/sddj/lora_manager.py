"""LoRA discovery and listing — thin wrapper over ResourceManager."""

from __future__ import annotations

from .config import settings
from .resource_manager import ResourceManager

_mgr = ResourceManager("LoRA", settings.loras_dir)

list_loras = _mgr.list
resolve_lora_path = _mgr.resolve
