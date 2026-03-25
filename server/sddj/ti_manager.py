"""Textual Inversion embedding discovery — thin wrapper over ResourceManager."""

from __future__ import annotations

from .config import settings
from .resource_manager import ResourceManager

_mgr = ResourceManager("embedding", settings.embeddings_dir)

list_embeddings = _mgr.list
resolve_embedding_path = _mgr.resolve
