"""LoRA fuse/unfuse/dynamo-reset lifecycle management."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .config import settings
from .lora_manager import resolve_lora_path

log = logging.getLogger("pixytoon.lora_fuser")

# Monotonic counter avoids PEFT adapter name reuse (PEFT retains
# stale names after fuse+unload, causing "already in use" errors).
_adapter_counter = 0


class LoRAFuser:
    """Manages pixel art LoRA fusing into pipeline weights."""

    def __init__(self) -> None:
        self.current_name: Optional[str] = None
        self.current_weight: float = 0.0

    def set_lora(self, pipe, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch pixel art LoRA (fused into weights, no PEFT runtime)."""
        global _adapter_counter

        # Validate new LoRA path BEFORE unfusing the old one
        if name is not None:
            resolve_lora_path(name)  # raises ValueError if invalid

        had_lora = self.current_name is not None

        # Unfuse previous pixel art LoRA if any
        if had_lora:
            try:
                pipe.unfuse_lora()
            except Exception as e:
                log.warning("Failed to unfuse pixel art LoRA '%s': %s — state may be corrupted",
                            self.current_name, e)
                raise
            try:
                pipe.unload_lora_weights()
            except Exception as e:
                log.warning("Failed to unload LoRA weights (unfuse already done): %s", e)
            # Both operations succeeded (or unload failed non-critically) — update tracking state
            self.current_name = None
            self.current_weight = 0.0

        if name is None:
            if had_lora and settings.enable_torch_compile:
                try:
                    torch._dynamo.reset()
                except Exception as e:
                    log.warning("Dynamo reset failed (non-critical): %s", e)
                log.info("Dynamo cache reset after LoRA removal")
            return

        path = resolve_lora_path(name)
        log.info("Loading pixel art LoRA: %s (weight=%.2f)", name, weight)

        # Use unique adapter name each time — PEFT retains stale names
        # after fuse+unload, so reusing the same name causes conflicts
        _adapter_counter += 1
        adapter_name = f"pixel_art_{_adapter_counter}"

        try:
            pipe.load_lora_weights(
                str(path),
                adapter_name=adapter_name,
            )
            pipe.fuse_lora(lora_scale=weight)
        except Exception:
            # Cleanup loaded but unfused weights
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            raise
        pipe.unload_lora_weights()
        self.current_name = name
        self.current_weight = weight

        if settings.enable_torch_compile:
            try:
                torch._dynamo.reset()
            except Exception as e:
                log.warning("Dynamo reset failed (non-critical): %s", e)
            log.info("Dynamo cache reset after LoRA weight change (will recompile on next generation)")
