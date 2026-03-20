"""LoRA fuse/unfuse/dynamo-reset lifecycle management."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .config import settings
from .lora_manager import resolve_lora_path

log = logging.getLogger("pixytoon.lora_fuser")


class LoRAFuser:
    """Manages pixel art LoRA fusing into pipeline weights."""

    def __init__(self) -> None:
        self.current_name: Optional[str] = None
        self.current_weight: float = 0.0

    def set_lora(self, pipe, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch pixel art LoRA (fused into weights, no PEFT runtime)."""
        had_lora = self.current_name is not None

        # Unfuse previous pixel art LoRA if any
        if had_lora:
            try:
                pipe.unfuse_lora()
                pipe.unload_lora_weights()
                self.current_name = None
                self.current_weight = 0.0
            except Exception as e:
                log.warning("Failed to unfuse pixel art LoRA '%s': %s — state may be corrupted",
                            self.current_name, e)
                # Do NOT reset current_name/weight — they reflect the actual model state
                raise

        if name is None:
            if had_lora and settings.enable_torch_compile:
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass
                log.info("Dynamo cache reset after LoRA removal")
            return

        path = resolve_lora_path(name)
        log.info("Loading pixel art LoRA: %s (weight=%.2f)", name, weight)
        try:
            pipe.load_lora_weights(
                str(path),
                adapter_name="pixel_art",
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
            except Exception:
                pass
            log.info("Dynamo cache reset after LoRA weight change (will recompile on next generation)")
