"""LoRA fuse/unfuse lifecycle management.

When enable_lora_hotswap is active (diffusers SOTA 2025), LoRA swaps bypass
torch.compile recompilation entirely — no dynamo.reset() needed.
Falls back to dynamo.reset() when hotswap unavailable.

Weight snapshot: stores original UNet weights (CPU, ~1.7GB for SD1.5) before
the first style LoRA fuse.  On unfuse, restores from snapshot instead of
unfuse_lora() to prevent numerical drift after N fuse/unfuse cycles.
"""

from __future__ import annotations

import itertools
import logging
from typing import Optional

import torch

from .config import settings
from .lora_manager import resolve_lora_path

log = logging.getLogger("sddj.lora_fuser")

# Thread-safe monotonic counter avoids PEFT adapter name reuse (PEFT retains
# stale names after fuse+unload, causing "already in use" errors).
_adapter_counter = itertools.count(1)


class LoRAFuser:
    """Manages style LoRA fusing into pipeline weights."""

    def __init__(self) -> None:
        self.current_name: Optional[str] = None
        self.current_weight: float = 0.0
        self._original_unet_state: Optional[dict] = None

    def _ensure_snapshot(self, pipe) -> None:
        """Snapshot UNet weights to CPU before the first style LoRA fuse."""
        if self._original_unet_state is None:
            self._original_unet_state = {
                k: v.cpu().clone() for k, v in pipe.unet.state_dict().items()
            }
            log.info("UNet weight snapshot captured (CPU)")

    def _restore_weights(self, pipe) -> None:
        """Restore UNet to exact original weights from CPU snapshot."""
        if self._original_unet_state is not None:
            pipe.unet.load_state_dict(self._original_unet_state, assign=False)

    def _needs_dynamo_reset(self) -> bool:
        """Return True if dynamo.reset() is needed after LoRA change."""
        return settings.enable_torch_compile and not settings.enable_lora_hotswap

    def set_lora(self, pipe, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch style LoRA (fused into weights, no PEFT runtime)."""

        # Validate new LoRA path BEFORE unfusing the old one
        if name is not None:
            resolve_lora_path(name)  # raises ValueError if invalid

        had_lora = self.current_name is not None

        # Unfuse previous style LoRA: restore from snapshot (avoids numerical drift)
        if had_lora:
            try:
                self._restore_weights(pipe)
            except Exception:
                # Fallback to unfuse_lora() if snapshot restore fails
                try:
                    pipe.unfuse_lora()
                except Exception as e:
                    log.warning("Failed to unfuse style LoRA '%s': %s",
                                self.current_name, e)
                    raise
            try:
                pipe.unload_lora_weights()
            except Exception as e:
                log.warning("Failed to unload LoRA weights (unfuse already done): %s", e)
            self.current_name = None
            self.current_weight = 0.0

        if name is None:
            if had_lora and self._needs_dynamo_reset():
                try:
                    torch._dynamo.reset()
                except Exception as e:
                    log.warning("Dynamo reset failed (non-critical): %s", e)
                log.info("Dynamo cache reset after LoRA removal")
            return

        # Capture pre-fuse UNet weights (once, before first style LoRA)
        self._ensure_snapshot(pipe)

        path = resolve_lora_path(name)
        log.info("Loading style LoRA: %s (weight=%.2f)", name, weight)

        # Use unique adapter name each time — PEFT retains stale names
        # after fuse+unload, so reusing the same name causes conflicts
        adapter_name = f"style_{next(_adapter_counter)}"

        try:
            pipe.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=adapter_name,
                local_files_only=True,
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

        if self._needs_dynamo_reset():
            try:
                torch._dynamo.reset()
            except Exception as e:
                log.warning("Dynamo reset failed (non-critical): %s", e)
            log.info("Dynamo cache reset after LoRA weight change (will recompile on next generation)")
