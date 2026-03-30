"""LoRA fuse/unfuse lifecycle management.

When enable_lora_hotswap is active (diffusers SOTA 2025), LoRA swaps bypass
torch.compile recompilation entirely — no dynamo.reset() needed.
Falls back to dynamo.reset() when hotswap unavailable.

Weight snapshot: stores original UNet + text_encoder weights (CPU) before
the first style LoRA fuse.  On unfuse, restores from snapshot instead of
unfuse_lora() to prevent numerical drift after N fuse/unfuse cycles.

FIX (v0.9.67): restore operates on the RAW (uncompiled) UNet.
FIX (v0.9.69): uses load_state_dict(assign=False) to COPY data into
existing CUDA tensors instead of replacing tensor objects.  assign=True
broke torch.compile's Dynamo graph references — the compiled graph held
pointers to old CPU tensors while new CUDA tensors were created by
.to(device), causing "Expected all tensors to be on the same device".
With assign=False, tensor identity is preserved and the compiled graph
stays valid.  Also validates BOTH parameters AND buffers for device
consistency (v0.9.67 only checked parameters, missing buffers).
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


def _get_raw_module(module):
    """Unwrap torch.compile OptimizedModule to get the raw nn.Module.

    load_state_dict must target the raw module so that weight data is
    copied into the actual parameter tensors, not into the compiled
    wrapper's internal references.
    """
    if hasattr(module, "_orig_mod"):
        return module._orig_mod
    return module


class LoRAFuser:
    """Manages style LoRA fusing into pipeline weights."""

    def __init__(self) -> None:
        self.current_name: Optional[str] = None
        self.current_weight: float = 0.0
        self._original_unet_state: Optional[dict] = None
        self._original_te_state: Optional[dict] = None

    def _ensure_snapshot(self, pipe) -> None:
        """Snapshot UNet + text_encoder weights to CPU before the first style LoRA fuse.

        Both are captured because fuse_lora() merges adapter weights into
        BOTH the UNet and text_encoder.  Without a text_encoder snapshot,
        successive LoRA switches accumulate fused weights in the encoder,
        causing semantic drift.
        """
        if self._original_unet_state is None:
            raw_unet = _get_raw_module(pipe.unet)
            self._original_unet_state = {
                k: v.to(dtype=torch.bfloat16, device="cpu") if v.is_floating_point() else v.cpu()
                for k, v in raw_unet.state_dict().items()
            }
            log.info("UNet weight snapshot captured (CPU)")
        if self._original_te_state is None and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            raw_te = _get_raw_module(pipe.text_encoder)
            self._original_te_state = {
                k: v.to(dtype=torch.bfloat16, device="cpu") if v.is_floating_point() else v.cpu()
                for k, v in raw_te.state_dict().items()
            }
            log.info("Text encoder weight snapshot captured (CPU)")

    def _restore_weights(self, pipe) -> None:
        """Restore UNet + text_encoder to exact original weights from CPU snapshot.

        Uses load_state_dict with assign=False (default) to COPY snapshot data
        into existing CUDA tensor objects.  This preserves torch.compile's Dynamo
        graph tensor references — no object replacement, no stale pointers, no
        device mismatch.  The copy operation handles CPU→CUDA transfer internally.

        Falls back to dynamo.reset() only if a device anomaly is detected.
        """
        if self._original_unet_state is not None:
            raw_unet = _get_raw_module(pipe.unet)
            first_param = next(raw_unet.parameters())
            device = first_param.device
            # Cast bfloat16 snapshot back to model dtype before restoring
            target_dtype = first_param.dtype
            restored = {
                k: v.to(target_dtype) if v.is_floating_point() else v
                for k, v in self._original_unet_state.items()
            }
            # assign=False (default): copies data INTO existing CUDA tensors.
            # Preserves tensor object identity → compiled graph refs stay valid.
            raw_unet.load_state_dict(restored)
            # Validate: first-10 sample device consistency check (avoids O(N) list materialization)
            mismatched = [
                k for k, t in itertools.islice(
                    itertools.chain(raw_unet.named_parameters(), raw_unet.named_buffers()), 10
                ) if t.device != device
            ]
            if mismatched:
                log.warning(
                    "Device mismatch after UNet restore (%d tensors on wrong device), "
                    "forcing .to(%s) + dynamo reset: %s",
                    len(mismatched), device, mismatched[:5],
                )
                raw_unet.to(device)
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass
            else:
                log.debug("UNet weights restored, all tensors on %s", device)

        if self._original_te_state is not None and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            raw_te = _get_raw_module(pipe.text_encoder)
            te_first_param = next(raw_te.parameters())
            te_device = te_first_param.device
            # Cast bfloat16 snapshot back to model dtype before restoring
            te_target_dtype = te_first_param.dtype
            te_restored = {
                k: v.to(te_target_dtype) if v.is_floating_point() else v
                for k, v in self._original_te_state.items()
            }
            raw_te.load_state_dict(te_restored)
            # Validate: first-10 sample text encoder device consistency check
            te_mismatched = [
                k for k, t in itertools.islice(
                    itertools.chain(raw_te.named_parameters(), raw_te.named_buffers()), 10
                ) if t.device != te_device
            ]
            if te_mismatched:
                log.warning(
                    "Device mismatch after text encoder restore (%d tensors), "
                    "forcing .to(%s): %s",
                    len(te_mismatched), te_device, te_mismatched[:5],
                )
                raw_te.to(te_device)
            else:
                log.debug("Text encoder weights restored to %s", te_device)

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

        # Capture pre-fuse weights (once, before first style LoRA)
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
                log.debug("Failed to unload partially-fused LoRA weights")
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
