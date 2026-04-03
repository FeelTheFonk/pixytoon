"""LoRA lifecycle management — set_adapters (fast path) with fuse/unfuse fallback.

Primary path (peft >= 0.18.1): Uses set_adapters() for ~1ms LoRA switches.
Adapters are loaded once and cached; subsequent switches only change the active
adapter name and weight via set_adapters().

Fallback path (older peft): Uses fuse_lora()/unfuse_lora() with CPU weight
snapshots to prevent numerical drift.  Costs 200-400ms per switch.

When enable_lora_hotswap is active (diffusers SOTA 2025), LoRA swaps bypass
torch.compile recompilation entirely — no dynamo.reset() needed.
Falls back to dynamo.reset() when hotswap unavailable.

Weight snapshot (fallback only): stores original UNet + text_encoder weights
(CPU) before the first style LoRA fuse.  On unfuse, restores from snapshot
instead of unfuse_lora() to prevent numerical drift after N fuse/unfuse cycles.

FIX (v0.9.67): restore operates on the RAW (uncompiled) UNet.
FIX (v0.9.69): uses load_state_dict(assign=False) to COPY data into
existing CUDA tensors instead of replacing tensor objects.
FIX (v0.9.93): set_adapters() fast path — eliminates 200-400ms fuse/unfuse
overhead per LoRA switch.
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

# ── peft version detection ─────────────────────────────────
# set_adapters() requires peft >= 0.18.1 and diffusers support.
# Detected once at import to avoid per-call overhead.
_USE_SET_ADAPTERS: bool = False
try:
    import peft
    _peft_version = tuple(int(x) for x in peft.__version__.split(".")[:3])
    if _peft_version >= (0, 18, 1):
        _USE_SET_ADAPTERS = True
        log.info("peft %s detected — using fast set_adapters() path", peft.__version__)
    else:
        log.info("peft %s < 0.18.1 — using fuse/unfuse fallback path", peft.__version__)
except ImportError:
    log.info("peft not installed — using fuse/unfuse fallback path")
except Exception:
    log.info("peft version detection failed — using fuse/unfuse fallback path")


def _sanitize_adapter_name(lora_name: str) -> str:
    """Convert a LoRA filename to a valid PEFT adapter name.

    PEFT adapter names must be valid Python identifiers.  Strips extension,
    replaces non-alphanumeric chars with underscores.
    """
    import re
    name = lora_name.rsplit(".", 1)[0] if "." in lora_name else lora_name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = "lora_" + name
    return name


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
    """Manages style LoRA application into pipeline.

    Fast path (set_adapters): loads adapter once, switches via set_adapters().
    Fallback (fuse/unfuse): fuses weights, restores from CPU snapshot on switch.
    """

    def __init__(self) -> None:
        self.current_name: Optional[str] = None
        self.current_weight: float = 0.0
        # Fallback-only state
        self._original_unet_state: Optional[dict] = None
        self._original_te_state: Optional[dict] = None
        # Fast path: track loaded adapter names to avoid redundant loads
        self._loaded_adapters: set[str] = set()

    # ── Fast path: set_adapters() ────────────────────────────

    def _set_lora_fast(self, pipe, name: Optional[str], weight: float) -> None:
        """Apply LoRA using set_adapters (peft >= 0.18.1). ~1ms vs 200-400ms fuse/unfuse."""
        if name is None:
            # Disable all adapters
            if self.current_name is not None:
                try:
                    pipe.set_adapters([], [])
                except Exception:
                    # Some diffusers versions don't accept empty list — disable instead
                    try:
                        pipe.disable_lora()
                    except Exception as e:
                        log.warning("Failed to disable LoRA adapters: %s", e)
            self.current_name = None
            self.current_weight = 0.0
            return

        path = resolve_lora_path(name)
        adapter_name = _sanitize_adapter_name(name)

        # Load adapter if not already cached in the pipeline
        if adapter_name not in self._loaded_adapters:
            log.info("Loading LoRA adapter '%s' from %s", adapter_name, path)
            try:
                pipe.load_lora_weights(
                    str(path.parent),
                    weight_name=path.name,
                    adapter_name=adapter_name,
                    local_files_only=True,
                )
                self._loaded_adapters.add(adapter_name)
            except Exception as e:
                log.error("Failed to load LoRA adapter '%s': %s", adapter_name, e)
                raise

        # Activate the adapter with the desired weight (~1ms)
        try:
            pipe.set_adapters([adapter_name], [weight])
            log.debug("set_adapters('%s', weight=%.2f) — ~1ms switch", adapter_name, weight)
        except Exception as e:
            log.error("set_adapters failed for '%s': %s", adapter_name, e)
            raise

        self.current_name = name
        self.current_weight = weight

    # ── Fallback path: fuse/unfuse with snapshot ─────────────

    def _ensure_snapshot(self, pipe) -> None:
        """Snapshot UNet + text_encoder weights to CPU before the first style LoRA fuse.

        Both are captured because fuse_lora() merges adapter weights into
        BOTH the UNet and text_encoder.  Without a text_encoder snapshot,
        successive LoRA switches accumulate fused weights in the encoder,
        causing semantic drift.

        Stores in the model's native dtype (float16 for SD1.5) instead of
        bfloat16 — eliminates the bf16→fp16 conversion at each restore,
        saving ~200-400ms per LoRA switch.  Both are 2 bytes/param, so
        memory usage is identical.
        """
        if self._original_unet_state is None:
            raw_unet = _get_raw_module(pipe.unet)
            target_dtype = next(raw_unet.parameters()).dtype
            self._original_unet_state = {
                k: v.to(dtype=target_dtype, device="cpu") if v.is_floating_point() else v.cpu()
                for k, v in raw_unet.state_dict().items()
            }
            log.info("UNet weight snapshot captured (CPU, dtype=%s)", target_dtype)
        if self._original_te_state is None and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            raw_te = _get_raw_module(pipe.text_encoder)
            te_dtype = next(raw_te.parameters()).dtype
            self._original_te_state = {
                k: v.to(dtype=te_dtype, device="cpu") if v.is_floating_point() else v.cpu()
                for k, v in raw_te.state_dict().items()
            }
            log.info("Text encoder weight snapshot captured (CPU, dtype=%s)", te_dtype)

    def _restore_weights(self, pipe) -> None:
        """Restore UNet + text_encoder to exact original weights from CPU snapshot.

        Uses load_state_dict with assign=False (default) to COPY snapshot data
        into existing CUDA tensor objects.  This preserves torch.compile's Dynamo
        graph tensor references — no object replacement, no stale pointers, no
        device mismatch.  The copy operation handles CPU→CUDA transfer internally.

        Snapshot is stored in the model's native dtype (float16), so no dtype
        conversion is needed — the dict is passed directly to load_state_dict.

        Falls back to dynamo.reset() only if a device anomaly is detected.
        """
        if self._original_unet_state is not None:
            raw_unet = _get_raw_module(pipe.unet)
            first_param = next(raw_unet.parameters())
            device = first_param.device
            # No dtype conversion needed — snapshot already in model's native dtype.
            # assign=False (default): copies data INTO existing CUDA tensors.
            # Preserves tensor object identity → compiled graph refs stay valid.
            raw_unet.load_state_dict(self._original_unet_state)
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
            # No dtype conversion needed — snapshot already in model's native dtype.
            raw_te.load_state_dict(self._original_te_state)
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

    def _set_lora_fuse(self, pipe, name: Optional[str], weight: float) -> None:
        """Load or switch style LoRA via fuse/unfuse (fallback for older peft)."""

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

    # ── Public API ───────────────────────────────────────────

    def set_lora(self, pipe, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch style LoRA.

        Dispatches to set_adapters() fast path (peft >= 0.18.1) or
        fuse/unfuse fallback (older peft) based on runtime detection.
        """
        # Validate new LoRA path BEFORE any state changes
        if name is not None:
            resolve_lora_path(name)  # raises ValueError if invalid

        if _USE_SET_ADAPTERS:
            self._set_lora_fast(pipe, name, weight)
        else:
            self._set_lora_fuse(pipe, name, weight)
