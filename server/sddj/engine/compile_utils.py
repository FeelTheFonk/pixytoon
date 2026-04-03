"""Compilation utilities — eager mode context manager for chain animations.

Provides a DRY context manager that handles the UNet swap + DeepCache suspend
sequence needed when running chain/audio-reactive animations in eager mode.
Previously duplicated in animation.py and audio_reactive.py.

Thread safety: UNet swap is protected by _pipeline_lock from pipeline_factory.
The lock is held for the entire duration of the eager block to prevent
concurrent generation from seeing a partially-swapped pipeline state.

Single-client invariant: SDDj currently serializes all generation through
a single asyncio.Lock in server.py.  The _pipeline_lock here is defense-in-depth
for future multi-client scenarios.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from ..animatediff_manager import get_uncompiled_unet
from .. import deepcache_manager
from ..pipeline_factory import _pipeline_lock

log = logging.getLogger("sddj.engine.compile")


@contextmanager
def eager_pipeline(pipe, img2img_pipe, controlnet_pipe, deepcache_helper,
                   controlnet_img2img_pipe=None):
    """Run a block with raw (uncompiled) UNet and DeepCache suspended.

    On enter:
      1. Acquire _pipeline_lock (prevents concurrent UNet access)
      2. Suspend DeepCache (context-managed)
      3. Swap compiled UNet → raw UNet on all pipelines

    On exit (even on exception):
      1. Restore compiled UNet on all pipelines
      2. DeepCache re-enables via its own context manager
      3. Release _pipeline_lock

    dynamo.reset() is NOT called here — it would force cold recompilation
    on every chain frame. Dynamo resets only when weights change (LoRAFuser).
    """
    raw_unet = get_uncompiled_unet(pipe)
    compiled_unet = pipe.unet

    with _pipeline_lock:
        with deepcache_manager.suspended(deepcache_helper):
            # Swap to raw UNet on all active pipelines
            pipe.unet = raw_unet
            img2img_pipe.unet = raw_unet
            if controlnet_pipe is not None:
                controlnet_pipe.unet = raw_unet
            if controlnet_img2img_pipe is not None:
                controlnet_img2img_pipe.unet = raw_unet
            try:
                yield
            finally:
                # Restore compiled UNet
                pipe.unet = compiled_unet
                img2img_pipe.unet = compiled_unet
                if controlnet_pipe is not None:
                    controlnet_pipe.unet = compiled_unet
                if controlnet_img2img_pipe is not None:
                    controlnet_img2img_pipe.unet = compiled_unet
