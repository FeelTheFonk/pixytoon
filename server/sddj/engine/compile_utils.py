"""Compilation utilities — eager mode context manager for chain animations.

Provides a DRY context manager that handles the UNet swap + DeepCache suspend +
dynamo reset sequence needed when running chain/audio-reactive animations in
eager mode.  Previously duplicated in animation.py and audio_reactive.py.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

from ..animatediff_manager import get_uncompiled_unet
from .. import deepcache_manager

log = logging.getLogger("sddj.engine.compile")


@contextmanager
def eager_pipeline(pipe, img2img_pipe, controlnet_pipe, deepcache_helper):
    """Run a block with raw (uncompiled) UNet and DeepCache suspended.

    On enter:
      1. Suspend DeepCache (context-managed)
      2. Swap compiled UNet → raw UNet on all pipelines
      3. Reset dynamo once

    On exit (even on exception):
      1. Reset dynamo
      2. Restore compiled UNet on all pipelines
      3. DeepCache re-enables via its own context manager
    """
    raw_unet = get_uncompiled_unet(pipe)
    compiled_unet = pipe.unet

    with deepcache_manager.suspended(deepcache_helper):
        # Swap to raw UNet on all active pipelines
        pipe.unet = raw_unet
        img2img_pipe.unet = raw_unet
        if controlnet_pipe is not None:
            controlnet_pipe.unet = raw_unet
        try:
            torch._dynamo.reset()
            yield
        finally:
            try:
                torch._dynamo.reset()
            except Exception:
                log.warning("torch._dynamo.reset() failed in eager pipeline cleanup")
            # Restore compiled UNet
            pipe.unet = compiled_unet
            img2img_pipe.unet = compiled_unet
            if controlnet_pipe is not None:
                controlnet_pipe.unet = compiled_unet
