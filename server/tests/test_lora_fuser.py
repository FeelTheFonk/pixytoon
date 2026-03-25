"""Tests for LoRA fuser lifecycle — snapshot, restore, hotswap, dynamo reset."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch
from collections import OrderedDict

import pytest


@pytest.fixture
def mock_settings():
    """Provide mock settings with all required fields."""
    with patch("sddj.lora_fuser.settings") as ms:
        ms.enable_torch_compile = True
        ms.enable_lora_hotswap = True
        ms.loras_dir = MagicMock()
        yield ms


@pytest.fixture
def mock_pipe():
    """Provide a mock pipeline with UNet state dict."""
    pipe = MagicMock()
    # Simulate a small UNet state dict
    import torch
    state = OrderedDict({
        "conv.weight": torch.randn(3, 3),
        "conv.bias": torch.randn(3),
    })
    pipe.unet.state_dict.return_value = state
    return pipe


def test_snapshot_captured_once(mock_settings, mock_pipe):
    """Weight snapshot is captured only once, before first style LoRA fuse."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)
    assert fuser._original_unet_state is not None
    first_snapshot = fuser._original_unet_state

    fuser._ensure_snapshot(mock_pipe)
    assert fuser._original_unet_state is first_snapshot  # same object — no re-capture


def test_restore_weights(mock_settings, mock_pipe):
    """Restore loads the snapshot back into UNet."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._ensure_snapshot(mock_pipe)
    fuser._restore_weights(mock_pipe)
    mock_pipe.unet.load_state_dict.assert_called_once()


def test_restore_no_snapshot(mock_settings, mock_pipe):
    """Restore with no snapshot is a no-op."""
    from sddj.lora_fuser import LoRAFuser
    fuser = LoRAFuser()
    fuser._restore_weights(mock_pipe)  # should not raise
    mock_pipe.unet.load_state_dict.assert_not_called()


def test_needs_dynamo_reset_with_hotswap(mock_settings):
    """Dynamo reset NOT needed when hotswap is enabled."""
    from sddj.lora_fuser import LoRAFuser
    mock_settings.enable_lora_hotswap = True
    mock_settings.enable_torch_compile = True
    fuser = LoRAFuser()
    assert fuser._needs_dynamo_reset() is False


def test_needs_dynamo_reset_without_hotswap(mock_settings):
    """Dynamo reset IS needed when hotswap is disabled."""
    from sddj.lora_fuser import LoRAFuser
    mock_settings.enable_lora_hotswap = False
    mock_settings.enable_torch_compile = True
    fuser = LoRAFuser()
    assert fuser._needs_dynamo_reset() is True


def test_needs_dynamo_reset_no_compile(mock_settings):
    """Dynamo reset NOT needed when torch.compile disabled."""
    from sddj.lora_fuser import LoRAFuser
    mock_settings.enable_lora_hotswap = False
    mock_settings.enable_torch_compile = False
    fuser = LoRAFuser()
    assert fuser._needs_dynamo_reset() is False
