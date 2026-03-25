"""Tests for vram_utils — centralized VRAM management utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_vram_cleanup_calls_gc_then_empty_cache():
    """gc.collect MUST run before empty_cache (frees Python refs first)."""
    from sddj.vram_utils import vram_cleanup
    call_order = []
    with (
        patch("sddj.vram_utils.gc.collect", side_effect=lambda: call_order.append("gc")),
        patch("sddj.vram_utils.torch.cuda.is_available", return_value=True),
        patch("sddj.vram_utils.torch.cuda.empty_cache", side_effect=lambda: call_order.append("empty")),
    ):
        vram_cleanup()
    assert call_order == ["gc", "empty"]


def test_vram_cleanup_no_cuda():
    """Must not crash when CUDA is unavailable."""
    from sddj.vram_utils import vram_cleanup
    with (
        patch("sddj.vram_utils.gc.collect"),
        patch("sddj.vram_utils.torch.cuda.is_available", return_value=False),
    ):
        vram_cleanup()  # should not raise


def test_get_vram_info_no_cuda():
    """Returns zeros when no CUDA device available."""
    from sddj.vram_utils import get_vram_info
    with patch("sddj.vram_utils.torch.cuda.is_available", return_value=False):
        used, free, total = get_vram_info()
    assert used == 0.0
    assert free == 0.0
    assert total == 0.0


def test_get_vram_info_with_cuda():
    """Returns correct values when CUDA is available."""
    from sddj.vram_utils import get_vram_info
    free_bytes = 4 * 1024 * 1024 * 1024  # 4GB free
    total_bytes = 8 * 1024 * 1024 * 1024  # 8GB total
    with (
        patch("sddj.vram_utils.torch.cuda.is_available", return_value=True),
        patch("sddj.vram_utils.torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)),
    ):
        used, free, total = get_vram_info()
    assert total == 8192.0
    assert free == 4096.0
    assert used == 4096.0


def test_move_to_cpu_none_safe():
    """move_to_cpu(None) must be a no-op."""
    from sddj.vram_utils import move_to_cpu
    move_to_cpu(None)  # should not raise


def test_move_to_cpu_non_module():
    """move_to_cpu(non-module) must be a no-op."""
    from sddj.vram_utils import move_to_cpu
    move_to_cpu("not a module")  # should not raise


def test_move_to_cpu_module():
    """move_to_cpu(module) calls .to('cpu')."""
    import torch.nn as nn
    from sddj.vram_utils import move_to_cpu
    module = MagicMock(spec=nn.Module)
    move_to_cpu(module)
    module.to.assert_called_once_with("cpu")


def test_check_vram_budget_no_cuda():
    """Budget check with no CUDA returns True (CPU mode OK)."""
    from sddj.vram_utils import check_vram_budget
    with patch("sddj.vram_utils.get_vram_info", return_value=(0, 0, 0)):
        assert check_vram_budget(required_mb=1000) is True


def test_check_vram_budget_sufficient():
    """Budget check passes when enough VRAM."""
    from sddj.vram_utils import check_vram_budget
    with patch("sddj.vram_utils.get_vram_info", return_value=(4096, 4096, 8192)):
        assert check_vram_budget(required_mb=1000, min_free_mb=512) is True


def test_check_vram_budget_insufficient():
    """Budget check fails when not enough VRAM."""
    from sddj.vram_utils import check_vram_budget
    with patch("sddj.vram_utils.get_vram_info", return_value=(7500, 692, 8192)):
        assert check_vram_budget(required_mb=500, min_free_mb=512) is False
