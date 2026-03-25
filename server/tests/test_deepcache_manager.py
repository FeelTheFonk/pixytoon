"""Tests for DeepCache manager — suspended context manager, create_helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_suspended_disables_then_reenables():
    """suspended() context manager disables on enter, re-enables on exit."""
    from sddj.deepcache_manager import suspended
    helper = MagicMock()
    with suspended(helper):
        helper.disable.assert_called_once()
        helper.enable.assert_not_called()
    helper.enable.assert_called_once()


def test_suspended_reenable_on_exception():
    """Re-enables DeepCache even on exception."""
    from sddj.deepcache_manager import suspended
    helper = MagicMock()
    with pytest.raises(RuntimeError):
        with suspended(helper):
            raise RuntimeError("test")
    helper.enable.assert_called_once()


def test_suspended_with_none_helper():
    """suspended(None) is a no-op — yields without error."""
    from sddj.deepcache_manager import suspended
    with suspended(None):
        pass  # should not raise


def test_create_returns_none_when_disabled():
    """create_helper returns None when DeepCache disabled in settings."""
    from sddj import deepcache_manager
    with patch.object(deepcache_manager, "settings") as ms:
        ms.enable_deepcache = False
        result = deepcache_manager.create_helper(MagicMock())
    assert result is None
