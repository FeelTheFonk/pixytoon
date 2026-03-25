"""Tests for eager_pipeline context manager — UNet swap + DeepCache suspend + dynamo reset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_components():
    """Create mock pipe, img2img_pipe, controlnet_pipe, deepcache_helper."""
    pipe = MagicMock()
    img2img_pipe = MagicMock()
    controlnet_pipe = MagicMock()
    deepcache_helper = MagicMock()

    raw_unet = MagicMock(name="raw_unet")
    compiled_unet = MagicMock(name="compiled_unet")
    pipe.unet = compiled_unet
    # Simulate _orig_mod for get_uncompiled_unet
    compiled_unet._orig_mod = raw_unet

    return pipe, img2img_pipe, controlnet_pipe, deepcache_helper, raw_unet, compiled_unet


def test_eager_pipeline_swaps_unet(mock_components):
    """Inside eager_pipeline, all pipes use the raw UNet."""
    pipe, img2img, cn, dc, raw_unet, compiled_unet = mock_components

    from sddj.engine.compile_utils import eager_pipeline
    with (
        patch("sddj.engine.compile_utils.deepcache_manager.suspended") as mock_suspended,
        patch("sddj.engine.compile_utils.torch._dynamo.reset"),
    ):
        mock_suspended.return_value.__enter__ = MagicMock()
        mock_suspended.return_value.__exit__ = MagicMock(return_value=False)

        with eager_pipeline(pipe, img2img, cn, dc):
            assert pipe.unet is raw_unet
            assert img2img.unet is raw_unet
            assert cn.unet is raw_unet


def test_eager_pipeline_restores_on_exit(mock_components):
    """After eager_pipeline exits, compiled UNet is restored."""
    pipe, img2img, cn, dc, raw_unet, compiled_unet = mock_components

    from sddj.engine.compile_utils import eager_pipeline
    with (
        patch("sddj.engine.compile_utils.deepcache_manager.suspended") as mock_suspended,
        patch("sddj.engine.compile_utils.torch._dynamo.reset"),
    ):
        mock_suspended.return_value.__enter__ = MagicMock()
        mock_suspended.return_value.__exit__ = MagicMock(return_value=False)

        with eager_pipeline(pipe, img2img, cn, dc):
            pass

    assert pipe.unet is compiled_unet
    assert img2img.unet is compiled_unet
    assert cn.unet is compiled_unet


def test_eager_pipeline_restores_on_exception(mock_components):
    """Compiled UNet is restored even if body raises."""
    pipe, img2img, cn, dc, raw_unet, compiled_unet = mock_components

    from sddj.engine.compile_utils import eager_pipeline
    with (
        patch("sddj.engine.compile_utils.deepcache_manager.suspended") as mock_suspended,
        patch("sddj.engine.compile_utils.torch._dynamo.reset"),
    ):
        mock_suspended.return_value.__enter__ = MagicMock()
        mock_suspended.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(RuntimeError):
            with eager_pipeline(pipe, img2img, cn, dc):
                raise RuntimeError("test")

    assert pipe.unet is compiled_unet


def test_eager_pipeline_none_controlnet(mock_components):
    """Works when controlnet_pipe is None."""
    pipe, img2img, _, dc, raw_unet, compiled_unet = mock_components

    from sddj.engine.compile_utils import eager_pipeline
    with (
        patch("sddj.engine.compile_utils.deepcache_manager.suspended") as mock_suspended,
        patch("sddj.engine.compile_utils.torch._dynamo.reset"),
    ):
        mock_suspended.return_value.__enter__ = MagicMock()
        mock_suspended.return_value.__exit__ = MagicMock(return_value=False)

        with eager_pipeline(pipe, img2img, None, dc):
            assert pipe.unet is raw_unet
