"""DeepCache lifecycle management — setup and context manager for toggle."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from .config import settings

log = logging.getLogger("pixytoon.deepcache")


def create_helper(pipe):
    """Create and enable a DeepCacheSDHelper for the given pipeline.

    Returns the helper on success, None if DeepCache is unavailable.
    """
    if not settings.enable_deepcache:
        return None
    try:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=settings.deepcache_interval,
            cache_branch_id=settings.deepcache_branch,
        )
        helper.enable()
        log.info("DeepCache enabled (interval=%d)", settings.deepcache_interval)
        return helper
    except Exception as e:
        log.warning("DeepCache unavailable: %s", e)
        if 'helper' in locals():
            try:
                helper.disable()
            except Exception:
                pass
        return None


def disable(helper) -> None:
    """Disable DeepCache on the given helper."""
    if helper is None:
        return
    helper.disable()
    log.info("DeepCache disabled")


def enable(helper) -> None:
    """Re-enable DeepCache on the given helper."""
    if helper is None:
        return
    helper.enable()
    log.info("DeepCache enabled")


@contextmanager
def suspended(helper):
    """Context manager: disable DeepCache on enter, re-enable on exit.

    If disable fails, logs a warning and proceeds — AnimateDiff/chain should
    run without DeepCache (5D latent shapes are incompatible with 4D caching).
    """
    if helper is None:
        yield
        return

    disabled = False
    try:
        helper.disable()
        disabled = True
        log.info("DeepCache temporarily disabled")
    except Exception as e:
        log.warning("Failed to disable DeepCache: %s — proceeding anyway", e)
    try:
        yield
    finally:
        if disabled:
            try:
                helper.enable()
                log.info("DeepCache re-enabled")
            except Exception as e:
                log.warning("Failed to re-enable DeepCache: %s", e)
