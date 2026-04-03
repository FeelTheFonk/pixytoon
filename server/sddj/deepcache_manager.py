"""DeepCache lifecycle management — setup, context manager, and mode-aware state."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

from .config import settings

log = logging.getLogger("sddj.deepcache")


def create_helper(pipe):
    """Create and enable a DeepCacheSDHelper for the given pipeline.

    Returns the helper on success, None if DeepCache is unavailable.
    """
    if not settings.enable_deepcache:
        return None
    helper = None
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
        if helper is not None:
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


class DeepCacheState:
    """Mode-aware DeepCache state — avoids redundant disable/enable cycles.

    Instead of suspending DeepCache per-call (100-300ms each toggle),
    tracks which mode suppressed it and only toggles on actual mode transitions.
    """

    __slots__ = ("helper", "is_active", "_suppressed_mode")

    def __init__(self, helper) -> None:
        self.helper = helper
        self.is_active = helper is not None
        self._suppressed_mode: Optional[str] = None

    def suppress_for(self, mode: str) -> bool:
        """Disable DeepCache for an incompatible mode.

        Returns True if state actually changed (disable was called).
        No-op if already suppressed for the same mode.
        """
        if self._suppressed_mode == mode:
            return False  # Already suppressed for this mode — zero overhead
        if self.helper is not None and self.is_active:
            try:
                self.helper.disable()
                self.is_active = False
                log.debug("DeepCache suppressed for mode=%s", mode)
            except Exception as e:
                log.warning("Failed to suppress DeepCache: %s", e)
        self._suppressed_mode = mode
        return True

    def restore(self) -> bool:
        """Re-enable DeepCache (e.g., returning to txt2img).

        Returns True if state actually changed (enable was called).
        """
        if not self.is_active and self.helper is not None:
            try:
                self.helper.enable()
                self.is_active = True
                self._suppressed_mode = None
                log.debug("DeepCache restored (txt2img mode)")
                return True
            except Exception as e:
                log.warning("Failed to restore DeepCache: %s", e)
        return False

