"""Shared test utilities for SDDj integration tests.

Eliminates duplication across test_animation.py, test_generate.py,
test_inpaint.py by providing:
  - WebSocket connection context manager with auto-close
  - Generic recv loop with safety counter and progress display
  - Test image/mask factories
  - Structured test result reporting
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from base64 import b64decode, b64encode
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_URL = "ws://127.0.0.1:9876/ws"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_output"
WS_MAX_SIZE = 50 * 1024 * 1024  # 50MB — must match server ws_max_size

DEFAULT_POST_PROCESS: dict[str, Any] = {
    "pixelate": {"enabled": True, "target_size": 64},
    "quantize_enabled": True,
    "quantize_method": "kmeans",
    "quantize_colors": 16,
    "dither": "none",
    "palette": {"mode": "auto"},
    "remove_bg": False,
}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    error: str = ""


def print_summary(results: list[TestResult]) -> bool:
    """Print test results and return True if all passed."""
    all_ok = all(r.passed for r in results)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'[ALL PASSED]' if all_ok else '[SOME FAILED]'} ({passed}/{total})")
    if not all_ok:
        for r in results:
            if not r.passed:
                print(f"  [FAIL] {r.name}: {r.error}")
    return all_ok


def write_report(results: list[TestResult], output_dir: Path | None = None) -> None:
    """Write JSON test report to output directory."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": [
            {
                "name": r.name,
                "passed": r.passed,
                "duration_s": round(r.duration_s, 3),
                **({"error": r.error} if r.error else {}),
            }
            for r in results
        ],
        "all_passed": all(r.passed for r in results),
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# WebSocket connection
# ---------------------------------------------------------------------------

@asynccontextmanager
async def ws_connection(url: str = SERVER_URL, connect_timeout: float = 10.0):
    """Connect to WebSocket server with auto-close on exit."""
    try:
        ws = await asyncio.wait_for(
            websockets.connect(url, max_size=WS_MAX_SIZE),
            timeout=connect_timeout,
        )
    except Exception as e:
        print(f"[FAIL] Could not connect to {url}: {e}")
        print("       Make sure the server is running (start.ps1 or uv run python run.py)")
        raise SystemExit(1)

    print("[OK]   Connected")
    try:
        yield ws
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Message recv helpers
# ---------------------------------------------------------------------------

MAX_MESSAGES = 5000  # Safety counter to prevent infinite loops


async def recv_generation_result(
    ws,
    *,
    timeout: float = 120.0,
    label: str = "generation",
) -> dict[str, Any] | None:
    """Receive messages until a 'result' or 'error' message arrives.

    Returns the result dict on success, None on error.
    Prints progress updates as they arrive.
    """
    count = 0
    while count < MAX_MESSAGES:
        count += 1
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        resp = json.loads(raw)

        if resp["type"] == "progress":
            pct = int(resp["step"] / resp["total"] * 100)
            frame_ctx = ""
            if resp.get("frame_index") is not None:
                frame_ctx = f" [Frame {resp['frame_index'] + 1}/{resp.get('total_frames', '?')}]"
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%){frame_ctx}")

        elif resp["type"] == "result":
            return resp

        elif resp["type"] == "error":
            print(f"[FAIL] {label} error: {resp['message']}")
            return None

    print(f"[FAIL] {label}: exceeded {MAX_MESSAGES} messages without result")
    return None


async def recv_animation_result(
    ws,
    *,
    expected_frames: int,
    timeout_per_frame: float = 60.0,
    label: str = "animation",
    output_prefix: str = "frame",
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Receive messages until 'animation_complete' arrives.

    Saves frame images to disk. Returns the complete response dict
    on success, None on error.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(exist_ok=True)
    timeout = expected_frames * timeout_per_frame
    frames_received = 0
    count = 0

    while count < MAX_MESSAGES:
        count += 1
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        resp = json.loads(raw)

        if resp["type"] == "progress":
            pct = int(resp["step"] / resp["total"] * 100)
            frame_ctx = ""
            if resp.get("frame_index") is not None:
                frame_ctx = f" [Frame {resp['frame_index'] + 1}/{resp.get('total_frames', '?')}]"
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%){frame_ctx}")

        elif resp["type"] == "animation_frame":
            frames_received += 1
            fname = f"{output_prefix}_{resp['frame_index']:02d}.png"
            (out / fname).write_bytes(b64decode(resp["image"]))
            print(f"       Frame {resp['frame_index'] + 1}/{resp['total_frames']} — "
                  f"{resp['time_ms']}ms, seed={resp['seed']} -> {fname}")

        elif resp["type"] == "animation_complete":
            if frames_received != expected_frames:
                print(f"[FAIL] {label} — expected {expected_frames} frames, got {frames_received}")
                return None
            print(f"[OK]   {label} done — {resp['total_frames']} frames, "
                  f"{resp['total_time_ms']}ms total")
            return resp

        elif resp["type"] == "error":
            print(f"[FAIL] {label} error: {resp['message']}")
            return None

    print(f"[FAIL] {label}: exceeded {MAX_MESSAGES} messages without completion")
    return None


# ---------------------------------------------------------------------------
# Test image/mask factories
# ---------------------------------------------------------------------------

def create_test_image(width: int = 512, height: int = 512,
                      color: tuple[int, int, int] = (50, 100, 200)) -> str:
    """Create a solid-color test image, return as base64 PNG."""
    img = Image.fromarray(
        np.full((height, width, 3), color, dtype=np.uint8), "RGB"
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


def create_test_mask(width: int = 512, height: int = 512,
                     region: tuple[int, int, int, int] = (192, 192, 320, 320)) -> str:
    """Create a binary mask (white rectangle on black), return as base64 PNG.

    region: (y1, x1, y2, x2) — the white area.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    y1, x1, y2, x2 = region
    mask[y1:y2, x1:x2] = 255
    img = Image.fromarray(mask, "L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")
