"""Standalone test: animation (chain + animatediff).

Usage:
    cd C:\\Users\\CleS\\Desktop\\pixytoon\\server
    uv run python ..\\scripts\\test_animation.py

Requires the PixyToon server to be running on ws://127.0.0.1:9876/ws.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from base64 import b64decode, b64encode
from io import BytesIO
from pathlib import Path

import numpy as np
import websockets
from PIL import Image


SERVER_URL = "ws://127.0.0.1:9876/ws"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_output"
FRAME_COUNT = 4


async def test_animation(ws, method: str, test_num: int, total: int) -> bool:
    """Run one animation test (chain or animatediff)."""
    print(f"\n[TEST] {test_num}/{total} — {method} animation ({FRAME_COUNT} frames)...")
    t0 = time.perf_counter()

    await ws.send(json.dumps({
        "action": "generate_animation",
        "method": method,
        "prompt": "pixel art, PixArFK, game sprite, walking cycle, sharp pixels",
        "mode": "txt2img",
        "width": 512,
        "height": 512,
        "seed": 42,
        "steps": 8,
        "cfg_scale": 5.0,
        "denoise_strength": 0.30,
        "frame_count": FRAME_COUNT,
        "frame_duration_ms": 100,
        "seed_strategy": "increment",
        "tag_name": f"test_{method}",
        "post_process": {
            "pixelate": {"enabled": True, "target_size": 64},
            "quantize_method": "kmeans",
            "quantize_colors": 16,
            "dither": "none",
            "palette": {"mode": "auto"},
            "remove_bg": False,
        },
    }))

    frames_received = 0
    complete = False

    # Generous timeout: 60s per frame for safety
    timeout = FRAME_COUNT * 60

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        resp = json.loads(raw)

        if resp["type"] == "progress":
            frame_ctx = ""
            if resp.get("frame_index") is not None:
                frame_ctx = f" [Frame {resp['frame_index'] + 1}/{resp.get('total_frames', '?')}]"
            pct = int(resp["step"] / resp["total"] * 100)
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%){frame_ctx}")

        elif resp["type"] == "animation_frame":
            frames_received += 1
            out = OUTPUT_DIR / f"test_{method}_frame{resp['frame_index']:02d}.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Frame {resp['frame_index'] + 1}/{resp['total_frames']} — "
                  f"{resp['time_ms']}ms, seed={resp['seed']} → {out.name}")

        elif resp["type"] == "animation_complete":
            elapsed = time.perf_counter() - t0
            print(f"[OK]   {method} animation done — {resp['total_frames']} frames, "
                  f"{resp['total_time_ms']}ms total, wall={elapsed:.1f}s")
            complete = True
            break

        elif resp["type"] == "error":
            print(f"[FAIL] {method} animation error: {resp['message']}")
            return False

    if not complete:
        print(f"[FAIL] {method} — never received animation_complete")
        return False

    if frames_received != FRAME_COUNT:
        print(f"[FAIL] {method} — expected {FRAME_COUNT} frames, got {frames_received}")
        return False

    print(f"[OK]   All {frames_received} frames received and saved")
    return True


def create_test_source() -> str:
    """Create a 512x512 solid blue test image, return as base64 PNG."""
    img = Image.fromarray(
        np.full((512, 512, 3), [50, 100, 200], dtype=np.uint8), "RGB"
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


async def test_chain_img2img(ws, test_num: int, total: int) -> bool:
    """Test chain animation starting from img2img (frame 0 = img2img)."""
    print(f"\n[TEST] {test_num}/{total} — chain img2img animation ({FRAME_COUNT} frames)...")
    t0 = time.perf_counter()

    source_b64 = create_test_source()

    await ws.send(json.dumps({
        "action": "generate_animation",
        "method": "chain",
        "prompt": "pixel art, PixArFK, game sprite, idle cycle, sharp pixels",
        "mode": "img2img",
        "width": 512,
        "height": 512,
        "seed": 100,
        "steps": 8,
        "cfg_scale": 5.0,
        "denoise_strength": 0.30,
        "frame_count": FRAME_COUNT,
        "frame_duration_ms": 100,
        "seed_strategy": "increment",
        "source_image": source_b64,
        "tag_name": "test_chain_img2img",
        "post_process": {
            "pixelate": {"enabled": True, "target_size": 64},
            "quantize_method": "kmeans",
            "quantize_colors": 16,
            "dither": "none",
            "palette": {"mode": "auto"},
            "remove_bg": False,
        },
    }))

    frames_received = 0
    complete = False
    timeout = FRAME_COUNT * 60

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        resp = json.loads(raw)

        if resp["type"] == "progress":
            frame_ctx = ""
            if resp.get("frame_index") is not None:
                frame_ctx = f" [Frame {resp['frame_index'] + 1}/{resp.get('total_frames', '?')}]"
            pct = int(resp["step"] / resp["total"] * 100)
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%){frame_ctx}")

        elif resp["type"] == "animation_frame":
            frames_received += 1
            out = OUTPUT_DIR / f"test_chain_img2img_frame{resp['frame_index']:02d}.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Frame {resp['frame_index'] + 1}/{resp['total_frames']} — "
                  f"{resp['time_ms']}ms, seed={resp['seed']} → {out.name}")

        elif resp["type"] == "animation_complete":
            elapsed = time.perf_counter() - t0
            print(f"[OK]   chain img2img animation done — {resp['total_frames']} frames, "
                  f"{resp['total_time_ms']}ms total, wall={elapsed:.1f}s")
            complete = True
            break

        elif resp["type"] == "error":
            print(f"[FAIL] chain img2img animation error: {resp['message']}")
            return False

    if not complete:
        print(f"[FAIL] chain img2img — never received animation_complete")
        return False

    if frames_received != FRAME_COUNT:
        print(f"[FAIL] chain img2img — expected {FRAME_COUNT} frames, got {frames_received}")
        return False

    print(f"[OK]   All {frames_received} frames received and saved")
    return True


async def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"[TEST] Connecting to {SERVER_URL}...")

    try:
        ws = await asyncio.wait_for(
            websockets.connect(SERVER_URL, max_size=50 * 1024 * 1024),
            timeout=10,
        )
    except Exception as e:
        print(f"[FAIL] Could not connect: {e}")
        print("       Make sure the server is running (start.bat or uv run python run.py)")
        return False

    print("[OK]   Connected")

    results = []

    # Test 1: Chain animation (txt2img base)
    results.append(await test_animation(ws, "chain", 1, 3))

    # Test 2: AnimateDiff animation
    results.append(await test_animation(ws, "animatediff", 2, 3))

    # Test 3: Chain animation (img2img base) — validates scheduler reset
    results.append(await test_chain_img2img(ws, 3, 3))

    await ws.close()

    all_passed = all(results)
    print(f"\n{'[ALL PASSED]' if all_passed else '[SOME FAILED]'}")
    return all_passed


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
