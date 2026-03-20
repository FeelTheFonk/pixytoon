"""Standalone test: inpainting (img2img + mask composite).

Usage:
    cd C:\\Users\\CleS\\Desktop\\pixytoon\\server
    uv run python ..\\scripts\\test_inpaint.py

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

import websockets
from PIL import Image
import numpy as np


SERVER_URL = "ws://127.0.0.1:9876/ws"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_output"


def create_test_source() -> str:
    """Create a 512x512 solid red test image, return as base64 PNG."""
    img = Image.fromarray(
        np.full((512, 512, 3), [200, 50, 50], dtype=np.uint8), "RGB"
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


def create_test_mask() -> str:
    """Create a 512x512 mask: white 128x128 center square, rest black."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Center square: (192, 192) to (320, 320) = 128x128
    mask[192:320, 192:320] = 255
    img = Image.fromarray(mask, "L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


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
    all_passed = True

    # ── Test 1: Inpaint ──────────────────────────────────────

    print("\n[TEST] 1/2 — inpaint generation...")
    source_b64 = create_test_source()
    mask_b64 = create_test_mask()

    t0 = time.perf_counter()
    await ws.send(json.dumps({
        "action": "generate",
        "prompt": "pixel art, PixArFK, blue crystal gem, sharp pixels",
        "mode": "inpaint",
        "width": 512,
        "height": 512,
        "seed": 42,
        "steps": 8,
        "cfg_scale": 5.0,
        "denoise_strength": 0.8,
        "source_image": source_b64,
        "mask_image": mask_b64,
        "post_process": {
            "pixelate": {"enabled": True, "target_size": 64},
            "quantize_method": "kmeans",
            "quantize_colors": 16,
            "dither": "none",
            "palette": {"mode": "auto"},
            "remove_bg": False,
        },
    }))

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        resp = json.loads(raw)
        if resp["type"] == "progress":
            pct = int(resp["step"] / resp["total"] * 100)
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%)")
        elif resp["type"] == "result":
            elapsed = time.perf_counter() - t0
            print(f"[OK]   inpaint done — {resp['time_ms']}ms, seed={resp['seed']}, "
                  f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
            out = OUTPUT_DIR / "test_inpaint.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Saved: {out}")
            break
        elif resp["type"] == "error":
            print(f"[FAIL] inpaint error: {resp['message']}")
            all_passed = False
            break

    # ── Test 2: Inpaint with low denoise ─────────────────────

    print("\n[TEST] 2/2 — inpaint with low denoise (0.3)...")
    t0 = time.perf_counter()
    await ws.send(json.dumps({
        "action": "generate",
        "prompt": "pixel art, green tree, sharp pixels",
        "mode": "inpaint",
        "width": 512,
        "height": 512,
        "seed": 99,
        "steps": 8,
        "cfg_scale": 5.0,
        "denoise_strength": 0.3,
        "source_image": source_b64,
        "mask_image": mask_b64,
        "post_process": {
            "pixelate": {"enabled": True, "target_size": 64},
            "quantize_method": "kmeans",
            "quantize_colors": 16,
            "dither": "none",
            "palette": {"mode": "auto"},
            "remove_bg": False,
        },
    }))

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        resp = json.loads(raw)
        if resp["type"] == "progress":
            pct = int(resp["step"] / resp["total"] * 100)
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%)")
        elif resp["type"] == "result":
            elapsed = time.perf_counter() - t0
            print(f"[OK]   inpaint (low denoise) done — {resp['time_ms']}ms, seed={resp['seed']}, "
                  f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
            out = OUTPUT_DIR / "test_inpaint_low_denoise.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Saved: {out}")
            break
        elif resp["type"] == "error":
            print(f"[FAIL] inpaint (low denoise) error: {resp['message']}")
            all_passed = False
            break

    await ws.close()
    print(f"\n{'[ALL PASSED]' if all_passed else '[SOME FAILED]'}")
    return all_passed


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
