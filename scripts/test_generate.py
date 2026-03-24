"""Standalone test: single-frame generation (txt2img + img2img).

Usage:
    cd C:\\Users\\CleS\\Desktop\\SDDj\\server
    uv run python ..\\scripts\\test_generate.py

Requires the SDDj server to be running on ws://127.0.0.1:9876/ws.
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
        print("       Make sure the server is running (start.ps1 or uv run python run.py)")
        return False

    print("[OK]   Connected")
    all_passed = True

    # ── Test 1: txt2img ──────────────────────────────────────

    print("\n[TEST] 1/3 — txt2img generation...")
    t0 = time.perf_counter()
    await ws.send(json.dumps({
        "action": "generate",
        "prompt": "pixel art, PixArFK, game sprite, a brave knight, sharp pixels",
        "mode": "txt2img",
        "width": 512,
        "height": 512,
        "seed": 42,
        "steps": 8,
        "cfg_scale": 5.0,
        "post_process": {
            "pixelate": {"enabled": True, "target_size": 64},
            "quantize_enabled": True,
            "quantize_method": "kmeans",
            "quantize_colors": 16,
            "dither": "none",
            "palette": {"mode": "auto"},
            "remove_bg": False,
        },
    }))

    result_b64 = None
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        resp = json.loads(raw)
        if resp["type"] == "progress":
            pct = int(resp["step"] / resp["total"] * 100)
            print(f"       Step {resp['step']}/{resp['total']} ({pct}%)")
        elif resp["type"] == "result":
            elapsed = time.perf_counter() - t0
            print(f"[OK]   txt2img done — {resp['time_ms']}ms, seed={resp['seed']}, "
                  f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
            out = OUTPUT_DIR / "test_txt2img.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Saved: {out}")
            result_b64 = resp["image"]
            break
        elif resp["type"] == "error":
            print(f"[FAIL] txt2img error: {resp['message']}")
            all_passed = False
            break

    # ── Test 2: img2img (using txt2img result) ───────────────

    if result_b64:
        print("\n[TEST] 2/3 — img2img generation (from txt2img result)...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "action": "generate",
            "prompt": "pixel art, PixArFK, game sprite, a brave knight with fire sword, sharp pixels",
            "mode": "img2img",
            "width": 512,
            "height": 512,
            "seed": 123,
            "steps": 8,
            "cfg_scale": 5.0,
            "denoise_strength": 0.5,
            "source_image": result_b64,
            "post_process": {
                "pixelate": {"enabled": True, "target_size": 64},
                "quantize_enabled": True,
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
                print(f"[OK]   img2img done — {resp['time_ms']}ms, seed={resp['seed']}, "
                      f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
                out = OUTPUT_DIR / "test_img2img.png"
                out.write_bytes(b64decode(resp["image"]))
                print(f"       Saved: {out}")
                break
            elif resp["type"] == "error":
                print(f"[FAIL] img2img error: {resp['message']}")
                all_passed = False
                break
    else:
        print("\n[SKIP] 2/3 — img2img (no source from previous test)")
        all_passed = False

    # ── Test 3: inpaint (using txt2img result) ───────────────

    if result_b64:
        print("\n[TEST] 3/3 — inpaint generation (from txt2img result)...")

        # Create a synthetic mask: white center square on black
        mask_arr = np.zeros((512, 512), dtype=np.uint8)
        mask_arr[192:320, 192:320] = 255
        mask_img = Image.fromarray(mask_arr, "L")
        buf = BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = b64encode(buf.getvalue()).decode("ascii")

        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "action": "generate",
            "prompt": "pixel art, PixArFK, game sprite, a golden crown, sharp pixels",
            "mode": "inpaint",
            "width": 512,
            "height": 512,
            "seed": 456,
            "steps": 8,
            "cfg_scale": 5.0,
            "denoise_strength": 0.7,
            "source_image": result_b64,
            "mask_image": mask_b64,
            "post_process": {
                "pixelate": {"enabled": True, "target_size": 64},
                "quantize_enabled": True,
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
                out = OUTPUT_DIR / "test_inpaint_from_gen.png"
                out.write_bytes(b64decode(resp["image"]))
                print(f"       Saved: {out}")
                break
            elif resp["type"] == "error":
                print(f"[FAIL] inpaint error: {resp['message']}")
                all_passed = False
                break
    else:
        print("\n[SKIP] 3/3 — inpaint (no source from previous test)")
        all_passed = False

    await ws.close()
    print(f"\n{'[ALL PASSED]' if all_passed else '[SOME FAILED]'}")
    return all_passed


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
