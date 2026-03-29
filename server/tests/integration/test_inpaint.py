"""Standalone test: inpainting (img2img + mask composite).

Usage:
    cd C:\\Users\\CleS\\Desktop\\SDDj\\server
    uv run python ..\\scripts\\test_inpaint.py

Requires the SDDj server to be running on ws://127.0.0.1:9876/ws.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from base64 import b64decode

from _test_lib import (
    DEFAULT_POST_PROCESS,
    OUTPUT_DIR,
    TestResult,
    create_test_image,
    create_test_mask,
    print_summary,
    recv_generation_result,
    write_report,
    ws_connection,
)


async def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results: list[TestResult] = []

    source_b64 = create_test_image(color=(200, 50, 50))
    mask_b64 = create_test_mask()

    async with ws_connection() as ws:

        # ── Test 1: Inpaint ──────────────────────────────────
        print("\n[TEST] 1/2 — inpaint generation...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "action": "generate",
            "prompt": "pixel art, PixArFK, blue crystal gem, sharp pixels",
            "mode": "inpaint",
            "width": 512, "height": 512,
            "seed": 42, "steps": 8, "cfg_scale": 5.0,
            "denoise_strength": 0.8,
            "source_image": source_b64,
            "mask_image": mask_b64,
            "post_process": DEFAULT_POST_PROCESS,
        }))

        resp = await recv_generation_result(ws, label="inpaint")
        elapsed = time.perf_counter() - t0

        if resp:
            print(f"[OK]   inpaint done — {resp['time_ms']}ms, seed={resp['seed']}, "
                  f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
            out = OUTPUT_DIR / "test_inpaint.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Saved: {out}")
            results.append(TestResult("inpaint", True, elapsed))
        else:
            results.append(TestResult("inpaint", False, elapsed, "No result"))

        # ── Test 2: Inpaint with low denoise ─────────────────
        print("\n[TEST] 2/2 — inpaint with low denoise (0.3)...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "action": "generate",
            "prompt": "pixel art, green tree, sharp pixels",
            "mode": "inpaint",
            "width": 512, "height": 512,
            "seed": 99, "steps": 8, "cfg_scale": 5.0,
            "denoise_strength": 0.3,
            "source_image": source_b64,
            "mask_image": mask_b64,
            "post_process": DEFAULT_POST_PROCESS,
        }))

        resp = await recv_generation_result(ws, label="inpaint (low denoise)")
        elapsed = time.perf_counter() - t0

        if resp:
            print(f"[OK]   inpaint (low denoise) done — {resp['time_ms']}ms, seed={resp['seed']}, "
                  f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
            out = OUTPUT_DIR / "test_inpaint_low_denoise.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Saved: {out}")
            results.append(TestResult("inpaint_low_denoise", True, elapsed))
        else:
            results.append(TestResult("inpaint_low_denoise", False, elapsed, "No result"))

    write_report(results)
    return print_summary(results)


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
