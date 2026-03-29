"""Standalone test: single-frame generation (txt2img + img2img + inpaint).

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
from base64 import b64decode
from pathlib import Path

from _test_lib import (
    DEFAULT_POST_PROCESS,
    OUTPUT_DIR,
    TestResult,
    create_test_mask,
    print_summary,
    recv_generation_result,
    write_report,
    ws_connection,
)


async def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results: list[TestResult] = []

    async with ws_connection() as ws:

        # ── Test 1: txt2img ──────────────────────────────────
        print("\n[TEST] 1/3 — txt2img generation...")
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "action": "generate",
            "prompt": "pixel art, PixArFK, game sprite, a brave knight, sharp pixels",
            "mode": "txt2img",
            "width": 512, "height": 512,
            "seed": 42, "steps": 8, "cfg_scale": 5.0,
            "post_process": DEFAULT_POST_PROCESS,
        }))

        resp = await recv_generation_result(ws, label="txt2img")
        elapsed = time.perf_counter() - t0
        result_b64 = None

        if resp:
            print(f"[OK]   txt2img done — {resp['time_ms']}ms, seed={resp['seed']}, "
                  f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
            out = OUTPUT_DIR / "test_txt2img.png"
            out.write_bytes(b64decode(resp["image"]))
            print(f"       Saved: {out}")
            result_b64 = resp["image"]
            results.append(TestResult("txt2img", True, elapsed))
        else:
            results.append(TestResult("txt2img", False, elapsed, "No result"))

        # ── Test 2: img2img ──────────────────────────────────
        if result_b64:
            print("\n[TEST] 2/3 — img2img generation (from txt2img result)...")
            t0 = time.perf_counter()
            await ws.send(json.dumps({
                "action": "generate",
                "prompt": "pixel art, PixArFK, game sprite, a brave knight with fire sword, sharp pixels",
                "mode": "img2img",
                "width": 512, "height": 512,
                "seed": 123, "steps": 8, "cfg_scale": 5.0,
                "denoise_strength": 0.5,
                "source_image": result_b64,
                "post_process": DEFAULT_POST_PROCESS,
            }))

            resp = await recv_generation_result(ws, label="img2img")
            elapsed = time.perf_counter() - t0

            if resp:
                print(f"[OK]   img2img done — {resp['time_ms']}ms, seed={resp['seed']}, "
                      f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
                out = OUTPUT_DIR / "test_img2img.png"
                out.write_bytes(b64decode(resp["image"]))
                print(f"       Saved: {out}")
                results.append(TestResult("img2img", True, elapsed))
            else:
                results.append(TestResult("img2img", False, elapsed, "No result"))
        else:
            print("\n[SKIP] 2/3 — img2img (no source)")
            results.append(TestResult("img2img", False, error="Skipped: no source"))

        # ── Test 3: inpaint ──────────────────────────────────
        if result_b64:
            print("\n[TEST] 3/3 — inpaint generation (from txt2img result)...")
            mask_b64 = create_test_mask()
            t0 = time.perf_counter()
            await ws.send(json.dumps({
                "action": "generate",
                "prompt": "pixel art, PixArFK, game sprite, a golden crown, sharp pixels",
                "mode": "inpaint",
                "width": 512, "height": 512,
                "seed": 456, "steps": 8, "cfg_scale": 5.0,
                "denoise_strength": 0.7,
                "source_image": result_b64,
                "mask_image": mask_b64,
                "post_process": DEFAULT_POST_PROCESS,
            }))

            resp = await recv_generation_result(ws, label="inpaint")
            elapsed = time.perf_counter() - t0

            if resp:
                print(f"[OK]   inpaint done — {resp['time_ms']}ms, seed={resp['seed']}, "
                      f"{resp['width']}x{resp['height']}, wall={elapsed:.1f}s")
                out = OUTPUT_DIR / "test_inpaint_from_gen.png"
                out.write_bytes(b64decode(resp["image"]))
                print(f"       Saved: {out}")
                results.append(TestResult("inpaint", True, elapsed))
            else:
                results.append(TestResult("inpaint", False, elapsed, "No result"))
        else:
            print("\n[SKIP] 3/3 — inpaint (no source)")
            results.append(TestResult("inpaint", False, error="Skipped: no source"))

    write_report(results)
    return print_summary(results)


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
