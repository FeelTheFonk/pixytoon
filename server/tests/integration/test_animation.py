"""Standalone test: animation (chain + animatediff).

Usage:
    cd C:\\Users\\CleS\\Desktop\\SDDj\\server
    uv run python ..\\scripts\\test_animation.py

Requires the SDDj server to be running on ws://127.0.0.1:9876/ws.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from _test_lib import (
    DEFAULT_POST_PROCESS,
    OUTPUT_DIR,
    TestResult,
    create_test_image,
    print_summary,
    recv_animation_result,
    write_report,
    ws_connection,
)


FRAME_COUNT = 4


async def _send_animation(ws, *, method: str, label: str, extra: dict | None = None) -> dict:
    """Build and send an animation request, return the request dict."""
    req = {
        "action": "generate_animation",
        "method": method,
        "prompt": "pixel art, PixArFK, game sprite, walking cycle, sharp pixels",
        "mode": "txt2img",
        "width": 512, "height": 512,
        "seed": 42, "steps": 8, "cfg_scale": 5.0,
        "denoise_strength": 0.30,
        "frame_count": FRAME_COUNT,
        "frame_duration_ms": 100,
        "seed_strategy": "increment",
        "tag_name": f"test_{label}",
        "post_process": DEFAULT_POST_PROCESS,
    }
    if extra:
        req.update(extra)
    await ws.send(json.dumps(req))
    return req


async def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results: list[TestResult] = []

    async with ws_connection() as ws:

        # ── Test 1: Chain txt2img ────────────────────────────
        print(f"\n[TEST] 1/4 — chain animation ({FRAME_COUNT} frames)...")
        t0 = time.perf_counter()
        await _send_animation(ws, method="chain", label="chain")
        resp = await recv_animation_result(
            ws, expected_frames=FRAME_COUNT, label="chain",
            output_prefix="test_chain_frame",
        )
        elapsed = time.perf_counter() - t0
        results.append(TestResult("chain_txt2img", resp is not None, elapsed,
                                  "" if resp else "No completion"))

        # ── Test 2: AnimateDiff ──────────────────────────────
        print(f"\n[TEST] 2/4 — animatediff animation ({FRAME_COUNT} frames)...")
        t0 = time.perf_counter()
        await _send_animation(ws, method="animatediff", label="animatediff")
        resp = await recv_animation_result(
            ws, expected_frames=FRAME_COUNT, label="animatediff",
            output_prefix="test_animatediff_frame",
        )
        elapsed = time.perf_counter() - t0
        results.append(TestResult("animatediff", resp is not None, elapsed,
                                  "" if resp else "No completion"))

        # ── Test 3: Chain img2img ────────────────────────────
        print(f"\n[TEST] 3/4 — chain img2img animation ({FRAME_COUNT} frames)...")
        source_b64 = create_test_image(color=(50, 100, 200))
        t0 = time.perf_counter()
        await _send_animation(ws, method="chain", label="chain_img2img",
                              extra={
                                  "mode": "img2img",
                                  "seed": 100,
                                  "source_image": source_b64,
                              })
        resp = await recv_animation_result(
            ws, expected_frames=FRAME_COUNT, label="chain_img2img",
            output_prefix="test_chain_img2img_frame",
        )
        elapsed = time.perf_counter() - t0
        results.append(TestResult("chain_img2img", resp is not None, elapsed,
                                  "" if resp else "No completion"))

        # ── Test 4: AnimateDiff Lightning ────────────────────
        print(f"\n[TEST] 4/4 — animatediff-lightning ({FRAME_COUNT} frames)...")
        t0 = time.perf_counter()
        await _send_animation(ws, method="animatediff", label="animatediff_lightning",
                              extra={
                                  "steps": 4,
                                  "cfg_scale": 2.0,
                                  "seed_strategy": "fixed",
                              })
        resp = await recv_animation_result(
            ws, expected_frames=FRAME_COUNT, label="animatediff-lightning",
            output_prefix="test_lightning_frame",
        )
        elapsed = time.perf_counter() - t0
        results.append(TestResult("animatediff_lightning", resp is not None, elapsed,
                                  "" if resp else "No completion"))

    write_report(results)
    return print_summary(results)


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
