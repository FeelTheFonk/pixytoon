"""FastAPI WebSocket server — entry point for the PixyToon server."""

from __future__ import annotations

import asyncio
import base64
import itertools
import json as json_stdlib
import logging
import struct
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .engine import DiffusionEngine, GenerationCancelled
from .protocol import (
    Action,
    AnimationCompleteResponse,
    AnimationFrameResponse,
    CleanupResponse,
    ErrorResponse,
    ListResponse,
    PongResponse,
    PresetDeletedResponse,
    PresetResponse,
    PresetSavedResponse,
    ProgressResponse,
    PromptResultResponse,
    RealtimeReadyResponse,
    RealtimeResultResponse,
    RealtimeStoppedResponse,
    Request,
)
from . import __version__
from . import lora_manager, palette_manager, ti_manager
from .postprocess import warmup_numba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pixytoon.server")

# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────

engine = DiffusionEngine()
_generate_lock: asyncio.Lock | None = None
_realtime_lock: asyncio.Lock | None = None  # protects _realtime_owner / _realtime_ws
_generating: dict[int, threading.Event] = {}  # connection id -> cancel event
_active_connections: set[WebSocket] = set()
_MAX_CONNECTIONS = 5
_ws_id_gen = itertools.count(1)  # thread-safe monotonic connection ID
_realtime_owner: int | None = None  # ws_id that owns realtime mode (None = free)
_realtime_ws: WebSocket | None = None  # WebSocket of the realtime owner (for auto-stop notify)
_realtime_timeout_task: asyncio.Task | None = None  # auto-stop timer


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _generate_lock, _realtime_lock
    _generate_lock = asyncio.Lock()
    _realtime_lock = asyncio.Lock()

    log.info("PixyToon server starting — loading diffusion engine...")
    loop = asyncio.get_running_loop()

    # Run engine load and Numba JIT warmup in parallel (independent tasks)
    async def _load_engine():
        await loop.run_in_executor(None, engine.load)

    async def _warmup_numba():
        log.info("Pre-compiling Numba JIT kernels...")
        await loop.run_in_executor(None, warmup_numba)
        log.info("Numba JIT warmup complete")

    await asyncio.gather(_load_engine(), _warmup_numba())

    log.info("Engine loaded. WebSocket ready on ws://%s:%d/ws", settings.host, settings.port)
    yield

    # Graceful shutdown: close active WebSocket connections
    for ws in list(_active_connections):
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except Exception:
            pass
    _active_connections.clear()

    engine.unload()
    log.info("Engine unloaded.")


app = FastAPI(title="PixyToon Server", version=__version__, lifespan=_lifespan)


# ─────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    # Accept first (WebSocket protocol requires accept before sending), then
    # enforce the connection limit and close with a proper error if exceeded.
    await websocket.accept()

    if len(_active_connections) >= _MAX_CONNECTIONS:
        await _send(websocket, ErrorResponse(
            code="MAX_CONNECTIONS", message="Too many connections"
        ))
        await websocket.close(code=1008, reason="Server at capacity")
        return
    _active_connections.add(websocket)

    ws_id = next(_ws_id_gen)
    _generating[ws_id] = threading.Event()
    log.info("Client connected: %s", websocket.client)

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg:
                raw = msg["text"]
                try:
                    req = Request.model_validate_json(raw)
                except Exception as e:
                    await _send(websocket, ErrorResponse(
                        code="INVALID_REQUEST",
                        message=f"Malformed request: {e}",
                    ))
                    continue
                try:
                    await _handle(websocket, req, ws_id)
                except Exception as e:
                    log.exception("Handler error for action '%s': %s", req.action, e)
                    await _send(websocket, ErrorResponse(
                        code="ENGINE_ERROR",
                        message=f"Handler error: {e}",
                    ))

            elif "bytes" in msg:
                # P0-L4: Binary frame from live painting client
                try:
                    await _handle_binary_frame(websocket, msg["bytes"], ws_id)
                except Exception as e:
                    log.warning("Binary frame error: %s", e)
                    await _send(websocket, ErrorResponse(
                        code="INVALID_REQUEST",
                        message=f"Binary frame error: {e}",
                    ))

            elif msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", websocket.client)
        if ws_id in _generating and _generating[ws_id].is_set():
            engine.cancel()
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        if ws_id in _generating and _generating[ws_id].is_set():
            engine.cancel()
    finally:
        _active_connections.discard(websocket)
        _generating.pop(ws_id, None)
        # Clean up realtime session if this client owned it
        await _cleanup_realtime(ws_id)


def _make_thread_callback(websocket: WebSocket, loop: asyncio.AbstractEventLoop, timeout: float = 1.0):
    """Create a thread-safe callback that sends responses via the event loop.

    Used by both generate and animate handlers to send progress/frame updates
    from the engine thread back through the WebSocket.
    """
    def callback(response) -> None:
        try:
            try:
                if websocket.application_state.name != "CONNECTED":
                    return
            except AttributeError:
                log.debug("WebSocket state check failed (Starlette internal changed) — skipping send")
                return
            fut = asyncio.run_coroutine_threadsafe(
                _send(websocket, response), loop
            )
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass  # Send failed or timed out — skip
        except Exception:
            pass
    return callback


def _parse_binary_frame(data: bytes) -> tuple[dict, bytes]:
    """Parse a binary live frame: 4-byte LE header length + JSON header + PNG data.

    Returns (header_dict, png_bytes).
    """
    if len(data) < 4:
        raise ValueError("Binary frame too short (< 4 bytes)")
    header_len = struct.unpack("<I", data[:4])[0]
    if len(data) < 4 + header_len:
        raise ValueError(f"Binary frame truncated: need {4 + header_len}, got {len(data)}")
    header_json = data[4:4 + header_len]
    png_data = data[4 + header_len:]
    header = json_stdlib.loads(header_json)
    return header, png_data


async def _handle_binary_frame(websocket: WebSocket, data: bytes, ws_id: int) -> None:
    """Handle a binary WebSocket frame (live painting fast path).

    The client sends PNG data as raw bytes instead of base64-in-JSON to eliminate
    the expensive pure-Lua base64 encoding step (~200ms on 512x512).
    Server-side base64 encoding of the PNG is done in C (Python stdlib) — negligible.
    """
    header, png_data = _parse_binary_frame(data)

    action = header.get("action")
    if action != "realtime_frame":
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST",
            message=f"Binary frames only support realtime_frame, got: {action!r}",
        ))
        return

    # Convert PNG bytes to base64 for the existing engine API
    image_b64 = base64.b64encode(png_data).decode("ascii")

    # Build a Request-compatible dict and dispatch to the existing handler
    req_data = {
        "action": "realtime_frame",
        "image": image_b64,
        "frame_id": header.get("frame_id", 0),
    }
    # Forward ROI fields if present
    for key in ("roi_x", "roi_y", "roi_w", "roi_h", "mask", "prompt"):
        if key in header:
            req_data[key] = header[key]

    try:
        req = Request.model_validate(req_data)
    except Exception as e:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST",
            message=f"Binary frame header invalid: {e}",
        ))
        return

    await _handle(websocket, req, ws_id)


async def _handle(websocket: WebSocket, req: Request, ws_id: int) -> None:
    """Dispatch request by action type."""
    try:
        if req.action == Action.PING:
            await _send(websocket, PongResponse())

        elif req.action == Action.CANCEL:
            if ws_id in _generating and _generating[ws_id].is_set():
                engine.cancel()
                # Don't send response here — the GenerationCancelled exception
                # handler will send CANCELLED when the generation actually stops.
            else:
                await _send(websocket, PongResponse())  # No-op — nothing to cancel

        elif req.action == Action.LIST_LORAS:
            items = lora_manager.list_loras()
            await _send(websocket, ListResponse(list_type="loras", items=items))

        elif req.action == Action.LIST_PALETTES:
            items = palette_manager.list_palettes()
            await _send(websocket, ListResponse(list_type="palettes", items=items))

        elif req.action == Action.LIST_CONTROLNETS:
            from . import pipeline_factory
            await _send(websocket, ListResponse(
                list_type="controlnets",
                items=[m.value for m in pipeline_factory.CONTROLNET_IDS],
            ))

        elif req.action == Action.LIST_EMBEDDINGS:
            items = ti_manager.list_embeddings()
            await _send(websocket, ListResponse(list_type="embeddings", items=items))

        elif req.action == Action.GENERATE_PROMPT:
            await _handle_generate_prompt(websocket, req)

        elif req.action == Action.LIST_PRESETS:
            from .presets_manager import presets_manager
            items = presets_manager.list_presets()
            await _send(websocket, ListResponse(list_type="presets", items=items))

        elif req.action == Action.GET_PRESET:
            await _handle_get_preset(websocket, req)

        elif req.action == Action.SAVE_PRESET:
            await _handle_save_preset(websocket, req)

        elif req.action == Action.DELETE_PRESET:
            await _handle_delete_preset(websocket, req)

        elif req.action == Action.CLEANUP:
            await _handle_cleanup(websocket)

        elif req.action == Action.REALTIME_START:
            await _handle_realtime_start(websocket, req, ws_id)

        elif req.action == Action.REALTIME_FRAME:
            await _handle_realtime_frame(websocket, req, ws_id)

        elif req.action == Action.REALTIME_UPDATE:
            await _handle_realtime_update(websocket, req, ws_id)

        elif req.action == Action.REALTIME_STOP:
            await _handle_realtime_stop(websocket, ws_id)

        elif req.action == Action.GENERATE:
            # Block if realtime mode is active
            async with _realtime_lock:
                rt_busy = _realtime_owner is not None
            if rt_busy:
                await _send(websocket, ErrorResponse(
                    code="REALTIME_BUSY",
                    message="Cannot generate while real-time mode is active",
                ))
                return
            gen_req = req.to_generate_request()
            loop = asyncio.get_running_loop()

            on_progress = _make_thread_callback(websocket, loop, timeout=1.0)

            # Serialize GPU access — pipeline is NOT thread-safe
            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            async with _generate_lock:
                _generating[ws_id].set()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: engine.generate(gen_req, on_progress=on_progress),
                        ),
                        timeout=settings.generation_timeout,
                    )
                except asyncio.TimeoutError:
                    engine.cancel()
                    raise RuntimeError(
                        f"Generation timed out after {settings.generation_timeout:.0f}s"
                    )
                finally:
                    _generating[ws_id].clear()
            await _send(websocket, result)

        elif req.action == Action.GENERATE_ANIMATION:
            # Block if realtime mode is active
            async with _realtime_lock:
                rt_busy = _realtime_owner is not None
            if rt_busy:
                await _send(websocket, ErrorResponse(
                    code="REALTIME_BUSY",
                    message="Cannot animate while real-time mode is active",
                ))
                return
            anim_req = req.to_animation_request()

            # Server-side frame count validation (protocol allows 120, config may differ)
            if anim_req.frame_count > settings.max_animation_frames:
                await _send(websocket, ErrorResponse(
                    code="INVALID_REQUEST",
                    message=f"frame_count {anim_req.frame_count} exceeds max {settings.max_animation_frames}",
                ))
                return

            loop = asyncio.get_running_loop()

            on_anim_progress = _make_thread_callback(websocket, loop, timeout=1.0)
            on_anim_frame = _make_thread_callback(websocket, loop, timeout=2.0)

            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            # Auto-scale timeout for animation (30s per frame minimum)
            anim_timeout = max(
                settings.generation_timeout,
                anim_req.frame_count * 30,
            )
            async with _generate_lock:
                _generating[ws_id].set()
                try:
                    t0 = time.perf_counter()
                    frames = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: engine.generate_animation(
                                anim_req,
                                on_frame=on_anim_frame,
                                on_progress=on_anim_progress,
                            ),
                        ),
                        timeout=anim_timeout,
                    )
                    total_ms = int((time.perf_counter() - t0) * 1000)
                except asyncio.TimeoutError:
                    engine.cancel()
                    raise RuntimeError(
                        f"Animation timed out after {anim_timeout:.0f}s"
                    )
                finally:
                    _generating[ws_id].clear()

            await _send(websocket, AnimationCompleteResponse(
                total_frames=len(frames),
                total_time_ms=total_ms,
                tag_name=anim_req.tag_name,
            ))

        else:
            await _send(websocket, ErrorResponse(
                code="UNKNOWN_ACTION",
                message=f"Unknown action: {req.action!r}",
            ))

    except GenerationCancelled:
        log.info("Generation cancelled by client")
        try:
            await _send(websocket, ErrorResponse(code="CANCELLED", message="Generation cancelled"))
        except Exception:
            pass
    except torch.cuda.OutOfMemoryError as e:
        log.exception("CUDA OOM: %s", e)
        try:
            await _send(websocket, ErrorResponse(code="OOM", message=str(e)))
        except Exception:
            pass
    except Exception as e:
        log.exception("Handler error: %s", e)
        if "timed out" in str(e).lower():
            code = "TIMEOUT"
        else:
            code = "ENGINE_ERROR"
        try:
            await _send(websocket, ErrorResponse(code=code, message=str(e)))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# AUTO-PROMPT, PRESETS & CLEANUP HANDLERS
# ─────────────────────────────────────────────────────────────

async def _handle_generate_prompt(websocket: WebSocket, req: Request) -> None:
    from .prompt_generator import prompt_generator
    locked = req.locked_fields or {}
    template = req.prompt_template
    prompt, negative, components = prompt_generator.generate(locked, template)
    await _send(websocket, PromptResultResponse(
        prompt=prompt, negative_prompt=negative, components=components,
    ))


async def _handle_get_preset(websocket: WebSocket, req: Request) -> None:
    from .presets_manager import presets_manager
    name = req.preset_name
    if not name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="preset_name required"))
        return
    try:
        data = presets_manager.get_preset(name)
        await _send(websocket, PresetResponse(name=name, data=data))
    except FileNotFoundError as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))
    except ValueError as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))


async def _handle_save_preset(websocket: WebSocket, req: Request) -> None:
    from .presets_manager import presets_manager
    if not req.preset_name or not req.preset_data:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="preset_name and preset_data required"))
        return
    try:
        presets_manager.save_preset(req.preset_name, req.preset_data)
        await _send(websocket, PresetSavedResponse(name=req.preset_name))
    except ValueError as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))


async def _handle_delete_preset(websocket: WebSocket, req: Request) -> None:
    from .presets_manager import presets_manager
    if not req.preset_name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="preset_name required"))
        return
    try:
        presets_manager.delete_preset(req.preset_name)
        await _send(websocket, PresetDeletedResponse(name=req.preset_name))
    except (FileNotFoundError, ValueError) as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))


async def _handle_cleanup(websocket: WebSocket) -> None:
    async with _realtime_lock:
        rt_busy = _realtime_owner is not None
    if rt_busy:
        await _send(websocket, ErrorResponse(
            code="REALTIME_BUSY",
            message="Cannot cleanup while real-time mode is active"))
        return
    # Best-effort check: locked() is racy but acceptable here — we only use it
    # as a courtesy guard and never enter the lock afterward.
    if _generate_lock is not None and _generate_lock.locked():
        await _send(websocket, ErrorResponse(
            code="GPU_BUSY",
            message="Cannot cleanup while a generation is in progress"))
        return
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, engine.cleanup_resources)
    await _send(websocket, CleanupResponse(
        message=result["message"],
        freed_mb=result["freed_mb"],
    ))


# ─────────────────────────────────────────────────────────────
# REAL-TIME PAINT HANDLERS
# ─────────────────────────────────────────────────────────────

async def _handle_realtime_start(websocket: WebSocket, req: Request, ws_id: int) -> None:
    global _realtime_owner, _realtime_ws, _realtime_timeout_task

    async with _realtime_lock:
        if _realtime_owner is not None and _realtime_owner != ws_id:
            await _send(websocket, ErrorResponse(
                code="REALTIME_BUSY",
                message="Another client is using real-time mode",
            ))
            return

        if _generate_lock is None:
            raise RuntimeError("Server not fully initialized")

        # Best-effort check: locked() is racy but acceptable here — we only use it
        # as a courtesy guard and never enter the lock afterward.
        if _generate_lock.locked():
            await _send(websocket, ErrorResponse(
                code="GPU_BUSY",
                message="Cannot start real-time mode while a generation is in progress",
            ))
            return

        # Claim ownership inside the lock to prevent race condition
        _realtime_owner = ws_id
        _realtime_ws = websocket

    rt_req = req.to_realtime_start()
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None, lambda: engine.start_realtime(rt_req),
        )
        _reset_realtime_timeout()
        await _send(websocket, result)
    except Exception as e:
        # Release ownership on failure
        async with _realtime_lock:
            _realtime_owner = None
            _realtime_ws = None
        log.exception("Failed to start realtime: %s", e)
        await _send(websocket, ErrorResponse(code="ENGINE_ERROR", message=str(e)))


async def _handle_realtime_frame(websocket: WebSocket, req: Request, ws_id: int) -> None:
    async with _realtime_lock:
        is_owner = _realtime_owner == ws_id
    if not is_owner:
        await _send(websocket, ErrorResponse(
            code="REALTIME_NOT_ACTIVE",
            message="Real-time mode not active for this connection",
        ))
        return

    frame_req = req.to_realtime_frame()
    if not frame_req.image:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST",
            message="realtime_frame requires image",
        ))
        return

    # P2-O3: Acquire GPU lock with a short timeout instead of racy locked() check.
    # This gives the frame a brief window to wait for GPU availability.
    gpu_acquired = False
    if _generate_lock is not None:
        try:
            await asyncio.wait_for(_generate_lock.acquire(), timeout=0.5)
            gpu_acquired = True
        except asyncio.TimeoutError:
            await _send(websocket, ErrorResponse(
                code="GPU_BUSY",
                message="Frame skipped — GPU busy",
            ))
            return

    _reset_realtime_timeout()

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: engine.process_realtime_frame(
                frame_req.image,
                frame_req.frame_id,
                prompt_override=frame_req.prompt,
                mask_b64=frame_req.mask,
                roi_x=frame_req.roi_x,
                roi_y=frame_req.roi_y,
                roi_w=frame_req.roi_w,
                roi_h=frame_req.roi_h,
            ),
        )
        await _send(websocket, result)
    except torch.cuda.OutOfMemoryError as e:
        log.error("CUDA OOM during realtime frame: %s", e)
        await _send(websocket, ErrorResponse(code="OOM", message=str(e)))
    except Exception as e:
        log.warning("Realtime frame error: %s", e)
        await _send(websocket, ErrorResponse(code="ENGINE_ERROR", message=str(e)))
    finally:
        # P2-O3: Release GPU lock if we acquired it
        if gpu_acquired and _generate_lock is not None:
            _generate_lock.release()


async def _handle_realtime_update(websocket: WebSocket, req: Request, ws_id: int) -> None:
    async with _realtime_lock:
        is_owner = _realtime_owner == ws_id
    if not is_owner:
        return  # Silently ignore updates from non-owner

    update_req = req.to_realtime_update()
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, lambda: engine.update_realtime_params(update_req),
        )
    except Exception as e:
        log.exception("Realtime update error: %s", e)


async def _handle_realtime_stop(websocket: WebSocket, ws_id: int) -> None:
    async with _realtime_lock:
        is_owner = _realtime_owner == ws_id
    if not is_owner:
        await _send(websocket, ErrorResponse(
            code="REALTIME_NOT_ACTIVE",
            message="Real-time mode not active for this connection",
        ))
        return

    await _cleanup_realtime(ws_id)
    await _send(websocket, RealtimeStoppedResponse())


async def _cleanup_realtime(ws_id: int) -> None:
    """Stop realtime mode if owned by ws_id. Safe to call multiple times."""
    global _realtime_owner, _realtime_ws, _realtime_timeout_task

    async with _realtime_lock:
        if _realtime_owner != ws_id:
            return

        if _realtime_timeout_task is not None:
            _realtime_timeout_task.cancel()
            _realtime_timeout_task = None

        _realtime_owner = None
        _realtime_ws = None

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, engine.stop_realtime)
    except Exception as e:
        log.warning("Realtime cleanup error: %s", e)

    log.info("Realtime session cleaned up for ws_id=%d", ws_id)


def _reset_realtime_timeout() -> None:
    """Reset the auto-stop timer for realtime mode."""
    global _realtime_timeout_task

    if _realtime_timeout_task is not None:
        _realtime_timeout_task.cancel()

    async def _auto_stop():
        global _realtime_timeout_task
        await asyncio.sleep(settings.realtime_timeout)
        _realtime_timeout_task = None
        global _realtime_owner, _realtime_ws
        async with _realtime_lock:
            if _realtime_owner is None:
                return
            log.info("Realtime auto-stop: no frame for %.0fs", settings.realtime_timeout)
            ws = _realtime_ws
            _realtime_owner = None
            _realtime_ws = None
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, engine.stop_realtime)
        except Exception:
            pass
        # Notify client that realtime was auto-stopped
        if ws is not None:
            try:
                await _send(ws, RealtimeStoppedResponse(
                    message="Real-time mode auto-stopped (timeout)",
                ))
            except Exception:
                pass

    try:
        loop = asyncio.get_running_loop()
        _realtime_timeout_task = loop.create_task(_auto_stop())
    except RuntimeError:
        pass  # No running loop (shouldn't happen in normal operation)


async def _send(websocket: WebSocket, response) -> None:
    try:
        text = response.model_dump_json()
        await websocket.send_text(text)
    except (WebSocketDisconnect, RuntimeError):
        pass  # Client already disconnected or connection closing


# ─────────────────────────────────────────────────────────────
# HTTP HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "loaded": engine.is_loaded,
    })


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "pixytoon.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_max_size=50 * 1024 * 1024,  # 50MB max WebSocket message
    )


if __name__ == "__main__":
    main()
