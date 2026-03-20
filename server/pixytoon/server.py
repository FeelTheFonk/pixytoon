"""FastAPI WebSocket server — entry point for the PixyToon server."""

from __future__ import annotations

import asyncio
import logging
import threading
import time as _time
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
    ErrorResponse,
    ListResponse,
    PongResponse,
    ProgressResponse,
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
_generating: dict[int, threading.Event] = {}  # websocket id -> event
_active_connections: set[WebSocket] = set()
_MAX_CONNECTIONS = 5


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _generate_lock
    _generate_lock = asyncio.Lock()

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
    engine.unload()
    log.info("Engine unloaded.")


app = FastAPI(title="PixyToon Server", version=__version__, lifespan=_lifespan)


# ─────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    if len(_active_connections) >= _MAX_CONNECTIONS:
        await websocket.send_text(ErrorResponse(
            code="MAX_CONNECTIONS", message="Too many connections"
        ).model_dump_json())
        await websocket.close()
        return
    _active_connections.add(websocket)

    ws_id = id(websocket)
    _generating[ws_id] = threading.Event()
    log.info("Client connected: %s", websocket.client)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = Request.model_validate_json(raw)
            except Exception as e:
                await _send(websocket, ErrorResponse(
                    code="INVALID_REQUEST",
                    message=f"Malformed request: {e}",
                ))
                continue

            await _handle(websocket, req)

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", websocket.client)
        if _generating.get(ws_id, threading.Event()).is_set():
            engine.cancel()
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        if _generating.get(ws_id, threading.Event()).is_set():
            engine.cancel()
    finally:
        _active_connections.discard(websocket)
        _generating.pop(ws_id, None)


async def _handle(websocket: WebSocket, req: Request) -> None:
    """Dispatch request by action type."""
    try:
        if req.action == Action.PING:
            await _send(websocket, PongResponse())

        elif req.action == Action.CANCEL:
            ws_id = id(websocket)
            if _generating.get(ws_id, threading.Event()).is_set():
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

        elif req.action == Action.GENERATE:
            gen_req = req.to_generate_request()
            loop = asyncio.get_running_loop()

            def on_progress(p: ProgressResponse) -> None:
                try:
                    # Guard: skip if WebSocket is already closed
                    try:
                        if websocket.application_state.name != "CONNECTED":
                            return
                    except AttributeError:
                        return  # Starlette internal changed — skip safely
                    fut = asyncio.run_coroutine_threadsafe(
                        _send(websocket, p), loop
                    )
                    # Backpressure: wait briefly for send to complete
                    # to avoid unbounded future accumulation
                    try:
                        fut.result(timeout=1.0)
                    except Exception:
                        pass  # Send failed or timed out — skip this progress
                except Exception:
                    pass

            # Serialize GPU access — pipeline is NOT thread-safe
            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            ws_id = id(websocket)
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
            anim_req = req.to_animation_request()

            # Server-side frame count validation (protocol allows 120, config may differ)
            if anim_req.frame_count > settings.max_animation_frames:
                await _send(websocket, ErrorResponse(
                    code="INVALID_REQUEST",
                    message=f"frame_count {anim_req.frame_count} exceeds max {settings.max_animation_frames}",
                ))
                return

            loop = asyncio.get_running_loop()

            def on_anim_progress(p: ProgressResponse) -> None:
                try:
                    try:
                        if websocket.application_state.name != "CONNECTED":
                            return
                    except AttributeError:
                        return
                    fut = asyncio.run_coroutine_threadsafe(
                        _send(websocket, p), loop
                    )
                    try:
                        fut.result(timeout=1.0)
                    except Exception:
                        pass
                except Exception:
                    pass

            def on_anim_frame(f: AnimationFrameResponse) -> None:
                try:
                    try:
                        if websocket.application_state.name != "CONNECTED":
                            return
                    except AttributeError:
                        return
                    fut = asyncio.run_coroutine_threadsafe(
                        _send(websocket, f), loop
                    )
                    try:
                        fut.result(timeout=2.0)
                    except Exception:
                        pass
                except Exception:
                    pass

            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            # Auto-scale timeout for animation (30s per frame minimum)
            anim_timeout = max(
                settings.generation_timeout,
                anim_req.frame_count * 30,
            )
            ws_id = id(websocket)
            async with _generate_lock:
                _generating[ws_id].set()
                try:
                    t0 = _time.perf_counter()
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
                    total_ms = int((_time.perf_counter() - t0) * 1000)
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
                message=f"Unknown action: {req.action}",
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


async def _send(websocket: WebSocket, response) -> None:
    try:
        await websocket.send_text(response.model_dump_json())
    except (WebSocketDisconnect, RuntimeError):
        pass  # Client already disconnected


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
        ws_max_size=16 * 1024 * 1024,  # 16MB max WebSocket message
    )


if __name__ == "__main__":
    main()
