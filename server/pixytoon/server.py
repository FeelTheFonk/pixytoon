"""FastAPI WebSocket server — entry point for the PixyToon server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .engine import DiffusionEngine, GenerationCancelled
from .protocol import (
    Action,
    ErrorResponse,
    ListResponse,
    PongResponse,
    ProgressResponse,
    Request,
)
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


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _generate_lock
    _generate_lock = asyncio.Lock()

    log.info("PixyToon server starting — loading diffusion engine...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engine.load)

    # Pre-trigger Numba JIT compilation for Floyd-Steinberg dithering
    log.info("Pre-compiling Numba JIT kernels...")
    await loop.run_in_executor(None, warmup_numba)
    log.info("Numba JIT warmup complete")

    log.info("Engine loaded. WebSocket ready on ws://%s:%d/ws", settings.host, settings.port)
    yield
    engine.unload()
    log.info("Engine unloaded.")


app = FastAPI(title="PixyToon Server", version="0.1.0", lifespan=_lifespan)


# ─────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
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
        engine.cancel()
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        engine.cancel()


async def _handle(websocket: WebSocket, req: Request) -> None:
    """Dispatch request by action type."""
    try:
        if req.action == Action.PING:
            await _send(websocket, PongResponse())

        elif req.action == Action.LIST_LORAS:
            items = lora_manager.list_loras()
            await _send(websocket, ListResponse(list_type="loras", items=items))

        elif req.action == Action.LIST_PALETTES:
            items = palette_manager.list_palettes()
            await _send(websocket, ListResponse(list_type="palettes", items=items))

        elif req.action == Action.LIST_CONTROLNETS:
            await _send(websocket, ListResponse(list_type="controlnets", items=[
                "controlnet_openpose",
                "controlnet_canny",
                "controlnet_scribble",
                "controlnet_lineart",
            ]))

        elif req.action == Action.LIST_EMBEDDINGS:
            items = ti_manager.list_embeddings()
            await _send(websocket, ListResponse(list_type="embeddings", items=items))

        elif req.action == Action.GENERATE:
            gen_req = req.to_generate_request()
            loop = asyncio.get_running_loop()

            def on_progress(p: ProgressResponse) -> None:
                try:
                    # Guard: skip if WebSocket is already closed
                    if websocket.application_state.name != "CONNECTED":
                        return
                    fut = asyncio.run_coroutine_threadsafe(
                        _send(websocket, p), loop
                    )
                    def _on_done(f):
                        exc = f.exception()
                        if exc:
                            log.warning("Progress send failed: %s", exc)
                    fut.add_done_callback(_on_done)
                except Exception:
                    pass

            # Serialize GPU access — pipeline is NOT thread-safe
            if _generate_lock is None:
                raise RuntimeError("Server not fully initialized")
            async with _generate_lock:
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
            await _send(websocket, result)

    except GenerationCancelled:
        log.info("Generation cancelled by client")
        try:
            await _send(websocket, ErrorResponse(code="CANCELLED", message="Generation cancelled"))
        except Exception:
            pass
    except Exception as e:
        log.exception("Handler error: %s", e)
        if "cancelled" in str(e).lower():
            code = "CANCELLED"
        elif "out of memory" in str(e).lower():
            code = "OOM"
        elif "timed out" in str(e).lower():
            code = "TIMEOUT"
        else:
            code = "ENGINE_ERROR"
        try:
            await _send(websocket, ErrorResponse(code=code, message=str(e)))
        except Exception:
            pass  # WebSocket already closed


async def _send(websocket: WebSocket, response) -> None:
    await websocket.send_text(response.model_dump_json())


# ─────────────────────────────────────────────────────────────
# HTTP HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> JSONResponse:
    from . import __version__
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
    )


if __name__ == "__main__":
    main()
