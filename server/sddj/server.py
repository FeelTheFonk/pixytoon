"""FastAPI WebSocket server — entry point for the SDDj server."""

from __future__ import annotations

import asyncio
import atexit
import itertools
import logging
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .engine import DiffusionEngine, GenerationCancelled
from .protocol import (
    Action,
    AnimationCompleteResponse,
    AudioAnalysisResponse,
    AudioReactiveCompleteResponse,
    ChoreographyPresetDetailResponse,
    ChoreographyPresetsListResponse,
    CleanupResponse,
    ErrorResponse,
    ExpressionPresetDetailResponse,
    ExpressionPresetsListResponse,
    ListResponse,
    ModulationPresetsResponse,
    ModulationPresetDetailResponse,
    PaletteDeletedResponse,
    PaletteSavedResponse,
    PongResponse,
    PresetDeletedResponse,
    PresetResponse,
    PresetSavedResponse,
    ProgressResponse,
    PromptResultResponse,
    Request,
    ShutdownResponse,
    StemsAvailableResponse,
    ExportMp4Response,
    ExportMp4ErrorResponse,
)
from . import __version__
from . import lora_manager, palette_manager, ti_manager
from .postprocess import warmup_numba
from .auto_calibrate import recommend_preset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sddj.server")

_SUPPORTED_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})


async def _validate_audio_path(websocket: WebSocket, audio_path: str) -> str | None:
    """Validate audio path: realpath, existence, extension, size.

    Returns validated real path on success, or None after sending error.
    """
    real_path = os.path.realpath(audio_path)
    if not os.path.isfile(real_path):
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=f"Audio file not found: {audio_path}"))
        return None
    ext = os.path.splitext(real_path)[1].lower()
    if ext not in _SUPPORTED_AUDIO_EXTS:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=f"Unsupported audio format: {ext}"))
        return None
    size_mb = os.path.getsize(real_path) / (1024 * 1024)
    if size_mb > settings.audio_max_file_size_mb:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST",
            message=f"Audio file too large: {size_mb:.0f}MB (max {settings.audio_max_file_size_mb}MB)"))
        return None
    return real_path

# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────

engine = DiffusionEngine()
_generate_lock: asyncio.Lock | None = None
_generating: dict[int, threading.Event] = {}  # connection id -> cancel event
_active_connections: set[WebSocket] = set()
_MAX_CONNECTIONS = 5
_ws_id_gen = itertools.count(1)  # thread-safe monotonic connection ID

# Actions that run long enough to block the receive loop.
# During these, we keep receiving cancel/ping messages concurrently.
_LONG_RUNNING_ACTIONS: frozenset[str] = frozenset({
    Action.GENERATE,
    Action.GENERATE_ANIMATION,
    Action.GENERATE_AUDIO_REACTIVE,
    Action.ANALYZE_AUDIO,
})


_PID_FILE = Path(__file__).resolve().parent.parent / "sddj.pid"


def _write_pid() -> None:
    """Write current PID to file for orphan detection."""
    try:
        _PID_FILE.write_text(str(os.getpid()))
    except Exception as e:
        log.warning("Failed to write PID file: %s", e)


def _remove_pid() -> None:
    """Remove PID file on clean shutdown."""
    try:
        _PID_FILE.unlink(missing_ok=True)
    except Exception as e:
        log.debug("PID cleanup failed: %s", e)


# Register atexit as a safety net (covers non-fatal exits)
atexit.register(_remove_pid)
atexit.register(lambda: __import__("sddj.vram_utils", fromlist=["vram_cleanup"]).vram_cleanup())


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _generate_lock
    _generate_lock = asyncio.Lock()

    _write_pid()
    log.info("SDDj server starting — loading diffusion engine...")
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
    _remove_pid()
    log.info("Engine unloaded.")


app = FastAPI(title="SDDj Server", version=__version__, lifespan=_lifespan)


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

    gen_task: asyncio.Task | None = None  # tracks the in-flight long-running handler

    try:
        while True:
            # ── Concurrent receive: if a long-running task is in progress,
            # we race between receiving new messages and the task completing.
            # This allows cancel/ping messages to be processed during generation.
            if gen_task is not None:
                recv_future = asyncio.ensure_future(websocket.receive())
                done, pending = await asyncio.wait(
                    {gen_task, recv_future},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Handle received message (cancel/ping) during generation
                if recv_future in done:
                    msg = recv_future.result()
                    await _handle_msg_during_gen(websocket, msg, ws_id)

                # Check if the long-running task completed
                if gen_task in done:
                    # Propagate any exception from the handler task
                    exc = gen_task.exception()
                    gen_task = None
                    if exc is not None:
                        raise exc
                    # Cancel the pending receive if the task finished first
                    if recv_future in pending:
                        recv_future.cancel()
                        try:
                            await recv_future
                        except (asyncio.CancelledError, Exception):
                            pass
                else:
                    # Task still running, loop back to keep receiving
                    continue

            # ── Normal receive: no long-running task in progress
            else:
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

                    # Long-running actions: start as a background task so we
                    # can keep receiving cancel/ping messages.
                    if req.action in _LONG_RUNNING_ACTIONS:
                        gen_task = asyncio.create_task(
                            _handle(websocket, req, ws_id)
                        )
                    else:
                        try:
                            await _handle(websocket, req, ws_id)
                        except Exception as e:
                            log.exception("Handler error for action '%s': %s", req.action, e)
                            await _send(websocket, ErrorResponse(
                                code="ENGINE_ERROR",
                                message=f"Handler error: {e}",
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
        # Cancel any in-flight generation task on disconnect
        if gen_task is not None and not gen_task.done():
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, Exception):
                pass
        _active_connections.discard(websocket)
        _generating.pop(ws_id, None)


async def _handle_msg_during_gen(
    websocket: WebSocket, msg: dict, ws_id: int,
) -> None:
    """Handle messages received while a long-running generation task is active.

    Only cancel, ping, and shutdown are processed — everything else is ignored
    to prevent concurrent handler conflicts.
    """
    if msg.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect()

    if "text" not in msg:
        return  # Ignore binary frames during generation

    try:
        req = Request.model_validate_json(msg["text"])
    except Exception:
        return  # Ignore malformed messages during generation

    if req.action == Action.CANCEL:
        if ws_id in _generating and _generating[ws_id].is_set():
            engine.cancel()
            log.info("Cancel received during generation (ws_id=%d)", ws_id)
            await _send(websocket, ProgressResponse(step=0, total=0))
        else:
            await _send(websocket, PongResponse())
    elif req.action == Action.PING:
        await _send(websocket, PongResponse())
    elif req.action == Action.SHUTDOWN:
        # Cancel current generation then shut down
        if ws_id in _generating and _generating[ws_id].is_set():
            engine.cancel()
        await _handle_shutdown(websocket, ws_id)


def _make_thread_callback(websocket: WebSocket, loop: asyncio.AbstractEventLoop, timeout: float = 1.0):
    """Create a thread-safe callback that sends responses via the event loop.

    Fire-and-forget: schedules the WS send on the async loop but does NOT block
    the engine thread waiting for completion. This eliminates the GPU idle time
    that occurred when .result(timeout) would stall the diffusion thread for
    every frame callback.
    """
    def callback(response) -> None:
        try:
            try:
                if websocket.application_state.name != "CONNECTED":
                    return
            except AttributeError:
                return
            asyncio.run_coroutine_threadsafe(_send(websocket, response), loop)
        except Exception as e:
            log.debug("Frame callback send failed: %s", e)
    return callback


async def _handle(websocket: WebSocket, req: Request, ws_id: int) -> None:
    """Dispatch request by action type."""
    try:
        if req.action == Action.PING:
            await _send(websocket, PongResponse())

        elif req.action == Action.CANCEL:
            if ws_id in _generating and _generating[ws_id].is_set():
                engine.cancel()
                # ACK immediately so client knows cancel was received.
                # The GenerationCancelled exception handler will still send
                # the final CANCELLED error when the generation actually stops.
                await _send(websocket, ProgressResponse(step=0, total=0))
            else:
                await _send(websocket, PongResponse())  # No-op — nothing to cancel

        elif req.action == Action.LIST_LORAS:
            items = lora_manager.list_loras()
            await _send(websocket, ListResponse(list_type="loras", items=items))

        elif req.action == Action.LIST_PALETTES:
            items = palette_manager.list_palettes()
            await _send(websocket, ListResponse(list_type="palettes", items=items))

        elif req.action == Action.SAVE_PALETTE:
            await _handle_save_palette(websocket, req)

        elif req.action == Action.DELETE_PALETTE:
            await _handle_delete_palette(websocket, req)

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

        elif req.action == Action.ANALYZE_AUDIO:
            await _handle_analyze_audio(websocket, req)

        elif req.action == Action.CHECK_STEMS:
            await _handle_check_stems(websocket)

        elif req.action == Action.LIST_MODULATION_PRESETS:
            await _handle_list_modulation_presets(websocket)

        elif req.action == Action.GET_MODULATION_PRESET:
            await _handle_get_modulation_preset(websocket, req)

        elif req.action == Action.LIST_EXPRESSION_PRESETS:
            await _handle_list_expression_presets(websocket)

        elif req.action == Action.GET_EXPRESSION_PRESET:
            await _handle_get_expression_preset(websocket, req)

        elif req.action == Action.LIST_CHOREOGRAPHY_PRESETS:
            await _handle_list_choreography_presets(websocket)

        elif req.action == Action.GET_CHOREOGRAPHY_PRESET:
            await _handle_get_choreography_preset(websocket, req)

        elif req.action == Action.LIST_PROMPT_SCHEDULES:
            await _handle_list_prompt_schedules(websocket)

        elif req.action == Action.GET_PROMPT_SCHEDULE:
            await _handle_get_prompt_schedule(websocket, req)

        elif req.action == Action.SAVE_PROMPT_SCHEDULE:
            await _handle_save_prompt_schedule(websocket, req)

        elif req.action == Action.DELETE_PROMPT_SCHEDULE:
            await _handle_delete_prompt_schedule(websocket, req)

        elif req.action == Action.GENERATE_AUDIO_REACTIVE:
            await _handle_generate_audio_reactive(websocket, req, ws_id)

        elif req.action == Action.EXPORT_MP4:
            await _handle_export_mp4(websocket, req)

        elif req.action == Action.SHUTDOWN:
            await _handle_shutdown(websocket, ws_id)
            return  # Connection will close after shutdown

        elif req.action == Action.GENERATE:
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
                    frame_count = await asyncio.wait_for(
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
                total_frames=frame_count,
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
    randomness = getattr(req, 'randomness', 0) or 0
    prompt, negative, components = prompt_generator.generate(
        locked, template, randomness=randomness,
        subject_type=req.subject_type,
        mode=req.prompt_mode,
        exclude=req.exclude_terms,
    )
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


async def _handle_save_palette(websocket: WebSocket, req: Request) -> None:
    if not req.palette_save_name or not req.palette_save_colors:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="palette_save_name and palette_save_colors required"))
        return
    try:
        palette_manager.save_palette(req.palette_save_name, req.palette_save_colors)
        await _send(websocket, PaletteSavedResponse(name=req.palette_save_name))
    except ValueError as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))


async def _handle_delete_palette(websocket: WebSocket, req: Request) -> None:
    if not req.palette_save_name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="palette_save_name required"))
        return
    try:
        palette_manager.delete_palette(req.palette_save_name)
        await _send(websocket, PaletteDeletedResponse(name=req.palette_save_name))
    except (FileNotFoundError, ValueError) as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))


async def _handle_cleanup(websocket: WebSocket) -> None:
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
# SERVER LIFECYCLE HANDLER
# ─────────────────────────────────────────────────────────────

async def _handle_shutdown(websocket: WebSocket, ws_id: int) -> None:
    """Graceful server shutdown triggered by client."""
    # Refuse if a generation is in progress
    if _generate_lock is not None and _generate_lock.locked():
        await _send(websocket, ErrorResponse(
            code="GPU_BUSY",
            message="Cannot shutdown while a generation is in progress",
        ))
        return

    log.info("Shutdown requested by client ws_id=%d", ws_id)
    await _send(websocket, ShutdownResponse())

    # Close all active connections gracefully
    for ws in list(_active_connections):
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except Exception:
            pass

    # Trigger uvicorn graceful shutdown — platform-safe
    _request_shutdown()


# ─────────────────────────────────────────────────────────────
# AUDIO REACTIVITY HANDLERS
# ─────────────────────────────────────────────────────────────

async def _handle_analyze_audio(websocket: WebSocket, req: Request) -> None:
    audio_req = req.to_analyze_audio_request()
    if not audio_req.audio_path:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="audio_path required"))
        return
    real_path = await _validate_audio_path(websocket, audio_req.audio_path)
    if real_path is None:
        return

    loop = asyncio.get_running_loop()
    try:
        analysis = await loop.run_in_executor(
            None,
            lambda: engine.analyze_audio(
                real_path, audio_req.fps, audio_req.enable_stems,
            ),
        )
        # Detect which stem features are present
        stem_names = []
        for feat in analysis.feature_names:
            parts = feat.split("_", 1)
            if parts[0] not in ("global",) and parts[0] not in stem_names:
                stem_names.append(parts[0])

        recommended = recommend_preset(analysis)

        await _send(websocket, AudioAnalysisResponse(
            duration=analysis.duration,
            total_frames=analysis.total_frames,
            features=analysis.feature_names,
            bpm=analysis.bpm,
            lufs=analysis.lufs,
            sample_rate=analysis.sample_rate,
            hop_length=settings.audio_hop_length,
            recommended_preset=recommended,
            stems_available=len(stem_names) > 0,
            stems=stem_names if stem_names else None,
            waveform=analysis.get_waveform_preview(100),
        ))
    except FileNotFoundError as e:
        await _send(websocket, ErrorResponse(code="INVALID_REQUEST", message=str(e)))
    except Exception as e:
        log.exception("Audio analysis failed: %s", e)
        await _send(websocket, ErrorResponse(code="ENGINE_ERROR", message=str(e)))


async def _handle_check_stems(websocket: WebSocket) -> None:
    available = engine.stems_available()
    msg = "Stem separation ready" if available else (
        "Stem separation requires demucs. Install with: pip install demucs>=4.0"
    )
    await _send(websocket, StemsAvailableResponse(available=available, message=msg))


async def _handle_list_modulation_presets(websocket: WebSocket) -> None:
    from .modulation_engine import ModulationEngine
    presets = ModulationEngine.list_presets()
    await _send(websocket, ModulationPresetsResponse(presets=presets))


async def _handle_get_modulation_preset(websocket: WebSocket, req: Request) -> None:
    name = req.preset_name
    if not name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="preset_name required"))
        return
    from .modulation_engine import ModulationEngine
    try:
        slots = ModulationEngine.get_preset(name)
        slot_dicts = [
            {
                "source": s.source, "target": s.target,
                "min_val": s.min_val, "max_val": s.max_val,
                "attack": s.attack, "release": s.release,
                "enabled": s.enabled,
                "invert": s.invert,
            }
            for s in slots
        ]
        await _send(websocket, ModulationPresetDetailResponse(
            name=name, slots=slot_dicts))
    except ValueError as e:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=str(e)))


async def _handle_list_expression_presets(websocket: WebSocket) -> None:
    from .expression_presets import list_expression_presets
    presets = list_expression_presets()
    await _send(websocket, ExpressionPresetsListResponse(presets=presets))


async def _handle_get_expression_preset(websocket: WebSocket, req: Request) -> None:
    name = req.preset_name
    if not name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="preset_name required"))
        return
    from .expression_presets import get_expression_preset
    preset = get_expression_preset(name)
    if not preset:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=f"Unknown expression preset: {name}"))
        return
    await _send(websocket, ExpressionPresetDetailResponse(
        name=name,
        targets=preset["targets"],
        description=preset["description"],
        category=preset["category"],
    ))


async def _handle_list_choreography_presets(websocket: WebSocket) -> None:
    from .expression_presets import list_choreography_presets
    presets = list_choreography_presets()
    await _send(websocket, ChoreographyPresetsListResponse(presets=presets))


async def _handle_get_choreography_preset(websocket: WebSocket, req: Request) -> None:
    name = req.preset_name
    if not name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="preset_name required"))
        return
    from .expression_presets import get_choreography_preset
    choreo = get_choreography_preset(name)
    if not choreo:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=f"Unknown choreography preset: {name}"))
        return
    await _send(websocket, ChoreographyPresetDetailResponse(
        name=name,
        description=choreo["description"],
        slots=choreo.get("slots", []),
        expressions=choreo.get("expressions", {}),
    ))


async def _handle_list_prompt_schedules(websocket: WebSocket) -> None:
    from .prompt_schedule_presets import PromptSchedulePresetsManager
    mgr = PromptSchedulePresetsManager(settings.prompt_schedules_dir)
    items = mgr.list_presets()
    await _send(websocket, ListResponse(list_type="prompt_schedules", items=items))


async def _handle_get_prompt_schedule(websocket: WebSocket, req: Request) -> None:
    name = req.prompt_schedule_name
    if not name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="prompt_schedule_name required"))
        return
    from .prompt_schedule_presets import PromptSchedulePresetsManager
    mgr = PromptSchedulePresetsManager(settings.prompt_schedules_dir)
    try:
        data = mgr.get_preset(name)
        await _send(websocket, PresetResponse(name=name, data=data))
    except (FileNotFoundError, ValueError) as e:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=str(e)))


async def _handle_save_prompt_schedule(websocket: WebSocket, req: Request) -> None:
    name = req.prompt_schedule_name
    data = req.prompt_schedule_data
    if not name or not data:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST",
            message="prompt_schedule_name and prompt_schedule_data required"))
        return
    from .prompt_schedule_presets import PromptSchedulePresetsManager
    mgr = PromptSchedulePresetsManager(settings.prompt_schedules_dir)
    try:
        mgr.save_preset(name, data)
        await _send(websocket, PresetSavedResponse(name=name))
    except ValueError as e:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=str(e)))


async def _handle_delete_prompt_schedule(websocket: WebSocket, req: Request) -> None:
    name = req.prompt_schedule_name
    if not name:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="prompt_schedule_name required"))
        return
    from .prompt_schedule_presets import PromptSchedulePresetsManager
    mgr = PromptSchedulePresetsManager(settings.prompt_schedules_dir)
    try:
        mgr.delete_preset(name)
        await _send(websocket, PresetDeletedResponse(name=name))
    except (FileNotFoundError, ValueError) as e:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message=str(e)))


async def _handle_export_mp4(websocket: WebSocket, req: Request) -> None:
    """Export animation frames + audio to MP4 via ffmpeg."""
    from .video_export import export_mp4, find_ffmpeg

    # Validate ffmpeg availability
    ffmpeg = settings.ffmpeg_path or find_ffmpeg()
    if not ffmpeg:
        await _send(websocket, ExportMp4ErrorResponse(
            message="ffmpeg not found. Install ffmpeg and ensure it is in PATH."))
        return

    # Extract parameters from request
    output_dir = req.output_dir
    audio_path = req.audio_path
    fps = req.fps or 24.0
    scale_factor = req.scale_factor or 4
    quality = req.quality or "high"

    if not output_dir:
        await _send(websocket, ExportMp4ErrorResponse(
            message="output_dir is required"))
        return

    # Security: validate output_dir exists and contains SDDj frames
    real_dir = os.path.realpath(output_dir)
    if not os.path.isdir(real_dir):
        await _send(websocket, ExportMp4ErrorResponse(
            message=f"Output directory not found: {output_dir}"))
        return
    # Ensure dir contains .png frames (prevents arbitrary dir export)
    has_frames = any(f.endswith(".png") for f in os.listdir(real_dir))
    if not has_frames:
        await _send(websocket, ExportMp4ErrorResponse(
            message="Output directory contains no PNG frames"))
        return

    # Validate audio_path if provided
    if audio_path:
        audio_real = os.path.realpath(audio_path)
        if not os.path.isfile(audio_real):
            await _send(websocket, ExportMp4ErrorResponse(
                message=f"Audio file not found: {audio_path}"))
            return
        audio_ext = os.path.splitext(audio_real)[1].lower()
        if audio_ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            await _send(websocket, ExportMp4ErrorResponse(
                message=f"Unsupported audio format: {audio_ext}"))
            return

    # Build metadata from request
    export_meta: dict[str, str] = {}
    if hasattr(req, "prompt") and req.prompt:
        export_meta["comment"] = req.prompt[:256]
    export_meta["tool"] = "SDDj"

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: export_mp4(
                frame_dir=real_dir,
                audio_path=audio_path,
                fps=fps,
                scale_factor=scale_factor,
                quality=quality,
                metadata=export_meta,
                ffmpeg_path=ffmpeg if settings.ffmpeg_path else None,
            ),
        )
        await _send(websocket, ExportMp4Response(
            path=result.path,
            size_mb=result.size_mb,
            duration_s=result.duration_s,
        ))
    except Exception as e:
        log.exception("MP4 export failed: %s", e)
        await _send(websocket, ExportMp4ErrorResponse(message=str(e)))


async def _handle_generate_audio_reactive(
    websocket: WebSocket, req: Request, ws_id: int,
) -> None:
    audio_req = req.to_audio_reactive_request()
    if not audio_req.audio_path:
        await _send(websocket, ErrorResponse(
            code="INVALID_REQUEST", message="audio_path required"))
        return

    real_path = await _validate_audio_path(websocket, audio_req.audio_path)
    if real_path is None:
        return
    audio_req.audio_path = real_path

    loop = asyncio.get_running_loop()
    on_progress = _make_thread_callback(websocket, loop, timeout=1.0)
    on_frame = _make_thread_callback(websocket, loop, timeout=2.0)

    if _generate_lock is None:
        raise RuntimeError("Server not fully initialized")

    # Auto-scale timeout: base + 10s per frame (actual) + analysis overhead
    actual_frames = audio_req.fps * 300  # 5 min max audio at given fps
    anim_timeout = max(
        settings.generation_timeout,
        120 + actual_frames * 10,  # 2 min base + 10s/frame
    )

    async with _generate_lock:
        _generating[ws_id].set()
        try:
            t0 = time.perf_counter()
            frame_count = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: engine.generate_audio_reactive(
                        audio_req,
                        on_frame=on_frame,
                        on_progress=on_progress,
                    ),
                ),
                timeout=anim_timeout,
            )
            total_ms = int((time.perf_counter() - t0) * 1000)
        except asyncio.TimeoutError:
            engine.cancel()
            raise RuntimeError(
                f"Audio-reactive generation timed out after {anim_timeout:.0f}s"
            )
        finally:
            _generating[ws_id].clear()

    await _send(websocket, AudioReactiveCompleteResponse(
        total_frames=frame_count,
        total_time_ms=total_ms,
        tag_name=audio_req.tag_name,
    ))


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
    from .vram_utils import get_vram_info
    used, free, total = get_vram_info()
    extra = engine.get_status() if engine.is_loaded else {}
    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "loaded": engine.is_loaded,
        "vram_used_mb": used,
        "vram_free_mb": free,
        "vram_total_mb": total,
        **extra,
    })


@app.post("/shutdown")
async def http_shutdown() -> JSONResponse:
    """HTTP shutdown endpoint — used by start.ps1 for graceful stop."""
    if _generate_lock is not None and _generate_lock.locked():
        return JSONResponse({"error": "GPU_BUSY"}, status_code=503)

    log.info("HTTP shutdown requested")
    for ws in list(_active_connections):
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except Exception:
            pass

    # Schedule graceful shutdown after response is sent
    asyncio.get_running_loop().call_later(0.5, _request_shutdown)
    return JSONResponse({"status": "shutting_down"})


def _request_shutdown() -> None:
    """Platform-safe uvicorn shutdown.

    On POSIX, SIGINT triggers uvicorn's graceful shutdown handler.
    On Windows, os.kill(SIGINT) sends CTRL_C_EVENT to the entire console
    group, which can crash parent processes (Aseprite, PowerShell).
    We use SIGBREAK on Windows (only targets the current process) and
    fall back to sys.exit(0) if SIGBREAK is unavailable.
    """
    pid = os.getpid()
    if sys.platform == "win32":
        sig = getattr(signal, "SIGBREAK", None)
        if sig is not None:
            os.kill(pid, sig)
        else:
            os._exit(0)
    else:
        os.kill(pid, signal.SIGINT)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "sddj.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_max_size=50 * 1024 * 1024,  # 50MB max WebSocket message
    )


if __name__ == "__main__":
    main()
