# PixyToon

![demo1](https://github.com/user-attachments/assets/0ca19204-fdcd-46fb-86d1-23a48226f9ef)

---

Local SOTA pixel art generation and animation for Aseprite via Stable Diffusion 1.5 + AnimateDiff.

---

## Quick Start

```
setup.ps1         <- One-click: install deps, download models, build extension
start.ps1         <- One-click: launch server + Aseprite
```

## Architecture

```
Aseprite (Lua WebSocket) <-> PixyToon Server (Python FastAPI)
                                  |
                            SD1.5 + Hyper-SD + DeepCache + FreeU v2 + torch.compile
                                  |
                    AnimateDiff + Frame Chain + Audio-Reactive (Chain/AnimateDiff)
                                  |
                   Audio Analyzer (librosa) + Stem Separator (demucs, CPU)
                                  |
                     Modulation Engine (synth matrix + expressions)
                                  |
                            Pixel Art Post-Processing Pipeline
```

## Project Structure

```
pixytoon/
├── setup.ps1                    # One-click setup (PowerShell 7)
├── start.ps1                    # One-click start (PowerShell 7)
├── bin/
│   └── aseprite/                # Compiled Aseprite v1.3.17
│       └── aseprite.exe
├── dist/                        # Built .aseprite-extension
├── extension/
│   ├── package.json             # Extension manifest
│   ├── keys/
│   │   └── pixytoon.aseprite-keys  # F5 hotkey for Live Send
│   └── scripts/
│       ├── json.lua             # Pure Lua JSON parser
│       ├── pixytoon.lua         # Plugin entry point (init/exit + module loader)
│       ├── pixytoon_base64.lua  # Base64 encode/decode (pure Lua)
│       ├── pixytoon_state.lua   # Constants + shared mutable state
│       ├── pixytoon_utils.lua   # Temp files, image I/O, timer helper
│       ├── pixytoon_settings.lua # Save/load/apply settings (JSON)
│       ├── pixytoon_ws.lua      # WebSocket transport + connection mgmt
│       ├── pixytoon_capture.lua # Image capture (active layer, flattened, mask)
│       ├── pixytoon_request.lua # Request builders (parse, attach, build)
│       ├── pixytoon_import.lua  # Import result, animation frame, live preview
│       ├── pixytoon_live.lua    # Live paint system (event-driven, F5 hotkey, ROI)
│       ├── pixytoon_handler.lua # Response dispatch table
│       └── pixytoon_dialog.lua  # Dialog construction (tabs + actions)
├── scripts/
│   ├── build_extension.py       # Package -> .aseprite-extension
│   ├── download_models.py       # Download all models from HuggingFace
│   ├── test_generate.py         # Test: txt2img, img2img, inpaint
│   ├── test_animation.py        # Test: chain, animatediff, chain-img2img
│   └── test_inpaint.py          # Test: inpaint (high/low denoise)
├── docs/
│   ├── GUIDE.md                 # Complete user guide
│   ├── COOKBOOK.md               # Recipes and workflows
│   ├── LIVE-PAINT.md            # Live paint technical deep-dive
│   ├── AUDIO-REACTIVITY.md      # Audio reactivity guide (presets, expressions, BPM)
│   ├── API-REFERENCE.md         # WebSocket protocol specification
│   ├── CONFIGURATION.md         # Environment variables reference
│   └── TROUBLESHOOTING.md       # Common issues and solutions
├── server/
│   ├── pyproject.toml           # Dependencies (torch CUDA 12.8)
│   ├── run.py                   # Entry point
│   ├── palettes/                # 7 preset palettes (JSON)
│   ├── presets/                 # Built-in and user-saved generation presets
│   ├── data/prompts/            # Auto-prompt generator data files
│   ├── models/                  # Downloaded models (auto-populated)
│   │   ├── loras/               # User LoRA files (.safetensors)
│   │   └── embeddings/          # Textual Inversion embeddings
│   └── pixytoon/                # Python package
│       ├── __init__.py          # Package version
│       ├── config.py            # Pydantic Settings (env vars)
│       ├── protocol.py          # WebSocket schemas (Pydantic v2)
│       ├── engine.py            # SD1.5 SOTA pipeline orchestrator
│       ├── server.py            # FastAPI WebSocket + HTTP server
│       ├── pipeline_factory.py  # Pipeline construction, compile, scheduler
│       ├── animatediff_manager.py # AnimateDiff lifecycle (load/inject/eject)
│       ├── deepcache_manager.py # DeepCache toggle + suspend/resume
│       ├── lora_fuser.py        # LoRA fuse/unfuse with dynamo reset
│       ├── freeu_applicator.py  # FreeU v2 application
│       ├── postprocess.py       # 6-stage pixel art pipeline
│       ├── image_codec.py       # Base64 encode/decode, resize, composite
│       ├── rembg_wrapper.py     # Background removal (CPU, lazy-load)
│       ├── validation.py        # Shared input validation (path traversal guard)
│       ├── lora_manager.py      # LoRA discovery (path-validated)
│       ├── ti_manager.py        # Textual Inversion discovery
│       ├── palette_manager.py   # Palette loading (path-validated)
│       ├── presets_manager.py   # Preset save/load/delete
│       ├── prompt_generator.py  # Auto-prompt generation from templates
│       ├── audio_analyzer.py   # Audio feature extraction + BPM detection (librosa)
│       ├── audio_cache.py      # Disk cache for audio analysis (NPZ)
│       ├── stem_separator.py   # Optional stem separation (demucs, CPU)
│       ├── modulation_engine.py # Modulation matrix + expressions (simpleeval)
│       ├── auto_calibrate.py   # Audio-based preset recommendation
│       ├── prompt_schedule.py  # Per-segment prompt resolution
│       └── video_export.py    # MP4 export via ffmpeg (nearest-neighbor, audio mux)
```

## Features

- **txt2img / img2img / inpaint / ControlNet** — OpenPose, Canny, Scribble, Lineart (v1.1)
- **Live Paint** (v0.3.0, v0.6.0 rewrite) — Event-driven SD-assisted painting: Auto mode (sends after each brush stroke) + Manual mode (F5 hotkey), zero CPU when idle, ROI dirty-region detection, debounced stroke detection via `sprite.events:on('change')`
- **Sequence Output** (v0.6.1) — Choose per-generation output: new layer (default) or new frame in the timeline — ideal for img2img iteration and loop workflows
- **Loop Mode** (v0.4.0) — Continuous generation with random seeds for rapid variation exploration
- **Random Loop** (v0.5.0) — Continuous generation with auto-randomized prompts; lock subject/elements while randomizing the rest
- **Auto-Prompt Generator** (v0.4.0) — Randomize creative prompts from curated templates with lockable fields
- **Lock Subject** (v0.5.0) — Keep a fixed subject (character, object) while randomizing style, mood, lighting, etc.
- **Presets** (v0.4.0) — Save/load generation settings; built-in presets for pixel art, anime, character, landscape, and more
- **Audio Reactivity** (v0.7.0-v0.7.3) — Synth-style modulation matrix: map audio features (RMS, onset, spectral, beat, per-stem, sub-bass, upper-mid, presence) to inference parameters (denoise, CFG, noise, ControlNet, seed, palette shift, frame cadence). BPM detection + auto-calibration recommends best preset. 20 built-in presets (genre/style/complexity/target). Prompt scheduling for per-segment prompts. Deforum-inspired math expressions with BPM variable. Optional CPU stem separation (demucs). AnimateDiff + Audio mode for temporal coherence (16-frame chunked batches with overlap blending).
- **Animation** — Dual-method: Frame Chain (img2img chaining) + AnimateDiff (motion module temporal consistency)
- **AnimateDiff** — Motion adapter v1-5-3, FreeInit support, auto DeepCache disable/re-enable, ControlNet compatible
- **LoRA stacking** — Hyper-SD (speed) + pixel art LoRA (style, ±2.0 weight range)
- **Textual Inversion** — EasyNegative for quality (auto-loaded from `server/models/embeddings/`)
- **CLIP skip 2** — Skips last encoder layer for better stylized output
- **Fast generation** — Hyper-SD (8 steps) + DeepCache + FreeU v2 + torch.compile (~2-5s on RTX 4060 after warmup)
- **Post-processing** — Pixelate, K-Means/Octree/MedianCut quantize, CIELAB palette enforcement (KD-Tree), Floyd-Steinberg (Numba JIT) / Bayer dithering, bg removal (BiRefNet)
- **Startup warmup** — Pre-compiles torch + Numba JIT on boot (first real generation is fast)
- **Health check** — `GET /health` for readiness polling
- **Concurrency safe** — GPU access serialized via asyncio lock
- **MP4 Export** (v0.7.3) — Export audio-reactive animations as MP4 with nearest-neighbor upscaling and embedded audio. Quality presets: web, high, archive, raw. Requires ffmpeg.
- **Cancellation** (v0.6.1, v0.7.3 fix) — Robust multi-layered cancel: immediate server ACK, 30s safety timer fallback, GPU cleanup on timeout; concurrent receive loop handles cancel during long-running generations (audio-reactive, AnimateDiff). Works across all modes.
- **Auto-reconnect** (v0.6.1) — Exponential backoff reconnection (2s → 30s max) with heartbeat pong watchdog (3× interval unresponsive → disconnect + reconnect)
- **Generation timeout** — Configurable max time per generation (default 10min, auto-scaled for animation); sends cancel to server to free the GPU

## Performance Stack

| Optimization | Role | Impact |
|-------------|------|--------|
| **Hyper-SD LoRA** (8-step CFG, fused at 0.8) | Fewer diffusion steps | ~2-3x faster generation |
| **DeepCache** (interval=3) | Feature caching between steps | ~2.3x additional speedup |
| **FreeU v2** (s1=0.9 s2=0.2 b1=1.5 b2=1.6) | Free quality boost | No speed cost |
| **CLIP skip 2** | Skip last CLIP layer | Better stylized output |
| **torch.compile** (default) | UNet Triton codegen | ~20-30% faster inference |
| **AnimateDiff** (motion adapter v1-5-3) | Temporally consistent animation | Motion module ~97MB |
| **FreeInit** (optional, 2 iters) | Improved AnimateDiff temporal consistency | ~2x AnimateDiff time |
| **PyTorch SDP** (native, auto-active) | Fused attention kernels (FlashAttention2) | Memory + speed |
| **VAE slicing + tiling** | Batched VAE decode | Lower VRAM peak |
| **fp16 inference** | Half-precision throughout | ~50% VRAM reduction |

> **torch.compile modes:** `default` mode is used by default — fast compilation with Triton codegen.
> `max-autotune` benchmarks every kernel candidate for peak throughput but adds minutes to startup.
> `reduce-overhead` uses CUDAGraphs which is **incompatible** with DeepCache's dynamic
> control flow (skip/compute branches). If you set `PIXYTOON_COMPILE_MODE=reduce-overhead`,
> you must disable DeepCache (`PIXYTOON_ENABLE_DEEPCACHE=False`).

### Attention Mechanism

PyTorch ≥ 2.0 uses `scaled_dot_product_attention` (SDP) **by default** in diffusers — no configuration needed. SDP auto-dispatches to the best available kernel:

- **FlashAttention2** kernels (integrated in PyTorch ≥ 2.2) — fused, memory-efficient
- **Math fallback** — for unsupported head dimensions

A separate `flash-attn` package is **not required** — PyTorch ≥ 2.1 includes FA2 natively via SDP. xformers is superseded by native SDP and provides no benefit on PyTorch ≥ 2.0.

SageAttention (thu-ml) targets long-sequence attention and offers minimal gains on SD1.5's short UNet sequences (77 tokens, small head dims). Not justified given installation complexity.

## Requirements

- **GPU**: NVIDIA >= 8GB VRAM (10GB for AnimateDiff + ControlNet; tested RTX 4060)
- **CUDA**: 12.8
- **Python**: 3.11-3.13
- **uv**: Package manager
- **Visual Studio 2022**: C++ Desktop Development workload (for torch.compile / Triton)

## WebSocket Protocol

See **[API Reference](docs/API-REFERENCE.md)** for the complete WebSocket protocol specification.

## Post-Processing Pipeline

Executed in strict order:

1. **Background Removal** — rembg (u2net / birefnet-general / bria-rmbg)
2. **Pixelation** — NEAREST downscale to target size
3. **Color Quantization** — KMeans / Median Cut / Octree
4. **Palette Enforcement** — CIELAB nearest neighbor (KD-Tree)
5. **Dithering** — Floyd-Steinberg (Numba JIT) or Bayer
6. **Alpha Cleanup** — Binary threshold

## Configuration

See **[Configuration](docs/CONFIGURATION.md)** for all `PIXYTOON_*` environment variables.

## Troubleshooting

See **[Troubleshooting](docs/TROUBLESHOOTING.md)** for the full list. Most common:

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce resolution, disable torch.compile, enable VAE tiling |
| torch.compile fails | Install Visual Studio 2022 C++ workload |
| Slow first generation | Normal: torch.compile + Numba JIT warm up on first run |
| "Server unresponsive" | Heartbeat watchdog — auto-reconnect will kick in |
| Cancel doesn't stop | 30s safety timer auto-unlocks UI; check server terminal |

## Documentation

| Document | Description |
|----------|-------------|
| **[User Guide](docs/GUIDE.md)** | First launch, modes, parameters, post-processing, LoRA, performance |
| **[Cookbook](docs/COOKBOOK.md)** | Tested recipes by creative intention |
| **[Live Paint Guide](docs/LIVE-PAINT.md)** | Real-time SD-assisted painting |
| **[Audio Reactivity](docs/AUDIO-REACTIVITY.md)** | Audio-driven animation — modulation matrix, presets, expressions, BPM sync |
| **[API Reference](docs/API-REFERENCE.md)** | WebSocket protocol specification |
| **[Configuration](docs/CONFIGURATION.md)** | Environment variables reference |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |

## License

MIT
