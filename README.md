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
                    AnimateDiff + Frame Chain + Audio-Reactive Chain
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
│   └── LIVE-PAINT.md            # Live paint technical deep-dive
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
│       ├── audio_analyzer.py   # Audio feature extraction (librosa)
│       ├── audio_cache.py      # Disk cache for audio analysis (NPZ)
│       ├── stem_separator.py   # Optional stem separation (demucs, CPU)
│       └── modulation_engine.py # Modulation matrix + expressions (simpleeval)
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
- **Audio Reactivity** (v0.7.0) — Synth-style modulation matrix: map audio features (RMS, onset, spectral bands, per-stem) to inference parameters (denoise, CFG, noise, ControlNet, seed). Deforum-inspired math expressions. Pre-computed offline analysis via librosa. Optional CPU stem separation (demucs). Built-in presets: energetic, ambient, bass_driven.
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
- **Cancellation** (v0.6.1) — Robust multi-layered cancel: immediate server ACK, 30s safety timer fallback, GPU cleanup on timeout; works across all modes (txt2img, img2img, inpaint, ControlNet, animation, live)
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

Connect to `ws://127.0.0.1:9876/ws`. All messages are JSON.

### Actions

| Action               | Description                        |
|----------------------|------------------------------------|
| `ping`               | Health check, returns `pong`       |
| `cancel`             | Cancel in-progress generation (server ACK + GPU cleanup) |
| `generate`           | Run single-frame generation        |
| `generate_animation` | Run multi-frame animation          |
| `list_loras`         | List available LoRAs               |
| `list_palettes`      | List available palettes            |
| `list_controlnets`   | List available ControlNet modes    |
| `list_embeddings`    | List available TI embeddings       |
| `realtime_start`     | Start real-time paint session       |
| `realtime_frame`     | Send canvas frame for live processing |
| `realtime_update`    | Hot-update realtime parameters      |
| `realtime_stop`      | Stop real-time paint session        |
| `generate_prompt`    | Generate a random prompt using templates |
| `list_presets`       | List available presets               |
| `get_preset`         | Load a preset by name                |
| `save_preset`        | Save current settings as preset      |
| `delete_preset`      | Delete a user preset                 |
| `cleanup`            | Free GPU VRAM and run garbage collection |
| `analyze_audio`      | Analyze audio file, extract per-frame features |
| `generate_audio_reactive` | Generate audio-reactive animation   |
| `check_stems`        | Check if stem separation (demucs) is available |
| `list_modulation_presets` | List built-in modulation presets    |

### Generate Request

```json
{
  "action": "generate",
  "prompt": "pixel art character",
  "negative_prompt": "blurry, photorealistic",
  "mode": "txt2img | img2img | inpaint | controlnet_*",
  "width": 512,
  "height": 512,
  "seed": -1,
  "steps": 8,
  "cfg_scale": 5.0,
  "denoise_strength": 1.0,
  "clip_skip": 2,
  "lora": { "name": "pixelart_redmond", "weight": 1.0 },
  "negative_ti": [{ "name": "EasyNegative", "weight": 1.0 }],
  "post_process": {
    "pixelate": { "enabled": true, "target_size": 128 },
    "quantize_method": "kmeans",
    "quantize_colors": 32,
    "dither": "none",
    "palette": { "mode": "auto" },
    "remove_bg": false
  }
}
```

#### Inpaint Mode

For inpainting, set `mode` to `"inpaint"` and include `source_image` (base64 PNG of the original image) and `mask_image` (base64 PNG where white=repaint, black=keep):

```json
{
  "action": "generate",
  "prompt": "pixel art, golden crown",
  "mode": "inpaint",
  "width": 512,
  "height": 512,
  "seed": -1,
  "steps": 8,
  "cfg_scale": 5.0,
  "denoise_strength": 0.7,
  "source_image": "<base64 PNG>",
  "mask_image": "<base64 PNG, white=repaint, black=keep>",
  "post_process": { "..." }
}
```

### Animation Request

```json
{
  "action": "generate_animation",
  "method": "chain",
  "prompt": "pixel art walk cycle, PixArFK",
  "mode": "controlnet_scribble",
  "width": 512,
  "height": 512,
  "seed": -1,
  "steps": 8,
  "cfg_scale": 5.0,
  "denoise_strength": 0.30,
  "control_image": "<base64 PNG>",
  "frame_count": 8,
  "frame_duration_ms": 100,
  "seed_strategy": "increment",
  "tag_name": "walk",
  "enable_freeinit": false,
  "freeinit_iterations": 2,
  "post_process": { "..." }
}
```

| Field               | Values                               | Default       |
|---------------------|--------------------------------------|---------------|
| `method`            | `chain`, `animatediff`               | `chain`       |
| `frame_count`       | 2 - 120                              | `8`           |
| `frame_duration_ms` | 50 - 2000                            | `100`         |
| `seed_strategy`     | `fixed`, `increment`, `random`       | `increment`   |
| `tag_name`          | string or null                       | `null`        |
| `enable_freeinit`   | boolean                              | `false`       |
| `freeinit_iterations` | 1 - 3                              | `2`           |

### Real-Time Paint Request

Start a live paint session:

```json
{
  "action": "realtime_start",
  "prompt": "pixel art character",
  "negative_prompt": "blurry, photorealistic",
  "width": 512,
  "height": 512,
  "seed": -1,
  "steps": 4,
  "cfg_scale": 2.5,
  "denoise_strength": 0.5,
  "clip_skip": 2,
  "lora": { "name": "pixelart_redmond", "weight": 1.0 },
  "post_process": { "..." }
}
```

Send canvas frames for processing:

```json
{
  "action": "realtime_frame",
  "image": "<base64 PNG — current canvas>",
  "frame_id": 1,
  "prompt": "optional prompt override",
  "mask": "<base64 mask of dirty region (optional)>",
  "roi_x": 0, "roi_y": 0, "roi_w": 128, "roi_h": 128
}
```

Hot-update parameters mid-session (all fields optional):

```json
{
  "action": "realtime_update",
  "prompt": "new prompt",
  "negative_prompt": "blurry, photorealistic",
  "denoise_strength": 0.6,
  "steps": 3,
  "cfg_scale": 3.0,
  "clip_skip": 2,
  "seed": 42
}
```

Stop the session:

```json
{ "action": "realtime_stop" }
```

| Field               | Values / Constraint                  | Default       |
|---------------------|--------------------------------------|---------------|
| `steps`             | 2 - 8                                | `4`           |
| `cfg_scale`         | 1.0 - 10.0                           | `2.5`         |
| `denoise_strength`  | 0.05 - 0.95                          | `0.5`         |
| `clip_skip`         | 1 - 12                               | `2`           |

### Audio-Reactive Request

Analyze audio first, then generate animation with per-frame parameter modulation:

```json
{
  "action": "analyze_audio",
  "audio_path": "C:/path/to/audio.wav",
  "fps": 24.0,
  "enable_stems": false
}
```

```json
{
  "action": "generate_audio_reactive",
  "audio_path": "C:/path/to/audio.wav",
  "fps": 24.0,
  "enable_stems": false,
  "modulation_slots": [
    {
      "source": "global_rms",
      "target": "denoise_strength",
      "min_val": 0.2,
      "max_val": 0.7,
      "attack": 2,
      "release": 8,
      "enabled": true
    }
  ],
  "expressions": null,
  "modulation_preset": null,
  "prompt": "pixel art character",
  "mode": "txt2img",
  "width": 512,
  "height": 512,
  "steps": 8,
  "cfg_scale": 5.0,
  "denoise_strength": 0.30,
  "frame_duration_ms": 42,
  "post_process": { "..." }
}
```

| Modulation Source    | Description                    |
|----------------------|--------------------------------|
| `global_rms`         | Overall energy                 |
| `global_onset`       | Transient / attack strength    |
| `global_centroid`    | Spectral brightness            |
| `global_low`         | Low-frequency energy (20-300Hz)  |
| `global_mid`         | Mid-frequency energy (300-2kHz)  |
| `global_high`        | High-frequency energy (2k-16kHz) |
| `drums_rms/onset`    | Per-stem (requires `enable_stems`) |
| `bass_rms/onset`     | Per-stem (requires `enable_stems`) |
| `vocals_rms/onset`   | Per-stem (requires `enable_stems`) |
| `other_rms/onset`    | Per-stem (requires `enable_stems`) |

| Modulation Target    | Range           | Description                |
|----------------------|-----------------|----------------------------|
| `denoise_strength`   | 0.05 – 0.95     | Frame-to-frame change      |
| `cfg_scale`          | 1.0 – 30.0      | Prompt adherence           |
| `noise_amplitude`    | 0.0 – 1.0       | Additive latent noise      |
| `controlnet_scale`   | 0.0 – 2.0       | ControlNet influence       |
| `seed_offset`        | 0 – 1000         | Per-frame seed variation   |

### Response Types

| Type                 | Fields                                                              |
|----------------------|---------------------------------------------------------------------|
| `progress`           | `step`, `total`, `frame_index` (opt), `total_frames` (opt)         |
| `result`             | `image` (b64 PNG), `seed`, `time_ms`, `width`, `height`            |
| `animation_frame`    | `frame_index`, `total_frames`, `image` (b64 PNG), `seed`, `time_ms`, `width`, `height` |
| `animation_complete` | `total_frames`, `total_time_ms`, `tag_name` (opt)                  |
| `error`              | `code` (`ENGINE_ERROR`, `OOM`, `CANCELLED`, `TIMEOUT`, `INVALID_REQUEST`, `MAX_CONNECTIONS`, `UNKNOWN_ACTION`, `REALTIME_BUSY`, `REALTIME_NOT_ACTIVE`, `GPU_BUSY`), `message` |
| `list`               | `list_type`, `items`                                                |
| `pong`               | (no fields)                                                         |
| `realtime_ready`     | `message`                                                           |
| `realtime_result`    | `image` (b64 PNG), `latency_ms`, `frame_id`, `width`, `height`, `roi_x` (opt), `roi_y` (opt) |
| `realtime_stopped`   | `message`                                                           |
| `prompt_result`      | `prompt`, `negative_prompt`, `components`                           |
| `preset`             | `name`, `data`                                                      |
| `preset_saved`       | `name`                                                              |
| `preset_deleted`     | `name`                                                              |
| `cleanup_done`       | `message`, `freed_mb`                                               |
| `audio_analysis`     | `duration`, `total_frames`, `features` (list), `stems_available`, `stems` (opt) |
| `audio_reactive_frame` | `frame_index`, `total_frames`, `image` (b64), `seed`, `time_ms`, `width`, `height`, `params_used` |
| `audio_reactive_complete` | `total_frames`, `total_time_ms`, `tag_name` (opt)               |
| `stems_available`    | `available`, `message`                                              |
| `modulation_presets` | `presets` (list of names)                                           |

### Input Validation

| Field       | Constraint           |
|-------------|---------------------|
| `width`     | 64 - 2048 (rounded to ×8) |
| `height`    | 64 - 2048 (rounded to ×8) |
| `steps`     | 1 - 100              |
| `cfg_scale` | 0.0 - 30.0           |
| `clip_skip` | 1 - 12               |
| `denoise_strength` | 0.0 - 1.0     |
| `lora.weight` | -2.0 - 2.0 (negative LoRA) |
| `target_size` | 8 - 512              |
| `colors`    | 2 - 256              |

## Post-Processing Pipeline

Executed in strict order (non-negotiable):

1. **Background Removal** — rembg with u2net (CPU, fast ~3-4s). Configurable: birefnet-general (best edges), bria-rmbg (SOTA quality)
2. **Pixelation** — NEAREST downscale to target size (aspect-ratio preserving)
3. **Color Quantization** — MiniBatchKMeans (batch=4096) / PIL Median Cut / Octree
4. **Palette Enforcement** — CIELAB nearest neighbor via cached KD-Tree
5. **Dithering** — Floyd-Steinberg (Numba JIT) or Bayer (2×2, 4×4, 8×8)
6. **Alpha Cleanup** — Binary threshold (no semi-transparency in pixel art)

## Environment Variables

All prefixed with `PIXYTOON_`. Example: `PIXYTOON_PORT=8080`.

| Variable                   | Default                               | Description                     |
|----------------------------|---------------------------------------|---------------------------------|
| `HOST`                     | `127.0.0.1`                           | Server bind address             |
| `PORT`                     | `9876`                                | Server port                     |
| `MODELS_DIR`               | `server/models`                       | Root models directory           |
| `CHECKPOINTS_DIR`          | `server/models/checkpoints`           | SD checkpoint cache             |
| `LORAS_DIR`                | `server/models/loras`                 | LoRA files (.safetensors)       |
| `EMBEDDINGS_DIR`           | `server/models/embeddings`            | Textual Inversion embeddings    |
| `PALETTES_DIR`             | `server/palettes`                     | Palette JSON files              |
| `PRESETS_DIR`              | `server/presets`                      | Generation presets              |
| `PROMPTS_DATA_DIR`         | `server/data/prompts`                 | Auto-prompt generator data      |
| `DEFAULT_CHECKPOINT`       | `Lykon/dreamshaper-8`                 | SD1.5 checkpoint                |
| `HYPER_SD_REPO`            | `ByteDance/Hyper-SD`                  | Hyper-SD HuggingFace repo       |
| `HYPER_SD_LORA_FILE`       | `Hyper-SD15-8steps-CFG-lora.safetensors` | Hyper-SD LoRA filename       |
| `HYPER_SD_FUSE_SCALE`      | `0.8`                                 | Hyper-SD LoRA fusion scale      |
| `DEFAULT_STEPS`            | `8`                                   | Default inference steps         |
| `DEFAULT_CFG`              | `5.0`                                 | Default CFG scale               |
| `DEFAULT_WIDTH`            | `512`                                 | Default output width            |
| `DEFAULT_HEIGHT`           | `512`                                 | Default output height           |
| `DEFAULT_CLIP_SKIP`        | `2`                                   | CLIP skip layers (2 = stylized) |
| `DEFAULT_PIXEL_LORA`       | `auto`                                | Default pixel LoRA (`auto` = first found, `""` = none) |
| `DEFAULT_PIXEL_LORA_WEIGHT`| `1.0`                                 | Default LoRA fuse weight        |
| `ENABLE_TORCH_COMPILE`     | `True`                                | UNet compilation (requires Triton + MSVC) |
| `COMPILE_MODE`             | `default`                             | torch.compile mode (`default` / `max-autotune` / `reduce-overhead`) |
| `ENABLE_DEEPCACHE`         | `True`                                | DeepCache acceleration          |
| `ENABLE_FREEU`             | `True`                                | FreeU v2 quality boost          |
| `ENABLE_ATTENTION_SLICING` | `True`                                | Attention slicing (PyTorch < 2.0 fallback) |
| `ENABLE_VAE_TILING`        | `True`                                | VAE tiling for large images     |
| `ENABLE_WARMUP`            | `True`                                | Warmup generation at startup    |
| `DEEPCACHE_INTERVAL`       | `3`                                   | DeepCache skip interval         |
| `DEEPCACHE_BRANCH`         | `0`                                   | DeepCache branch ID             |
| `FREEU_S1`                 | `0.9`                                 | FreeU v2 skip scale 1           |
| `FREEU_S2`                 | `0.2`                                 | FreeU v2 skip scale 2           |
| `FREEU_B1`                 | `1.5`                                 | FreeU v2 backbone scale 1       |
| `FREEU_B2`                 | `1.6`                                 | FreeU v2 backbone scale 2       |
| `GENERATION_TIMEOUT`       | `600.0`                               | Max seconds per generation      |
| `REMBG_MODEL`              | `u2net`                                 | Background removal model (u2net / birefnet-general / bria-rmbg) |
| `REMBG_ON_CPU`             | `True`                                | Run rembg on CPU                |
| `DEFAULT_ANIM_FRAMES`      | `8`                                   | Default animation frame count   |
| `DEFAULT_ANIM_DURATION_MS` | `100`                                 | Default frame duration (ms)     |
| `DEFAULT_ANIM_DENOISE`     | `0.30`                                | Default animation denoise       |
| `MAX_ANIMATION_FRAMES`     | `120`                                 | Max frames per animation        |
| `ANIMATEDIFF_MODEL`        | `guoyww/animatediff-motion-adapter-v1-5-3` | AnimateDiff motion adapter |
| `ENABLE_FREEINIT`          | `False`                               | FreeInit for AnimateDiff        |
| `FREEINIT_ITERATIONS`      | `2`                                   | FreeInit iteration count        |
| `REALTIME_TIMEOUT`         | `300.0`                               | Auto-stop if no frame for N seconds |
| `REALTIME_DEFAULT_STEPS`   | `4`                                   | Default realtime inference steps |
| `REALTIME_DEFAULT_CFG`     | `2.5`                                 | Default realtime CFG scale      |
| `REALTIME_DEFAULT_DENOISE` | `0.5`                                 | Default realtime denoise strength |
| `REALTIME_ROI_PADDING`     | `32`                                  | Padding around ROI crop (pixels)  |
| `REALTIME_ROI_MIN_SIZE`    | `64`                                  | Minimum ROI dimension (pixels)    |
| `AUDIO_CACHE_DIR`          | `""` (temp dir)                       | Cache directory for audio analysis |
| `AUDIO_MAX_FILE_SIZE_MB`   | `500`                                 | Max audio file size (MB)          |
| `AUDIO_MAX_FRAMES`         | `3600`                                | Max frames per audio animation    |
| `AUDIO_DEFAULT_FPS`        | `24.0`                                | Default audio analysis FPS        |
| `AUDIO_DEFAULT_ATTACK`     | `2`                                   | Default EMA attack frames         |
| `AUDIO_DEFAULT_RELEASE`    | `8`                                   | Default EMA release frames        |
| `STEM_MODEL`               | `htdemucs`                            | Demucs model for stem separation  |
| `STEM_DEVICE`              | `cpu`                                 | Stem separation device (always CPU) |

## HTTP Endpoints

| Method | Path      | Description       |
|--------|-----------|-------------------|
| `GET`  | `/health` | Server readiness  |

## Troubleshooting

| Problem                        | Solution                                                    |
|--------------------------------|-------------------------------------------------------------|
| Server won't start             | Check `uv run python run.py` output for errors              |
| CUDA OOM                       | Reduce resolution, disable torch.compile, enable VAE tiling |
| Port already in use            | Change `PIXYTOON_PORT` or kill existing process              |
| Aseprite can't connect         | Ensure server is running, check firewall on 127.0.0.1:9876  |
| LoRA not found                 | Place `.safetensors` file in `server/models/loras/`          |
| Slow first generation          | Normal: torch.compile + Numba JIT warm up on first run      |
| torch.compile fails            | Install Visual Studio 2022 C++ workload; ensure Triton installed |
| "Not enough SMs"               | Harmless Triton warning on consumer GPUs, can be ignored    |
| CUDAGraphs tensor overwrite    | Uses `default` compile mode. If using `reduce-overhead`, disable DeepCache |
| Generation timed out           | Increase `PIXYTOON_GENERATION_TIMEOUT` or reduce steps/resolution |
| LoRA change is slow            | Expected: LoRA weight change triggers recompilation (~30-60s once) |
| AnimateDiff OOM                | AnimateDiff needs ~8-10GB VRAM; reduce `frame_count` or resolution |
| AnimateDiff slow first run     | Motion adapter downloads on first use (~97MB); subsequent runs use cache |
| Chain animation hangs          | Fixed in v0.1.5: dynamo.reset + scheduler reset + RGBA→RGB fix     |
| Cancel button doesn't stop     | Server-side cancel ACK + 30s safety timer auto-unlocks UI; check server terminal |
| "Server unresponsive"          | Heartbeat watchdog detected no pong for 90s; auto-reconnect kicks in |
| "Reconnecting in Xs"           | Normal: exponential backoff (2s→30s); server is unreachable, will auto-retry |
| Live Paint not starting        | Ensure no generation is in progress (GPU_BUSY); check server logs  |
| Live Paint high latency        | Reduce steps (2-3), reduce resolution, ensure no other GPU load    |
| Live Paint auto-stopped        | Session times out after 5min of inactivity (configurable via `PIXYTOON_REALTIME_TIMEOUT`) |

## Documentation

| Document | Description |
|----------|-------------|
| **[User Guide](docs/GUIDE.md)** | First launch, modes, parameters, post-processing, LoRA, performance, troubleshooting |
| **[Cookbook](docs/COOKBOOK.md)** | Tested recipes by creative intention — characters, environments, items, animation, palettes |
| **[Live Paint Guide](docs/LIVE-PAINT.md)** | Dedicated guide for real-time SD-assisted painting — techniques, workflow, parameters |

## License

MIT
