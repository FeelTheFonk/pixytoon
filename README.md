# PixyToon

Local SOTA pixel art generation and animation for Aseprite via Stable Diffusion 1.5 + AnimateDiff.

## Quick Start

```
setup.bat         <- One-click: install deps, download models, build extension
start.bat         <- One-click: launch server + Aseprite
```

## Architecture

```
Aseprite (Lua WebSocket) <-> PixyToon Server (Python FastAPI)
                                  |
                            SD1.5 + Hyper-SD + DeepCache + FreeU v2 + torch.compile
                                  |
                            AnimateDiff (motion module) + Frame Chain
                                  |
                            Pixel Art Post-Processing Pipeline
```

## Project Structure

```
pixytoon/
├── setup.bat                    # One-click setup
├── start.bat                    # One-click start (health-check polling)
├── bin/
│   └── aseprite/                # Compiled Aseprite v1.3.17
│       └── aseprite.exe
├── dist/                        # Built .aseprite-extension
├── extension/
│   ├── package.json             # Extension manifest
│   └── scripts/
│       ├── json.lua             # Pure Lua JSON parser
│       └── pixytoon.lua         # Dialog UI + WebSocket client
├── scripts/
│   ├── build_extension.py       # Package -> .aseprite-extension
│   ├── download_models.py       # Download all models from HuggingFace
│   ├── test_generate.py         # Test: txt2img, img2img, inpaint
│   ├── test_animation.py        # Test: chain, animatediff, chain-img2img
│   └── test_inpaint.py          # Test: inpaint (high/low denoise)
├── server/
│   ├── pyproject.toml           # Dependencies (torch CUDA 12.8)
│   ├── run.py                   # Entry point
│   ├── palettes/                # 7 preset palettes (JSON)
│   └── pixytoon/                # Python package
│       ├── __init__.py          # Package version (0.2.0)
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
│       ├── lora_manager.py      # LoRA discovery (path-validated)
│       ├── ti_manager.py        # Textual Inversion discovery
│       └── palette_manager.py   # Palette loading (path-validated)
```

## Features

- **txt2img / img2img / inpaint / ControlNet** — OpenPose, Canny, Scribble, Lineart (v1.1)
- **Animation** — Dual-method: Frame Chain (img2img chaining) + AnimateDiff (motion module temporal consistency)
- **AnimateDiff** — Motion adapter v1-5-3, FreeInit support, auto DeepCache disable/re-enable, ControlNet compatible
- **LoRA stacking** — Hyper-SD (speed) + pixel art LoRA (style, ±2.0 weight range)
- **Textual Inversion** — EasyNegative / FastNegativeV2 for quality (auto-loaded from `server/models/embeddings/`)
- **CLIP skip 2** — Skips last encoder layer for better stylized output
- **Fast generation** — Hyper-SD (8 steps) + DeepCache + FreeU v2 + torch.compile (~2-5s on RTX 4060 after warmup)
- **Post-processing** — Pixelate, K-Means/Octree/MedianCut quantize, CIELAB palette enforcement (KD-Tree), Floyd-Steinberg (Numba JIT) / Bayer dithering, bg removal (BiRefNet)
- **Startup warmup** — Pre-compiles torch + Numba JIT on boot (first real generation is fast)
- **Health check** — `GET /health` for readiness polling
- **Concurrency safe** — GPU access serialized via asyncio lock
- **Cancellation** — Generation stops cleanly on WebSocket disconnect
- **Generation timeout** — Configurable max time per generation (default 5min, auto-scaled for animation)

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

A separate `flash-attn` package is **not required** — PyTorch 2.10 includes FA2 natively via SDP. xformers is superseded by native SDP and provides no benefit on PyTorch ≥ 2.0.

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
| `cancel`             | Cancel in-progress generation      |
| `generate`           | Run single-frame generation        |
| `generate_animation` | Run multi-frame animation          |
| `list_loras`         | List available LoRAs               |
| `list_palettes`      | List available palettes            |
| `list_controlnets`   | List available ControlNet modes    |
| `list_embeddings`    | List available TI embeddings       |

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

### Response Types

| Type                 | Fields                                                              |
|----------------------|---------------------------------------------------------------------|
| `progress`           | `step`, `total`, `frame_index` (opt), `total_frames` (opt)         |
| `result`             | `image` (b64 PNG), `seed`, `time_ms`, `width`, `height`            |
| `animation_frame`    | `frame_index`, `total_frames`, `image` (b64 PNG), `seed`, `time_ms`, `width`, `height` |
| `animation_complete` | `total_frames`, `total_time_ms`, `tag_name` (opt)                  |
| `error`              | `code` (`ENGINE_ERROR`, `OOM`, `CANCELLED`, `TIMEOUT`, `INVALID_REQUEST`, `MAX_CONNECTIONS`, `UNKNOWN_ACTION`), `message` |
| `list`               | `list_type`, `items`                                                |
| `pong`               | (no fields)                                                         |

### Input Validation

| Field       | Constraint           |
|-------------|---------------------|
| `width`     | 64 - 2048 (rounded to ×8) |
| `height`    | 64 - 2048 (rounded to ×8) |
| `steps`     | 1 - 100              |
| `cfg_scale` | 0.0 - 30.0           |
| `clip_skip` | 1 - 12               |
| `denoise`   | 0.0 - 1.0            |
| `lora.weight` | -2.0 - 2.0 (negative LoRA) |
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
| `DEFAULT_CHECKPOINT`       | `Lykon/dreamshaper-8`                 | SD1.5 checkpoint                |
| `HYPER_SD_REPO`            | `ByteDance/Hyper-SD`                  | Hyper-SD HuggingFace repo       |
| `HYPER_SD_LORA_FILE`       | `Hyper-SD15-8steps-CFG-lora.safetensors` | Hyper-SD LoRA filename       |
| `HYPER_SD_FUSE_SCALE`      | `0.8`                                 | Hyper-SD LoRA fusion scale      |
| `DEFAULT_STEPS`            | `8`                                   | Default inference steps         |
| `DEFAULT_CFG`              | `5.0`                                 | Default CFG scale               |
| `DEFAULT_CLIP_SKIP`        | `2`                                   | CLIP skip layers (2 = stylized) |
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

## License

MIT
