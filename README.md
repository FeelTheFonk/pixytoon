# PixyToon

Local SOTA pixel art generation for Aseprite via Stable Diffusion 1.5.

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
│   └── download_models.py       # Download all models from HuggingFace
├── server/
│   ├── pyproject.toml           # Dependencies (torch CUDA 12.8)
│   ├── run.py                   # Entry point
│   ├── palettes/                # 7 preset palettes (JSON)
│   └── pixytoon/                # Python package
│       ├── config.py            # Pydantic Settings (env vars)
│       ├── protocol.py          # WebSocket schemas (Pydantic)
│       ├── engine.py            # SD1.5 SOTA pipeline
│       ├── server.py            # FastAPI WebSocket + HTTP server
│       ├── postprocess.py       # 6-stage pixel art pipeline
│       ├── rembg_wrapper.py     # Background removal (CPU)
│       ├── lora_manager.py      # LoRA discovery
│       ├── ti_manager.py        # Textual Inversion discovery
│       └── palette_manager.py   # Palette loading
└── docs/                        # Research notes
```

## Features

- **txt2img / img2img / ControlNet** — OpenPose, Canny, Scribble, Lineart (v1.1)
- **LoRA stacking** — Hyper-SD (speed) + pixel art LoRA (style, ±2.0 weight range)
- **Textual Inversion** — EasyNegative / FastNegativeV2 for quality (auto-loaded from `server/models/embeddings/`)
- **CLIP skip 2** — Skips last encoder layer for better stylized output
- **Fast generation** — Hyper-SD (8 steps) + DeepCache + FreeU v2 + torch.compile (~2-5s on RTX 4060 after warmup)
- **Post-processing** — Pixelate, K-Means/Octree/MedianCut quantize, CIELAB palette enforcement (KD-Tree), Floyd-Steinberg (Numba JIT) / Bayer dithering, bg removal (BiRefNet)
- **Startup warmup** — Pre-compiles torch + Numba JIT on boot (first real generation is fast)
- **Health check** — `GET /health` for readiness polling
- **Concurrency safe** — GPU access serialized via asyncio lock
- **Cancellation** — Generation stops cleanly on WebSocket disconnect
- **Generation timeout** — Configurable max time per generation (default 5min)

## Performance Stack

| Optimization | Role | Impact |
|-------------|------|--------|
| **Hyper-SD LoRA** (8-step CFG, fused at 0.8) | Fewer diffusion steps | ~2-3x faster generation |
| **DeepCache** (interval=3) | Feature caching between steps | ~2.3x additional speedup |
| **FreeU v2** (s1=0.9 s2=0.2 b1=1.5 b2=1.6) | Free quality boost | No speed cost |
| **CLIP skip 2** | Skip last CLIP layer | Better stylized output |
| **torch.compile** (default) | UNet Triton codegen | ~20-30% faster inference |
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

- **GPU**: NVIDIA >= 8GB VRAM (tested RTX 4060)
- **CUDA**: 12.8
- **Python**: 3.11-3.13
- **uv**: Package manager
- **Visual Studio 2022**: C++ Desktop Development workload (for torch.compile / Triton)

## WebSocket Protocol

Connect to `ws://127.0.0.1:9876/ws`. All messages are JSON.

### Actions

| Action             | Description                     |
|--------------------|---------------------------------|
| `ping`             | Health check, returns `pong`    |
| `generate`         | Run generation pipeline         |
| `list_loras`       | List available LoRAs            |
| `list_palettes`    | List available palettes         |
| `list_controlnets` | List available ControlNet modes |
| `list_embeddings`  | List available TI embeddings    |

### Generate Request

```json
{
  "action": "generate",
  "prompt": "pixel art character",
  "negative_prompt": "blurry, photorealistic",
  "mode": "txt2img",
  "width": 512,
  "height": 512,
  "seed": -1,
  "steps": 8,
  "cfg_scale": 5.0,
  "denoise_strength": 0.75,
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

### Response Types

| Type       | Fields                                          |
|------------|------------------------------------------------|
| `progress` | `step`, `total`                                 |
| `result`   | `image` (b64 PNG), `seed`, `time_ms`, `width`, `height` |
| `error`    | `code` (`ENGINE_ERROR`, `OOM`, `CANCELLED`, `TIMEOUT`), `message` |
| `list`     | `list_type` ("loras"/"palettes"/"controlnets"/"embeddings"), `items` |
| `pong`     | (no fields)                                     |

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

1. **Background Removal** — rembg with BiRefNet-general (CPU, configurable)
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
| `DEFAULT_QUANTIZE_COLORS`  | `32`                                  | Default palette colors          |
| `DEFAULT_PIXEL_LORA_WEIGHT`| `1.0`                                 | Default LoRA fuse weight        |
| `DEFAULT_NEGATIVE_TI`      | `auto`                                | Auto-load TI embeddings         |
| `ENABLE_TORCH_COMPILE`     | `True`                                | UNet compilation (requires Triton + MSVC) |
| `COMPILE_MODE`             | `default`                             | torch.compile mode (`default` / `max-autotune` / `reduce-overhead`) |
| `ENABLE_DEEPCACHE`         | `True`                                | DeepCache acceleration          |
| `ENABLE_FREEU`             | `True`                                | FreeU v2 quality boost          |
| `ENABLE_ATTENTION_SLICING` | `True`                                | Attention slicing (PyTorch < 2.0 fallback) |
| `ENABLE_VAE_TILING`        | `True`                                | VAE tiling for large images     |
| `ENABLE_WARMUP`            | `True`                                | Warmup generation at startup    |
| `GENERATION_TIMEOUT`       | `300.0`                               | Max seconds per generation      |
| `REMBG_MODEL`              | `birefnet-general`                    | Background removal model        |
| `REMBG_ON_CPU`             | `True`                                | Run rembg on CPU                |

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
| CUDAGraphs tensor overwrite    | Fixed in v0.1.0: uses `max-autotune` mode. If using `reduce-overhead`, disable DeepCache |
| Generation timed out           | Increase `PIXYTOON_GENERATION_TIMEOUT` or reduce steps/resolution |
| LoRA change is slow            | Expected: LoRA weight change triggers recompilation (~30-60s once) |

## License

MIT
