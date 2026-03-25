# SDDj

---

Local SOTA generation and animation for Aseprite via Stable Diffusion + AnimateDiff.

---

## Quick Start

```
setup.ps1         <- One-click: install deps, download models, build extension
start.ps1         <- One-click: launch server + Aseprite
```

## Architecture

See **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** for detailed Mermaid diagrams of the system design, DSP routing, and inference stack.

## Project Structure

- `server/` — Fast FastAPI/PyTorch backend. Handles all heavy lifting (diffusion, DSP, rendering).
- `extension/` — Lightweight Lua scripting API extension running inside Aseprite.

See **[ARCHITECTURE.md](docs/ARCHITECTURE.md#project-structure)** for the complete exhaustive module hierarchy.

## Features

### Generation

| Feature | Version | Description |
|---------|---------|-------------|
| **txt2img / img2img / inpaint** | v0.3.0 | Core generation modes |
| **[ControlNet](docs/GUIDE.md#controlnet)** | v0.3.0 | OpenPose, Canny, Scribble, Lineart (v1.1) |
| **Sequence Output** | v0.6.1 | Output as new layer (default) or new frame in the timeline |
| **[Loop Mode](docs/GUIDE.md#loop-mode)** | v0.4.0 | Continuous generation with random/increment seeds |
| **[Random Loop](docs/GUIDE.md#random-loop)** | v0.5.0 | Auto-randomized prompts; lock subject while randomizing style |
| **[Auto-Prompt Generator](docs/GUIDE.md#auto-prompt-generator)** | v0.4.0 | 9-phase composition engine with curated templates |
| **[Presets](docs/GUIDE.md#presets)** | v0.4.0 | Save/load generation settings; 7 built-in presets |
| **Randomness Slider** | v0.7.7 | 0-20 scale controlling prompt diversity |
| **Contextual Action Button** | v0.7.7 | Single button adapts to active tab (GENERATE / ANIMATE / AUDIO GEN) |
| **Dedicated Pipeline Sliders** | v0.7.7 | Animation and Audio tabs have independent Steps, CFG, Strength |
| **Palette CRUD** | v0.7.9 | Save/delete custom palettes from the UI |

### Animation

| Feature | Version | Description |
|---------|---------|-------------|
| **Frame Chain** | v0.3.0 | img2img chaining for walk cycles, simple loops |
| **AnimateDiff** | v0.3.0 | Motion module (v1-5-3) for temporal consistency, FreeInit support |
| **[AnimateDiff-Lightning](docs/GUIDE.md#animatediff-lightning-v0941)** | v0.9.41 | 10× faster animation via adversarial distillation (2/4/8-step) |

### [Audio Reactivity](docs/AUDIO-REACTIVITY.md)

| Feature | Version | Description |
|---------|---------|-------------|
| **Modulation Matrix** | v0.7.0 | Synth-style source→target routing with attack/release EMA |
| **Audio Features** | v0.9.35 | 34 comprehensive rhythmic and spectral features |
| **[24 Presets](docs/AUDIO-REACTIVITY.md#presets-reference)** | v0.7.1+ | Genre/style/complexity/target/motion, auto-calibration |
| **[Motion/Camera](docs/AUDIO-REACTIVITY.md#motion--camera-v074)** | v0.7.4 | Deforum-like pan/zoom/rotate + perspective tilt (faux 3D) |
| **AnimateDiff + Audio** | v0.7.3 | 16-frame temporal batches with overlap blending |
| **Prompt Scheduling** | v0.7.1 | Per-segment prompts + audio-linked auto-generation |
| **[MP4 Export](docs/AUDIO-REACTIVITY.md#mp4-export-v073)** | v0.7.3 | Nearest-neighbor upscaling, audio mux, quality presets |
| **Stem Separation** | v0.7.0 | Optional CPU stem separation via demucs |

### Pipeline & Quality

| Feature | Description |
|---------|-------------|
| **LoRA stacking** | Hyper-SD (speed) + style LoRA (±2.0 weight range) |
| **Textual Inversion** | EasyNegative auto-loaded from `server/models/embeddings/` |
| **CLIP skip 2** | Better stylized output (configurable 1-12) |
| **[Post-processing](docs/GUIDE.md#post-processing-pipeline)** | Pixelate, quantize, CIELAB palette, dithering, bg removal |
| **Cancellation** | Multi-layered cancel with server ACK + 30s safety timer |
| **Auto-reconnect** | Exponential backoff (2s→30s) + heartbeat watchdog |
| **Concurrency safe** | GPU access serialized via asyncio lock |

## Performance Stack

| Optimization | Role | Impact |
|-------------|------|--------|
| **Hyper-SD LoRA** (8-step CFG, fused at 0.8) | Fewer diffusion steps | ~2-3x faster generation |
| **DeepCache** (interval=3) | Feature caching between steps | ~2.3x additional speedup |
| **FreeU v2** (s1=0.9 s2=0.2 b1=1.5 b2=1.6) | Free quality boost | No speed cost |
| **CLIP skip 2** | Skip last CLIP layer | Better stylized output |
| **torch.compile** (default) | UNet Triton codegen | ~20-30% faster inference |
| **AnimateDiff** (motion adapter v1-5-3) | Temporally consistent animation | Motion module ~97MB |
| **AnimateDiff-Lightning** (optional) | Adversarial distillation (4-step) | ~10x faster AnimateDiff |
| **FreeInit** (optional, 2 iters) | Improved AnimateDiff temporal consistency | ~2x AnimateDiff time |
| **PyTorch SDP** (native, auto-active) | Fused attention kernels (FlashAttention2) | Memory + speed |
| **VAE slicing + tiling** | Batched VAE decode | Lower VRAM peak |
| **fp16 inference** | Half-precision throughout | ~50% VRAM reduction |

> **torch.compile modes:** `default` mode is used by default — fast compilation with Triton codegen.
> `max-autotune` benchmarks every kernel candidate for peak throughput but adds minutes to startup.
> `reduce-overhead` uses CUDAGraphs which is **incompatible** with DeepCache's dynamic
> control flow (skip/compute branches). If you set `SDDJ_COMPILE_MODE=reduce-overhead`,
> you must disable DeepCache (`SDDJ_ENABLE_DEEPCACHE=False`).



## Requirements

> **Windows Only**: SDDj relies on PowerShell 7 and Visual Studio 2022 C++ workloads which restrict operation to Windows environments.

- **GPU**: NVIDIA >= 4GB VRAM (txt2img/img2img/audio at 512x512). 8GB+ recommended for AnimateDiff/ControlNet
- **CUDA**: 12.8
- **Python**: 3.11-3.13
- **uv**: Package manager
- **Visual Studio 2022**: C++ Desktop Development workload (for torch.compile / Triton)

## WebSocket Protocol

See **[API Reference](docs/API-REFERENCE.md)** for the complete WebSocket protocol specification.

## Post-Processing Pipeline

> **Note**: Post-processing is optional and primarily designed for pixel art / retro-gaming styles. For anime, illustration, concept art, or realistic styles, you can disable pixelation and use raw SD output.

Executed in strict order:

1. **Background Removal** — rembg (u2net / birefnet-general / bria-rmbg)
2. **Pixelation** — NEAREST downscale to target size
3. **Color Quantization** — KMeans / Median Cut / Octree
4. **Palette Enforcement** — CIELAB nearest neighbor (KD-Tree)
5. **Dithering** — Floyd-Steinberg (Numba JIT) or Bayer
6. **Alpha Cleanup** — Binary threshold

## Configuration

See **[Configuration](docs/CONFIGURATION.md)** for all `SDDJ_*` environment variables.

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
| **[Audio Reactivity](docs/AUDIO-REACTIVITY.md)** | Audio-driven animation — modulation matrix, presets, expressions, BPM sync |
| **[API Reference](docs/API-REFERENCE.md)** | WebSocket protocol specification |
| **[Configuration](docs/CONFIGURATION.md)** | Environment variables reference |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |

## License

MIT
