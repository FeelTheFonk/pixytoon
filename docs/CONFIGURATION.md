# Configuration

All environment variables are prefixed with `PIXYTOON_`. Example: `PIXYTOON_PORT=8080`.

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Network

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `9876` | Server port |

## Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `server/models` | Root models directory |
| `CHECKPOINTS_DIR` | `server/models/checkpoints` | SD checkpoint cache |
| `LORAS_DIR` | `server/models/loras` | LoRA files (.safetensors) |
| `EMBEDDINGS_DIR` | `server/models/embeddings` | Textual Inversion embeddings |
| `PALETTES_DIR` | `server/palettes` | Palette JSON files |
| `PRESETS_DIR` | `server/presets` | Generation presets |
| `PROMPTS_DATA_DIR` | `server/data/prompts` | Auto-prompt generator data |

## Model

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CHECKPOINT` | `Lykon/dreamshaper-8` | SD1.5 checkpoint |
| `HYPER_SD_REPO` | `ByteDance/Hyper-SD` | Hyper-SD HuggingFace repo |
| `HYPER_SD_LORA_FILE` | `Hyper-SD15-8steps-CFG-lora.safetensors` | Hyper-SD LoRA filename |
| `HYPER_SD_FUSE_SCALE` | `0.8` | Hyper-SD LoRA fusion scale |
| `DEFAULT_PIXEL_LORA` | `auto` | Default pixel LoRA (`auto` = first found, `""` = none) |
| `DEFAULT_PIXEL_LORA_WEIGHT` | `1.0` | Default LoRA fuse weight |

## Generation Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_STEPS` | `8` | Default inference steps |
| `DEFAULT_CFG` | `5.0` | Default CFG scale |
| `DEFAULT_WIDTH` | `512` | Default output width |
| `DEFAULT_HEIGHT` | `512` | Default output height |
| `DEFAULT_CLIP_SKIP` | `2` | CLIP skip layers (2 = stylized) |

## Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TORCH_COMPILE` | `True` | UNet compilation (requires Triton + MSVC) |
| `COMPILE_MODE` | `default` | torch.compile mode (`default` / `max-autotune` / `reduce-overhead`) |
| `ENABLE_DEEPCACHE` | `True` | DeepCache acceleration |
| `ENABLE_FREEU` | `True` | FreeU v2 quality boost |
| `ENABLE_ATTENTION_SLICING` | `True` | Attention slicing (PyTorch < 2.0 fallback) |
| `ENABLE_VAE_TILING` | `True` | VAE tiling for large images |
| `ENABLE_WARMUP` | `True` | Warmup generation at startup |
| `DEEPCACHE_INTERVAL` | `3` | DeepCache skip interval |
| `DEEPCACHE_BRANCH` | `0` | DeepCache branch ID |
| `FREEU_S1` | `0.9` | FreeU v2 skip scale 1 |
| `FREEU_S2` | `0.2` | FreeU v2 skip scale 2 |
| `FREEU_B1` | `1.5` | FreeU v2 backbone scale 1 |
| `FREEU_B2` | `1.6` | FreeU v2 backbone scale 2 |
| `REMBG_MODEL` | `u2net` | Background removal model (u2net / birefnet-general / bria-rmbg) |
| `REMBG_ON_CPU` | `True` | Run rembg on CPU |

## Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_TIMEOUT` | `600.0` | Max seconds per generation |
| `REALTIME_TIMEOUT` | `300.0` | Auto-stop if no frame for N seconds |

## Animation

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_ANIM_FRAMES` | `8` | Default animation frame count |
| `DEFAULT_ANIM_DURATION_MS` | `100` | Default frame duration (ms) |
| `DEFAULT_ANIM_DENOISE` | `0.30` | Default animation denoise |
| `MAX_ANIMATION_FRAMES` | `120` | Max frames per animation |
| `ANIMATEDIFF_MODEL` | `guoyww/animatediff-motion-adapter-v1-5-3` | AnimateDiff motion adapter |
| `ENABLE_FREEINIT` | `False` | FreeInit for AnimateDiff |
| `FREEINIT_ITERATIONS` | `2` | FreeInit iteration count |

## Real-Time Paint

| Variable | Default | Description |
|----------|---------|-------------|
| `REALTIME_DEFAULT_STEPS` | `4` | Default realtime inference steps |
| `REALTIME_DEFAULT_CFG` | `2.5` | Default realtime CFG scale |
| `REALTIME_DEFAULT_DENOISE` | `0.5` | Default realtime denoise strength |
| `REALTIME_ROI_PADDING` | `32` | Padding around ROI crop (pixels) |
| `REALTIME_ROI_MIN_SIZE` | `64` | Minimum ROI dimension (pixels) |

## Audio

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_CACHE_DIR` | `""` (temp dir) | Cache directory for audio analysis |
| `AUDIO_MAX_FILE_SIZE_MB` | `500` | Max audio file size (MB) |
| `AUDIO_MAX_FRAMES` | `3600` | Max frames per audio animation |
| `AUDIO_DEFAULT_FPS` | `24.0` | Default audio analysis FPS |
| `AUDIO_DEFAULT_ATTACK` | `2` | Default EMA attack frames |
| `AUDIO_DEFAULT_RELEASE` | `8` | Default EMA release frames |
| `STEM_MODEL` | `htdemucs` | Demucs model for stem separation |
| `STEM_DEVICE` | `cpu` | Stem separation device (always CPU) |
| `FFMPEG_PATH` | `""` (auto-detect) | Path to ffmpeg binary for MP4 export |

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
