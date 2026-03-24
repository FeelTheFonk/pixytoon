# Configuration

All environment variables are prefixed with `SDDJ_`. Example: `SDDJ_PORT=8080`.

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Network

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_HOST` | `127.0.0.1` | Server bind address |
| `SDDJ_PORT` | `9876` | Server port |

## Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_MODELS_DIR` | `server/models` | Root models directory |
| `SDDJ_CHECKPOINTS_DIR` | `server/models/checkpoints` | SD checkpoint cache |
| `SDDJ_LORAS_DIR` | `server/models/loras` | LoRA files (.safetensors) |
| `SDDJ_EMBEDDINGS_DIR` | `server/models/embeddings` | Textual Inversion embeddings |
| `SDDJ_PALETTES_DIR` | `server/palettes` | Palette JSON files |
| `SDDJ_PRESETS_DIR` | `server/presets` | Generation presets |
| `SDDJ_PROMPTS_DATA_DIR` | `server/data/prompts` | Auto-prompt generator data |

## Model

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_DEFAULT_CHECKPOINT` | `Lykon/dreamshaper-8` | SD1.5 checkpoint |
| `SDDJ_HYPER_SD_REPO` | `ByteDance/Hyper-SD` | Hyper-SD HuggingFace repo |
| `SDDJ_HYPER_SD_LORA_FILE` | `Hyper-SD15-8steps-CFG-lora.safetensors` | Hyper-SD LoRA filename |
| `SDDJ_HYPER_SD_FUSE_SCALE` | `0.8` | Hyper-SD LoRA fusion scale |
| `SDDJ_DEFAULT_STYLE_LORA` | `auto` | Default style LoRA (`auto` = first found, `""` = none) |
| `SDDJ_DEFAULT_STYLE_LORA_WEIGHT` | `1.0` | Default LoRA fuse weight |

## Generation Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_DEFAULT_STEPS` | `8` | Default inference steps |
| `SDDJ_DEFAULT_CFG` | `5.0` | Default CFG scale |
| `SDDJ_DEFAULT_WIDTH` | `512` | Default output width |
| `SDDJ_DEFAULT_HEIGHT` | `512` | Default output height |
| `SDDJ_DEFAULT_CLIP_SKIP` | `2` | CLIP skip layers (2 = stylized) |

## Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_ENABLE_TORCH_COMPILE` | `True` | UNet compilation (requires Triton + MSVC) |
| `SDDJ_COMPILE_MODE` | `default` | torch.compile mode (`default` / `max-autotune` / `reduce-overhead`) |
| `SDDJ_ENABLE_DEEPCACHE` | `True` | DeepCache acceleration |
| `SDDJ_ENABLE_FREEU` | `True` | FreeU v2 quality boost |
| `SDDJ_ENABLE_ATTENTION_SLICING` | `True` | Attention slicing (PyTorch < 2.0 fallback) |
| `SDDJ_ENABLE_VAE_TILING` | `True` | VAE tiling for large images |
| `SDDJ_ENABLE_WARMUP` | `True` | Warmup generation at startup |
| `SDDJ_DEEPCACHE_INTERVAL` | `3` | DeepCache skip interval |
| `SDDJ_DEEPCACHE_BRANCH` | `0` | DeepCache branch ID |
| `SDDJ_FREEU_S1` | `0.9` | FreeU v2 skip scale 1 |
| `SDDJ_FREEU_S2` | `0.2` | FreeU v2 skip scale 2 |
| `SDDJ_FREEU_B1` | `1.5` | FreeU v2 backbone scale 1 |
| `SDDJ_FREEU_B2` | `1.6` | FreeU v2 backbone scale 2 |
| `SDDJ_REMBG_MODEL` | `u2net` | Background removal model (u2net / birefnet-general / bria-rmbg) |
| `SDDJ_REMBG_ON_CPU` | `True` | Run rembg on CPU |

## Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_GENERATION_TIMEOUT` | `600.0` | Max seconds per generation |

## Animation

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_MAX_ANIMATION_FRAMES` | `120` | Max frames per animation |
| `SDDJ_ANIMATEDIFF_MODEL` | `guoyww/animatediff-motion-adapter-v1-5-3` | AnimateDiff motion adapter |
| `SDDJ_ENABLE_FREEINIT` | `False` | FreeInit for AnimateDiff |
| `SDDJ_FREEINIT_ITERATIONS` | `2` | FreeInit iteration count |

## Audio

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_AUDIO_CACHE_DIR` | `""` (temp dir) | Cache directory for audio analysis |
| `SDDJ_AUDIO_MAX_FILE_SIZE_MB` | `500` | Max audio file size (MB) |
| `SDDJ_AUDIO_MAX_FRAMES` | `3600` | Max frames per audio animation |
| `SDDJ_AUDIO_DEFAULT_ATTACK` | `2` | Default EMA attack frames |
| `SDDJ_AUDIO_DEFAULT_RELEASE` | `8` | Default EMA release frames |
| `SDDJ_STEM_MODEL` | `htdemucs` | Demucs model for stem separation |
| `SDDJ_STEM_DEVICE` | `cpu` | Stem separation device (always CPU) |
| `SDDJ_FFMPEG_PATH` | `""` (auto-detect) | Path to ffmpeg binary for MP4 export |

## Temporal Coherence

| Variable | Default | Description |
|----------|---------|-------------|
| `SDDJ_DISTILLED_STEP_SCALE_CAP` | `2` | Max step multiplier for distilled models (Hyper-SD). E.g. cap=2 means steps=8 can scale to at most 16 |
| `SDDJ_COLOR_COHERENCE_STRENGTH` | `0.5` | LAB color matching between consecutive frames. 0 = disabled, 0.3-0.7 recommended. Prevents color drift in chains |
| `SDDJ_AUTO_NOISE_COUPLING` | `True` | Auto noise-denoise coupling (Deforum pattern). Less denoise → more stabilizing noise for smoother transitions |
| `SDDJ_OPTICAL_FLOW_BLEND` | `0.0` | Optical flow temporal blending strength. 0 = disabled, 0.1-0.3 recommended. Adds ~10-20ms per frame |

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
