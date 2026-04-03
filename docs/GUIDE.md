# SDDj User Guide

## Interface

The dialog has **4 tabs** and an adaptive action button (**GENERATE** / **ANIMATE** / **AUDIO GEN**).

| Tab | Purpose |
|-----|---------|
| **Generate** | Mode, prompt, parameters, LoRA, IP-Adapter, ControlNet guidance |
| **Post-Process** | Background removal, pixelation, quantization, palette, dithering |
| **Animation** | Chain / AnimateDiff, frame count, prompt schedule, interpolation |
| **Audio** | Audio file, analysis, modulation matrix, presets, expressions, MP4 export |

**Connection panel** at the top: server URL (`ws://127.0.0.1:9876/ws`), status, connect/disconnect, refresh resources, GPU cleanup. Auto-reconnect with exponential backoff (2s-30s). Heartbeat every 30s.

---

## Generation Modes

| Mode | Input | Best for |
|------|-------|----------|
| `txt2img` | Text only | Starting from scratch |
| `img2img` | Active layer | Transforming existing artwork |
| `inpaint` | Active layer + mask | Fixing/adding specific regions |
| `controlnet_canny` | Edge-detected layer | Converting clean lineart |
| `controlnet_scribble` | Rough sketch layer | Transforming quick sketches |
| `controlnet_openpose` | Pose stick figure | Character poses |
| `controlnet_lineart` | Line drawing layer | Colorizing line drawings |
| `controlnet_qrcode` | QR code / pattern | Embedding QR codes, optical illusions |

**Inpaint mask detection** (priority order): active selection > layer named `Mask` > active layer alpha.

**ControlNet**: draw guide on a layer, make it active, select mode, prompt, generate. Models lazy-load on first use (~700 MB). Requires ~8 GB VRAM.

---

## Parameters

### Core

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Steps** | 8 | 1-100 | Iterations. 6-8 pixel art, 8-12 anime, 10-15 realistic. Hyper-SD makes 8 match 25+ standard |
| **CFG Scale** | 5.0 | 0.0-30.0 | Prompt adherence. 1-3 creative, 3-5 balanced, 7-10 strict |
| **Denoise Strength** | 1.0 | 0.0-1.0 | Input preservation (img2img/inpaint/anim). 0.3 subtle, 0.5 balanced, 1.0 = full |
| **Clip Skip** | 2 | 1-12 | 2 for stylized/anime/pixel art, 1 for realistic |
| **Seed** | -1 | -1 or int | -1 = random. Same seed + params = same result |
| **Size** | 512x512 | 64-2048 | SD 1.5 native is 512x512. Above 768 risks duplicated compositions |
| **Guidance Rescale** | 0 | 0.0-1.0 | Oversaturation control at high CFG |
| **CN Guidance Start** | 0.0 | 0.0-1.0 | When ControlNet conditioning begins (all CN modes) |
| **CN Guidance End** | 1.0 | 0.0-1.0 | When ControlNet conditioning stops (all CN modes) |

### Scheduler

| Scheduler | Best For |
|-----------|----------|
| **DPM++ SDE Karras** (default) | General purpose, pixel art |
| DPM++ 2M Karras | Smooth gradients, illustration |
| Euler Ancestral | Creative variation |
| Euler | Fast, deterministic |
| UniPC | Speed-critical workflows |
| DDIM | Legacy deterministic |
| LMS | Smooth textures |

All use `timestep_spacing="trailing"` for Hyper-SD compatibility.

### Seed Techniques

| Technique | Effect |
|-----------|--------|
| Same seed, different prompt | Keeps composition, changes subject/style |
| Adjacent seeds (seed+1, +2) | Similar but slightly different compositions |
| Same seed, different CFG | Varies prompt adherence |
| Same seed, different strength | Controls input preservation (img2img) |

Actual seed shown in status bar and layer name (`SDDj #<seed>`).

---

## Post-Processing Pipeline

Six optional stages, primarily designed for pixel art. For non-pixel-art styles, leave pixelation off and colors at 256.

| # | Stage | Settings | Notes |
|---|-------|----------|-------|
| 1 | **Background Removal** | `Remove BG` checkbox | u2net on CPU (no VRAM cost) |
| 2 | **Pixelate** | Target 8-512 px (longest edge) | **Nearest** = sharp point-sampling. **Box** = area averaging. **PixelOE** = contrast-aware (best edges). Defaults: 32 retro, 64 classic, 128 standard |
| 3 | **Color Quantize** | Method + color count (2-256) | **KMeans** = best grouping. **Median Cut** = fast. **Octree** = tree-based. **Octree OKLAB** = perceptually uniform. Classic pixel art: 8-32 colors |
| 4 | **Palette Enforce** | Auto / Preset / Custom hex | OKLAB perceptual distance matching |
| 5 | **Dithering** | Algorithm selection | **None**, **Floyd-Steinberg** (Numba-accelerated), **Bayer 2x2/4x4/8x8**. OKLAB-aware, alpha-aware (transparent BG untouched) |
| 6 | **Alpha Cleanup** | Automatic with Remove BG | Binarizes alpha: fully opaque or fully transparent |

### Built-in Palettes

| Palette | Colors | Character |
|---------|--------|-----------|
| **PICO-8** | 16 | Fantasy console, vibrant, recognizable |
| **Game Boy** | 4 | Green monochrome, maximum constraint |
| **NES** | 54 | Classic 8-bit warmth |
| **SNES** | 256 | Full 16-bit range |
| **C64** | 16 | Earthy, muted, nostalgic |
| **Endesga 32** | 32 | Modern pixel art standard, warm, versatile |
| **Endesga 64** | 64 | Extended Endesga |

> [!TIP]
> **PICO-8** and **Endesga 32** are the most versatile starting points.

**Custom palette**: set palette to Custom, paste hex codes: `#1a1c2c #5d275d #b13e53 ...`
Saved palettes appear alongside built-in presets (stored as JSON in `server/palettes/`).

---

## Animation

### Chain Animation

Each frame is generated via img2img from the previous frame. Frame 0 uses txt2img (or the active layer for img2img/ControlNet modes).

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Frames | 8 | 2-256 | Frame count |
| Duration | 100 ms | 50-2000 ms | Time per frame |
| Strength | 0.30 | 0.0-1.0 | Frame-to-frame change. 0.20-0.35 for coherent motion |
| Seed Mode | increment | fixed / increment / random | Seed strategy per frame |
| Tag Name | (empty) | text | Aseprite tag for the animation range |

### AnimateDiff

Temporal motion module generates all frames at once with temporal attention for fluid, coherent motion.

**AnimateDiff-Lightning** (default): ByteDance's distilled model. 4 steps, CFG 2.0, EulerDiscrete scheduler. Max 32 frames (FreeNoise incompatible with Lightning).

| Feature | Purpose |
|---------|---------|
| **FreeInit** | Improves temporal consistency (doubles gen time). Incompatible with Lightning |
| **FreeNoise** | Sliding window for long sequences on non-Lightning models. Context length 16, stride 4 |
| **FreeU** | Quality enhancement. Defaults: b1=1.2, b2=1.4, s1=0.9, s2=0.2 (tuned for pixel art) |

### Frame Interpolation

Post-generation interpolation: **None / 2x / 3x / 4x**. RIFE optical flow primary, `Image.blend` linear fallback if RIFE unavailable. Install `rife-ncnn-vulkan-python` for proper quality.

---

## Audio Reactivity

### Quick Start

1. Open **Audio** tab, select audio file (.wav, .mp3, .flac, .ogg, .m4a, .aac)
2. Click **Analyze** -- displays duration, BPM, features, auto-selects recommended preset
3. Click **AUDIO GEN**

Audio tab has its own Steps (8), CFG (5.0), and Strength (0.50) sliders. Supports all generation modes and both animation methods.

### Analysis Pipeline

Audio is loaded mono at 44100 Hz. Extraction yields **34 features per frame** normalized [0, 1] with K-weighting (ITU-R BS.1770). BPM detected via librosa (or madmom RNN). Results cached 24h (config-aware invalidation).

### Modulation Matrix

**Sources** (all normalized [0, 1]):

| Category | Sources |
|----------|---------|
| **Energy** | `global_rms` (K-weighted loudness) |
| **Transient** | `global_onset` (SuperFlux attacks) |
| **Spectral** | `global_centroid` (brightness), `spectral_contrast`, `spectral_flatness`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flux` |
| **Rhythmic** | `global_beat` (BPM-aligned impulse) |
| **9-band frequency** | `sub_bass` (20-60Hz), `bass` (60-150), `low_mid` (150-400), `mid` (400-2k), `upper_mid` (2-4k), `presence` (4-8k), `brilliance` (8-12k), `air` (12-20k), `ultrasonic` (20-22k) |
| **Compat aliases** | `global_low` (sub+bass+low_mid avg), `global_high` (presence+brilliance+air+ultra avg) |
| **Chromagram** | `chroma_C` through `chroma_B` (12 pitch classes), `chroma_energy` |

All prefixed `global_`. Per-stem features available with demucs (`drums_*`, `bass_*`, `vocals_*`, `other_*`).

**Targets:**

| Target | Range | Effect |
|--------|-------|--------|
| `denoise_strength` | 0.20-0.95 | Frame change intensity |
| `cfg_scale` | 1.0-30.0 | Prompt adherence |
| `noise_amplitude` | 0.0-1.0 | Additive latent noise (visual turbulence) |
| `controlnet_scale` | 0.0-2.0 | ControlNet conditioning strength |
| `seed_offset` | 0-1000 | Per-frame seed variation |
| `palette_shift` | 0.0-1.0 | Hue rotation |
| `frame_cadence` | 1-8 | Frame skip (higher = fewer frames) |
| `lora_weight` | 0.0-2.0 | LoRA influence |

Each slot has **Attack** (1-30, response speed) and **Release** (1-60, decay speed) EMA smoothing, plus an **Invert** toggle.

### Motion Targets

| Target | Range | Effect |
|--------|-------|--------|
| `motion_x` | -5.0 to 5.0 | Horizontal pan (pixels) |
| `motion_y` | -5.0 to 5.0 | Vertical pan (pixels) |
| `motion_zoom` | 0.92-1.08 | Zoom (1.0 = none, >1 = in) |
| `motion_rotation` | -2.0 to 2.0 | Rotation (degrees) |
| `motion_tilt_x` | -3.0 to 3.0 | Perspective pitch (homography warp) |
| `motion_tilt_y` | -3.0 to 3.0 | Perspective yaw (homography warp) |

> [!NOTE]
> Motion is auto-dampened by denoise strength and rate-limited per channel. Border replication + Lanczos4 interpolation. Conservative pixel art ranges: +/-2-3 px, zoom 0.99-1.01, rotation +/-1 deg, tilt +/-1.5 deg.

### Presets

| Preset | Style | Key Mapping |
|--------|-------|-------------|
| `one_click_easy` | Minimal | rms->denoise |
| `beginner_balanced` | Starting point | rms->denoise, onset->cfg |
| `rhythmic_pulse` | Beat-synced | beat->denoise, onset->cfg, beat->zoom |
| `bass_driven` | Low-end reactive | low->denoise, high->cfg, low->motion_y |
| `electronic_pulse` | EDM / techno | beat->denoise, onset->cfg, high->noise, beat->zoom |
| `rock_energy` | Rock / metal | rms->denoise, onset->cfg, low->seed, rms->motion_x |
| `hiphop_bounce` | Hip-hop / trap | low->denoise, beat->cfg, onset->noise, low->motion_y |
| `classical_flow` | Orchestral | rms->denoise, centroid->cfg, rms->motion_x |
| `ambient_drift` | Drone / ambient | rms->denoise, centroid->cfg, mid->noise, rms->zoom |
| `glitch_chaos` | Experimental | onset->denoise, high->cfg, beat->seed, rms->noise, onset->rotation |
| `smooth_morph` | Gentle transitions | rms->denoise, centroid->cfg, rms->zoom |
| `atmospheric` | Cinematic | rms->denoise, mid->cfg, high->noise, mid->motion_x |
| `cinematic_sweep` | Full camera | rms->denoise, low->pan, beat->zoom, centroid->rotation+tilt |
| `full_cinematic` | All 6 motion channels | rms->denoise, low->x, mid->y, beat->zoom, centroid->rotation, low->tilt |
| `spectral_sculptor` | Timbral sculpting | flatness->noise, contrast->denoise, bandwidth->cfg |
| `voyage_psychedelic` | Full psychedelic | 8 slots: all motion + palette_shift + chroma |

Additional presets: `abstract_noise`, `controlnet_reactive`, `seed_scatter`, `noise_sculpt`, `gentle_drift`, `pulse_zoom`, `slow_rotate`, `cinematic_tilt`, `zoom_breathe`, `parallax_drift`, `tonal_drift`, `ultra_precision`, `micro_reactive`, `intelligent_drift`, `reactive_pause`, plus 4 voyage variants.

### Expressions

Math formulas that override slot values. Enable via **Advanced > Custom Expressions**.

**Variables**: `t` (frame), `max_f` (total frames), `fps`, `s` (seconds = t/fps), `bpm`, plus all audio features (`global_rms`, `global_onset`, etc.).

**Functions**: `sin`, `cos`, `abs`, `min`, `max`, `sqrt`, `pow`, `clamp(x,lo,hi)`, `lerp(a,b,t)`, `smoothstep(e0,e1,x)`, `remap(x,a,b,c,d)`, `where(cond,a,b)`, `easeIn/Out/InOut`, `bounce`, `elastic`, `pingpong`, `hash1d`, `smoothnoise`, and more.

**Examples:**
```
0.2 + 0.4 * abs(sin(s * 3.14159 * bpm / 60))     # Denoise pulsing at BPM
where(global_beat > 0.3, 0.5, 0.0)                # Noise only on beats
1.0 + 0.02 * global_beat                           # Zoom pulse on beats
lerp(0.15, 0.65, t / max_f)                        # Gradual denoise increase
```

**30 expression presets** across 5 categories (rhythmic, temporal, spectral, easing, camera) via the Expr Preset dropdown. **7 camera choreography presets** (orbit, dolly zoom, crane, wandering voyage, etc.) via Camera Journey.

### AnimateDiff + Audio

Combines AnimateDiff temporal attention with audio modulation for superior temporal coherence.

1. Timeline divided into **16-frame chunks** with **4-frame overlap**
2. Modulation parameters **averaged per chunk**
3. Overlaps **alpha-blended** for smooth transitions

Set Method to `animatediff` in the Audio tab. Minimum 16 frames. Best for longer sequences.

### Stems

Optional CPU-based stem separation via demucs (`htdemucs`). Install: `uv add "demucs>=4.0"`. ~20-60s per minute of audio, cached 24h. Produces 4 stems (drums, bass, vocals, other), each with all 34 features.

### MP4 Export

Click **Export MP4** after generating. Requires ffmpeg in PATH.

| Quality | CRF | Scale | Best for |
|---------|-----|-------|----------|
| `web` | 23 | 4x | Social media |
| `high` | 17 | 4x | Sharing |
| `archive` | 12 | 8x | Maximum quality |
| `raw` | 0 | 1x | Lossless |

Pixel art upscaled with nearest-neighbor. Metadata (prompt, seed) embedded.

---

## Prompt Scheduling

Define keyframes to evolve prompts over time. Works in **all generation modes** (Generate, Animation, Audio).

```
[0]
pixel art forest, morning light

[5]
blend: 3
pixel art ocean, sunset

[100%]
pixel art volcano, dramatic sky
```

**Timing**: `[12]` (frame), `[2.5s]` (seconds), `[50%]` (percentage). **Transitions**: `blend: N` for crossfade. **Negative**: `--` separator. **Per-keyframe overrides**: `weight`, `denoise_strength`, `cfg_scale`, `steps`.

See **[Prompt Schedule DSL](PROMPT_SCHEDULE_DSL.md)** for the full grammar, all 8 transition types (hard_cut, blend, linear_blend, ease_in/out, ease_in_out, cubic, slerp), and validation rules.

### Built-in Presets

| Preset | Structure | Transitions |
|--------|-----------|-------------|
| `evolving_3act` | 3 keyframes | All hard_cut |
| `style_morph_4` | 4 keyframes | 3 blends (2-frame window) |
| `beat_alternating` | 2 keyframes | All hard_cut |
| `slow_drift` | 2 keyframes | 1 blend (4-frame window) |
| `rapid_cuts_6` | 6 keyframes | All hard_cut |

Presets are structural (empty prompts). Fill manually or use **Auto-Fill** (server-side PromptGenerator).

### Random Schedule Generator

The **Random** button generates a complete prompt schedule in one click. 7 profiles: `gentle` (2-3 KF, smooth), `dynamic` (3-5 KF, mixed), `rhythmic` (4-6 KF, hard cuts), `cinematic` (3-4 KF, ease curves), `dreamy` (2-3 KF, slerp), `chaos` (5-8 KF, all types), `minimal` (2 KF, single blend). Controlled by the **Randomness** slider (0-20) and **Lock Subject** checkbox.

---

## LoRA & Models

Hyper-SD is a speed LoRA fused permanently into the pipeline. Style-neutral, enables 8-step generation. No user action needed.

### Style LoRA

1. Drop `.safetensors` in `server/models/loras/`
2. Select from the **LoRA** dropdown
3. **Weight**: 1.0 = full, 0.5 = half, 0.0 = disabled, negative = invert

Multi-LoRA stacking: the server supports loading multiple LoRAs. Weight independently.

> [!NOTE]
> LoRA changes trigger torch.compile recompilation (~30-60s once per combination) unless `SDDJ_ENABLE_LORA_HOTSWAP=True` (default).

### Textual Inversion Embeddings

Drop `.safetensors` or `.pt` in `server/models/embeddings/`. Enable via **Neg. Embeddings** checkbox.

### Changing Checkpoint

Set `SDDJ_DEFAULT_CHECKPOINT` in `server/.env`. Any SD 1.5-compatible checkpoint works. See [REFERENCE.md](REFERENCE.md#configuration) for all model env vars.

---

## IP-Adapter

Conditions generation on a **reference image** instead of (or alongside) text. Load an image and select a mode:

| Mode | Effect | Best for |
|------|--------|----------|
| **full** | Transfers subject + style + composition | Reproducing a reference faithfully |
| **style** | Transfers only visual style | Applying a look to a different subject |
| **composition** | Transfers only spatial layout | Maintaining structure with new content |

**Scale** (0.0-2.0): strength of image conditioning. Start at 0.5-0.7. Lazy-loaded on first use. Requires ~2 GB additional VRAM.

---

## Recipes

### Pixel Art Parameter Matrix

| Intent | Pixelate | Colors | Extras |
|--------|----------|--------|--------|
| Character sprite | 64 | 16-24 | Remove BG |
| Tiny icon (32x32) | 32 | 8-12 | Remove BG, CFG 6-7 |
| Enemy / monster | 48-64 | 12-20 | Remove BG |
| UI element | 48-64 | 6-10 | Custom palette |
| Side-scroller BG | 128-192 | 32-48 | 768x512, Bayer 4x4 |
| Top-down tile | 64-128 | 16-24 | Same palette for set |
| Interior / room | 128 | 24-32 | Floyd-Steinberg |
| Hi-fi pixel art | 192-256 | 48-64 | Steps 10-12, Floyd-Steinberg |
| Retro Game Boy | 32-64 | 4 | Bayer 2x2, GB palette |
| Retro NES | 64 | 12-16 | NES palette |
| Retro PICO-8 | 64-128 | 16 | PICO-8 palette |

For **non-pixel-art** (anime, illustration, realistic): pixelate OFF, colors 256, no quantize, no dither. Increase steps to 10-15, adjust clip_skip (1 for realistic, 2 for anime).

### NPC Sprite Variations

Use a fixed seed and change one prompt detail at a time:
```
Seed 42: pixel art, townsperson, blue shirt    <- base
Seed 42: pixel art, townsperson, red shirt     <- same composition, different color
Seed 43: pixel art, townsperson, blue shirt    <- different pose
```

### ControlNet Workflows

| Workflow | Mode | Prompt tip |
|----------|------|-----------|
| Lineart to pixel art | `controlnet_canny` | `pixel art, colored game sprite` |
| Sketch to sprite | `controlnet_scribble` | Most forgiving mode |
| Pose to character | `controlnet_openpose` | Include action/pose in prompt |
| Coloring lineart | `controlnet_lineart` | `colored version, vibrant colors, flat shading` |

### QR Code Art

Use `controlnet_qrcode` with: Steps 20, CFG 7-10, CN conditioning scale 1.3-1.5, CN guidance end 0.7-0.85. Control image should be high-contrast QR (black on white, centered, quiet zone). Works well with architectural/landscape prompts. Test scannability after generation.

### Color Control Tips

- Quantize to 8-16 colors first, **then** enforce palette
- Add dithering (Floyd-Steinberg) after enforcement for smooth gradients
- When colors shift too much: increase color count (48+), then enforce
- Use a consistent palette across an entire tileset

---

## Performance

| Feature | Effect | Env var to disable |
|---------|--------|--------------------|
| **Hyper-SD** | 8 steps = 25+ standard | Built-in, cannot disable |
| **DeepCache** | Feature reuse (~2x faster) | `SDDJ_ENABLE_DEEPCACHE=false` |
| **FreeU v2** | Quality boost, zero speed cost | `SDDJ_ENABLE_FREEU=false` |
| **torch.compile** | Triton UNet codegen (~20-30% faster) | `SDDJ_ENABLE_TORCH_COMPILE=false` |
| **TF32** | Ampere+ ~15-30% free speedup | `SDDJ_ENABLE_TF32=false` |
| **Attention slicing** | Reduces VRAM peak | `SDDJ_ENABLE_ATTENTION_SLICING=false` |
| **VAE tiling** | Large images without OOM | `SDDJ_ENABLE_VAE_TILING=false` |
| **UNet quantization** | fp8 (Ada) / int8 (Ampere) | `SDDJ_ENABLE_UNET_QUANTIZATION=false` |
| **LoRA hotswap** | No recompile on LoRA switch | `SDDJ_ENABLE_LORA_HOTSWAP=false` |
| **Frame compression** | In-memory animation frame compression | `SDDJ_ENABLE_FRAME_COMPRESSION=false` |
| **EquiVDM noise** | Temporally coherent noise | `SDDJ_EQUIVDM_NOISE=false` |

See [REFERENCE.md](REFERENCE.md#configuration) for all environment variables.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Black image | Check prompt. Test: `a red dragon, fantasy art` |
| Blurry / not pixelated | Enable Pixelate in Post-Process (target 64-128) |
| Wrong colors | Try palette enforcement or adjust `quantize_colors` |
| Generation timed out | Increase `SDDJ_GENERATION_TIMEOUT` (default 600s) or reduce steps/resolution |
| CUDA OOM | Reduce resolution. Disable torch.compile if needed |
| torch.compile fails | Install VS 2022 with C++ Desktop Development. Ensure Triton installed |
| AnimateDiff OOM | Needs ~8-10 GB VRAM. Reduce frame count or resolution |
| Chain animation flicker | Lower strength to 0.20-0.35 |
| Animation too jittery | Enable `SDDJ_OPTICAL_FLOW_BLEND=0.2`, lower denoise (0.15-0.25) |
| Color drift in long chains | Increase `SDDJ_COLOR_COHERENCE_STRENGTH` (0.3-0.7) |
| LoRA not found | Place `.safetensors` in `server/models/loras/` |
| Audio modulation too subtle | Increase min/max range or use wider preset (`glitch_chaos`) |
| Audio modulation too aggressive | Increase release frames, decrease max range |
| MP4 export fails | Install ffmpeg and ensure it is in PATH |
| Cancel does not stop | Server ACK + 30s safety timer auto-unlocks |
| Slow first generation | Normal: torch.compile + Numba JIT warm up (~30-60s) |
| "Not enough SMs" warning | Harmless Triton warning on consumer GPUs -- ignore |
| Size > 768 duplicates faces | Generate at 512, upscale in Aseprite |
| CFG > 10 artifacts | Keep 3-7. Sweet spot ~5.0 with Hyper-SD |
| High img2img strength | Start at 0.3-0.4 for preservation |
| Dithering looks noisy | Set a palette or reduce colors (8-32) first |
