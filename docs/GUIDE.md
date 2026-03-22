# SDDj User Guide

> From first launch to advanced generation — everything you need to create images and animations with Stable Diffusion in Aseprite.

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [First Launch](#first-launch)
- [Connection](#connection)
- [How It Works](#how-it-works)
- [Modes](#modes)
- [Output Mode](#output-mode)
- [Generation Parameters](#generation-parameters)
- [Post-Processing Pipeline](#post-processing-pipeline)
- [LoRA and Models](#lora-and-models)
- [ControlNet](#controlnet)
- [Loop Mode](#loop-mode)
- [Auto-Prompt Generator](#auto-prompt-generator)
- [Presets](#presets)
- [Seeds and Reproducibility](#seeds-and-reproducibility)
- [Audio Reactivity](#audio-reactivity)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before using SDDj, make sure you have:

- **Aseprite** (v1.3+) — compiled or purchased
- **NVIDIA GPU** with at least 8 GB VRAM (10 GB for AnimateDiff + ControlNet)
- **`setup.ps1` already run** — this installs all dependencies and downloads models (~10 GB)

> [!NOTE]
> If you haven't run `setup.ps1` yet, do it first. It handles everything: Python environment, model downloads, extension install.

---

## First Launch

1. **Run `start.ps1`** at the project root
2. A terminal opens — the server loads the SD model (~30s first time)
3. Aseprite launches automatically once the server is ready
4. In Aseprite: **File > Scripts > SDDj** (compiled) or it appears in Extensions (purchased)
5. Click **Connect** in the SDDj dialog

The status bar shows "Connected" and the available resources (LoRAs, palettes, embeddings) load automatically.

> [!TIP]
> If Aseprite is already open, the server still starts. Just run the script manually via File > Scripts.

> [!NOTE]
> If the server connection drops, SDDj automatically reconnects with exponential backoff (2s, 4s, 8s... up to 30s). You don't need to click Connect again.

```mermaid
sequenceDiagram
    participant User
    participant start.ps1
    participant Server
    participant Aseprite

    User->>start.ps1: Run
    start.ps1->>Server: Launch (uv run python run.py)
    start.ps1->>start.ps1: Poll /health every 3s
    Server-->>start.ps1: {"status": "ok", "loaded": true}
    start.ps1->>Aseprite: Launch aseprite.exe
    User->>Aseprite: File > Scripts > SDDj
    Aseprite->>Server: WebSocket connect
    Server-->>Aseprite: pong + resources
```

---

## Connection

The top section of the SDDj dialog manages the server connection.

| Control | What it does |
|---------|-------------|
| **Server** | WebSocket URL (default: `ws://127.0.0.1:9876/ws`) |
| **Status** | Current connection state and generation progress |
| **Connect / Disconnect** | Toggle the server connection |
| **Refresh Resources** | Re-fetch LoRAs, palettes, embeddings, and presets from the server |
| **Cleanup GPU** | Free GPU VRAM and run garbage collection (only when idle) |

### Auto-Reconnect

If the server connection drops unexpectedly (crash, network issue), SDDj automatically reconnects using exponential backoff: 2s, 4s, 8s, 16s, up to a maximum of 30s between attempts. The status bar shows the countdown and attempt number.

Clicking **Disconnect** manually disables auto-reconnect. Clicking **Connect** re-enables it.

### Heartbeat Watchdog

A heartbeat ping is sent every 30 seconds. If the server doesn't respond with a pong within 90 seconds (3× the interval), the connection is considered dead and auto-reconnect triggers. This detects silent server crashes that don't properly close the WebSocket.

---

## How It Works

SDDj is a bridge between Aseprite and a local Stable Diffusion server running on your GPU.

```mermaid
flowchart LR
    A[Aseprite<br>Lua extension] -->|WebSocket<br>JSON| B[SDDj Server<br>Python FastAPI]
    B --> C[Stable Diffusion 1.5<br>+ Hyper-SD acceleration]
    C --> D[Post-Processing<br>pipeline (optional)]
    D -->|base64 PNG| A
```

- You describe what you want (prompt) and configure parameters in the dialog
- The server generates an image using Stable Diffusion 1.5
- The image goes through a post-processing pipeline
- The result appears as a new layer (or new frame in sequence mode) in your Aseprite sprite

Everything runs **locally on your machine**. No cloud, no API key, no internet needed after setup.

---

## Modes

The SDDj dialog has **5 tabs**: Generate, Post-Process, Animation, Live, and Audio. The **Generate tab** includes a **Mode** dropdown with 7 generation modes: txt2img, img2img, inpaint, and 4 ControlNet variants (openpose, canny, scribble, lineart).

> [!NOTE]
> The Strength slider is hidden in txt2img mode (it doesn't apply). In other modes, the Mode label shows a hint: "Mode (needs mask)" for inpaint, "Mode (needs layer)" for img2img and ControlNet.

### Generate (txt2img)

Creates a new image from a text prompt alone.

- Select mode **txt2img** in the Generate tab
- Type your prompt (what you want to see)
- Click **GENERATE**
- The result appears as a new layer

This is the simplest mode. Use it when you're starting from scratch.

### Img2Img

Transforms an existing image based on your prompt. Uses your **active layer** as the source.

- Draw something on a layer (even rough shapes or colors)
- Select mode **img2img**
- Adjust **Strength** (denoise):
  - `0.3` — subtle changes, keeps most of your drawing
  - `0.5` — balanced transformation
  - `0.8` — heavy reinterpretation, mostly SD-generated
  - `1.0` — completely ignores your input (same as txt2img)
- Click **GENERATE**

> [!TIP]
> Img2Img is powerful for iterating. Generate once with txt2img, then refine with img2img at low strength.

### Inpaint

Regenerates a specific area while keeping the rest intact.

The mask (what to repaint) is detected automatically in this order:

1. **Active selection** — if you have a selection in Aseprite, it becomes the mask
2. **"Mask" layer** — a layer named `Mask` or `mask` where white = repaint
3. **Active layer alpha** — any non-transparent pixel on the active layer

White = repaint, Black = keep.

### Animation

Generates multi-frame animations. Two methods available:

| Method | How it works | Best for |
|--------|-------------|----------|
| **Chain** | Generates frame 1, then uses each frame as img2img input for the next | Walk cycles, simple loops, controlled motion |
| **AnimateDiff** | Uses a motion module for temporal consistency across all frames at once | Fluid motion, complex animations |

Parameters in the Animation tab:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| Frames | 8 | Number of frames to generate |
| Duration | 100ms | Time per frame in the animation |
| Strength | 0.30 | How much each frame changes (chain) or overall denoise (AnimateDiff) |
| Seed Mode | increment | `fixed` = same seed, `increment` = seed+1 per frame, `random` = random per frame |
| Tag Name | (empty) | Creates an Aseprite tag for the animation range |
| FreeInit | off | AnimateDiff only — improves temporal consistency (doubles generation time) |

### Live Paint

Real-time SD-assisted painting. See [the dedicated Live Paint guide](LIVE-PAINT.md).

---

## Output Mode

The **Output** dropdown in the Generate tab controls where results are placed:

| Mode | Behavior |
|------|----------|
| **layer** (default) | Each result creates a new layer on the current frame |
| **sequence** | Each result creates a new frame in the timeline (like animation output) |

**When to use sequence mode:**

- **Loop + img2img**: See each iteration as a timeline frame — scrub through to compare
- **Rapid txt2img exploration**: Generate 20 variations and review them as an animation
- **Reference sheets**: Stack character poses in the timeline

When the loop ends (or you cancel), sequence frames are finalized with 100ms duration each. The sequence layer is named `SDDj Seq #<seed>`.

> [!TIP]
> Sequence mode is especially powerful with Loop Mode. Set output to "sequence", enable Loop, and each generation becomes a new frame you can scrub through in Aseprite's timeline.

---

### Loop Mode

Enable **Loop Mode** to continuously generate images with the same settings.

1. Check the "Loop Mode" checkbox
2. Choose a **Loop Seed** mode: `random` (new seed each time) or `increment` (seed +1 each iteration)
3. Optionally set **Output** to `sequence` to place each result as a timeline frame
4. Click **Generate**
5. Images generate one after another automatically
6. Click **Cancel** to stop the loop (partial results are kept)

### Random Loop

Enable **Random Loop** alongside Loop Mode for fully automated creative exploration.
Each iteration generates a new random prompt before generating the image.

1. Check "Loop Mode" **and** "Random Loop"
2. Optionally enable **Lock Subject** and enter a fixed subject (e.g., "warrior character")
3. Click **Generate**
4. Each iteration: random prompt is generated → image is generated → repeat
5. Click **Cancel** to stop

Lock Subject keeps your chosen subject constant while randomizing style, mood,
lighting, camera angle, and other creative elements.

### Auto-Prompt Generator

Click **Randomize** to generate a creative prompt from curated templates.
The generator combines quality tags, subjects, styles, lighting, moods,
and camera angles for diverse results.

**Lock Subject**: Check the "Lock Subject" checkbox and enter a subject
(e.g., "armored knight" or "pixel art cat") to keep it fixed while randomizing all other fields.
This is useful for exploring variations of the same character or object.

### Presets

Save and load generation settings as presets.

- Select a preset from the dropdown to load its settings
- Click **Save** to save current settings with a custom name
- Click **Del** to remove a user-created preset

Built-in presets: pixel_art, anime, character, landscape, concept_art, illustration, realistic.

---

## Generation Parameters

These are the core parameters that control what the model generates.

### Prompt

Describes what you want. Be specific and descriptive. SDDj works with any style — pixel art, anime, illustration, concept art, realistic, watercolor, abstract, and more.

**General example:**
```
a majestic dragon soaring over mountains, fantasy art, dramatic lighting
```

**Anime/illustration example:**
```
anime illustration, magical forest, glowing mushrooms, ethereal lighting
```

**Pixel art style example:**
```
pixel art, game sprite, warrior character, sword, shield, sharp pixels
```

The default negative prompt blocks common quality issues (blurry, deformed, low quality, artifacts). You can add to it but rarely need to replace it. For pixel art specifically, consider adding `antialiased, smooth gradients` to the negative prompt.

> [!TIP]
> Start your prompt with a style keyword (e.g. `pixel art`, `anime`, `watercolor`) if you have a matching style LoRA. This anchors the style before adding subject details.

### Steps

How many iterations the model performs. More steps = more detail, but slower.

| Steps | Use case |
|-------|----------|
| 4 | Live Paint (real-time speed) |
| 6-8 | Pixel art — fast and effective with Hyper-SD |
| 8-12 | Anime / illustration — good detail with stylized models |
| 10-15 | Realistic / concept art — extra detail for complex scenes |
| 20+ | Rarely needed with Hyper-SD — diminishing returns |

With Hyper-SD enabled (default), 8 steps produces results comparable to 25+ steps on a standard pipeline. The step counts above are guidelines — experiment to find what works best for your checkpoint and style LoRA.

### CFG Scale

"Classifier-Free Guidance" — how strictly the model follows your prompt.

| CFG | Effect |
|-----|--------|
| 1-3 | Very creative / loose interpretation (Live Paint default: 2.5) |
| 3-5 | Balanced — follows prompt while remaining natural |
| **5.0** | **Default** |
| 7-10 | Strict prompt following, can look over-saturated |
| 15+ | Extreme — artifacts and burnt colors, avoid |

### Denoise Strength

Only relevant for img2img, inpaint, and animation. Controls how much the model changes the input.

| Strength | Effect |
|----------|--------|
| 0.1-0.2 | Barely changes anything — subtle color/light adjustments |
| 0.3 | Light transformation — keeps composition, changes details |
| 0.5 | Balanced — recognizable source with significant SD changes |
| 0.7-0.8 | Heavy transformation — the model dominates, source is a vague guide |
| **1.0** | **Default** — full generation (effectively txt2img) |

### Clip Skip

Controls which CLIP encoder layer interprets your prompt.

| Value | Best for |
|-------|----------|
| 1 | Realistic / photorealistic checkpoints, detailed literal prompts |
| **2** | **Stylized, anime, illustration, pixel art** (default) — works well for most SD 1.5 models |
| 3+ | Very abstract interpretation (experimental) |

Most SD 1.5 models trained on anime, illustration, or stylized content expect clip_skip=2. If you switch to a realistic checkpoint (e.g., Realistic Vision), try clip_skip=1 for better results.

### Seed

Controls the random starting point. Same seed + same parameters = same result.

| Value | Behavior |
|-------|----------|
| **-1** | **Random seed** (default) — different result each time |
| Any number | Fixed seed — reproducible result |

The actual seed used is shown in the status bar after generation and in the layer name.

### Size

The generation resolution. Higher = more detail but slower and more VRAM.

| Size | Use case |
|------|----------|
| 32x32 — 96x96 | Tiny assets (below server minimum 64x64 — may be rejected) |
| 128x128 | Small assets |
| 256x256 | Medium assets |
| 384x384 | Medium-large assets |
| **512x512** | **Default** — best quality/speed ratio for SD 1.5 (all styles) |
| 512x768 / 768x512 | Rectangular formats (portraits / landscapes) |
| 768x768 | Large scenes (needs more VRAM) |

> [!NOTE]
> 512x512 is the native resolution for SD 1.5 and works well for all styles — pixel art, anime, illustration, concept art, and realistic. For small pixel art output (32x32, 48x48), generate at 512x512 and use the post-processing **Pixelate** to downscale to your target size.

> [!WARNING]
> SD 1.5 was trained on 512x512. Going above 768 often produces duplicated compositions or artifacts. For large scenes, generate at 512x512 and upscale in Aseprite.

---

## Post-Processing Pipeline

> **Note**: Post-processing (pixelation, color quantization, dithering, palette mapping) is primarily designed for pixel art and retro-gaming styles. For other styles (anime, illustration, realistic, concept art), you can leave post-processing at defaults or disable pixelation entirely by setting the target size to 512 (same as input).

After generation, the image passes through a 6-stage post-processing pipeline. Each stage is optional and configured in the **Post-Process** tab.

```mermaid
flowchart LR
    A[SD Output<br>512x512] --> B[Background<br>Removal]
    B --> C[Pixelate<br>NEAREST]
    C --> D[Color<br>Quantize]
    D --> E[Palette<br>Enforce]
    E --> F[Dithering]
    F --> G[Alpha<br>Cleanup]
    G --> H[Final<br>Output]
```

### 1. Background Removal

Removes the background, leaving only the subject on a transparent layer. Uses the `u2net` model running on CPU (so it doesn't compete with the GPU for VRAM).

Enable via the **Remove BG** checkbox.

### 2. Pixelation

Downscales the image to your target pixel art size using **NEAREST neighbor** interpolation (no anti-aliasing — mandatory for clean pixels).

| Target Size | Result |
|-------------|--------|
| 8 | Extreme low-res, abstract (minimum) |
| 32 | Very chunky, retro (NES-era) |
| 64 | Classic pixel art |
| **128** | **Default** — good detail while still reading as pixel art |
| 256 | Detailed pixel art, almost painterly |
| 512 | No pixelation (effectively disabled) |

> [!TIP]
> The target size is the longest edge. A 512x512 image at target 128 becomes 128x128 with correct aspect ratio.

### 3. Color Quantization

Reduces the number of unique colors in the image.

| Method | Speed | Quality | Best for |
|--------|-------|---------|----------|
| **KMeans** | Medium | Best color grouping | Default — most accurate palette extraction |
| Median Cut | Fast | Good | Quick iterations, large images |
| Octree | Fast | Good | Alternative to Median Cut |

**Colors** (2-256): How many colors to keep. Classic pixel art uses 8-32 colors.

### 4. Palette Enforcement

Forces all colors to match a specific palette using CIELAB color distance (perceptually accurate).

| Mode | Description |
|------|-------------|
| **Auto** | No enforcement — keeps quantized colors as-is |
| Preset | Uses one of the built-in palettes |
| Custom | Uses hex codes you provide (e.g., `#FF0000 #00FF00 #0000FF`) |

<details>
<summary>Built-in palettes</summary>

| Palette | Colors | Style |
|---------|--------|-------|
| **PICO-8** | 16 | Fantasy console, vibrant and limited |
| **NES** | 54 | Nintendo Entertainment System |
| **SNES** | 256 | Super Nintendo |
| **Game Boy** | 4 | Green monochrome |
| **C64** | 16 | Commodore 64 |
| **Endesga 32** | 32 | Modern pixel art, warm tones |
| **Endesga 64** | 64 | Extended modern pixel art |

</details>

> [!TIP]
> Palette presets are designed for retro-gaming aesthetics. For non-pixel-art styles (anime, illustration, realistic, concept art), use **palette_mode=Auto** or disable palette mapping to preserve the model's full color range.

### 5. Dithering

Simulates intermediate colors using dot patterns. Applied after palette enforcement.

| Mode | Effect |
|------|--------|
| **None** | No dithering (clean flat colors) — default |
| Floyd-Steinberg | Error-diffusion dithering — smooth gradients, organic feel |
| Bayer 2x2 | Ordered pattern — retro, Game Boy style |
| Bayer 4x4 | Larger ordered pattern — classic pixel art dithering |
| Bayer 8x8 | Large pattern — very visible, stylistic choice |

> [!NOTE]
> Floyd-Steinberg dithering is accelerated via Numba JIT. The first run compiles the kernel (~2s), subsequent runs are near-instant.

### 6. Alpha Cleanup

Automatically binarizes the alpha channel: pixels are either fully opaque or fully transparent. This is essential for pixel art and useful for any style where clean edges are desired.

Only applies when Remove BG is enabled.

---

## LoRA and Models

### What is a LoRA?

A LoRA (Low-Rank Adaptation) is a small file that adjusts the SD model's style without replacing it entirely. Think of it as a "style filter" you can mix in.

SDDj uses two types of LoRAs:

1. **Hyper-SD** (built-in, permanent) — A **speed LoRA** that accelerates generation. It is style-neutral and does not bias toward any particular aesthetic. It's what allows 8-step generation to look as good as 25+ steps. You don't need to manage this.

2. **Style LoRA** (user-configurable, optional) — Steers the model toward a specific visual style. Examples: pixel art LoRA, anime LoRA, watercolor LoRA, etc. Style LoRAs are **not required** — the base checkpoint already produces good results for many styles. Placed in `server/models/loras/`.

### Using LoRAs

- Drop `.safetensors` files in `server/models/loras/`
- They appear in the **LoRA** dropdown in the Generate tab
- **(default)** uses the first LoRA found automatically
- **Weight** controls intensity: `1.0` = full effect, `0.5` = half, `0.0` = disabled, negative = inverts

> [!WARNING]
> Changing LoRA or weight triggers a model recompilation (~30-60s) because torch.compile needs to rebuild the computation graph. This only happens once per LoRA/weight combination.

### Textual Inversion Embeddings

Embeddings like **EasyNegative** improve quality by encoding complex "what to avoid" concepts in a single token.

- Drop `.safetensors` or `.pt` files in `server/models/embeddings/`
- Enable via **Neg. Embeddings** checkbox
- They're added to the negative prompt automatically

### Changing Checkpoint

Edit `server/.env` to change the base model:

```
SDDJ_DEFAULT_CHECKPOINT=Lykon/dreamshaper-8
```

Any SD 1.5-compatible checkpoint works. The model downloads from HuggingFace on first launch if not cached.

---

## ControlNet

ControlNet modes let you guide the model using a reference image from your active layer.

| Mode | Input | Best for |
|------|-------|----------|
| **controlnet_canny** | Edge-detected image | Converting clean line art to finished artwork |
| **controlnet_scribble** | Rough sketch | Transforming quick sketches into detailed images |
| **controlnet_openpose** | Pose stick figure | Character poses (draw a simple skeleton) |
| **controlnet_lineart** | Line drawing | Colorizing and rendering line drawings in any style |

How to use:

1. Draw your guide on a layer (edges, sketch, pose, or line art)
2. Make sure that layer is **active**
3. Select the corresponding ControlNet mode
4. Write a prompt describing the desired result
5. Click **GENERATE**

> [!NOTE]
> ControlNet models are lazy-loaded — the first time you use a mode, it downloads the model (~700 MB). Subsequent uses are instant. ControlNet needs ~10 GB total VRAM.

---

## Seeds and Reproducibility

### Reproducing a result

After generation, the status bar shows: `Done (2450ms, seed=1234567890)`.
The layer is also named `SDDj #1234567890`.

To reproduce: enter that seed number in the Seed field, keep all other parameters identical, and generate again. The result will be pixel-for-pixel identical.

### Exploring variations

- **Same seed, different prompt:** Keeps the composition but changes the subject/style
- **Adjacent seeds** (seed, seed+1, seed+2): Similar but slightly different compositions
- **Same seed, slightly different CFG:** Varies how strictly the prompt is followed
- **Same seed, different strength (img2img):** Controls how much of the original is preserved

---

## Performance

### First launch vs. subsequent runs

| What | First time | After warmup |
|------|-----------|-------------|
| Model loading | ~30s | ~10s (cached) |
| torch.compile | ~30s (compiles UNet) | Instant (cached between sessions*) |
| Numba JIT | ~2s (compiles Floyd-Steinberg) | Instant (cached) |
| Generation (512x512, 8 steps) | ~5-8s | ~2-4s |

*torch.compile cache persists across sessions if the model hasn't changed.

### What each optimization does

| Feature | Effect | Can disable? |
|---------|--------|-------------|
| **Hyper-SD** | 8 steps instead of 25+ | No (built-in) |
| **DeepCache** | Caches features between steps (~2x faster) | Yes: `SDDJ_ENABLE_DEEPCACHE=false` |
| **FreeU v2** | Better quality at no speed cost | Yes: `SDDJ_ENABLE_FREEU=false` |
| **torch.compile** | Triton codegen for UNet (~20-30% faster) | Yes: `SDDJ_ENABLE_TORCH_COMPILE=false` |
| **Attention slicing** | Reduces VRAM peak | Yes: `SDDJ_ENABLE_ATTENTION_SLICING=false` |
| **VAE tiling** | Handles large images without OOM | Yes: `SDDJ_ENABLE_VAE_TILING=false` |

### VRAM usage

| Operation | Approximate VRAM |
|-----------|-----------------|
| Idle (model loaded) | ~4 GB |
| Generate 512x512 | ~6 GB |
| Generate 768x768 | ~8 GB |
| AnimateDiff 8 frames | ~8-10 GB |
| AnimateDiff + ControlNet | ~10+ GB |

> [!WARNING]
> If you hit OOM (Out of Memory), reduce resolution first. If that's not enough, disable `torch.compile` — it trades VRAM for speed.

---

## Audio Reactivity

> v0.7.0 — Synth-style modulation matrix. v0.7.1 — BPM detection, 20 presets, auto-calibration, prompt schedule. v0.7.3 — New bands (sub-bass, upper-mid, presence), new targets (palette shift, frame cadence), AnimateDiff + Audio mode, MP4 export, waveform preview. v0.7.4 — Audio-reactive motion/camera (smooth Deforum-like pan/zoom/rotate), frame limit control, 4 motion presets, 14 presets enriched with motion. v0.7.7 — Contextual action button, universal randomize, randomness slider (0-20), dedicated per-pipeline sliders, audio-linked randomness (auto-generates varied prompt segments from music structure).

The **Audio** tab drives generation parameters from audio features. Select a file, click **Analyze** (auto-detects BPM, shows waveform preview, and recommends a preset), then click the action button (shows **AUDIO GEN** when the Audio tab is active). The Audio tab has its own dedicated Steps, CFG, and Strength sliders. Supports all modes (txt2img, img2img, inpaint, ControlNet) and both animation methods:

- **Frame Chain**: Traditional img2img chaining — fast, per-frame control
- **AnimateDiff + Audio**: 16-frame temporal batches with overlap blending — superior coherence for longer sequences

After generating, click **Export MP4** to create a video with the audio track embedded (requires ffmpeg).

> [!TIP]
> **v0.7.7**: The new **randomness slider** (0-20) and **audio-linked randomness** let audio features drive prompt variation. The system auto-generates varied prompt segments from the music's structure, creating unique visuals that evolve with the song.

For the complete guide — modulation matrix, all 24 presets, custom expressions, motion/camera, prompt scheduling, tips: see **[Audio Reactivity Guide](AUDIO-REACTIVITY.md)**.

---

## Troubleshooting

For the complete troubleshooting reference, see **[Troubleshooting](TROUBLESHOOTING.md)**.

Common Aseprite-specific issues:

| Problem | Solution |
|---------|----------|
| **"Connection failed"** | Is the server running? Auto-reconnect will retry automatically |
| **Black image** | Check your prompt. Try a simple test like `a red dragon, fantasy art` or `pixel art, character` |
| **Blurry / not pixelated** | For pixel art: enable Pixelate in Post-Process, set target size to 64-128. For other styles, blurriness is normal at low step counts — try increasing Steps |

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
