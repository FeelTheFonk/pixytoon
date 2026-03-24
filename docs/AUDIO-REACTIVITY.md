# Audio Reactivity Guide

Generate SD-driven animations where inference parameters are modulated in real-time by audio features. This system is inspired by Deforum's audio-reactive workflows, brought to Aseprite.

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Overview

Audio reactivity maps characteristics of an audio file (energy, beats, spectral content) to generation parameters (denoise strength, CFG scale, noise, seed). The result: animations that breathe, pulse, and evolve with the music.

**Architecture:**

```
Audio File (.wav/.mp3/.flac/.ogg)
    |
    v
Audio Analyzer (librosa)
    |  - RMS energy, onset strength, spectral centroid
    |  - Multi-band energy (low/mid/high + sub_bass/upper_mid/presence)
    |  - BPM detection + beat signal
    |  - Waveform preview (100-point RMS)
    |  - Optional: per-stem features (demucs)
    v
Modulation Engine (synth-style matrix)
    |  - Source -> Target routing with min/max range
    |  - Attack/Release EMA smoothing
    |  - Custom math expressions (simpleeval)
    v
Parameter Schedule (per-frame or per-chunk values)
    |
    +--- Frame Chain (img2img from previous frame)
    |      - Frame 0: txt2img / img2img / inpaint / ControlNet
    |      - Frame 1+: motion warp (pan/zoom/rotate/tilt) -> img2img chain with modulated params
    |
    +--- AnimateDiff + Audio (v0.7.3+)
    |      - 16-frame temporal batches with 4-frame overlap
    |      - Per-chunk averaged parameters
    |      - Alpha-blended inter-batch transitions
    |      - Per-frame motion warp (pan/zoom/rotate/tilt) applied post-generation
    |      - FreeInit on first chunk (optional)
    |
    v
Post-Processing Pipeline (+ palette shift) -> Aseprite Timeline
    |
    v  (optional)
MP4 Export (ffmpeg, nearest-neighbor scaling, audio mux)
```

## Quick Start

1. **Connect** to the server
2. Open the **Audio** tab
3. **Select** an audio file (.wav, .mp3, .flac, .ogg)
4. Click **Analyze** — displays duration, frame count, BPM, available features, and auto-selects the recommended preset
5. Optionally set **Max Frames** to limit the number of generated frames (0 = all)
6. Click **AUDIO GEN** — generates an animation frame-by-frame

That's it. The auto-calibration system picks the best preset for your audio.

## Workflow

### 1. Audio Analysis

Click **Analyze** after selecting a file. The server:

- Loads the audio (mono, 22050 Hz)
- Extracts 10 global features (11 with beat detection) normalized to [0, 1]
- Detects BPM via `librosa.beat.beat_track`
- Optionally separates stems (drums, bass, vocals, other) via demucs — adds 8 more features
- Caches results for 24 hours (subsequent analyses of the same file are instant)

The status bar shows: `12.5s | 300 frames | 8 features | 128 BPM`

### 2. Preset Selection

After analysis, the system auto-selects the recommended preset based on audio characteristics. You can override by choosing from the dropdown.

### 3. Generation

Click **AUDIO GEN**. Each frame:
1. Reads modulated parameters from the schedule
2. Resolves the prompt (static or from prompt schedule)
3. Generates via img2img from the previous frame (or txt2img for frame 0)
4. Applies post-processing
5. Imports into Aseprite timeline

The cancel button works at any point — the server acknowledges immediately.

## Modulation Matrix

### Sources

Audio features extracted per frame, normalized to [0, 1]:

| Source | Description | Best For |
|--------|-------------|----------|
| `global_rms` | Overall energy (loudness) | General reactivity |
| `global_onset` | Transient/attack strength | Beat-driven effects |
| `global_centroid` | Spectral brightness | Timbral changes |
| `global_low` | Low-frequency energy (20-300 Hz) | Bass-driven effects |
| `global_mid` | Mid-frequency energy (300-2 kHz) | Melodic content |
| `global_high` | High-frequency energy (2k-16 kHz) | Hi-hat, cymbal reactivity |
| `global_sub_bass` | Sub-bass energy (20-60 Hz) | Deep bass, kick drums |
| `global_upper_mid` | Upper-mid energy (2-4 kHz) | Vocal presence, guitar |
| `global_presence` | Presence energy (4-8 kHz) | Clarity, sibilance |
| `global_beat` | Beat impulse (BPM-aligned) | Rhythmic sync |

With stems enabled (demucs, CPU):

| Source | Description |
|--------|-------------|
| `drums_rms` / `drums_onset` | Drum track energy and attacks |
| `bass_rms` / `bass_onset` | Bass line energy and attacks |
| `vocals_rms` / `vocals_onset` | Vocal track energy and attacks |
| `other_rms` / `other_onset` | Everything else |

### Targets

Inference parameters that can be modulated per-frame:

| Target | Range | Effect |
|--------|-------|--------|
| `denoise_strength` | 0.05 - 0.95 | How much each frame changes from the previous. Higher = more change. |
| `cfg_scale` | 1.0 - 30.0 | How closely the model follows the prompt. Higher = more prompt adherence. |
| `noise_amplitude` | 0.0 - 1.0 | Additive noise injected before generation. Creates visual turbulence. |
| `controlnet_scale` | 0.0 - 2.0 | ControlNet conditioning strength (if using ControlNet mode). |
| `seed_offset` | 0 - 1000 | Offset added to base seed. Creates visual jumps between frames. |
| `palette_shift` | 0.0 - 1.0 | Audio-driven hue rotation. Shifts the color palette per frame. |
| `frame_cadence` | 1 - 8 | Frame skip cadence. Higher = fewer generated frames (GPU savings). |
| `motion_x` | -5.0 - 5.0 | Horizontal camera pan (pixels). Smooth Deforum-like 2D warp. |
| `motion_y` | -5.0 - 5.0 | Vertical camera pan (pixels). Smooth Deforum-like 2D warp. |
| `motion_zoom` | 0.92 - 1.08 | Camera zoom (1.0 = none, >1 = in, <1 = out). Compounds over frames. |
| `motion_rotation` | -2.0 - 2.0 | Camera rotation (degrees). Smooth planar rotation. |
| `motion_tilt_x` | -3.0 - 3.0 | Perspective pitch (degrees). Faux 3D via homography warp. |
| `motion_tilt_y` | -3.0 - 3.0 | Perspective yaw (degrees). Faux 3D via homography warp. |

> **Motion anti-spaghetti**: Motion amplitude is automatically scaled by `denoise_strength` (clamped 0.15-0.8) — lower denoise = less motion. Frame-to-frame deltas are rate-limited per channel (`MOTION_MAX_DELTA`) with total motion budget enforcement. Border replication and Lanczos4 interpolation ensure clean edges.

### Attack / Release

Each slot has attack and release frames for asymmetric EMA smoothing:

- **Attack** (1-30): How fast the parameter responds to rising audio. Low = snappy, high = gradual.
- **Release** (1-60): How fast it returns when audio drops. Low = choppy, high = smooth tails.

Typical values: attack=2, release=8 (responsive but smooth).

## Presets Reference

### Genre-Specific

| Preset | Slots | Best For |
|--------|-------|----------|
| `electronic_pulse` | beat->denoise, onset->cfg, high->noise, beat->motion_zoom | EDM, techno, synth music |
| `rock_energy` | rms->denoise, onset->cfg, low->seed, rms->motion_x | Rock, metal, live instruments |
| `hiphop_bounce` | low->denoise, beat->cfg, onset->noise, low->motion_y | Hip-hop, trap, bass music |
| `classical_flow` | rms->denoise, centroid->cfg, rms->motion_x | Orchestral, piano, acoustic |
| `ambient_drift` | rms->denoise, centroid->cfg, mid->noise, mid->motion_x, rms->motion_zoom | Ambient, drone, meditation |

### Style-Specific

| Preset | Slots | Best For |
|--------|-------|----------|
| `glitch_chaos` | onset->denoise, high->cfg, beat->seed, rms->noise, onset->motion_rot | Glitch art, experimental, aggressive |
| `smooth_morph` | rms->denoise, centroid->cfg, rms->motion_zoom | Gentle transitions, slow evolve |
| `rhythmic_pulse` | beat->denoise, onset->cfg, beat->motion_zoom | Beat-synced pulsing |
| `atmospheric` | rms->denoise, mid->cfg, high->noise, mid->motion_x | Moody, cinematic, textural |
| `abstract_noise` | rms->noise, onset->denoise, centroid->seed, high->cfg, high->motion_rot, onset->motion_x, high->tilt_x | Abstract, generative, noisy |

### Complexity Levels

| Preset | Slots | Best For |
|--------|-------|----------|
| `one_click_easy` | 1 (rms->denoise) | First-time users, simple reactivity |
| `beginner_balanced` | 2 (rms->denoise, onset->cfg) | Good starting point |
| `intermediate_full` | 3+1 (rms->denoise, onset->cfg, low->noise, beat->motion_zoom) | Rich modulation |
| `advanced_max` | 4+3 (all targets including seed, low->motion_x, beat->motion_zoom, low->tilt_x) | Maximum expressiveness |

### Target-Specific

| Preset | Slots | Best For |
|--------|-------|----------|
| `controlnet_reactive` | rms->cn_scale, onset->denoise | ControlNet + audio |
| `seed_scatter` | onset->seed, rms->denoise | Visual variety per beat |
| `noise_sculpt` | rms->noise, onset->denoise, centroid->cfg, rms->motion_zoom | Noise-driven textures |

### Motion / Camera (v0.7.4+)

| Preset | Slots | Best For |
|--------|-------|----------|
| `gentle_drift` | rms->denoise, low->motion_x, mid->motion_y | Slow horizontal/vertical drift |
| `pulse_zoom` | rms->denoise, beat->motion_zoom | Beat-synced zoom pulse |
| `slow_rotate` | rms->denoise, centroid->motion_rotation | Gentle rotation driven by timbre |
| `cinematic_sweep` | rms->denoise, low->motion_x, beat->zoom, centroid->rotation, centroid->tilt_y | Full cinematic camera |
| `cinematic_tilt` | rms->denoise, low->tilt_x, centroid->tilt_y | Perspective tilt from bass/timbre |
| `zoom_breathe` | rms->denoise, rms->motion_zoom | Gentle RMS-driven zoom oscillation |
| `parallax_drift` | rms->denoise, low->motion_x, mid->tilt_x | Parallax effect (pan + tilt) |
| `full_cinematic` | rms->denoise, low->motion_x, mid->motion_y, beat->zoom, centroid->rotation, low->tilt_x | All 6 motion channels |

> Many existing presets (electronic_pulse, rock_energy, hiphop_bounce, ambient_drift, etc.) were enriched with subtle motion in v0.7.4. `cinematic_sweep`, `advanced_max`, and `abstract_noise` were enriched with perspective tilt in v0.9.34. The motion is automatic and smooth — no configuration needed.

### Legacy

| Preset | Description |
|--------|-------------|
| `energetic` | Original v0.7.0 preset (rms->denoise, onset->cfg, rms->motion_x) |
| `ambient` | Original v0.7.0 preset (rms->denoise, centroid->cfg, centroid->motion_x) |
| `bass_driven` | Original v0.7.0 preset (low->denoise, high->cfg, low->motion_y) |

## Auto-Calibration

When you click **Analyze**, the server examines audio characteristics and recommends the best preset:

| Audio Characteristic | Recommended Preset |
|---------------------|-------------------|
| Very quiet, minimal dynamics | `ambient_drift` |
| Fast BPM (>120) + bright spectrum | `electronic_pulse` |
| Fast BPM (>120) + dark spectrum | `hiphop_bounce` |
| High onset + loud peaks | `rock_energy` |
| Bass-heavy | `bass_driven` |
| Strong dynamic variation + onsets | `rhythmic_pulse` |
| Low energy + minimal variation | `classical_flow` |
| Very percussive | `glitch_chaos` |
| Other | `beginner_balanced` |

The recommendation auto-selects in the dropdown. You can always override.

## Custom Expressions

Expressions override slot values with mathematical formulas. Enable via **Advanced > Custom Expressions**.

### Available Functions

`sin`, `cos`, `tan`, `abs`, `min`, `max`, `sqrt`, `exp`, `log`, `pow`, `floor`, `ceil`, `clamp(x, lo, hi)`, `lerp(a, b, t)`, `smoothstep(edge0, edge1, x)`, `where(cond, a, b)`

### Available Variables

| Variable | Description |
|----------|-------------|
| `t` | Frame index (0, 1, 2, ...) |
| `max_f` | Total frame count |
| `fps` | Frames per second |
| `s` | Seconds elapsed (`t / fps`) |
| `bpm` | Detected BPM of the audio |
| `global_rms` | Current frame's RMS value |
| `global_onset` | Current frame's onset value |
| `global_centroid` | Current frame's centroid value |
| `global_low` / `mid` / `high` | Band energies |
| `global_sub_bass` / `upper_mid` / `presence` | Extended band energies |
| `global_beat` | Beat signal |
| *(per-stem vars)* | Available if stems enabled |

### Expression Examples

```
# Denoise pulsing at BPM
0.2 + 0.4 * abs(sin(s * 3.14159 * bpm / 60))

# CFG follows spectral brightness with floor
max(3.0, 5.0 + 4.0 * global_centroid)

# Noise only on beats
where(global_beat > 0.3, 0.5, 0.0)

# Gradual denoise increase over time
lerp(0.15, 0.65, t / max_f)

# Smooth CFG transition based on RMS
smoothstep(0.2, 0.8, global_rms) * 8.0 + 2.0

# Seed offset driven by onset (visual jumps on hits)
floor(global_onset * 500)

# Combined: low freqs drive denoise, high freqs add variance
clamp(0.3 * global_low + 0.2 * global_high, 0.1, 0.8)

# Alternating intensity every 4 beats
0.3 + 0.3 * abs(sin(s * 3.14159 * bpm / 240))

# ControlNet scale breathing with RMS
lerp(0.5, 1.5, global_rms)

# Exponential onset response
0.1 + 0.7 * pow(global_onset, 2.0)

# Motion: gentle horizontal drift following bass
global_low * 3.0 - 1.5

# Motion: zoom pulse on beats (for motion_zoom target)
1.0 + 0.02 * global_beat

# Motion: slow rotation driven by spectral centroid
sin(s * 0.5) * global_centroid * 1.0
```

## AnimateDiff + Audio (v0.7.3)

Frame Chain generates each frame independently from the previous one — no temporal awareness. AnimateDiff + Audio combines AnimateDiff's temporal attention (16-frame window) with audio-driven parameter modulation for superior temporal coherence.

### How It Works

1. The audio timeline is divided into **chunks of 16 frames** with **4-frame overlap**
2. Modulation parameters are **averaged per chunk** (e.g., denoise_strength is the mean over the 16 frames)
3. Each chunk is generated via AnimateDiff with the averaged parameters
4. Overlap frames are **alpha-blended** for smooth inter-chunk transitions
5. FreeInit can be applied to the first chunk for improved initialization

### When to Use

| Method | Best For |
|--------|----------|
| **Frame Chain** | Short clips (<5s), maximum per-frame control, fast iteration |
| **AnimateDiff + Audio** | Longer sequences, temporal coherence, smoother motion |

### Usage

1. In the **Audio** tab, set **Method** to `animatediff`
2. Optionally enable **FreeInit** for the first chunk
3. Click **AUDIO GEN** as usual

### Limitations

- Parameters are constant within each 16-frame chunk (averaged)
- Minimum useful sequence: 16 frames (shorter falls back to single chunk)
- Slightly slower than chain due to temporal attention computation

## MP4 Export (v0.7.3)

Export your audio-reactive animation as an MP4 video with the audio track embedded.

### Requirements

**ffmpeg** must be installed and in your PATH. Download from [ffmpeg.org](https://ffmpeg.org/download.html).

### Usage

After generating an audio-reactive animation, click **Export MP4** in the Audio tab. The button appears after generation completes.

### Quality Presets

| Quality | CRF | ffmpeg Preset | Scale | Best For |
|---------|-----|---------------|-------|----------|
| `web` | 23 | medium | 4x | Social media, small file size |
| `high` | 17 | slow | 4x | Sharing, good quality/size ratio |
| `archive` | 12 | veryslow | 8x | Archival, maximum quality |
| `raw` | 0 | ultrafast | 1x | Lossless, no scaling |

Pixel art is upscaled with **nearest-neighbor interpolation** (no blur). The output includes metadata (prompt, seed) embedded in the MP4.

## Prompt Schedule

Enable **Advanced > Prompt Schedule** to use different prompts for different time ranges.

Format: `start-end` in seconds (e.g., `0-10`) paired with a prompt.

**Example workflow:**
- T1: `0-8` / `serene forest, green canopy, morning light`
- T2: `8-16` / `dark cave, glowing crystals, underground`
- T3: `16-24` / `volcanic landscape, flowing lava, dramatic sky`

The default prompt (from the Generate tab) is used for any time not covered by segments. Overlapping segments use the highest-weight segment.

### Audio-Linked Randomness (v0.7.7)

When the **Randomness** slider (0-20) is set above 0 and no manual prompt segments are defined, SDDj auto-generates varied prompt segments aligned to the music's structure:

- **Onset detection**: Segment boundaries are placed at musical onset peaks (transients, beats)
- **BPM snapping**: Boundaries snap to the nearest beat grid position
- **Subject preservation**: The subject from your base prompt is locked; surrounding descriptors vary

| Randomness | Segments | Effect |
|------------|----------|--------|
| 0          | 0        | Single prompt throughout |
| 1-5        | 2        | Subtle variation between sections |
| 6-10       | 3        | Moderate: three distinct scenes |
| 11-15      | 4-5      | Wild: frequent visual shifts |
| 16-20      | 6-8      | Chaos: rapid scene changes |

Longer audio gets more segments (capped at 12). This creates a "music video" effect where scenes change in sync with the music — without requiring manual prompt scheduling.

**Example**: Base prompt `a fox in a magical forest` with randomness=12 on a 30s track might produce:
- 0-7s: `masterpiece, a fox in a magical forest, ethereal lighting, misty`
- 7-15s: `masterpiece, a fox in a magical forest, neon glow, cyberpunk style`
- 15-22s: `masterpiece, a fox in a magical forest, watercolor, soft pastels`
- 22-30s: `masterpiece, a fox in a magical forest, dramatic storm, dark atmosphere`

## Seed Control

**Fixed seed**: Set a specific seed in the Generate tab — same base for all frames, only seed_offset varies.

**Random seed per frame**: Check "Random seed per frame" — injects a seed_offset expression: `t * 7 + floor(global_rms * 500)`. Each frame gets a unique seed influenced by audio energy.

**Manual control**: Use a modulation slot with `seed_offset` target, or write a custom expression.

## img2img / ControlNet with Audio

The generation mode (txt2img, img2img, inpaint, ControlNet) is set in the **Generate** tab and applies to audio-reactive generation too.

- **txt2img**: Frame 0 generated from scratch, frames 1+ chain via img2img
- **img2img**: Frame 0 uses the active Aseprite layer as source, frames 1+ chain
- **inpaint**: Frame 0 uses source + mask, frames 1+ chain
- **ControlNet**: Frame 0 uses control image, frames 1+ chain via img2img

**Tip**: img2img mode is great for audio-reactive generation starting from an existing image — the animation evolves from your artwork.

## Stem Separation

Optional CPU-based stem separation via demucs (htdemucs model).

**Setup**: `pip install demucs>=4.0`

**Performance**: ~20-60 seconds per minute of audio (first run). Results are cached for 24 hours.

**Available stems**: drums, bass, vocals, other — each provides `_rms` and `_onset` features.

**When to use**: When you want specific instrument reactivity (e.g., drums driving denoise, vocals driving CFG).

## Tips and Best Practices

### Getting Started
- Start with **Analyze + auto-recommended preset** — no configuration needed
- Use `one_click_easy` if you want minimal but visible reactivity
- Use `beginner_balanced` for a good balance of reactivity and stability

### Denoise Strength
- **Low range (0.10-0.30)**: Stable, gradual evolution — good for ambient, backgrounds
- **High range (0.40-0.80)**: Dynamic, each frame distinctly different — good for energetic music
- **Rule of thumb**: Keep max below 0.70 for coherent animations
- **Sub-floor blending** (v0.8.7): When audio modulation drives denoise below the quality floor (≥2 denoising steps), the engine generates at the floor and blends the result toward the source image — preserving full audio dynamic range without quality loss. Quiet passages now produce correctly attenuated visual change instead of being clamped.

### CFG Scale
- **Low range (2-5)**: Dreamy, less prompt-bound — more abstract
- **High range (6-12)**: Prompt-faithful, clear subject — more literal
- **Modulated by onset**: Creates "attention peaks" on beats

### Noise Amplitude
- Use sparingly (0.0-0.3 typical) — too much creates visual chaos
- Best paired with low denoise — adds texture without destroying coherence

### Seed Offset
- Small range (0-100): Subtle frame-to-frame variation
- Large range (0-1000): Dramatic visual jumps, especially on beats

### Motion / Camera
- Motion targets (`motion_x/y/zoom/rotation/tilt_x/tilt_y`) create smooth Deforum-like camera movement
- Perspective tilt (`motion_tilt_x/y`) adds faux 3D pitch/yaw via homography warp
- Motion is **automatically dampened** by denoise strength — low denoise = minimal movement
- Frame-to-frame deltas are **rate-limited** per channel to prevent saccade from audio transients
- Use motion presets (`gentle_drift`, `pulse_zoom`, `slow_rotate`, `cinematic_sweep`) for quick results
- Use tilt presets (`cinematic_tilt`, `parallax_drift`, `full_cinematic`) for perspective effects
- For custom motion, use high attack (4-6) and release (20-30) for ultra-smooth movement
- Combining `motion_zoom` with `global_beat` creates satisfying pulse-zoom effects on beats
- Keep ranges conservative: ±2-3px translation, zoom 0.99-1.01, rotation ±1deg, tilt ±1.5deg for pixel art

### Attack/Release Tuning
- **Fast music**: attack=1, release=3-6 (snappy response)
- **Slow music**: attack=4-8, release=15-30 (gradual, flowing)
- **Percussive**: attack=1, release=1-2 (match transient shape)
- **Ambient**: attack=6-10, release=20-30 (glacial, atmospheric)

### Performance
- Audio analysis: ~2-5 seconds for a typical file (cached after first run)
- Frame generation: ~2-5 seconds per frame at 512x512 (same as regular generation)
- A 10-second clip at 24 FPS = 240 frames = ~8-20 minutes total

### Common Pitfalls
- Don't set denoise max above 0.90 — the animation becomes incoherent
- Don't use all 4 slots targeting the same parameter — they average, reducing range
- Don't forget to analyze before generating — the server needs the analysis cache
- Very short audio clips (<2s) may not provide enough dynamic range for visible modulation

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
