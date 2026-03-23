# API Reference

WebSocket protocol specification for SDDj.

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Connection

Connect to `ws://127.0.0.1:9876/ws`. All messages are JSON. Maximum 5 concurrent connections.

## Actions

| Action               | Description                        |
|----------------------|------------------------------------|
| `ping`               | Health check, returns `pong`       |
| `cancel`             | Cancel in-progress generation (server ACK + GPU cleanup) |
| `generate`           | Run single-frame generation        |
| `generate_animation` | Run multi-frame animation          |
| `list_loras`         | List available LoRAs               |
| `list_palettes`      | List available palettes            |
| `save_palette`       | Save a custom palette              |
| `delete_palette`     | Delete a user palette              |
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
| `export_mp4`            | Export frames + audio to MP4 (requires ffmpeg) |
| `shutdown`              | Graceful server shutdown              |

## Generate Prompt Request

```json
{
  "action": "generate_prompt",
  "locked_fields": { "subject": "a pixel cat" },
  "randomness": 10
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `locked_fields` | `object` | `{}` | Categories to keep fixed (e.g. `{"subject": "..."}`) |
| `prompt_template` | `string` | `null` | Custom template with `{category}` placeholders |
| `randomness` | `int 0-20` | `0` | Diversity level: 0=standard, 5=subtle, 10=moderate, 15=wild (rare items + random template), 20=chaos (combines multiple items per category) |

**Response**: `prompt_result` with `prompt`, `negative_prompt`, and `components` dict.

## Generate Request

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

### Inpaint Mode

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

## Animation Request

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
| `method`            | `chain`, `animatediff`, `animatediff_audio` | `chain` |
| `frame_count`       | 2 - 120                              | `8`           |
| `frame_duration_ms` | 50 - 2000                            | `100`         |
| `seed_strategy`     | `fixed`, `increment`, `random`       | `increment`   |
| `tag_name`          | string or null                       | `null`        |
| `enable_freeinit`   | boolean                              | `false`       |
| `freeinit_iterations` | 1 - 3                              | `2`           |

## Real-Time Paint Request

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
  "image": "<base64 PNG - current canvas>",
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

## Audio-Reactive Request

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
  "max_frames": null,
  "method": "chain",
  "enable_freeinit": false,
  "freeinit_iterations": 2,
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
  "prompt_segments": [
    { "start_second": 0, "end_second": 10, "prompt": "forest at dawn" },
    { "start_second": 10, "end_second": 20, "prompt": "castle in the sky" }
  ],
  "randomness": 0,
  "prompt": "fantasy landscape",
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

### Modulation Sources

| Source              | Description                    |
|---------------------|--------------------------------|
| `global_rms`         | Overall energy                 |
| `global_onset`       | Transient / attack strength    |
| `global_centroid`    | Spectral brightness            |
| `global_low`         | Low-frequency energy (20-300Hz)  |
| `global_mid`         | Mid-frequency energy (300-2kHz)  |
| `global_high`        | High-frequency energy (2k-16kHz) |
| `global_sub_bass`    | Sub-bass energy (20-60 Hz)     |
| `global_upper_mid`   | Upper-mid energy (2-4 kHz)     |
| `global_presence`    | Presence energy (4-8 kHz)      |
| `global_beat`        | Beat impulse (BPM-aligned)     |
| `drums_rms/onset`    | Per-stem (requires `enable_stems`) |
| `bass_rms/onset`     | Per-stem (requires `enable_stems`) |
| `vocals_rms/onset`   | Per-stem (requires `enable_stems`) |
| `other_rms/onset`    | Per-stem (requires `enable_stems`) |

### Modulation Targets

| Target              | Range           | Description                |
|----------------------|-----------------|----------------------------|
| `denoise_strength`   | 0.05 - 0.95     | Frame-to-frame change      |
| `cfg_scale`          | 1.0 - 30.0      | Prompt adherence           |
| `noise_amplitude`    | 0.0 - 1.0       | Additive latent noise      |
| `controlnet_scale`   | 0.0 - 2.0       | ControlNet influence       |
| `seed_offset`        | 0 - 1000         | Per-frame seed variation   |
| `palette_shift`      | 0.0 - 1.0        | Audio-driven hue rotation  |
| `frame_cadence`      | 1.0 - 8.0        | Frame skip cadence         |
| `motion_x`           | -5.0 - 5.0       | Horizontal pan (pixels)    |
| `motion_y`           | -5.0 - 5.0       | Vertical pan (pixels)      |
| `motion_zoom`        | 0.95 - 1.05      | Zoom factor (1.0 = none)   |
| `motion_rotation`    | -2.0 - 2.0       | Rotation (degrees)         |

### Audio-Linked Randomness

When `randomness` > 0 and no manual `prompt_segments` are provided, the server auto-generates varied prompt segments aligned to the audio's musical structure (onset peaks, BPM). Each segment receives a unique prompt variation derived from the base prompt's subject.

| `randomness` | Segments | Description |
|--------------|----------|-------------|
| `0`          | 0        | Single prompt throughout (default) |
| `1-5`        | 2        | Subtle: two variations |
| `6-10`       | 3        | Moderate: three scenes |
| `11-15`      | 4-5      | Wild: frequent changes |
| `16-20`      | 6-8      | Chaos: rapid shifts |

Longer audio increases segments proportionally (×1 per extra minute, capped at 12). Boundaries snap to the BPM beat grid when available.

## Palette Management

Save a custom palette:

```json
{
  "action": "save_palette",
  "palette_save_name": "my_palette",
  "palette_save_colors": ["#FF0000", "#00FF00", "#0000FF"]
}
```

Delete a palette:

```json
{
  "action": "delete_palette",
  "palette_save_name": "my_palette"
}
```

| Field                | Type           | Description                        |
|----------------------|----------------|------------------------------------|
| `palette_save_name`  | `string`       | Palette name (alphanumeric, max 256 chars) |
| `palette_save_colors`| `string[]`     | Hex color codes (`#RRGGBB` or `#RGB`)      |

**Responses**: `palette_saved` (with `name`) or `palette_deleted` (with `name`). Both also trigger a `list` response refreshing the palette list.

## Export MP4 Request

Requires ffmpeg in PATH. Export animation frames + audio to a single MP4 file.

```json
{
  "action": "export_mp4",
  "output_dir": "C:/path/to/frames/",
  "audio_path": "C:/path/to/audio.wav",
  "fps": 24.0,
  "scale_factor": 4,
  "quality": "high"
}
```

| Field           | Values                            | Default  |
|-----------------|-----------------------------------|----------|
| `output_dir`    | Path to directory with frame_*.png | required |
| `audio_path`    | Path to audio file (optional)     | `null`   |
| `fps`           | 1.0 - 120.0                      | `24.0`   |
| `scale_factor`  | 1 - 8                            | `4`      |
| `quality`       | `web`, `high`, `archive`, `raw`   | `high`   |

## Response Types

| Type                 | Fields                                                              |
|----------------------|---------------------------------------------------------------------|
| `progress`           | `step`, `total`, `frame_index` (opt), `total_frames` (opt)         |
| `result`             | `image` (b64 PNG), `seed`, `time_ms`, `width`, `height`            |
| `animation_frame`    | `frame_index`, `total_frames`, `image` (b64 PNG), `seed`, `time_ms`, `width`, `height` |
| `animation_complete` | `total_frames`, `total_time_ms`, `tag_name` (opt)                  |
| `error`              | `code`, `message`                                                   |
| `list`               | `list_type`, `items`                                                |
| `pong`               | (no fields)                                                         |
| `realtime_ready`     | `message`                                                           |
| `realtime_result`    | `image` (b64 PNG), `latency_ms`, `frame_id`, `width`, `height`, `roi_x` (opt), `roi_y` (opt) |
| `realtime_stopped`   | `message`                                                           |
| `prompt_result`      | `prompt`, `negative_prompt`, `components`                           |
| `preset`             | `name`, `data`                                                      |
| `preset_saved`       | `name`                                                              |
| `preset_deleted`     | `name`                                                              |
| `palette_saved`      | `name`                                                              |
| `palette_deleted`    | `name`                                                              |
| `cleanup_done`       | `message`, `freed_mb`                                               |
| `audio_analysis`     | `duration`, `total_frames`, `features`, `bpm`, `recommended_preset`, `stems_available`, `stems`, `waveform` (opt) |
| `audio_reactive_frame` | `frame_index`, `total_frames`, `image`, `seed`, `time_ms`, `width`, `height`, `params_used` |
| `audio_reactive_complete` | `total_frames`, `total_time_ms`, `tag_name` (opt)               |
| `stems_available`    | `available`, `message`                                              |
| `modulation_presets` | `presets` (list of names)                                           |
| `export_mp4_complete`| `path`, `size_mb`, `duration_s`                                     |
| `export_mp4_error`   | `message`                                                           |
| `shutdown_ack`       | `message`                                                           |

### Error Codes

| Code                 | Meaning                                  |
|----------------------|------------------------------------------|
| `ENGINE_ERROR`       | Internal generation failure              |
| `OOM`                | CUDA out of memory                       |
| `CANCELLED`          | Generation cancelled by client           |
| `TIMEOUT`            | Generation timed out                     |
| `INVALID_REQUEST`    | Malformed or invalid parameters          |
| `MAX_CONNECTIONS`    | Server connection limit reached (5)      |
| `UNKNOWN_ACTION`     | Unrecognized action type                 |
| `REALTIME_BUSY`      | Live paint session already active        |
| `REALTIME_NOT_ACTIVE`| No active live paint session             |
| `GPU_BUSY`           | GPU is occupied by another operation     |

## Input Validation

| Field       | Constraint           |
|-------------|---------------------|
| `width`     | 64 - 2048 (rounded to x8) |
| `height`    | 64 - 2048 (rounded to x8) |
| `steps`     | 1 - 100              |
| `cfg_scale` | 0.0 - 30.0           |
| `clip_skip` | 1 - 12               |
| `denoise_strength` | 0.0 - 1.0     |
| `lora.weight` | -2.0 - 2.0 (negative LoRA) |
| `target_size` | 8 - 512              |
| `colors`    | 2 - 256              |

## HTTP Endpoints

| Method | Path      | Description       |
|--------|-----------|-------------------|
| `GET`  | `/health` | Server readiness  |

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
