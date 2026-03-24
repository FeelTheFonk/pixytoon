# Changelog

All notable changes to SDDj are documented here.

## [0.9.3] — 2026-03

### Fixed
- **Post-processing applied unconditionally** — color quantization (KMeans 32 colors) ran on every image even when disabled, silently degrading output quality
- `PixelateSpec.enabled` defaulted to `True` in the protocol model while the Lua UI defaulted to `false` — mismatch caused hidden pixelation when presets omitted the field

### Added
- `quantize_enabled` flag in `PostProcessSpec` — explicit opt-in for color quantization (default: `false`)
- "Quantize Colors" checkbox in post-process UI tab
- Fast-path bypass in `postprocess.apply()` — returns image untouched if no processing flags are active
- Preset loading/saving for `quantize_enabled` state
- 3 new unit tests: passthrough identity, default-spec passthrough, quantize-disabled color preservation

### Changed
- **Default output is now raw SD quality** — zero compression, zero color limitation unless explicitly enabled
- All 7 preset JSON files updated with explicit `quantize_enabled` field

---

## [0.9.2] — 2026-03

### Fixed
- Version drift: all manifests synchronized (were stuck at 0.8.9 since v0.8.8)
- `AudioReactiveRequest` field ordering normalized (fields declared before validators)
- `frame_duration_ms` lower bound inconsistency unified to 30ms across all request models
- Missing palette save count limit added (max 100, parity with presets)
- `round8` edge case: clamp minimum to 8 (prevent 0-size from tiny inputs)
- Hue shift guard for 0-pixel images
- Export handler uses typed attribute access instead of `getattr`
- Duplicated audio validation extracted into shared helper
- Added `.gemini/` to `.gitignore`

---

## [0.9.1] — 2026-03

### Changed
- Optimized generation-to-display pipeline

---

## [0.9.0] — 2026-03

### Fixed
- Cleanup refresh timer and queue in cancel safety timeout
- Eliminated C stack overflow in audio reactive chain mode

### Changed
- Real-time frame display with decoupled refresh timer
- Cleaned dead Live Paint traces from documentation
- Added temporal coherence config and expanded troubleshooting docs

---

## [0.8.9] — 2026-03

### Added
- Temporal coherence engine: LAB color matching, auto noise coupling, optical flow blending
- Distilled step scale cap for Hyper-SD models

### Changed
- Engine refactored from single `engine.py` to modular `engine/` package (core, animation, audio_reactive, helpers)
- Enforced 100% offline mode — no HuggingFace fetches at runtime
- Eliminated `uv run` at runtime — direct venv Python execution
- Hardened all Lua modules against stack overflow

### Removed
- Live Paint mode (event-driven real-time painting) — removed entirely

---

## [0.8.8] — 2026-02

### Fixed
- Audio reactivity bypass after parameter change
- Combobox selection preservation

---

## [0.8.7] — 2026-02

### Fixed
- Sub-floor blending: audio modulation below the denoising quality floor now smoothly attenuates instead of clamping

---

## [0.7.9] — 2025

### Added
- Palette CRUD: save/delete custom palettes from the UI (persist as JSON)

## [0.7.7] — 2025

### Added
- Contextual action button adapts to active tab (GENERATE, ANIMATE, AUDIO GEN)
- Universal randomize across all pipelines
- Randomness slider (0-20 scale)
- Dedicated per-pipeline Steps/CFG/Strength sliders (Animation + Audio)
- Audio-linked randomness: auto-generates prompt segments from musical structure

## [0.7.4] — 2025

### Added
- Audio-reactive motion/camera: smooth Deforum-like pan, zoom, rotation
- 7-layer anti-spaghetti protection for motion warp
- 4 dedicated motion presets + 14 existing presets enriched with motion
- Frame limit control (0 = all, or exact count)

## [0.7.3] — 2025

### Added
- AnimateDiff + Audio: 16-frame temporal batches with overlap blending
- MP4 export with nearest-neighbor upscaling and audio mux
- Sub-bass, upper-mid, presence frequency bands
- Palette shift and frame cadence modulation targets

### Fixed
- Cancellation works during long-running generations (concurrent receive)

## [0.7.0] — 2025

### Added
- Audio reactivity: synth-style modulation matrix
- 10 audio feature sources, 5 modulation targets
- Attack/release EMA smoothing
- Custom math expressions (simpleeval)
- BPM detection + auto-calibration
- Stem separation (demucs, CPU)
- 20 built-in modulation presets

## [0.6.1] — 2025

### Added
- Sequence output mode (new layer vs new frame)
- Cancellation with server ACK + 30s safety timer
- Auto-reconnect with exponential backoff (2s → 30s)
- Heartbeat pong watchdog (3× interval)

## [0.5.0] — 2025

### Added
- Random loop: auto-randomized prompts per iteration
- Lock subject: keep fixed subject while randomizing style

## [0.4.0] — 2025

### Added
- Loop mode: continuous generation
- Auto-prompt generator from curated templates
- Presets: save/load generation settings

## [0.3.0] — 2025

### Added
- Initial release
- txt2img, img2img, inpaint, ControlNet (OpenPose, Canny, Scribble, Lineart)
- Frame Chain + AnimateDiff animation
- Hyper-SD + DeepCache + FreeU v2 + torch.compile acceleration
- 6-stage post-processing pipeline (pixelate, quantize, palette, dither, rembg, alpha)
