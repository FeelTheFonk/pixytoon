# Changelog

All notable changes to SDDj are documented here.

## [0.9.35] — 2026-03

### Added
- **Pinnacle audio DSP pipeline** — complete rewrite of `audio_analyzer.py`
  - 44100 Hz sample rate (full 22.05 kHz Nyquist, was 22050)
  - 256 hop length (~172 Hz feature rate, was 512 / ~43 Hz — 4× improvement)
  - 4096 n_fft (93 ms window preserved via upsampled rate)
  - 256 mel bands (was 128 — 2× frequency resolution)
- **9-band frequency segmentation** — sub_bass, bass, low_mid, mid, upper_mid, presence, brilliance, air, ultrasonic (was 6 bands). Backward-compatible aliases `global_low`, `global_mid`, `global_high` preserved
- **5 new spectral timbral features** — `spectral_contrast`, `spectral_flatness`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flux`
- **12-bin CQT chromagram** — individual pitch classes (C through B) + aggregate `chroma_energy`
- **SuperFlux onset detection** — vibrato suppression via configurable `lag` and `max_size` parameters
- **ITU-R BS.1770 K-weighting pre-filter** — perceptual loudness weighting for energy-based features (configurable, enabled by default)
- **Savitzky-Golay smoothing** — causal, right-edge aligned polynomial filter as alternative to EMA (better transient preservation). Selectable via `audio_smoothing_mode` config
- **Optional madmom RNN beat tracking** — auto-detected at runtime, falls back to librosa. Manual install: `pip install madmom`
- **Integrated LUFS measurement** — per-file reference loudness via pyloudnorm, exposed in `AudioAnalysisResponse`
- **Percentile-clipped normalization** — prevents single-spike distortion on onset and flux features (99th percentile)
- **Full stem feature expansion** — stems now get all 34 features (was only rms/onset)
- 4 new modulation presets: `spectral_sculptor`, `tonal_drift`, `ultra_precision`, `micro_reactive`
- 4 new auto-calibrate genres using spectral features for nuanced detection
- `AudioAnalysisResponse` gains `lufs`, `sample_rate`, `hop_length` fields
- 10 new DSP config settings with validation: `audio_sample_rate`, `audio_hop_length`, `audio_n_fft`, `audio_n_mels`, `audio_perceptual_weighting`, `audio_smoothing_mode`, `audio_beat_backend`, `audio_superflux_lag`, `audio_superflux_max_size`
- 14 new unit tests (40 total for audio analyzer)
- `pyloudnorm>=0.1` added to core dependencies
- Lua frontend: 34 audio sources in dropdown (was 10), 4 new presets

### Changed
- `auto_calibrate.py` decision tree uses spectral_flatness, spectral_contrast, spectral_flux, chroma_energy, and brilliance for more accurate genre detection
- Cache key now includes `sr`, `hop_length`, `n_fft`, `n_mels`, `perceptual_weighting` — DSP config changes auto-invalidate stale caches
- STFT computed once and reused for all spectral features (was computed twice, ~15% compute saved)
- Test warnings suppressed: librosa `n_fft` and pitch tuning warnings on short test WAV files

## [0.9.34] — 2026-03

### Fixed
- **Zoom inversion bug** — `cv2.warpAffine` uses inverse mapping; `zoom > 1.0` was producing zoom OUT instead of zoom IN. Fixed by inverting the scale factor before building the affine matrix (with div-by-zero guard).
- Total motion threshold now uses the corrected inverted zoom value for accurate negligible-motion detection.

### Added
- **Perspective tilt** — faux 3D camera pitch/yaw via `cv2.warpPerspective` homography warp (`apply_perspective_tilt()`). Uses 3D rotation matrices (Rx·Ry) projected through a pinhole camera model: `H = K · R · K⁻¹`. Same denoise-correlation pattern and safety guards as affine warp.
- `motion_tilt_x` and `motion_tilt_y` modulation targets (±3.0 degrees)
- **Motion rate limiting** — `MOTION_MAX_DELTA` dict clamps frame-to-frame delta per motion channel. Total motion budget enforcement: if combined deltas exceed budget, all channels are scaled proportionally. Prevents saccade/jerk from rapid audio transients.
- 4 new presets: `cinematic_tilt`, `zoom_breathe`, `parallax_drift`, `full_cinematic`
- `cinematic_sweep`, `advanced_max`, `abstract_noise` enriched with tilt targets
- Frontend: tilt expression entries, tilt slider scaling, tilt settings persistence, new presets in dropdown

### Changed
- `motion_zoom` range widened from (0.95, 1.05) to (0.92, 1.08) — more expressive zoom while staying within safe corridor
- Frontend slider scaling updated for new zoom range: `0.92 + mn * 0.16` (was `0.95 + mn * 0.10`)
- Inverse scaling `to_pct()` updated accordingly

## [0.9.33] — 2026-03

### Added
- **9-phase prompt composition engine** — subject type awareness, generation modes (standard/art_focus/character/chaos), artist coherence via tag-bucketed selection, CLIP token budgeting (65-token soft cap), auto-negative matching, exclusion filtering
- 6 new data categories from OBP CSVs: pose (186), outfit (557), accessory (232), material (113), background (243), descriptor (700)
- Subject type classification: 1032 subjects across 5 types (humanoid/animal/landscape/object/concept) with keyword heuristic inference
- Artist tag system: 881 artists tagged across 108 style categories for coherence-aware selection
- 4 new prompt templates: character, material_study, scene_bg, descriptor_rich
- 3 new protocol fields: `subject_type`, `prompt_mode`, `exclude_terms` (all optional, backward compatible)
- Data audit script (`scripts/audit_data.py`) for JSON validation and cross-file dedup checking

### Changed
- `prompt_generator.py` fully rewritten as multi-phase pipeline (from simple template-based sampling)
- Token budget trimming uses regex-escaped values for robustness
- Artist tag matching uses word-boundary set intersection (not substring)

## [0.9.32] — 2026-03

### Fixed
- **Audio-reactive spaghetti artifacts during silence** — sub-floor denoising, unresolvable auto-noise coupling, and compounding motion warp artifacts during low-activity segments
  - Raised `denoise_strength` lower bound in `TARGET_RANGES` to 0.20
  - Raised `min_val` floor to ≥0.30 across all 27 modulation presets
  - Gated auto-noise coupling below `denoise_strength < 0.35` (both chain and AnimateDiff loops)
  - Added motion warp kill-switch below `denoise_strength < 0.25`
  - Switched `cv2.warpAffine` border mode from `REFLECT_101` to `REPLICATE`
  - Raised motion warp scale clamp from 0.10 to 0.15
- **Preset selection did not update UI sliders** — modulation preset selection was purely server-side, causing UI/server parameter desync
- Corrected misleading "Deforum pattern" reference in `auto_noise_coupling` docstring

### Added
- `GET_MODULATION_PRESET` server action — returns slot details for client-side hydration
- Preset hydration: selecting a modulation preset now populates all UI sliders via inverse-scaled slot values
- Auto-switch to `(custom)` when any modulation slot field is manually edited

### Changed
- Frontend slider minimums aligned with server-side safety floors: `anim_denoise` 5→20, `audio_denoise` 0→20, slot default min 15→30

## [0.9.31] — 2026-03

### Added
- Unit tests for `auto_calibrate.py` — 10 test cases covering every decision tree branch (ambient, electronic, hiphop, rock, bass, rhythmic, classical, glitch, default, empty-features safety)
- Centralized warning suppression in `__init__.py` — 15 filters covering diffusers, transformers, torch, PEFT, and audioread for a spotless console from boot

### Fixed
- README: clarified palette directory comment (removed ambiguous "7 preset palettes" count)
- Removed duplicate warning suppression block from `engine/core.py` (now in `__init__.py`)
- Suppressed Python 3.13 `aifc`/`sunau` deprecation warnings from audioread (librosa transitive dep) in pytest config

### Changed
- Version scheme: sub-versions within 0.9.3x (dizaines) for polish increments

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
