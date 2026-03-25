# Changelog

## [0.9.50] — 2026-03
### Changed
- **DRY: Unified generation helpers** — `compute_effective_denoise()` and `make_step_callback()` (previously dead code in `helpers.py`) now wired into `animation.py` and `audio_reactive.py`, replacing 8 inline duplications.
- **DRY: Frame processing helpers** — extracted `apply_temporal_coherence()`, `apply_frame_motion()`, `apply_noise_injection()` into `helpers.py`, replacing 7 copy-pasted blocks across chain and AnimateDiff loops.
- **DRY: Unified `ResourceManager`** — new generic `resource_manager.py` replaces cloned `lora_manager.py` (43→11 LOC) and `ti_manager.py` (43→11 LOC) with thin wrappers.
- **DRY: Protocol `BaseGenerationParams`** — 15 shared fields extracted into base class; `GenerateRequest`, `AnimationRequest`, `AudioReactiveRequest` now inherit. `_check_generation_mode_images()` extracted as shared validator.
- **Stale imports cleaned** — removed 7 unused imports across `animation.py` and `audio_reactive.py` (`match_color_lab`, `apply_optical_flow_blend`, `apply_motion_warp`, `apply_perspective_tilt`, `numpy`).

### Fixed
- **`auto_calibrate.py` dead branch** — both branches of `avg_chroma > 0.4` returned `"classical_flow"`; high-chroma path now returns `"atmospheric"`.
- **Version drift** — harmonized Lua extension version (was 0.9.48) with server (0.9.50).

### Added
- `test_helpers.py` — 25 tests covering all engine helper functions (previously untested dead code).
- `test_resource_manager.py` — 7 tests for unified ResourceManager (list, resolve, extensions, path traversal guard).
- Test suite: 483 → 509 tests (+26).

## [0.9.49] — 2026-03
### Added
- **Centralized VRAM management** — new `vram_utils.py` module: `vram_cleanup()`, `get_vram_info()`, `move_to_cpu()`, `check_vram_budget()`. Single source of truth for GPU memory management; all ad-hoc `gc.collect()`/`empty_cache()` patterns eliminated.
- **`eager_pipeline` context manager** — new `engine/compile_utils.py`: DRY UNet swap + DeepCache suspend + dynamo reset for chain/audio-reactive animations (replaced 2×26-line duplicated blocks → 3 lines each).
- **UNet weight snapshot** — `LoRAFuser` captures pre-fuse UNet weights on CPU and restores from snapshot on unfuse, preventing numerical drift after repeated fuse/unfuse cycles.
- **`get_status()` engine method** — reports loaded models, current LoRA, DeepCache state; exposed via `/health` endpoint.
- **VRAM budget guard** — pre-flight check before ControlNet lazy-load; triggers cleanup when free VRAM is below threshold (`vram_min_free_mb` config).
- **Load retry** — `DiffusionEngine.load()` retries once with VRAM cleanup on transient failures.
- **Path traversal guards** — resolved LoRA/TI paths validated to stay inside their configured directories (blocks `../` escape and symlink escape).
- **7 new config keys** — `enable_tf32`, `compile_dynamic`, `enable_lora_hotswap`, `max_lora_rank`, `enable_cpu_offload`, `vram_min_free_mb`, `quantize_unet`.
- **24 new unit tests** — `test_vram_utils.py` (10), `test_lora_fuser.py` (6), `test_deepcache_manager.py` (4), `test_compile_utils.py` (4).
- **DRY helpers** — `compute_effective_denoise()`, `make_step_callback()` in `engine/helpers.py`.

### Changed
- **TF32 + high matmul precision** — enabled by default on Ampere+ GPUs (~15-30% free speedup).
- **LoRA hotswap** — `enable_lora_hotswap()` called before first `load_lora_weights()`, eliminating ~15-25s torch.compile recompilation on LoRA switches; conditional `dynamo.reset()` only when hotswap is unavailable.
- **`torch.compile` dynamic shapes** — `dynamic=True` conditionally enabled when DeepCache is disabled (incompatible combination documented).
- **AnimateDiff DRY** — extracted `_apply_lightning_scheduler()` and `_apply_freeu_if_enabled()` methods (3×15-line blocks → 2 method calls each).
- **`.to("cpu")` before unload** — all `unload()` methods now move models to CPU before nullifying, ensuring immediate VRAM release instead of waiting for Python GC.
- **`/health` endpoint enriched** — returns VRAM used/free/total, loaded models list, current LoRA, DeepCache state.
- **atexit handler** — uses centralized `vram_cleanup()` instead of standalone `empty_cache()`.

### Fixed
- **Runtime crash in OOM handlers** — `gc.collect()` calls in `core.py`, `animation.py`, `audio_reactive.py` OOM handlers would crash because `gc` was not imported after Phase 0 refactoring; replaced with `vram_cleanup()`.
- **Unused `import gc`** — removed from `audio_reactive.py` (was dangling after eager_pipeline refactor).
- **Version drift** — harmonized Lua extension version (was 0.9.47) with server (0.9.49).

## [0.9.48] — 2026-03
### Changed
- **FPS-based audio frame timing** — Replaced the mathematical­ly incorrect ms-based `audio_frame_duration` slider (30–100ms) with the existing FPS combobox as sole timing source. Expanded FPS options to all professional rates: `23.976`, `25`, `29.97`, `50`, `59.94`. Frame durations are computed via Bresenham-style integer accumulation (`expected_ms - elapsed_ms`) for zero cumulative drift.
- **PCHIP upsampling + max-pooling downsampling** — `_resample_to_fps` now uses `PchipInterpolator` (shape-preserving, no overshoot) for upsampling and vectorized envelope max-pooling for downsampling, preserving transient peak amplitude.
- **Chunked async frame finalization** — All handlers (`animation_complete`, `audio_reactive_complete`, `error`) use `chunked_finalize_durations` with Timer-based async yielding to prevent UI freeze on large timelines.
- **FFmpeg muxing hardening** — `Fraction`-based framerate representation, `-vsync 1` strict CFR, `-tune animation` for pixel art, conditional AAC encoding (320kbps) for `.wav` inputs with stream copy for pre-encoded audio.

### Fixed
- **Orphaned `audio_frame_duration` references** — Removed stale slider references from `sddj_settings.lua` (save/load/apply) and `sddj_request.lua` (request payload) that would crash Aseprite after the slider was removed from the UI.
- **Handler code duplication** — Extracted `reset_anim_state()` helper, eliminating 5× duplicated 9-line cleanup blocks across `animation_complete`, `audio_reactive_complete`, and `error` handlers.
- **Error handler UI freeze** — Replaced inline synchronous frame-duration loop in `error` handler with async `chunked_finalize_durations` for consistency and non-blocking behavior.
- **Resampling O(n²) bottleneck** — Vectorized `np.searchsorted` calls in `_resample_to_fps` downsampling branch (was per-element Python loop).

## [0.9.47] — 2026-03
### Changed
- **AnimateDiff-Lightning Default Migration** — Elevated ByteDance's AnimateDiff-Lightning to be the out-of-the-box default model, replacing the classic v1.5.3 adapter.
- Achieved ultimate SOTA out-of-the-box performance with optimized 4-step generation and EulerDiscrete scheduler auto-engagement.
- Conducted an exhaustive cross-module audit guaranteeing zero edge-case overrides between frontend UI payloads and backend enforcement.

## [0.9.46] — 2026-03
### Added
- Complete Codebase Hardening: 100% test passing, Ruff compliant.
- Documentation Refactor: Complete structural purge, edge-cases coverage, and API payload synchronization.
- `AUDIO-REFERENCE.md` split for reading clarity.
- Interactive Table of Contents in `COOKBOOK.md`.

### Fixed
- Harmonized versioning between Lua extension (0.9.39) and Python Engine (0.9.45) to unified 0.9.46.
- Resolved WebSocket URL hardcoding in documentation.
- Clarified `animatediff` vs `animatediff_audio` backend aliases.


All notable changes to SDDj are documented here.

## [0.9.45] — 2026-03

### Changed
- **Pre-Release Audit** — Eradicated hidden edge cases before 0.9.45 deployment.
- **Portability Hardening** — Removed absolute `C:\` paths from all data processing scripts (`classify_subjects.py`, `build_prompt_data.py`, `build_artist_tags.py`, `audit_data.py`), ensuring cross-platform stability.
- **Model Preflight Safety** — `start.ps1` now explicitly verifies local model weights exist before launching in `HF_HUB_OFFLINE` mode, intercepting cryptic HuggingFace offline crashes if `setup.ps1` was skipped or aborted.
- **CI/CD Stabilization** — Added `ruff` to explicit dev dependencies in `pyproject.toml` to prevent static analysis failures on fresh installs.

## [0.9.44] — 2026-03

### Fixed
- **Exhaustive Deep Audit Remediation ** — Extensive cross-component architecture review completed with 100/100 performance/rectification validation.
- **Denoise lower bound** — `breathing_calm` choreography preset floor raised to 0.30, preventing Hyper-SD quality drop.
- **Audio stem separation sampling rate** — Unified default `target_sr` to 44100Hz aligning with engine DSP output.
- **Cache Persistence** — Fixed temporal caching flaw where `lufs` metric dropped during `audio_cache` serialization.
- **Zero-std extraction crash protection** — Added safeguard against `ref_std` in `match_color_lab` to prevent flat outputs from uniform reference images.
- **Type Coercion** — Added explicit integer boundary mapping for `frame_cadence` inside the modulation schedule processor.
- **Metadata String Safety** — Blocked command-line injection surface by strictly validating ffmpeg metadata keys against a hardened allowlist.
- Removed unused assignments and completed strict `ruff` static analysis compliance.

### Added
- 4 additional test integration modules covering new zero-std guards, caching constraints, and coercion limits.
- Complete expression parser validation suite testing all 32 presets (25 expressions + 7 choreographies) for syntax continuity and math soundness (`test_expression_presets.py`). Test suite footprint reaches 450 assertions.

## [0.9.43] — 2026-03
- **Expression Template Library** — 30 curated expression presets in 5 categories (rhythmic, temporal, spectral, easing, camera) via `expression_presets.py`; server API actions `list_expression_presets` / `get_expression_preset`
- **Camera Choreography Meta-Presets** — 7 multi-target presets (orbit journey, dolly zoom vertigo, crane ascending, wandering voyage, hypnotic spiral, breathing calm, staccato cuts) coordinating modulation slots + math expressions; server API actions `list_choreography_presets` / `get_choreography_preset`
- **14 new math functions** in `ExpressionEvaluator`: easing (`easeIn`, `easeOut`, `easeInOut`, `easeInCubic`, `easeOutCubic`), animation (`bounce`, `elastic`), utility (`step`, `fract`, `remap`, `pingpong`, `hash1d`, `smoothnoise`, `sign`, `atan2`, `mix`)
- **Slot inversion** — `invert` boolean on `ModulationSlot` / `ModulationSlotSpec`; when enabled, source feature is inverted (1−x) before min/max mapping — enables ducking effects and inverse-coupling
- **6 new modulation presets**: 4 voyage journeys (`voyage_serene`, `voyage_exploratory`, `voyage_dramatic`, `voyage_psychedelic`) and 2 rest-aware presets (`intelligent_drift`, `reactive_pause`)
- **6 modulation slots** — expanded from 4; default slot count set to 2 with motion-oriented defaults for slots 5-6
- **Choreography combobox** ("Camera Journey") in Lua UI — selects and hydrates both modulation slots and expression fields simultaneously
- **Expression preset combobox** — dynamically populated from server; auto-fills expression fields on selection
- **Invert checkbox** per modulation slot in Lua UI
- 4 new protocol actions, 4 new response models (`ExpressionPresetsListResponse`, `ExpressionPresetDetailResponse`, `ChoreographyPresetsListResponse`, `ChoreographyPresetDetailResponse`)
- 5 new Lua response handlers (`expression_presets_list`, `expression_preset_detail`, `choreography_preset_detail`, `choreography_presets_list`, updated `modulation_preset_detail`)
- ~50 new tests in `test_expression_presets.py` + `test_protocol.py` invert tests

### Fixed
- **Invert field not forwarded** — `audio_reactive.py` `ModulationSlot` construction from `ModulationSlotSpec` was missing `invert=s.invert`, causing slot inversion to silently fail during audio-reactive generation

### Changed
- **AUDIO-REACTIVITY.md** updated: Available Functions expanded from 16 to 30, added spectral variable rows, new sections for Slot Inversion / Expression Library / Choreography
- **API-REFERENCE.md** updated: 4 new actions, `invert` field on modulation slot, 5 new response types
- Frontend source dropdown hydration expanded to cover 6 slots (was 4)
- Slot default enable state: only slots 1-2 enabled by default (was all 4)

## [0.9.42] — 2026-03

### Changed
- **Documentation Overhaul** — Massive, multi-level audit of the entire SDDj documentation suite to ensure clarity, optimization, and completeness.
  - **API-REFERENCE**: added missing `motion_tilt_x`/`motion_tilt_y` targets, `get_modulation_preset` action, `modulation_preset_detail` response, `encoding` field on frames; corrected `motion_zoom` and `frame_duration_ms` numeric ranges; aligned Modulation Sources with full 34-feature list; added `subject_type`/`prompt_mode`/`exclude_terms` to generate_prompt docs.
  - **CONFIGURATION**: documented all 4 AnimateDiff-Lightning environment variables (`SDDJ_ANIMATEDIFF_LIGHTNING_STEPS`, `SDDJ_ANIMATEDIFF_LIGHTNING_CFG`, `SDDJ_ANIMATEDIFF_MOTION_LORA_STRENGTH`, `SDDJ_ANIMATEDIFF_LIGHTNING_FREEU`); added `.env` priority explanation.
  - **GUIDE**: added AnimateDiff-Lightning documentation; added Quick Reference Card for top 5 workflows; replaced redundant built-in palette list with cross-reference to Cookbook.
  - **COOKBOOK**: eliminated 150+ lines of redundant post-processing boilerplate by standardizing a reusable "non-pixel-art default" block (Pixelate OFF, Colors 256, Palette Auto).
  - **AUDIO-REACTIVITY**: fixed `denoise_strength` lower bound range (0.20); corrected stated stem features count.
  - **README**: refactored unreadable features list into a structured, categorized table with deep-links to documentation; added AnimateDiff-Lightning to Performance Stack; added Version/Python/CUDA shields.
  - **TROUBLESHOOTING**: removed historical version clutter (e.g., "fixed in v0.8.7") to focus exclusively on current behavior.
  - Standardized all CHANGELOG date formats to ISO 8601 (YYYY-MM).

### Added
- **CONTRIBUTING.md** — Developer guide covering repository structure, `uv` environment setup, Ruff code style, and PR process.
- **ARCHITECTURE.md** — Module-level system design covering Lua ↔ Python WS flow, inference optimizations (DeepCache/Hyper-SD), and DSP pipeline routing.

## [0.9.41] — 2026-03

### Added
- **AnimateDiff-Lightning support** (ByteDance) — 10× faster animation via progressive adversarial distillation (2/4/8-step checkpoints)
  - Auto-detection via `is_animatediff_lightning` config property
  - `EulerDiscreteScheduler` (trailing, linear, `clip_sample=False`) auto-applied to all AnimateDiff pipelines
  - Lightning-optimal CFG (default 2.0 — preserves negative prompt effectiveness)
  - Step count enforcement aligned to checkpoint distillation target
  - FreeInit force-disabled with log warning (incompatible with distilled models)
  - Conditional FreeU toggle (`animatediff_lightning_freeu` setting)
- New config: `SDDJ_ANIMATEDIFF_LIGHTNING_STEPS`, `SDDJ_ANIMATEDIFF_LIGHTNING_CFG`, `SDDJ_ANIMATEDIFF_MOTION_LORA_STRENGTH`, `SDDJ_ANIMATEDIFF_LIGHTNING_FREEU`
- Download script: `--animatediff-lightning` flag with `HF_HUB_OFFLINE` guard
- `pipeline_factory.create_lightning_scheduler()` utility
- AnimateDiff-Lightning integration test in `test_animation.py`

## [0.9.40] — 2026-03

### Fixed
- **DeepCache crash on img2img/inpaint** — `ValueError: 311 is not in list` caused by DeepCache's `wrapped_forward` looking up timesteps in the txt2img scheduler while img2img uses a different scheduler with a truncated schedule (`strength < 1.0` + `scale_steps_for_denoise`). Fixed by suspending DeepCache around `_img2img` and `_inpaint` calls via `deepcache_manager.suspended()`. Animation and audio-reactive paths were already correct (they suspend DeepCache as part of the UNet swap + dynamo reset flow)
- **Flaky `test_negative_prompt_default`** — test asserted `"worst quality" in negative` but auto-negative matching may return a specialized set (pixel_art, anime, etc.) depending on the randomly generated prompt; fixed to assert non-empty negative instead

## [0.9.39] — 2026-03

### Fixed
- **Silent frame drops undetected** — `animation_frame` and `audio_reactive_frame` handlers now track frame index continuity and warn when gaps are detected (fire-and-forget WebSocket sends can silently fail under load)
- **`audio_reactive_complete` missing frame count validation** — added parity with `animation_complete` to compare received vs expected frame count and warn on mismatch
- **Decode failure silent count mismatch** — `import_animation_frame` now increments a `decode_failures` counter when image decode fails, surfacing cumulative failure count in status and completion messages
- **Preset handler missing fields** — loading a preset now correctly restores `remove_bg`, `palette.mode`, `palette.name` (when preset mode), and LoRA `name`/`weight` settings; previously only post-process pixelate/quantize/dither were restored
- **Audio analysis dropped fields** — `lufs`, `sample_rate`, and `hop_length` from `AudioAnalysisResponse` are now stored in `PT.audio` and displayed in the audio status bar (LUFS shown when > -90)
- **List stale selection silent** — palette, LoRA, and preset combobox handlers now notify the user when a previously-selected item disappears from an updated resource list
- **Server frame callback logging** — `_make_thread_callback` bare `except: pass` replaced with `log.debug` for post-mortem visibility into dropped frames
- **State reset consistency** — new `last_frame_index` and `decode_failures` tracking fields are reset in all 5 state reset paths (animation_complete, audio_reactive_complete, error, disconnect, gen_timeout)
- **`PT.audio` undeclared fields** — added `lufs`, `sample_rate`, `hop_length` initial values to `PT.audio` state table for declaration consistency

## [0.9.38] — 2026-03

### Fixed
- **Double warmup / recompilation on first generation** — `_warmup()` was disabling DeepCache before the dummy generation, causing `torch.compile` to trace the UNet with original forwards; when DeepCache re-enabled afterward, all 30+ wrapped block forwards triggered dynamo guard failures and a full ~15-25s recompilation on the first real `generate()` call. Warmup now runs with DeepCache active (matching real generation state), plus noop `callback_on_step_end` for graph parity and post-warmup cache flush to prevent stale feature leakage.

## [0.9.37] — 2026-03

### Fixed
- **Lock Subject universality** — centralized all inline `locked_fields` construction into a single `PT.build_locked_fields()` helper with whitespace trim; eliminated 5 redundant inline patterns across dialog, request builder, and handler
- **Animation tab ignored Lock Subject** — `trigger_animate()` now injects the locked subject into the animation prompt (with duplicate-prevention guard), ensuring subject persistence across all animation frames
- **Audio-reactive lost locked subject** — `AudioReactiveRequest` now carries `locked_fields` through the protocol; `auto_generate_segments()` uses explicit locked subject instead of heuristic comma-split extraction (fixes short-subject misidentification)
- **Metadata did not persist lock state** — `build_generation_meta()` and `build_animation_meta()` now store `lock_subject` and `fixed_subject`; `apply_metadata()` restores both fields on load

### Added
- 6 new unit tests: locked_fields propagation in `AudioReactiveRequest` (4 tests), explicit locked subject override in `auto_generate_segments` (2 tests)

## [0.9.36] — 2026-03

### Fixed
- **Lock Subject in audio mode** — `fixed_subject` is now injected into the prompt sent for audio-reactive generation, ensuring the server's prompt schedule correctly preserves the user's locked subject across auto-generated segments
- **Randomize before audio hang** — pre-validates that audio analysis is complete before dispatching randomize+generate in audio mode; previously caused a silent UI hang if audio was not yet analyzed
- **Export MP4 button stale state** — disconnect now resets `export_mp4_btn` and clears `audio.last_output_dir`, preventing orphaned enabled state after connection loss

### Added
- **Dedicated audio tag name** — audio tab now has its own `audio_tag` entry field instead of borrowing from the Animation tab's `anim_tag`; persisted in settings
- **FPS 4 and 60** — expanded audio FPS dropdown with time-lapse (4) and high-fluidity (60) options

### Changed
- **Loop controls disabled in audio tab** — Loop and Random Loop checkboxes are now grayed out when the audio tab is active (no loop logic exists in the audio handler; was silently ignored)

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

## [0.7.9] — 2025-12

### Added
- Palette CRUD: save/delete custom palettes from the UI (persist as JSON)

## [0.7.7] — 2025-11

### Added
- Contextual action button adapts to active tab (GENERATE, ANIMATE, AUDIO GEN)
- Universal randomize across all pipelines
- Randomness slider (0-20 scale)
- Dedicated per-pipeline Steps/CFG/Strength sliders (Animation + Audio)
- Audio-linked randomness: auto-generates prompt segments from musical structure

## [0.7.4] — 2025-10

### Added
- Audio-reactive motion/camera: smooth Deforum-like pan, zoom, rotation
- 7-layer anti-spaghetti protection for motion warp
- 4 dedicated motion presets + 14 existing presets enriched with motion
- Frame limit control (0 = all, or exact count)

## [0.7.3] — 2025-09

### Added
- AnimateDiff + Audio: 16-frame temporal batches with overlap blending
- MP4 export with nearest-neighbor upscaling and audio mux
- Sub-bass, upper-mid, presence frequency bands
- Palette shift and frame cadence modulation targets

### Fixed
- Cancellation works during long-running generations (concurrent receive)

## [0.7.0] — 2025-08

### Added
- Audio reactivity: synth-style modulation matrix
- 10 audio feature sources, 5 modulation targets
- Attack/release EMA smoothing
- Custom math expressions (simpleeval)
- BPM detection + auto-calibration
- Stem separation (demucs, CPU)
- 20 built-in modulation presets

## [0.6.1] — 2025-07

### Added
- Sequence output mode (new layer vs new frame)
- Cancellation with server ACK + 30s safety timer
- Auto-reconnect with exponential backoff (2s → 30s)
- Heartbeat pong watchdog (3× interval)

## [0.5.0] — 2025-06

### Added
- Random loop: auto-randomized prompts per iteration
- Lock subject: keep fixed subject while randomizing style

## [0.4.0] — 2025-05

### Added
- Loop mode: continuous generation
- Auto-prompt generator from curated templates
- Presets: save/load generation settings

## [0.3.0] — 2025-04

### Added
- Initial release
- txt2img, img2img, inpaint, ControlNet (OpenPose, Canny, Scribble, Lineart)
- Frame Chain + AnimateDiff animation
- Hyper-SD + DeepCache + FreeU v2 + torch.compile acceleration
- 6-stage post-processing pipeline (pixelate, quantize, palette, dither, rembg, alpha)
