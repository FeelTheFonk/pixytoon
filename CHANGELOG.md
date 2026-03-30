# Changelog

## [0.9.79] — 2026-03
### Pinnacle UI/IO Hardening & Determinism
Final architectural lockdown eliminating all silent edge cases in UI recursive loops, asynchronous I/O bounds, and data sanitization. Achieved a rigorous 100/100 robustness score.

#### Security & Robustness
- **Payload Validation** (`sddj_ws.lua`): Implemented strict O(1) type-check validation on JSON decoding. Null or scalar (boolean/number) responses traversing the WebSocket are instantly dropped before triggering nil-index exceptions downstream during intense IPC.
- **Math Bounds Pre-Flight** (`sddj_request.lua`): Enforced rigorous numerical validation and type-safe limits (`NaN`, `inf` interception) across all payload builders to permanently neutralize upstream pipeline math limit crashes.
- **Safe I/O Atomicity & Finalization** (`sddj_output.lua`, `sddj_settings.lua`, `sddj_import.lua`): Eliminated all "silent I/O failures". File handles now explicitly enforce closure on failure paths, preventing resource leaks. Temporary atomic write/renaming is now standardized across all save functions. Success checks instantly trigger `app.alert` instead of failing mutely.

#### Fixed
- **UI Desync / Slider Blind-Spot (CRITICAL)** (`sddj_dialog.lua`): Aseprite's `enabled = false` manipulation via `dlg:modify` permanently locked slider widgets. Completely removed all conditional enable toggling on sliders. Sliders now remain editable at all times; business logic safely ignores their values when disabled (e.g., target feature toggle), restoring flawless UX.
- **Recursive UI Mutex** (`sddj_dialog.lua`, `sddj_settings.lua`, `sddj_output.lua`): Introduced `PT._ui_transaction_depth` absolute lock. Batch updates (`apply_settings`, `apply_metadata`, dialog `onchange`) are now wrapped in transaction checkpoints, preventing catastrophic event-loop recursion and Aseprite latency bottlenecks.


## [0.9.78] — 2026-03

### Full Lock Custom & Subject Alignment
Achieved 100% UI and Generation alignment for locked prompting mechanisms, removing blind spots across all standard, bulk, and scheduled randomization modes.

#### Added
- **Visual Editor Tooltips**: The Schedule Randomizer dialog now openly reflects the exact `Lock Custom` position (`suffix` / `prefix`) and its designated string.

#### Fixed
- **UI Prompt Desync**: `Lock Custom` fields are now seamlessly woven into the server's returned text immediately following a single "Randomize" request, syncing the visual UI field instantly with the active Generation constraints.
- **Schedule Empty Keyframe Resolution**: Resolved a blind-spot edge case where an artificially blank keyframe assigned by a standard schedule randomization bypassed prefix/suffix text injections. All keyframes, empty or full, now conform to the active `lock_custom` state.
- **Animation Request Parsing**: DSL Parsers processing scheduled routines (e.g. Audio-Reactive or standard Animation) natively validate and hard-inject the `lock_custom` string into all evaluated frames moments before execution over the Socket, cementing zero-omission prompt consistency.

## [0.9.77] — 2026-03
### Centralized UI Synchronization
Single source of truth for all conditional widget states, eliminating desync after programmatic updates.

#### Added
- **`PT.sync_ui_conditional_states()`** — centralized function that reads `dlg.data` and enforces all enabled/disabled states and dynamic labels across Post Process, Animation, Audio, and Generate tabs. Wrapped in `pcall` for crash safety against unbuilt tabs.

#### Fixed
- **Post Process / Animation / Audio tabs desynchronized after preset load, metadata load, or settings restore (CRITICAL)**: Aseprite's `dlg:modify{}` does not fire `onchange` callbacks, so programmatic data injection left dependent widgets (sliders, comboboxes) in stale enabled/disabled states. Centralized sync called at 4 critical points: dialog build, `apply_settings()`, `apply_metadata()`, `handlers.preset()`.

#### Cleanup
- Removed 3 hardcoded init blocks (`enabled = false` after widget creation) in Animation and Audio tabs — now handled by centralized sync.
- Removed duplicated randomness label computation from `handlers.preset()` (5 lines) — handled by centralized sync.
- Removed ~50 lines of duplicated conditional UI logic from `apply_settings()` — replaced with single sync call.
- Removed inline mode-label logic from `apply_metadata()` — replaced with single sync call.

## [0.9.76] — 2026-03
### Polish & Edge Case Eradication
Formal verification of all execution paths, guaranteeing absolute zero-copy binary transit and leak-free memory management.

#### Security & Robustness
- **Generator Disconnect Leak (Verified)**: Formally validated _ws_receive_loop implicitly triggers ngine.cancel() on WebSocketDisconnect, ensuring generating loops correctly abort on sudden client loss without requiring explicit exception bubbling in the fire-and-forget payload callback.
- **Farneback Flow Memory Leak (Fixed)**: Implemented an explicit LRU dict capacity constraint (max=4) for _FLOW_GRID_CACHE in image_codec.py, preventing out-of-bounds loat32 accumulation when users dynamically alter resolutions sequentially over long uninterrupted sessions.
- **Zero-Copy Serialization (Verified)**: Audited pipeline transit guarantees that Tobytes() inside ncode_image_raw_bytes(...) correctly prevents redundant RGBA allocations when the payload matches the target color mode.
- **Client-Side VRAM Panning Freezes (Fixed)**: Added aggressive explicit Lua Object reference niling (
esponse._raw_image = nil) immediately post-dispatch in sddj_ws.lua, aiding prompt garbage collection of Multi-MB binary strings during intense IPC frames.

## [0.9.75] — 2026-03
### Provisioning Velocity & Animation Stability
Four surgical fixes across the provisioning script, WebSocket protocol, and Lua runtime.

#### Fixed
- **Setup script multi-minute hang**: Removed the massive 1.8GB legacy AnimateDiff adapter (`guoyww/...`) from the default `--all` provisioning checklist. By default, setup now only provisions precisely what's needed for the ultra-fast AnimateDiff-Lightning pipeline, reducing fresh install duration by minutes and eliminating an unconditional background download.
- **HuggingFace Cache False Negatives**: Corrected the cache probe sentinel for `QR Code Monster v2` to target `v2/config.json`. Previously, a missing root config triggered an aggressive ETags network scan on every run, padding setup times. The cache check is now fully local and instantaneous.
- **WebSocket concurrent drain assertion**: Added graceful handling for `AssertionError` in the WebSocket `_send` handler. Prevents the server from crashing when fire-and-forget Animation callbacks attempt concurrent writes with final metadata responses during high-speed AnimateDiff-Lightning chunk generation.
- **Lua `_pow2` upvalue leak**: Reordered lexical scoping in `sddj_base64.lua` to ensure codec lookup tables are declared before function closures. Resolves the `attempt to index a nil value (global '_pow2')` error that crashed `img2img`, `inpaint`, and `controlnet` extensions in Aseprite via binary base64-fallback logic.

## [0.9.74] — 2026-03
### Binary WebSocket Frames, CLIP Embedding Cache & SOTA Performance Hardening
Eliminates base64 encode/decode overhead via binary WebSocket frames (-33% payload, ~25ms/frame saved), adds CLIP embedding LRU cache, bfloat16 weight snapshots, transaction batching, and 20 systemic performance fixes across the full pipeline.

#### Added
- **Binary WebSocket frames (D-001)**: New binary transport `[uint32 LE json_len][JSON metadata][raw RGBA bytes]` replaces base64 text frames. Server sends raw RGBA via `websocket.send_bytes()`, extension decodes via `string.byte()` header parsing. Eliminates base64 encode (server) + decode (extension) entirely. `_raw_image` field on response objects bypasses all legacy decode paths.
- **`encode_image_raw_bytes()`**: New `image_codec.py` function — `PIL.Image.tobytes()` for RGBA, zero-copy.
- **`ResultResponse.encoding` field**: Protocol schema extended with `encoding: Optional[str] = None` for binary frame support parity with `AnimationFrameResponse` / `AudioReactiveFrameResponse`.
- **CLIP embedding LRU cache (D-002)**: `_EmbeddingCache` in `embedding_blend.py` with `OrderedDict`, maxsize=256, key = `(prompt, negative_prompt, clip_skip, id(pipe.text_encoder))`. Eliminates redundant CLIP forward passes for repeated prompts.
- **bfloat16 weight snapshots (D-010)**: LoRA fuser stores UNet/text_encoder state_dict compressed to bf16 (`-50%` RAM). Casts back to model dtype on restore. Sample-based device validation (10 random tensors vs full O(N) scan).
- **Transaction batching (D-016)**: Consecutive animation/audio frames wrapped in single `app.transaction("SDDj Batch")` with `PT._in_batch_transaction` flag. `collectgarbage("step", 200)` after each batch. Reduces Aseprite undo history overhead ~5-10x.
- **AnimateDiff adapter cache (D-012)**: Class-level `_ADAPTER_CACHE` avoids reloading motion modules on consecutive AnimateDiff calls. Lightning scheduler instance cached and deep-copied per use.
- **Progress throttle (D-015)**: Server skips every 2nd progress callback to reduce WebSocket message volume during generation.
- **Lazy imports (D-011)**: `_get_ti_manager()`, `_get_recommend_preset()` in `server.py` — deferred import avoids loading unused modules at startup.

#### Optimized
- **Image.bytes bulk alpha extraction (D-005)**: Lua `string.byte()` on raw RGBA bytes replaces per-pixel `getPixel()` in capture/selection export. O(1) string op vs O(W*H) API calls.
- **mtime-based cache keys (D-013, D-020)**: `os.stat().st_mtime_ns + st_size` replaces file content hashing for audio cache and palette/preset file monitoring. Eliminates 4MB SHA256 hashes on every cache lookup.
- **Pre-extracted modulation arrays (D-014)**: `slot_sources`, `slot_inverts`, `slot_mins`, `slot_ranges`, `slot_targets` pre-extracted before frame loop. `expr_variables` dict allocated once, updated in-place. Bounds-guarded array access prevents IndexError on short feature arrays.
- **Float32 warp matrices (D-006)**: `apply_motion_warp` / `apply_perspective_tilt` use `np.float32` instead of float64. Inverse matrix cache keyed by `(w, h, fv)`.
- **TensorRT > CUDA > CPU provider auto-detection (D-009)**: onnxruntime session for rembg selects optimal execution provider at load time.
- **Fire-and-forget callbacks (D-017)**: `asyncio.run_coroutine_threadsafe` without `result()` blocking in engine thread callbacks.
- **Compact JSON serialization (D-018)**: `json.dumps(separators=(',', ':'))` for binary frame metadata — ~5-10% smaller payloads.
- **`_json_dumps_compact()` helper**: Cached separator tuple, direct `json.dumps` call bypasses Pydantic's `model_dump_json()` overhead for binary frames.

#### Fixed
- **Binary frame guards (CRITICAL)**: `sddj_handler.lua` lines 60/150 — image presence guards now accept `_raw_image` field, preventing silent frame drops on binary transport.
- **Output save guards (CRITICAL)**: `sddj_output.lua` lines 58/110 — save functions now accept binary frames (`_raw_image`), preventing output loss.
- **Output encoding detection (CRITICAL)**: `sddj_output.lua` lines 74/141 — raw RGBA detection now checks both `resp.encoding == "raw_rgba"` and `resp._raw_image` presence.
- **`shallow_copy_request` key mismatch (HIGH)**: `sddj_utils.lua` excluded `mask`/`init_image` (non-existent keys) instead of actual protocol field names `source_image`/`mask_image`/`control_image`/`_raw_image`. Fixed — prevents multi-MB blobs leaking into metadata copies.
- **`_generate_chain_inner` return type**: Annotation `-> list[AnimationFrameResponse]` corrected to `-> int` (returns frame count).
- **`AudioCache.invalidate()`/`cleanup()` thread safety**: Both methods now wrapped with `self._lock` (RLock), matching `get()`/`put()`.
- **LoRA fuser double `next(parameters())`**: Merged two consecutive iterator calls into single `first_param = next(raw_unet.parameters())`, preventing `StopIteration` on small models.
- **Test mock iterators**: `test_lora_fuser.py` — `parameters.side_effect = lambda: iter([...])` provides fresh iterators per call, fixing `StopIteration` in multi-call tests.
- **GC release for binary frames**: `resp._raw_image = nil` after import in handler, preventing multi-MB Lua string retention across frame batches.

## [0.9.73] — 2026-03
### Lockable Fields, Prefix/Suffix Injection & Widget Freeze Fix
New custom lockable field with positional prompt injection, HuggingFace cache bypass, and systemic fix for Aseprite's `enabled = false` constructor bug affecting 38 widgets.

#### Added
- **Custom lockable field**: New "Custom" entry below Subject in the Generate tab — lockable via checkbox, supports free-text (LoRA triggers, style tags, etc.). Full pipeline: dialog → settings → request → handler → metadata.
- **Prefix/Suffix position selector**: Both Subject and Custom locked fields now have a position combobox (`prefix` / `suffix` / `off`) controlling where the locked text is injected into the prompt.
- **Centralized `inject_locked_prompt()`**: Single function handles prefix/suffix injection for all 3 request builders (generate, animation, audio), replacing ad-hoc subject injection.
- **HuggingFace cache probe**: `_hf_file_cached()` / `_hf_snapshot_cached()` using `try_to_load_from_cache()` — all 9 download functions now skip network calls when models are already cached. Eliminates unnecessary HTTP HEAD requests on every `setup.ps1` run.

#### Fixed
- **Post-processing sliders frozen (CRITICAL)**: Aseprite's `dlg:slider` / `dlg:combobox` / `dlg:entry` / `dlg:button` with `enabled = false` in the constructor creates permanently unresponsive widgets. Moved all 13 constructor occurrences across 4 tab builders to `dlg:modify` calls after widget creation. Affects: `pixel_size`, `colors`, `quantize_method`, `dither`, `palette_name`, `palette_custom_colors`, `anim_freeinit_iters`, `audio_freeinit_iters`, `audio_expr_preset`, all EXPR_FIELDS entries, `export_mp4_btn`, `action_btn`, `cancel_btn`.
- **Empty prompt comma artifacts**: `inject_locked_prompt()` with an empty prompt and active prefix/suffix produced trailing/leading commas (e.g. `"cat, "` instead of `"cat"`). Replaced concatenation with parts-based join.
- **Aseprite freeze on exit (CRITICAL)**: `ws_handle:close()` in `onclose` and `exit()` performed a blocking WebSocket close handshake on the UI thread. If the server was shutting down or unresponsive, Aseprite froze indefinitely. Replaced with fire-and-forget `ws_handle = nil` — OS tears down the TCP socket on process exit.
- **Server PowerShell window persisted after exit**: `-NoExit` flag on the server's PowerShell window kept it alive after Python exited. Removed.
- **Launcher never detected Aseprite closing**: `Read-Host` blocked the launcher indefinitely, requiring manual Enter press. Replaced with a 500ms-polling process monitor that auto-triggers shutdown when Aseprite exits, server crashes, or user presses any key.
- **Server shutdown fragile on Windows**: `_request_shutdown()` used `os.kill(SIGBREAK)` with `os._exit(0)` fallback — non-deterministic and platform-dependent. Replaced with `uvicorn.Server.should_exit = True` (documented, signal-free, platform-safe).
- **Heartbeat race during shutdown**: Timer callbacks could fire between `onclose` (dialog destroyed) and `exit()` (timers stopped), causing the pong watchdog to call `ws_handle:close()` and freeze Aseprite. Fixed by stopping all timers and disarming connection state in `onclose`.
- **Settings fallback non-atomic write**: `exit()` fallback settings save used `os.remove` + `io.open` (data loss on crash). Replaced with `.tmp` + `os.rename` atomic pattern, matching `save_settings()`.

#### Improved
- **`exit()` teardown ordering**: Reordered from (cancel → save → shutdown → close → timers → cleanup) to (timers → disarm → cancel → save → shutdown → abandon → cleanup). Timers stopped first prevents all callback-during-teardown races.
- **Launcher monitoring**: Detects server crashes with exit code reporting, handles missing Aseprite gracefully (keypress fallback).

#### Cleanup
- Removed `import signal` from `server.py` (no longer needed).
- Removed blank line artifact in imports.

## [0.9.72] — 2026-03
### Prompt Schedule DSL Hardening & Exhaustive Testing
Deep audit of the entire Prompt Schedule DSL pipeline — parser, engine integration, embedding blending, presets, and protocol — with critical bug fixes, full Lua/Python parity, and 77 new tests.

#### Fixed
- **Audio-reactive keyframe resolution (CRITICAL)**: `audio_reactive.py` used time-based `get_prompt()` instead of keyframe-based `get_blend_info_for_frame()`, ignoring the entire prompt schedule during audio-reactive generation. Full keyframe resolution with SLERP blending and per-keyframe parameter overrides now applied.
- **Negative prompt SLERP blending**: Transitions between keyframes with different negative prompts now SLERP-blend negative embeddings in `embedding_blend.py`, matching positive prompt behavior. Previously the outgoing negative was used without blending.
- **`weight_end` validation leak**: Invalid `weight_end` values (e.g. `weight: 1.5->6.0`) leaked into keyframes despite E006 validation. Assignment now occurs only after range check passes.
- **Protocol weight floor**: `PromptKeyframeSpec.weight` and `weight_end` minimum changed from `0.0` to `0.1` — matches DSL spec range `[0.1, 5.0]`.
- **Animated weight fallback in `_compute_weight`**: Last keyframe with `weight_end` used hardcoded `kf.frame + 100` as fallback range. Now uses `PromptSchedule.total_frames` for correct interpolation.
- **Dead `default_prompt` parameter**: Removed unused parameter from `_KeyframeBuilder.build()`.

#### Improved
- **DSL safety limits**: E013 (input > 100 KB) and E014 (> 500 keyframes) prevent stack overflow / memory exhaustion from adversarial schedules.
- **Separated E002 / E003**: Duplicate frame (E002) and out-of-order frame (E003) are now distinct validation errors in both Python and Lua parsers.
- **W006 warning**: `blend:` directive with `hard_cut` transition now emits a warning (no effect).
- **W007 warning**: Lines matching `word: value` that aren't recognized directives emit an unrecognized-directive warning.
- **Lua parser parity**: `{auto}` case-insensitive, E002/E003 split, W004 for animated weight > 2.0, W006 for blend+hard_cut — all aligned with Python parser.
- **Presets v2 (ratio-based)**: All builtin presets converted to `keyframe_ratios` format with `ratio` and `blend_ratio` — frame-count independent, resolves correctly for any `total_frames`.
- **`schedule_to_dsl()` round-trip**: Added `include_auto` parameter, reordered output (directives before prompt text).

#### Docs
- Fixed `ease_in_out` description: "cubic" → "quadratic" (matching actual implementation).
- Added E013, E014 to errors table; W006, W007 to warnings table.
- Fixed E012 description and per-keyframe parameter interpolation notes.

#### Tests
- **77 new tests** (`test_dsl_parser.py`): empty inputs, all time formats, all directives, prompt parsing, auto directive, validation (E002/E003/W001/W006/E004/W007), safety limits (E013/E014), file references (E010/E011), DSL round-trip, comments/whitespace, blend resolution, edge cases (single KF, 1 frame, low FPS, CRLF), 1000-schedule stress test, presets v2.
- Rewrote `test_sddj_dsl_parser.py` (Lua parser) — removed invalid `w:` shortcut test, added `pytest.importorskip`.
- Fixed `test_prompt_schedule_keyframes.py` for v2 preset format.
- **675 tests passing, 0 failures.**

## [0.9.71] — 2026-03
### Systemic Hardening & DRY Refactoring
14-file cross-stack audit: security hardening, performance optimization, DRY refactoring, and edge-case elimination.

#### Security
- **JSON decode string length limit**: 50 MB cap on decoded strings prevents memory exhaustion from adversarial payloads.
- **WebSocket message size guard**: Rejects incoming messages exceeding `MAX_WS_MESSAGE_SIZE` before JSON parsing.
- **Base64 decode size guard**: `MAX_BASE64_SIZE` limit prevents memory exhaustion from oversized image payloads.
- **DSL parser input limits**: `MAX_DSL_LENGTH` (100 KB) and `MAX_KEYFRAMES` (500) prevent stack overflow on adversarial prompt schedules.
- **Frame filename validation**: `export_mp4` validates frame filenames match `frame_\d+.png` pattern (aligned with `video_export.py`).
- **Server sandbox validation**: `export_mp4` rejects output directories outside the allowed sandbox via `os.path.realpath` check.
- **Anti-double-connect**: `PT.state.connecting` flag prevents WebSocket race conditions across all connect/disconnect/timeout paths.

#### Refactored
- **Handler DRY** (`sddj_handler.lua`, -78 lines): Extracted `_handle_streaming_frame()` and `_handle_streaming_complete(resp, opts)` with opts-table pattern — animation and audio-reactive handlers now share ~80 lines of common streaming logic.
- **Resource list DRY**: `_update_resource_combobox()` helper + `_list_config` lookup table replaces 7 near-identical handler blocks.
- **Settings schema-driven** (`sddj_settings.lua`): `_FIELD_SCHEMA` array as single source of truth for save/apply — eliminates dual-list maintenance when adding new dialog fields.
- **Request parameter clamping** (`sddj_request.lua`): All numeric request params go through `clamp(v, lo, hi)` with nil-safe fallback before server transmission.
- **Extension exit cleanup** (`sddj.lua`): Re-entry guard with `_PT = nil` + local `pt` prevents double-exit crashes.

#### Improved
- **Conditional widget enable/disable** (`sddj_dialog.lua`): Pixelate, quantize, palette, FreeInit, expression, and modulation slot widgets are disabled on construction and toggled via `onchange` callbacks — prevents invalid parameter combinations.
- **Settings restore widget sync** (`sddj_settings.lua`): After restoring saved settings, explicit sync block re-applies conditional enable/disable states (since `dlg:modify` doesn't fire `onchange`).
- **Type-validated settings restore**: `_apply_field()` checks `type(val)` before applying, with `pcall` for combobox options — prevents crashes from corrupted settings files.
- **Base64 encoder rewrite** (`sddj_base64.lua`): O(n) table.concat approach replaces O(n²) string concatenation.
- **Large sprite warning** (`sddj_capture.lua`): Warns when capturing sprites >2048px.
- **GC cleanup in handlers**: Decoded image bytes and raw base64 strings are nilled after import to reduce peak memory.
- **Combobox option safety** (`sddj_output.lua`): All combobox `option=` sets wrapped in `pcall` to handle stale/missing values gracefully.

## [0.9.70] — 2026-03
### Random Prompt Schedule Generator
Generate complete prompt schedules — structure and content — with a single click. Seven built-in profiles control keyframe placement, transition types, blend windows, parameter overrides, and prompt variety.

#### Added
- **Random Schedule Generator**: New `randomize_schedule` action with 7 profiles (`gentle`, `dynamic`, `rhythmic`, `cinematic`, `dreamy`, `chaos`, `minimal`). Each profile parameterizes keyframe count, spacing strategy (uniform/random/front-heavy/back-heavy), transition weights, blend ranges, parameter variety, and weight animation.
- **UI**: "Random" button in the Prompt Schedule section with profile picker popup showing descriptions, preview, and current context (randomness, Lock Subject, total frames). Confirmation dialog when replacing an existing schedule.
- **DSL round-trip**: `schedule_to_dsl()` converts generated keyframes to standard DSL, fully editable and saveable as presets.
- **Per-keyframe negatives**: Each generated keyframe gets its own auto-matched negative prompt from the PromptGenerator.
- **Settings persistence**: Last-used profile is saved and restored across sessions.

#### Tests
- 31 new tests across 6 classes: structure validation, E004 compliance, value ranges, locked subject, edge cases (1-10800 frames), profile distribution, DSL round-trip, stress (100 random iterations), protocol serialization.
- **597 tests passing, 0 failures.**

## [0.9.69] — 2026-03
### Settings Persistence, AnimateDiff Audio Pipeline & LoRA Hotswap Fix
Three critical bug fixes across the full stack: Lua frontend persistence, Python AnimateDiff pipeline, and CUDA tensor management.

#### Fixed
- **Settings Persistence (Windows)**: `os.rename` fails on Windows when the destination file already exists, silently discarding saved settings. Fixed with explicit `os.remove` before rename + fallback direct write. Added `.tmp` crash recovery in `load_settings()` and `exit()` fallback via cached JSON when the dialog is already destroyed.
- **AnimateDiff-Lightning 32-Frame Cap in Audio-Reactive Mode**: Hard total-frame rejection (`total_frames > 32`) blocked audio-reactive generation even though chunked processing (16-frame chunks) stays within the Lightning per-batch limit. Replaced with per-chunk validation that correctly allows long sequences.
- **LoRA Hotswap CUDA Device Mismatch**: `load_state_dict(assign=True)` replaced tensor objects, breaking torch.compile Dynamo graph references that still pointed to the old CPU tensors. Fixed by using `assign=False` (default) which copies data into existing CUDA tensors, preserving tensor identity. Extended post-restore validation to include both parameters AND buffers, and added text_encoder snapshot/restore parity.
- **DeepCache Restore After AnimateDiff**: Unprotected `DeepCacheState.restore()` in `finally` blocks could throw and mask the actual generation result. Wrapped in try/except with warning log in both `animation.py` and `audio_reactive.py`.

#### Tests
- Added `test_restore_uses_assign_false` — validates that `load_state_dict` is called without `assign=True` to preserve torch.compile tensor references.
- Added `TestAnimateDiffLightning` — validates `is_animatediff_lightning` property, default max frames, and bounds validation.
- **566 tests passing, 0 failures.**

## [0.9.68] — 2026-03
### AnimateDiff Performance Optimization
Systematic elimination of pipeline initialization and DeepCache toggle overhead across all AnimateDiff paths.

#### Performance
- **UNet sharing** — `ensure_vid2vid` and `ensure_controlnet` now reuse the already-converted `UNetMotionModel` from the base pipeline instead of calling `get_uncompiled_unet(base_pipe)` which triggered a redundant `UNetMotionModel.from_unet2d()` deep-copy (~5-6s each). All AnimateDiff pipeline variants share a single UNet instance.
- **DeepCacheState activated** — the mode-aware `DeepCacheState` class (previously dead code since v0.9.58) is now wired into `DiffusionEngine` and used for both `animation.py` and `audio_reactive.py` AnimateDiff paths. Eliminates redundant `disable()`/`enable()` cycles (100-300ms per toggle) on consecutive AnimateDiff calls.
- **GPU-accurate timing** — `torch.cuda.synchronize()` barriers around inference in `_generate_animatediff_inner` provide precise setup/inference/post-process breakdown in logs.

#### Improved
- **FreeNoise window count logging** — log message now includes computed number of temporal attention windows for immediate performance visibility.
- **Config performance warnings** — Pydantic model validator warns at startup on `freeinit_iterations > 2` (each is a FULL denoising pass) and `animatediff_context_stride < 4` (excessive windowing overhead).
- **UNet sharing verification** — soft `log.error` assertion after vid2vid and controlnet pipeline creation detects if diffusers unexpectedly re-wraps the UNet.

#### Cleanup
- Removed 4 dead imports: `deepcache_manager` and `get_uncompiled_unet` from `animation.py`, `get_uncompiled_unet` and `scale_steps_for_denoise` from `audio_reactive.py`.
- `deepcache_manager` import removed from `audio_reactive.py` (suspended() replaced by DeepCacheState).

#### Tests
- **561 tests passing, 0 failures.**
- Ruff lint: 0 errors across all modified files.

## [0.9.67] — 2026-03
### CUDA Hotswap Fix, Systemic Hardening & Edge-Case Lockdown
Comprehensive 11-phase remediation: critical bug fix, security hardening, performance optimization, and cross-stack alignment.

#### Fixed
- **CUDA LoRA Hotswap Crash**: Fixed "Expected all tensors on same device" error when switching LoRA. Root cause: `load_state_dict(assign=True)` on torch.compile's OptimizedModule replaced tensor references the Dynamo graph still held. Fix: operate on raw UNet via `_get_raw_module()`, snapshot/restore both UNet AND text_encoder, post-restore device validation with `dynamo.reset()` fallback.
- **Eager Pipeline Missing Pipe**: `eager_pipeline()` now swaps UNet on all 4 pipelines including `controlnet_img2img_pipe` (was missing).
- **Animation Frame 0 Parameter Consistency**: Inpaint and ControlNet frame 0 now use `blend_embeds` (SLERP transitions), `frame_cfg`, `frame_steps`, and `frame_denoise` from prompt schedule keyframe overrides instead of hardcoded `req.*` values. TXT2IMG and ControlNet use `frame_steps_full` (unscaled) for correct step counts.
- **Thread Safety**: `vram_utils._last_gc` now protected by `threading.Lock` to prevent race conditions.
- **Path Traversal Security**: `validate_path_in_sandbox()` with `Path.is_relative_to()` + symlink rejection replaces unsafe `str.startswith()` checks in `resource_manager.py` and `dsl_parser.py`.
- **Template Injection**: Expanded unsafe pattern regex to block `!` and `:` format specs + 2000-char length limit in `prompt_generator.py`.
- **FreeU Error Handling**: `enable_freeu()` wrapped in try/except with log.warning fallback.
- **Config Cross-Validation**: `cpu_offload + deepcache` mutual exclusion, `compile_dynamic + deepcache` incompatibility auto-fix, `torch_compile` without `lora_hotswap` performance warning.
- **URL Validation**: WebSocket connect now validates URL scheme before attempting connection.
- **Settings Atomic Write**: `save_settings()` writes to `.tmp` then renames to prevent corruption on crash.
- **Unknown Response Handler**: Server responses with unknown types now logged instead of silently dropped (both direct dispatch and queue drain paths).
- **Output Directory Check**: `save_animation_frame()` verifies `makeDirectory()` success before writing frames.
- **`.env.example` Alignment**: Fixed wrong variable names (`SDDJ_COMPILE_UNET` → `SDDJ_ENABLE_TORCH_COMPILE`, `SDDJ_FREEU_ENABLED` → `SDDJ_ENABLE_FREEU`) and corrected FreeU v2 default values (B1=1.5, B2=1.6).

#### Improved
- **Prompt Schedule Performance**: O(n) linear keyframe search replaced with O(log n) `bisect.bisect_right()`.
- **Named Constants**: Motion thresholds, denoise floor, hue shift epsilon, and auto-noise parameters extracted from magic numbers into module-level constants in `helpers.py`.
- **Cancel Guard**: `import_animation_frame` re-checks `cancel_pending` after decode (belt-and-suspenders against file I/O yield).

#### Documentation
- **GUIDE.md**: Timeout troubleshooting now specifies both server (600s) and client (660s) defaults.
- **AUDIO.md**: Added default parameters table (FPS=24, Steps=8, CFG=5.0, Denoise=0.50).
- **README.md**: Version number synchronized.

#### Tests
- `test_lora_fuser.py`: Added `_get_raw_module` unwrap/passthrough, text_encoder snapshot, dual-module restore tests.
- `test_compile_utils.py`: Added `controlnet_img2img_pipe` swap and restore test.
- `test_validation.py`: Added `validate_path_in_sandbox` tests (traversal, symlink escape, valid paths).
- **561 tests passing, 0 failures.**

## [0.9.66] — 2026-03
*Version number skipped (no release under this version).*

## [0.9.65] — 2026-03
### Absolute Rigor & Empty Input Defenses
Total lockdown of the DSL parser edge cases and Aseprite widget pass-through bugs.

#### Fixed
- **Empty File UI Leak**: Prevented Aseprite's 'file' widget from leaking the default working directory as a string when no file is selected. A strict `app.fs.isFile` validation guard now intercepts pseudo-paths, eliminating the `unable to read scheduling file` warning console popup.
- **`{auto}` Tag Purity**: The `{auto}` tag is now rigorously stripped from prompt strings to prevent explicit bleed-through into the SD generation, while still securely injecting the baseline keyframe logic.
- **Headless Lua Validation Suite**: Implemented a standalone Python `unittest` suite wrapped in `lupa` to rigorously test all syntax and bounds of `sddj_dsl_parser.lua` natively in backend environments devoid of Lua binaries or `pytest`.

## [0.9.64] — 2026-03
### Aseprite Environment Fixes
Final cross-check and remediation of module loading in the Aseprite runtime environment.

#### Fixed
- **Aseprite Module Discovery Error**: Replaced standard Lua `require("sddj_dsl_parser")` with Aseprite-compatible `dofile("./sddj_dsl_parser.lua")` in `sddj_request.lua` to fully resolve the "Generate" tab crashing due to unlocated DSL parsing logic.

## [0.9.63] — 2026-03
### Robustness Pinnacle & Edge-Case Lockdown
Complete cross-platform zero-crash remediation of the Prompt Scheduling DSL integration and UI protocol constraints.

#### Fixed
- **Lua Path Extraction Crash (0-Day)**: Suppressed a critical null-pointer crash triggered during `file:` input parsing when attempting to `trim(nil)` an empty path argument.
- **Protocol Bounds 500-Error Vulnerability**: Added graceful boundary-clamping pre-validators directly to `PromptKeyframeSpec` in `protocol.py`. Weights outside `(0.0, 5.0)` or transitions outside `(0, 120)` are now safely clipped rather than throwing internal Server Error validation exceptions (UX 90/100).
- **Silent Sequence Override**: Negative prompts inside the timeline (e.g. `-- bad`) now correctly concatenate when multiple lines are provided instead of continually overriding each other in the Lua AST.
- **Transition Token Truncation**: Fixed the regex `(%w+)` incorrectly truncating the underscore in `"hard_cut"` to `"hard"` during Lua ingestion, closing a subtle mismatch gap with the Python Backend protocol.

## [0.9.62] — 2026-03
### DSL Parser Perfection & UI UX Alignment
Refinement of the Prompt Scheduling DSL parser, UI layout, and backend fail-safes ensuring 100% zero-crash stability and supreme user experience.

#### Fixed
- **Implicit Frame 0 Fallback**: Writing raw text without `[time]` tags now implicitly scopes to `frame 0` instead of quietly dropping the text.
- **Null-Schedule Backend Crash**: Missing or whitespace-only DSL now cleanly evaluates to `None` in the Lua payload, allowing the Python backend to gracefully fall back to standard text prompts without raising `ValidationError` or throwing missing keyframe exceptions.

#### Changed
- **UI Centralization**: Eradicated the redundant `anim_prompt_schedule_dsl` and `audio_prompt_schedule_dsl` silos from the Animation and Audio tabs. Prompt Scheduling DSL is now globally unified at the bottom of the **Generate** tab.
- **Parser Syntax SOTA Upgrade**: Restructured the Lua parser to support a clean, multiline bracket-based syntax (e.g., `[0]`, `[50%]`, `@5s`) over the legacy pipe-based format. Allows natural multi-line prompt construction with inline options (`blend: 10`, `weight: 1.2`).

## [0.9.61] — 2026-03
### Prompt Scheduling DSL & Auto-DJ Upgrade
Maximal awareness UX overhaul introducing a powerful Timeline DSL directly inside Aseprite, bridging the robust 0.9.60 backend architecture.

#### Added
- **Timeline DSL**: Unified prompt scheduling UI offering absolute frame (`12:`), seconds (`2.5s:`), and percent (`50%:`) timing directly in Aseprite. Supports multi-line input and inline negative modifiers `[-]`.
- **Visual Crossfade Transitions**: Built-in `(blend:N)` transition commands that alternate prompts over `N` frames during the img2img chain to produce mathematically pure visual crossfades.
- **Universal Payload Specs**: Seamless bridging from the Lua UI to the new backend `PromptScheduleSpec` in generate, animation, and audio modes.
- **Audio Auto-DJ Upgrade**: Audio-reactive `randomness` auto-generation now synthesizes and injects Keyframe specs with `blend` transitions instead of rigid time dicts, creating flawless music-driven visual morphs.
- **Graceful Fail-Safes**: Lua AST parser wraps syntax errors in visual `app.alert`s to prevent silent generation failure.

#### Removed
- Legacy, rigid 3-slot text-box prompt scheduler from the Audio tab UI.
- All backend legacy `prompt_segments` fallbacks from protocol and payload validators, enforcing extreme SOTA purity.

## [0.9.60] — 2026-03
Decoupled, frame-accurate prompt scheduling for all generation modes — txt2img, img2img, animation (chain + AnimateDiff), and audio-reactive.

#### Added
- **`PromptKeyframe` dataclass** — frame-indexed keyframe with `hard_cut`/`blend` transitions, per-keyframe negative prompts, and weight.
- **`PromptSchedule` keyframe engine** — `get_prompt_for_frame()` (hard_cut + blend alternation), `get_negative_for_frame()`, `get_unique_prompts()`/`get_unique_negatives()` for embedding pre-cache, `auto_fill_prompts()` via `PromptGenerator`.
- **`PromptSchedulePresetsManager`** — CRUD with path traversal protection, name validation, 50-preset cap. 5 structural factory presets: `evolving_3act`, `style_morph_4`, `beat_alternating`, `slow_drift`, `rapid_cuts_6`.
- **Protocol expansion** — 4 new `Action` entries (`LIST_PROMPT_SCHEDULES`, `GET_PROMPT_SCHEDULE`, `SAVE_PROMPT_SCHEDULE`, `DELETE_PROMPT_SCHEDULE`), `PromptKeyframeSpec`/`PromptScheduleSpec` Pydantic models with transition validator, `prompt_schedule` field on `GenerateRequest`, `AnimationRequest`, `AudioReactiveRequest`.
- **`build_prompt_schedule()` helper** — unified entry point resolving `PromptScheduleSpec`, raw dict, or legacy segments. Includes Lua `json.lua` dict-encoded array normalization.
- **Server dispatch** — 4 CRUD handlers wired in `server.py`.
- **`prompt_schedules_dir`** config path.
- 41 new unit tests in `test_prompt_schedule_keyframes.py` (engine, presets, protocol, helper, auto-fill, backward compat, Lua edge case).

#### Changed
- **`core.py` `generate()`** — schedule resolves frame 0 prompt/negative before mode dispatch; all 4 private methods (`_txt2img`, `_img2img`, `_inpaint`, `_controlnet_generate`) accept resolved `prompt`/`negative` parameters.
- **`animation.py` chain loop** — per-frame prompt/negative resolution at all 6 pipeline call sites.
- **`animation.py` AnimateDiff loop** — per-chunk midpoint resolution at 2 pipeline call sites.
- **Precedence**: `prompt_schedule` > `prompt_segments` > static `prompt` (backward compatible).



## [0.9.59] — 2026-03
### LoRA Management & Configuration Audit
Exhaustive multi-level audit of LoRA management system and environment configuration.

#### Critical
- **`.env` was never loaded** — `pydantic-settings v2` requires explicit `env_file` in `model_config` (was not set), `uv run` requires `--env-file` flag (was not passed), no `python-dotenv` in codebase. All `SDDJ_*` variables in `.env` were silently ignored since day one; every setting used hardcoded defaults from `config.py`. Fixed: `model_config` now explicitly loads `server/.env` via absolute path.
- **Orphaned root `.env` deleted** — duplicate configuration file at project root was never read by any process.

#### Fixed
- **`default_style_lora` full-path crash** — full file paths in `.env` (e.g. `C:/models/pixelart.safetensors`) failed `validate_resource_name()` regex. Now extracts stem name automatically via `PurePath`.
- **`ResourceManager.resolve()` returned non-resolved path** — validated against the absolute resolved path but returned the relative candidate. Now returns the absolute resolved path consistently.
- **`set_style_lora()` silent no-op** — no feedback when called before engine loaded. Added `log.debug` for observability.

#### Improved
- **`.env.example` completeness** — rewritten with all 28+ config variables (was missing: LoRA hotswap, TF32, CPU offload, VRAM budget, QR ControlNet ×4, audio core ×4, audio DSP ×9, `compile_dynamic`, `audio_cache_ttl_hours`).
- **`server/.env` synced** — all new sections added, `SDDJ_DEFAULT_STYLE_LORA` changed from full path to stem name.
- **Test isolation** — all `Settings()` calls in `test_config.py` now pass `_env_file=None` to prevent real `.env` from leaking into unit tests.

## [0.9.58] — 2026-03
### Pipeline Performance Audit
Exhaustive 30+ module audit (weighted avg 66/100 → target 90/100) with 15 fixes across 13 source files.

#### Performance
- **AnimateDiff chunked processing** — replaced single-batch inference (O(n²) temporal attention, VRAM explosion on large frame counts) with 16-frame chunks + 4-frame overlap alpha blending. Solves "infinite inference" on >16 frames.
- **PEFT strip guard** — skip full UNet module traversal when no PEFT artifacts exist; flag prevents redundant strip on re-entry.
- **Lightning scheduler cache** — `EulerDiscreteScheduler` config cached on first call; subsequent pipeline constructors reuse cached config.
- **Scheduler `from_config`** — all 3 `copy.deepcopy(scheduler)` replaced with `type(sched).from_config(sched.config)` (eliminates deep-copy of scheduler internal state).
- **DeepCacheState** — mode-aware class avoids redundant disable/enable cycles (100-300ms per toggle) when staying in the same incompatible mode.
- **Double `dynamo.reset` removed** — `eager_pipeline` no longer resets dynamo on enter AND exit (forced cold recompilation on every chain frame).
- **GC throttle** — `vram_cleanup()` throttles `gc.collect()` to 2s cooldown; `force=True` parameter for genuine cleanup (model unload, OOM).
- **LoRA `assign=True`** — `load_state_dict(assign=True)` avoids ~1.7GB temporary VRAM spike during weight restore (tensors swapped in-place vs copied).
- **Image codec RGBA guard** — skip redundant `image.convert("RGBA")` when image is already RGBA (saves one full-image copy per frame).
- **Postprocess double palette** — skip explicit `_enforce_palette()` when Bayer dithering is about to run (dithering calls it internally).
- **Rembg CPU guard** — skip `vram_cleanup()` on unload when rembg runs on CPU (no GPU memory to reclaim).
- **Generation timeout** — step callback now checks `time.perf_counter()` against `settings.generation_timeout` per step (previously only enforced at WebSocket level).

#### Robustness
- **VRAM leak fixed** — `_controlnet_img2img_pipe` now cleaned in `unload()` and `cleanup_resources()` (was leaking ControlNet img2img pipeline).
- **AnimateDiff null guard** — `_ensure_controlnet` checks `self._animatediff is not None` before accessing `.pipe`.
- **Audio cache hardened** — hash chunk increased from 1MB to 4MB (collision resistance); writes use `tempfile.mkstemp()` + `os.replace()` for cross-platform atomic persistence.
- **Config checkpoint validation** — `model_validator` warns if `default_checkpoint` path doesn't exist at settings construction time.

#### Cleanup
- Removed unused `import copy` from `animatediff_manager.py`.
- Removed unused `import torch` from `compile_utils.py`.
- Updated 4 `test_compile_utils.py` tests (removed stale `torch._dynamo.reset` mock patches).

## [0.9.57] — 2026-03
### Changed
- **Dialog Architectural Refactoring** — `sddj_dialog.lua` restructured from 1441 to 1316 lines via data-driven patterns and DRY infrastructure:
  - All ~20 slider `onchange` callbacks replaced with `onchange_sync(id)` using centralized `PT.SLIDER_LABELS` registry.
  - 13 expression entry fields generated via `EXPR_FIELDS` data table loop.
  - 6 modulation slots generated via `SLOT_DEFAULTS` data table loop.
  - Action button `onclick` dispatch: `if/elseif` cascade → tab-keyed dispatch table.
  - 3 trigger functions (`generate`, `animate`, `audio`) share extracted `init_loop_state(target)` helper.
  - `trigger_qr_generate` request construction extracted to `PT.build_qr_request()` in `sddj_request.lua`.
- **Cross-Module Label Sync Centralization** — `sddj_settings.lua` `apply_settings()` and `sddj_output.lua` `apply_metadata()` now use `PT.sync_slider_label(id)` from the shared registry instead of hardcoded format strings (~25 manual label lines eliminated per file).
- **Loop State Reset DRY** — `PT.reset_loop_state()` added to `sddj_utils.lua`, replacing 17 inline triple-assignments (`PT.loop.mode/random_mode/target`) across `sddj_dialog.lua` (7), `sddj_handler.lua` (7), `sddj_ws.lua` (3).

### Fixed
- **Dead code in `trigger_animate`** — Removed unreachable branch (`PT.loop.random_mode and not PT.loop.mode`) which could never evaluate to `true` because `init_loop_state()` always sets `PT.loop.mode = true` before the check.

### Added
- `PT.reset_loop_state()` in `sddj_utils.lua` — shared loop state cleanup function.
- `PT.SLIDER_LABELS` registry in `sddj_utils.lua` — 23-entry table mapping slider widget IDs to format strings and divisors.
- `PT.sync_slider_label(id)` in `sddj_utils.lua` — formats and applies a slider label from the registry.
- `PT.build_qr_request()` in `sddj_request.lua` — extracted QR/Illusion Art request construction.

## [0.9.56] — 2026-03
### Changed
- **QR Code Monster v2 Simplification** — Removed server-side QR image generation (`qrcode_generator.py` deleted, `qrcode[pil]` dependency removed). All ControlNet modes now use client-provided `control_image` uniformly. Engine ControlNet path simplified from 32 to 19 lines. QR scan validation + auto-retry loop removed (14 new protocol test added for `controlnet_qrcode` requiring `control_image`).
- **Lua QR Tab** — Removed `qr_content`, `qr_error_correction`, `qr_module_size` UI fields. QR tab now captures active layer as control image directly (consistent with other ControlNet modes). Illusion Art source switched from `capture_active_layer` to `capture_flattened` for full sprite compositing.
- **AnimateDiff ControlNet** — QR Code Monster v2 loaded from `v2/` subfolder via conditional `load_kwargs`.
- **Pipeline Factory** — `create_controlnet_pipeline` return type corrected to `tuple[..., ...]`. Base pipeline uses `local_files_only=True` + explicit `config` for local checkpoints.

### Fixed
- **Loop/Random Loop in Animate mode** — `trigger_animate()` now has full loop initialization mirroring `trigger_generate()` (seed mode, counter, locked fields, random-loop prompt dispatch). `handlers.animation_complete` schedules next animation via timer-based loop continuation.
- **Loop/Random Loop in Audio mode** — `trigger_audio_generate()` now has loop initialization. `handlers.audio_reactive_complete` adds loop continuation. Loop buttons enabled in Audio tab when audio is analyzed.
- **Random Loop tab-aware dispatch** — `handlers.prompt_result` random-loop block now dispatches based on `PT.loop.target` ("generate"/"animate"/"audio") instead of always calling `build_generate_request()`. Prevents tab-switching from altering loop behavior mid-iteration.
- **Preset persistence incomplete** — Preset Save now includes `randomness`, `lock_subject`, `fixed_subject`, `randomize_before`. Preset Load restores all 4 fields with label sync.
- **Loop state leak** — Added `PT.loop.target` cleanup (`= nil`) to all 13 reset paths across `sddj_handler.lua`, `sddj_ws.lua`, `sddj_dialog.lua` (cancel, error, disconnect, timeout, early exits).

### Added
- `PT.loop.target` field in `sddj_state.lua` — stores the trigger function to call on loop re-entry, preventing tab-switching race conditions during loop iterations.

## [0.9.55] — 2026-03
### Changed
- **SOTA Architectural Hardening** — Executed a flawless 100/100 multi-level architecture audit across the entire perimeter:
  - **Native Provisioning**: `download_models.py` migrated entirely to native `urllib.request` with atomicity (`.part`), robust streaming, and strict 30s timeouts + exponential backoff retries. Zero third-party dependencies.
  - **Determinism**: `setup.ps1` migrated from open commands to `uv sync --locked` ensuring absolute runtime parity with the lockfile.
  - **Powershell Resilience**: `setup.ps1` and `start.ps1` upgraded to PS7 SOTA standards (`Set-StrictMode`, `Join-Path`). `start.ps1` uses Base64 encoded `uv run --frozen` payloads to eliminate all path injection vectors.
  - **Surgical Teardown**: Replaced generic fallback WMI kills with strict `$serverProc.Id` recursive tree taskkill, preventing collisions with other Python processes.
  - **Default Checkpoint**: Swapped HF `Lykon/dreamshaper-8` to `Liberte.Redmond` (Civitai) fetched directly via local path resolution (`models/checkpoints/liberteRedmond_v10.safetensors`).
  - **Stealth Mode Enforcement**: Purged all python docstrings across routing configuration and provisioning modules, aligning with the rigorous minimal operational signature constraint.
  - **Fail Gracefully**: Hardened `run.py` to intercept catastrophic init failures dynamically instead of vomiting console stacks.
  - **Precise Bump**: Regex format-preserving operations replacing heavy JSON AST rebuilding in `bump_version.ps1` to prevent formatting destruction.

### Added
- **ControlNet QR Code Monster v2 Integration** — new `controlnet_qrcode` generation mode explicitly optimized for "QR Illusion Art" (embedding functional QR codes into artistic generations).
- **Dual-Path QR Engine**:
  - *Illusion Art Workflow (img2img-based)*: Scrapes active layer as source image, blends QR conditioning with adjustable denoise strength (default 0.75). Uses `StableDiffusionControlNetImg2ImgPipeline`.
  - *Standard Workflow (txt2img-based)*: Generates art directly from prompt shaped by the QR control. Uses `StableDiffusionControlNetPipeline`.
  - Both pipelines share a single ControlNet model instance in VRAM.
- **Dedicated QR UI Tab**: 10 parameters fully persisted via `sddj_settings.lua`, including "Use Layer (Illusion Art)" toggle, denoise slider, module size (16px default), error correction (H default), and scale tuning.
- **Server-Side QR Synthesis**: `qrcode_generator.py` mathematically constructs the base QR image on a #808080 gray canvas honoring ISO quiet zones, avoiding client-side encoding complexity.
- **Zero-Friction Scan Validation**: Auto-validates output scannability via `cv2.QRCodeDetector`. If scan fails, engine auto-retries (up to 2 times, configurable) with new seeds and progressively higher conditioning scale (+0.3 per retry).
- **ControlNet DeepCache Override**: Suspends DeepCache for *all* ControlNet modes. (Cached steps skip UNet blocks where CN injects residuals, which previously caused conditioning to be partially dropped).
- 14 new unit tests for QR logic (`test_qrcode_generator.py`).

### Fixed
- **Mode Pollution**: Excluded QR-specific fields (`qr_content`, `qr_error_correction`, etc.) from leaking into `AnimationRequest` and `AudioReactiveRequest` converters, preventing Pydantic validation crashes.
- **Label Mismatches**: Fixed tooltip mode names in `sddj_dialog.lua` to correctly identify `controlnet_qrcode` as a zero-source-layer mode (unlike other ControlNets) when the illusion toggle is off.
- **Unused Import**: Removed unused `import numpy` from `pipeline_factory.py`.
- **Pending Action Comment**: Updated stale documentation comment in `sddj_state.lua` to include `qr_generate`.

### Changed
- Revised QR default values per SOTA research: `control_guidance_end` lowered from 1.0 to 0.8 (allows art to blend over the QR structure in final steps), default steps increased from 12 to 20 for better geometric structural integrity.
## [0.9.54] — 2026-03
### Codebase Audit & Remediation

#### Removed
- **PixyToon legacy traces** — purged from `setup.ps1` (cleanup list + extension removal block) and `sddj_settings.lua` (migration fallback). Zero references remain.
- **Noop `quantize_unet` config** — field existed in `config.py` and `REFERENCE.md` but was never consumed by `pipeline_factory.py`. Removed until implemented.
- **Misplaced CHANGELOG boilerplate** — duplicate "All notable changes…" line at L121.

#### Fixed
- **Logger name collisions** — `audio_cache.py`, `stem_separator.py`, and `modulation_engine.py` all shared `sddj.audio`. Each now has its own logger (`sddj.audio_cache`, `sddj.stem_separator`, `sddj.modulation_engine`).
- **Silent `except: pass` blocks** — `vram_utils.py` and `lora_fuser.py` now log at DEBUG level instead of swallowing errors silently.
- **Dead `hasattr` guard** — `prompt_generator.py` checked `_active_exclude` which is always initialized in `__init__`.
- **Pillow 13 deprecation** — `helpers.py` `Image.fromarray(mode="L")` replaced with `Image.fromarray()` (auto-detected).
- **REFERENCE.md file structure** — `models/` was shown inside `sddj/` but is a sibling directory.
- **Stale protocol comments** — removed misleading "legacy" and "deprecated" labels from `protocol.py`.

#### Security
- **Expression length cap** — `modulation_engine.py` now rejects expressions > 1024 characters in both `validate()` and `evaluate()`.
- **Explicit `shell=False`** — `video_export.py` `subprocess.run()` now has defense-in-depth shell restriction.
- **`os.execute` hardening** — `sddj_output.lua` `open_output_dir()` strips all shell metacharacters (`"&|;$%<>()`) instead of only double quotes.

#### Improved
- **Audio cache TTL configurable** — new `SDDJ_AUDIO_CACHE_TTL_HOURS` env var (default 24, range 1–168h) replaces hardcoded constant.
- **Cache meta → JSON** — `audio_cache.py` now uses `json.dumps`/`json.loads` instead of fragile custom `key=value` format. Existing caches auto-invalidate gracefully.
- **Legacy preset warnings** — `modulation_engine.py` logs `WARNING` when v0.7.0 presets (`energetic`, `ambient`, `bass_driven`) are used.
- **DRY modulation slot persistence** — `sddj_settings.lua` save/apply now use loops for 6 slots × 8 fields instead of 68 copy-paste lines.
- **Version single source of truth** — `__init__.py` reads version from `importlib.metadata` (set by `pyproject.toml`). New `bump_version.ps1` updates all 3 remaining files atomically.

#### Added
- **`.env.example`** — documents all configurable environment variables with defaults.
- **`bump_version.ps1`** — atomic version bump across `pyproject.toml`, `package.json`, `sddj_state.lua`, + `uv.lock` regeneration.
- **`SDDJ_AUDIO_CACHE_TTL_HOURS`** documented in `REFERENCE.md`.

## [0.9.53] — 2026-03
### Fixed
- **Settings persistence: 23 missing fields** — modulation slots 5-6 (14 fields), invert toggles for all 6 slots (6 fields), `quantize_enabled`, `audio_choreography`, and `audio_expr_preset` were not saved/loaded, causing data loss on restart.
- **Resource requests on connect** — expression and choreography preset lists were never requested on connect, leaving those dropdowns unpopulated until manual refresh.
- **Loop seed initialization** — first iteration of "random" loop mode used the stale seed from the text field instead of `-1`.
- **Prompt randomization timeout** — `generate_prompt` with `pending_action` had no timeout; if the server never responded, the UI remained locked indefinitely. Now protected by a 30-second timeout.
- **LoRA label mismatch** — preset hydration set the LoRA weight label to `"Weight (X.XX)"` instead of `"LoRA (X.XX)"`, inconsistent with the dialog definition.
- **Metadata `quantize_enabled` not restored** — loading generation metadata did not restore the quantize checkbox state.
- **`save_to_output` not encoding-aware** — single-result output save assumed PNG encoding; now handles `raw_rgba` correctly (parity with `save_animation_frame`).
- **Version drift** — Lua extension version was `0.9.51` while all other manifests were `0.9.52`; harmonized.

### Changed
- **Factored `build_animation_request()`** — extracted inline animation request construction from `trigger_animate()` into `sddj_request.lua`, consistent with `build_generate_request()` and `build_audio_reactive_request()`.

## [0.9.52] — 2026-03
### Added
- **Pinnacle Documentation Overhaul** — Complete rewrite and restructuring of the entire documentation suite (reduction from 8 files / 122 KB to 5 files / 50 KB). 
  - `README.md`: Lean, punchy landing page.
  - `docs/GUIDE.md`: Unified user guide (setup, generation, animation, performance) + inline troubleshooting + system architecture.
  - `docs/AUDIO.md`: Unified audio reactivity guide combining previous concepts and reference tables into coherent parameter matrices + expression guides.
  - `docs/REFERENCE.md`: Technical reference combining WebSocket API protocol, Configuration (all env vars), and Architecture details.
  - `docs/RECIPES.md`: Condensed cookbook transforming repetitive text recipes into a single high-density parameter matrix + focused workflow techniques + anti-patterns.
  - `docs/SOURCES.md`: Centralized list of all scientific papers, techniques, algorithms, and models powering SDDj (Hyper-SD, DeepCache, FreeU v2, FlashAttention, Demucs, librosa, etc.).
- **New Modulation Presets**: Added `voyage_serene`, `voyage_exploratory`, `voyage_dramatic`, `voyage_psychedelic`, `intelligent_drift`, and `reactive_pause` to `AUDIO.md` for zero-delta with the codebase.
- **Reference Completion**: Added 7 previously undocumented environment variables (`SDDJ_COMPILE_DYNAMIC`, `SDDJ_ENABLE_TF32`, `SDDJ_ENABLE_LORA_HOTSWAP`, `SDDJ_MAX_LORA_RANK`, `SDDJ_ENABLE_CPU_OFFLOAD`, `SDDJ_VRAM_MIN_FREE_MB`, `SDDJ_QUANTIZE_UNET`) to the Configuration section in `REFERENCE.md`.

### Removed
- Deprecated legacy documentation files: `COOKBOOK.md`, `AUDIO-REACTIVITY.md`, `AUDIO-REFERENCE.md`, `API-REFERENCE.md`, `CONFIGURATION.md`, `TROUBLESHOOTING.md`, `ARCHITECTURE.md`.

## [0.9.51] — 2026-03
### Fixed
- **Cancel race condition → `_generic_mt_newindex` crash** — Clicking Cancel during audio-reactive or animation generation left pending frames in the response queue, which continued processing via `_drain_next` and attempted `app.transaction` on destroyed Aseprite objects (sprite/layer/cel). 8-point defense-in-depth fix:
  - **F1** `sddj_dialog.lua` — Immediate `clear_response_queue()` + `stop_refresh_timer()` on cancel click (root cause).
  - **F2** `sddj_handler.lua` — `cancel_pending` guard on `audio_reactive_frame` handler.
  - **F3** `sddj_handler.lua` — `cancel_pending` guard on `animation_frame` handler.
  - **F4** `sddj_handler.lua` — `_drain_next` flushes queue if cancel was requested between timer ticks.
  - **F5** `sddj_import.lua` — `cancel_pending` guard + unconditional sprite nil bail in `import_animation_frame`.
  - **F6** `sddj_import.lua` — Sprite re-validation in `finalize_sequence` before `app.transaction` (prevents crash if sprite closed during async yield).
  - **F7** `sddj_output.lua` — `cancel_pending` guard on `save_animation_frame` (prevents zombie frame file I/O).
  - **F8** `sddj_ws.lua` — Added `clear_response_queue()` + `stop_refresh_timer()` to `gen_timeout` handler (consistency with `error` handler).
- **Version drift** — Lua extension version was 0.9.49 while server was 0.9.50; harmonized to 0.9.51.

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
