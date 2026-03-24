"""Tests for modulation_engine — slots, expressions, schedule computation."""

from __future__ import annotations

import numpy as np
import pytest

from sddj.audio_analyzer import AudioAnalysis
from sddj.modulation_engine import (
    ExpressionEvaluator,
    ModulationEngine,
    ModulationSlot,
    MOTION_MAX_DELTA,
    ParameterSchedule,
    PRESETS,
    TARGET_RANGES,
)


# ─── Helpers ────────────────────────────────────────────────

def _make_analysis(n_frames=10, fps=24.0, features=None):
    """Create a test AudioAnalysis with controllable features."""
    if features is None:
        features = {
            "global_rms": np.full(n_frames, 0.5, dtype=np.float32),
            "global_onset": np.linspace(0, 1, n_frames, dtype=np.float32),
        }
    return AudioAnalysis(
        fps=fps,
        duration=n_frames / fps,
        total_frames=n_frames,
        sample_rate=22050,
        audio_path="test.wav",
        features=features,
    )


# ─── ParameterSchedule ─────────────────────────────────────

class TestParameterSchedule:
    def test_get_params_valid_index(self):
        schedule = ParameterSchedule(
            total_frames=3,
            frame_params=[{"a": 1.0}, {"a": 2.0}, {"a": 3.0}],
        )
        assert schedule.get_params(0) == {"a": 1.0}
        assert schedule.get_params(2) == {"a": 3.0}

    def test_get_params_out_of_range(self):
        schedule = ParameterSchedule(total_frames=1, frame_params=[{"x": 1.0}])
        assert schedule.get_params(-1) == {}
        assert schedule.get_params(5) == {}

    def test_get_chunk_params_average(self):
        """v0.7.3: get_chunk_params averages parameters over a frame range."""
        schedule = ParameterSchedule(
            total_frames=4,
            frame_params=[
                {"denoise_strength": 0.2, "cfg_scale": 4.0},
                {"denoise_strength": 0.4, "cfg_scale": 6.0},
                {"denoise_strength": 0.6, "cfg_scale": 8.0},
                {"denoise_strength": 0.8, "cfg_scale": 10.0},
            ],
        )
        chunk = schedule.get_chunk_params(0, 4)
        assert chunk["denoise_strength"] == pytest.approx(0.5)
        assert chunk["cfg_scale"] == pytest.approx(7.0)

    def test_get_chunk_params_partial_range(self):
        schedule = ParameterSchedule(
            total_frames=4,
            frame_params=[
                {"denoise_strength": 0.2},
                {"denoise_strength": 0.8},
                {"denoise_strength": 0.4},
                {"denoise_strength": 0.6},
            ],
        )
        chunk = schedule.get_chunk_params(1, 3)
        assert chunk["denoise_strength"] == pytest.approx(0.6)

    def test_get_chunk_params_empty(self):
        schedule = ParameterSchedule(total_frames=2, frame_params=[{}, {}])
        assert schedule.get_chunk_params(0, 2) == {}

    def test_get_chunk_params_out_of_range(self):
        schedule = ParameterSchedule(total_frames=2, frame_params=[{"a": 1.0}, {"a": 2.0}])
        assert schedule.get_chunk_params(5, 10) == {}
        assert schedule.get_chunk_params(2, 1) == {}  # start >= end


# ─── ExpressionEvaluator ───────────────────────────────────

class TestExpressionEvaluator:
    def test_simple_math(self):
        ev = ExpressionEvaluator()
        result = ev.evaluate("2 + 3", {})
        assert result == 5.0

    def test_with_variables(self):
        ev = ExpressionEvaluator()
        result = ev.evaluate("x * 2", {"x": 3.0})
        assert result == 6.0

    def test_clamp_function(self):
        ev = ExpressionEvaluator()
        assert ev.evaluate("clamp(5, 0, 3)", {}) == 3.0
        assert ev.evaluate("clamp(-1, 0, 3)", {}) == 0.0
        assert ev.evaluate("clamp(1.5, 0, 3)", {}) == 1.5

    def test_lerp_function(self):
        ev = ExpressionEvaluator()
        assert ev.evaluate("lerp(0, 10, 0.5)", {}) == pytest.approx(5.0)

    def test_smoothstep_function(self):
        ev = ExpressionEvaluator()
        assert ev.evaluate("smoothstep(0, 1, 0.5)", {}) == pytest.approx(0.5)
        assert ev.evaluate("smoothstep(0, 1, 0)", {}) == pytest.approx(0.0)
        assert ev.evaluate("smoothstep(0, 1, 1)", {}) == pytest.approx(1.0)

    def test_where_function(self):
        ev = ExpressionEvaluator()
        assert ev.evaluate("where(1, 10, 20)", {}) == 10.0
        assert ev.evaluate("where(0, 10, 20)", {}) == 20.0

    def test_trig_functions(self):
        ev = ExpressionEvaluator()
        assert ev.evaluate("sin(0)", {}) == pytest.approx(0.0)
        assert ev.evaluate("cos(0)", {}) == pytest.approx(1.0)

    def test_audio_feature_variables(self):
        ev = ExpressionEvaluator()
        result = ev.evaluate(
            "0.2 + 0.3 * drums_rms",
            {"drums_rms": 0.8, "t": 0, "max_f": 10, "fps": 24.0, "s": 0.0},
        )
        assert result == pytest.approx(0.44)

    def test_validate_valid(self):
        ev = ExpressionEvaluator()
        err = ev.validate("sin(t) * global_rms", ["global_rms"])
        assert err is None

    def test_validate_invalid_syntax(self):
        ev = ExpressionEvaluator()
        err = ev.validate("1 +/ 2", ["global_rms"])
        assert err is not None

    def test_validate_unknown_function(self):
        ev = ExpressionEvaluator()
        err = ev.validate("import('os')", [])
        assert err is not None


# ─── ModulationEngine ──────────────────────────────────────

class TestModulationEngine:
    def test_single_slot_constant_feature(self):
        """Feature=0.5 with min=0.2, max=0.8 → output=0.5."""
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=5, features={
            "global_rms": np.full(5, 0.5, dtype=np.float32),
        })
        slots = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8,
        )]
        schedule = engine.compute_schedule(analysis, slots)

        for i in range(5):
            params = schedule.get_params(i)
            assert params["denoise_strength"] == pytest.approx(0.5, abs=0.01)

    def test_slot_maps_zero_to_min(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([0.0], dtype=np.float32),
        })
        slots = [ModulationSlot(
            source="global_rms", target="cfg_scale",
            min_val=3.0, max_val=9.0,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        assert schedule.get_params(0)["cfg_scale"] == pytest.approx(3.0)

    def test_slot_maps_one_to_max(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([1.0], dtype=np.float32),
        })
        slots = [ModulationSlot(
            source="global_rms", target="cfg_scale",
            min_val=3.0, max_val=9.0,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        assert schedule.get_params(0)["cfg_scale"] == pytest.approx(9.0)

    def test_multi_slot_same_target_averages(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([1.0], dtype=np.float32),
            "global_onset": np.array([0.0], dtype=np.float32),
        })
        slots = [
            ModulationSlot(source="global_rms", target="denoise_strength",
                          min_val=0.2, max_val=0.8),
            ModulationSlot(source="global_onset", target="denoise_strength",
                          min_val=0.2, max_val=0.8),
        ]
        schedule = engine.compute_schedule(analysis, slots)
        # (0.8 + 0.2) / 2 = 0.5
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.5)

    def test_clamping_to_target_range(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([1.0], dtype=np.float32),
        })
        # Slot with min/max outside valid range
        slots = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.0, max_val=2.0,  # max > 0.95 limit
        )]
        schedule = engine.compute_schedule(analysis, slots)
        assert schedule.get_params(0)["denoise_strength"] <= 0.95

    def test_disabled_slot_ignored(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([1.0], dtype=np.float32),
        })
        slots = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8, enabled=False,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        assert schedule.get_params(0) == {}

    def test_unknown_source_ignored(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([0.5], dtype=np.float32),
        })
        slots = [ModulationSlot(source="nonexistent", target="cfg_scale")]
        schedule = engine.compute_schedule(analysis, slots)
        assert schedule.get_params(0) == {}

    def test_expression_overrides_slot(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([0.5], dtype=np.float32),
        })
        slots = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.1, max_val=0.9,
        )]
        expressions = {"denoise_strength": "0.42"}
        schedule = engine.compute_schedule(analysis, slots, expressions)
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.42)

    def test_expression_with_frame_vars(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=10, features={
            "global_rms": np.full(10, 0.5, dtype=np.float32),
        })
        expressions = {"cfg_scale": "5.0 + t"}
        schedule = engine.compute_schedule(analysis, [], expressions)
        assert schedule.get_params(0)["cfg_scale"] == pytest.approx(5.0)
        assert schedule.get_params(5)["cfg_scale"] == pytest.approx(10.0)

    def test_expression_clamped(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([0.5], dtype=np.float32),
        })
        expressions = {"cfg_scale": "999.0"}  # exceeds max 30.0
        schedule = engine.compute_schedule(analysis, [], expressions)
        assert schedule.get_params(0)["cfg_scale"] == pytest.approx(30.0)

    def test_seed_offset_integer(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([0.5], dtype=np.float32),
        })
        slots = [ModulationSlot(
            source="global_rms", target="seed_offset",
            min_val=0, max_val=100,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        val = schedule.get_params(0)["seed_offset"]
        assert val == float(int(val))  # must be integer-valued

    def test_empty_schedule_no_slots_no_expressions(self):
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=5)
        schedule = engine.compute_schedule(analysis, [])
        for i in range(5):
            assert schedule.get_params(i) == {}

    def test_per_slot_ema_uses_raw_features(self):
        """Per-slot EMA smoothing uses raw_features when available."""
        engine = ModulationEngine()
        # Raw: step function 0→1 at frame 5
        raw = np.zeros(20, dtype=np.float32)
        raw[5:] = 1.0
        # Smoothed: already heavily smoothed (simulates global EMA)
        smoothed = np.full(20, 0.3, dtype=np.float32)
        analysis = AudioAnalysis(
            fps=24.0, duration=20/24.0, total_frames=20,
            sample_rate=22050, audio_path="test.wav",
            features={"global_rms": smoothed},
            raw_features={"global_rms": raw},
        )
        slots = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8, attack=1, release=4,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        # Frame 0-4: feature=0 → denoise near min (0.2)
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.2, abs=0.01)
        # Frame 10+: feature has risen toward 1.0 via EMA → denoise well above 0.5
        d10 = schedule.get_params(10)["denoise_strength"]
        assert d10 > 0.5, f"Expected denoise > 0.5 at frame 10, got {d10}"

    def test_per_slot_ema_fast_attack(self):
        """attack=1 responds faster than attack=8 to a step input."""
        engine = ModulationEngine()
        raw = np.zeros(20, dtype=np.float32)
        raw[5:] = 1.0
        analysis = AudioAnalysis(
            fps=24.0, duration=20/24.0, total_frames=20,
            sample_rate=22050, audio_path="test.wav",
            features={"global_rms": raw.copy()},
            raw_features={"global_rms": raw},
        )
        # Fast attack
        slots_fast = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.0, max_val=1.0, attack=1, release=4,
        )]
        sched_fast = engine.compute_schedule(analysis, slots_fast)

        # Slow attack
        engine2 = ModulationEngine()
        slots_slow = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.0, max_val=1.0, attack=8, release=4,
        )]
        sched_slow = engine2.compute_schedule(analysis, slots_slow)

        # At frame 6 (1 frame after step), fast attack should be higher
        d_fast = sched_fast.get_params(6)["denoise_strength"]
        d_slow = sched_slow.get_params(6)["denoise_strength"]
        assert d_fast > d_slow, f"Fast ({d_fast}) should exceed slow ({d_slow})"

    def test_fallback_to_smoothed_when_no_raw(self):
        """When raw_features is empty, uses smoothed features (backward compat)."""
        engine = ModulationEngine()
        analysis = _make_analysis(n_frames=1, features={
            "global_rms": np.array([0.5], dtype=np.float32),
        })
        # raw_features is empty dict (default)
        assert analysis.raw_features == {}
        slots = [ModulationSlot(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        assert schedule.get_params(0)["denoise_strength"] == pytest.approx(0.5, abs=0.01)


# ─── Presets ────────────────────────────────────────────────

class TestPresets:
    def test_get_preset_energetic(self):
        slots = ModulationEngine.get_preset("energetic")
        assert len(slots) > 0
        assert all(isinstance(s, ModulationSlot) for s in slots)

    def test_get_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown modulation preset"):
            ModulationEngine.get_preset("nonexistent")

    def test_list_presets(self):
        presets = ModulationEngine.list_presets()
        assert "energetic" in presets
        assert "ambient" in presets
        assert "bass_driven" in presets

    def test_all_presets_have_valid_targets(self):
        for name, slots in PRESETS.items():
            for slot in slots:
                assert slot["target"] in TARGET_RANGES, \
                    f"Preset {name!r} has invalid target: {slot['target']}"

    def test_new_targets_in_ranges(self):
        """v0.7.3: palette_shift and frame_cadence must be in TARGET_RANGES."""
        assert "palette_shift" in TARGET_RANGES
        assert "frame_cadence" in TARGET_RANGES
        # palette_shift: 0-1
        assert TARGET_RANGES["palette_shift"] == (0.0, 1.0)
        # frame_cadence: 1-8
        assert TARGET_RANGES["frame_cadence"] == (1.0, 8.0)

    def test_motion_targets_in_ranges(self):
        """v0.7.4+: motion targets including tilt in TARGET_RANGES."""
        assert TARGET_RANGES["motion_x"] == (-5.0, 5.0)
        assert TARGET_RANGES["motion_y"] == (-5.0, 5.0)
        assert TARGET_RANGES["motion_zoom"] == (0.92, 1.08)
        assert TARGET_RANGES["motion_rotation"] == (-2.0, 2.0)
        assert TARGET_RANGES["motion_tilt_x"] == (-3.0, 3.0)
        assert TARGET_RANGES["motion_tilt_y"] == (-3.0, 3.0)

    def test_motion_presets_exist(self):
        """v0.7.4: 4 dedicated motion presets."""
        for name in ("gentle_drift", "pulse_zoom", "slow_rotate", "cinematic_sweep"):
            slots = ModulationEngine.get_preset(name)
            assert len(slots) > 0, f"Preset {name!r} has no slots"
            targets = {s.target for s in slots}
            # Each motion preset must have at least one motion target
            motion_targets = targets & {"motion_x", "motion_y", "motion_zoom", "motion_rotation",
                                        "motion_tilt_x", "motion_tilt_y"}
            assert len(motion_targets) > 0, f"Preset {name!r} has no motion targets"

    def test_perspective_presets_exist(self):
        """New perspective/advanced camera presets."""
        for name in ("cinematic_tilt", "zoom_breathe", "parallax_drift", "full_cinematic"):
            slots = ModulationEngine.get_preset(name)
            assert len(slots) > 0, f"Preset {name!r} has no slots"

    def test_cinematic_tilt_has_tilt(self):
        slots = ModulationEngine.get_preset("cinematic_tilt")
        targets = {s.target for s in slots}
        assert "motion_tilt_x" in targets or "motion_tilt_y" in targets

    def test_enriched_presets_have_tilt(self):
        """Enriched presets should have tilt targets."""
        for name in ("cinematic_sweep", "advanced_max", "abstract_noise"):
            slots = ModulationEngine.get_preset(name)
            targets = {s.target for s in slots}
            tilt_targets = targets & {"motion_tilt_x", "motion_tilt_y"}
            assert len(tilt_targets) > 0, f"Preset {name!r} should have tilt"

    def test_enriched_presets_have_motion(self):
        """v0.7.4: existing presets enriched with motion slots."""
        enriched = [
            "electronic_pulse", "rock_energy", "hiphop_bounce", "classical_flow",
            "ambient_drift", "glitch_chaos", "smooth_morph", "rhythmic_pulse",
            "atmospheric", "abstract_noise", "intermediate_full", "advanced_max",
            "noise_sculpt", "energetic", "ambient", "bass_driven",
        ]
        motion_set = {"motion_x", "motion_y", "motion_zoom", "motion_rotation",
                      "motion_tilt_x", "motion_tilt_y"}
        for name in enriched:
            slots = ModulationEngine.get_preset(name)
            targets = {s.target for s in slots}
            assert targets & motion_set, f"Preset {name!r} should have motion target"

    def test_presets_without_motion(self):
        """v0.7.4: beginner presets deliberately have no motion."""
        no_motion = ["one_click_easy", "beginner_balanced", "controlnet_reactive", "seed_scatter"]
        motion_set = {"motion_x", "motion_y", "motion_zoom", "motion_rotation",
                      "motion_tilt_x", "motion_tilt_y"}
        for name in no_motion:
            slots = ModulationEngine.get_preset(name)
            targets = {s.target for s in slots}
            assert not (targets & motion_set), f"Preset {name!r} should NOT have motion"

    def test_schedule_truncation_max_frames(self):
        """v0.7.4: max_frames truncation (simulates engine logic)."""
        schedule = ParameterSchedule(
            total_frames=100,
            frame_params=[{"denoise_strength": 0.5 + i * 0.001} for i in range(100)],
        )
        max_frames = 30
        schedule.total_frames = max_frames
        schedule.frame_params = schedule.frame_params[:max_frames]
        assert schedule.total_frames == 30
        assert len(schedule.frame_params) == 30
        assert schedule.get_params(29) == {"denoise_strength": pytest.approx(0.529)}

    def test_all_presets_denoise_min_val_above_floor(self):
        """All presets with denoise_strength must have min_val >= 0.30.

        Research: Hyper-SD 8-step needs >=0.35 for meaningful output.
        Floor 0.30 guarantees >=4 effective steps at steps=8/cap=2.
        """
        for name, slots in PRESETS.items():
            for slot in slots:
                if slot["target"] == "denoise_strength":
                    assert slot["min_val"] >= 0.30, (
                        f"Preset {name!r} has denoise min_val={slot['min_val']} < 0.30"
                    )

    def test_target_range_denoise_lower_bound(self):
        """TARGET_RANGES denoise_strength lower bound must be >= 0.20."""
        lo, hi = TARGET_RANGES["denoise_strength"]
        assert lo >= 0.20, f"Denoise lower bound {lo} < 0.20"
        assert hi <= 0.95

    def test_motion_rate_limiting_clamps_delta(self):
        """Rate limiting should clamp frame-to-frame motion deltas."""
        engine = ModulationEngine()
        # Step function: zoom jumps from 1.0 to 1.06 instantly at frame 5
        raw = np.zeros(20, dtype=np.float32)
        raw[5:] = 1.0
        analysis = AudioAnalysis(
            fps=24.0, duration=20/24.0, total_frames=20,
            sample_rate=22050, audio_path="test.wav",
            features={"global_rms": raw},
            raw_features={"global_rms": raw},
        )
        slots = [ModulationSlot(
            source="global_rms", target="motion_zoom",
            min_val=1.0, max_val=1.06, attack=1, release=1,
        )]
        schedule = engine.compute_schedule(analysis, slots)
        # Frame-to-frame delta should be clamped to MOTION_MAX_DELTA["motion_zoom"]
        max_d = MOTION_MAX_DELTA["motion_zoom"]
        for i in range(1, 20):
            prev = schedule.get_params(i - 1).get("motion_zoom", 1.0)
            cur = schedule.get_params(i).get("motion_zoom", 1.0)
            delta = abs(cur - prev)
            assert delta <= max_d + 1e-6, (
                f"Frame {i}: delta {delta:.4f} exceeds max {max_d}"
            )


# ─── Validate Expressions ──────────────────────────────────

class TestValidateExpressions:
    def test_valid_expression(self):
        engine = ModulationEngine()
        errors = engine.validate_expressions(
            {"denoise_strength": "0.2 + 0.3 * global_rms"},
            ["global_rms"],
        )
        assert errors == {}

    def test_invalid_expression(self):
        engine = ModulationEngine()
        errors = engine.validate_expressions(
            {"denoise_strength": "1 +/ 2"},
            ["global_rms"],
        )
        assert "denoise_strength" in errors

    def test_unknown_target(self):
        engine = ModulationEngine()
        errors = engine.validate_expressions(
            {"nonexistent_param": "1.0"},
            [],
        )
        assert "nonexistent_param" in errors
