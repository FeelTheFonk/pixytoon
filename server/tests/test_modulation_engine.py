"""Tests for modulation_engine — slots, expressions, schedule computation."""

from __future__ import annotations

import numpy as np
import pytest

from pixytoon.audio_analyzer import AudioAnalysis
from pixytoon.modulation_engine import (
    ExpressionEvaluator,
    ModulationEngine,
    ModulationSlot,
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
