"""Modulation engine — maps audio features to inference parameters.

Implements a synth-style modulation matrix where audio feature sources
are routed to inference parameter targets with configurable range and smoothing.
Also supports custom math expressions via simpleeval (sandboxed).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .audio_analyzer import AudioAnalysis

log = logging.getLogger("pixytoon.audio")

# Valid modulation targets and their clamping ranges
TARGET_RANGES: dict[str, tuple[float, float]] = {
    "denoise_strength": (0.05, 0.95),
    "cfg_scale": (1.0, 30.0),
    "noise_amplitude": (0.0, 1.0),
    "controlnet_scale": (0.0, 2.0),
    "seed_offset": (0.0, 1000.0),
}

# Built-in modulation presets — organized by category
PRESETS: dict[str, list[dict]] = {
    # ─── Genre-Specific ──────────────────────────────────────────
    "electronic_pulse": [
        {"source": "global_beat", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.65, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 10.0, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_high", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.3, "attack": 1, "release": 3, "enabled": True},
    ],
    "rock_energy": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.20, "max_val": 0.70, "attack": 2, "release": 5, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 8.0, "attack": 1, "release": 8, "enabled": True},
        {"source": "global_low", "target": "seed_offset",
         "min_val": 0.0, "max_val": 200.0, "attack": 2, "release": 10, "enabled": True},
    ],
    "hiphop_bounce": [
        {"source": "global_low", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.55, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_beat", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 9.0, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_onset", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.2, "attack": 1, "release": 5, "enabled": True},
    ],
    "classical_flow": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.10, "max_val": 0.40, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 4, "release": 15, "enabled": True},
    ],
    "ambient_drift": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.08, "max_val": 0.25, "attack": 8, "release": 30, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 5.0, "attack": 6, "release": 25, "enabled": True},
        {"source": "global_mid", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.15, "attack": 5, "release": 20, "enabled": True},
    ],
    # ─── Style-Specific ──────────────────────────────────────────
    "glitch_chaos": [
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.90, "attack": 1, "release": 2, "enabled": True},
        {"source": "global_high", "target": "cfg_scale",
         "min_val": 1.0, "max_val": 15.0, "attack": 1, "release": 2, "enabled": True},
        {"source": "global_beat", "target": "seed_offset",
         "min_val": 0.0, "max_val": 1000.0, "attack": 1, "release": 1, "enabled": True},
        {"source": "global_rms", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.8, "attack": 1, "release": 3, "enabled": True},
    ],
    "smooth_morph": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.10, "max_val": 0.35, "attack": 6, "release": 18, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 6.0, "attack": 5, "release": 15, "enabled": True},
    ],
    "rhythmic_pulse": [
        {"source": "global_beat", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.60, "attack": 1, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 9.0, "attack": 1, "release": 6, "enabled": True},
    ],
    "atmospheric": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.12, "max_val": 0.30, "attack": 4, "release": 25, "enabled": True},
        {"source": "global_mid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 6.5, "attack": 3, "release": 20, "enabled": True},
        {"source": "global_high", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.10, "attack": 4, "release": 15, "enabled": True},
    ],
    "abstract_noise": [
        {"source": "global_rms", "target": "noise_amplitude",
         "min_val": 0.05, "max_val": 0.60, "attack": 2, "release": 5, "enabled": True},
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.85, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_centroid", "target": "seed_offset",
         "min_val": 0.0, "max_val": 500.0, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_high", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 12.0, "attack": 1, "release": 3, "enabled": True},
    ],
    # ─── Complexity Levels ───────────────────────────────────────
    "one_click_easy": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.50, "attack": 3, "release": 10, "enabled": True},
    ],
    "beginner_balanced": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.50, "attack": 3, "release": 10, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 2, "release": 8, "enabled": True},
    ],
    "intermediate_full": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.55, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 8.0, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_low", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.2, "attack": 2, "release": 8, "enabled": True},
    ],
    "advanced_max": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.10, "max_val": 0.65, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 10.0, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_low", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.3, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_beat", "target": "seed_offset",
         "min_val": 0.0, "max_val": 300.0, "attack": 1, "release": 10, "enabled": True},
    ],
    # ─── Target-Specific ─────────────────────────────────────────
    "controlnet_reactive": [
        {"source": "global_rms", "target": "controlnet_scale",
         "min_val": 0.3, "max_val": 1.5, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.50, "attack": 2, "release": 6, "enabled": True},
    ],
    "seed_scatter": [
        {"source": "global_onset", "target": "seed_offset",
         "min_val": 0.0, "max_val": 800.0, "attack": 1, "release": 3, "enabled": True},
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.20, "max_val": 0.50, "attack": 2, "release": 8, "enabled": True},
    ],
    "noise_sculpt": [
        {"source": "global_rms", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.5, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.20, "max_val": 0.60, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 8.0, "attack": 3, "release": 10, "enabled": True},
    ],
    # ─── Legacy (backward-compatible) ────────────────────────────
    "energetic": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.20, "max_val": 0.70, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 9.0, "attack": 1, "release": 10, "enabled": True},
    ],
    "ambient": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.10, "max_val": 0.30, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 6.0, "attack": 3, "release": 15, "enabled": True},
    ],
    "bass_driven": [
        {"source": "global_low", "target": "denoise_strength",
         "min_val": 0.15, "max_val": 0.60, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_high", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 8.0, "attack": 1, "release": 5, "enabled": True},
    ],
}


@dataclass
class ModulationSlot:
    """A single routing in the modulation matrix."""
    source: str        # audio feature name
    target: str        # inference parameter name
    min_val: float = 0.0
    max_val: float = 1.0
    attack: int = 2
    release: int = 8
    enabled: bool = True


@dataclass
class ParameterSchedule:
    """Pre-computed per-frame parameter overrides for audio-reactive generation."""
    total_frames: int
    frame_params: list[dict[str, float]] = field(default_factory=list)

    def get_params(self, frame_idx: int) -> dict[str, float]:
        if 0 <= frame_idx < len(self.frame_params):
            return self.frame_params[frame_idx]
        return {}


class ExpressionEvaluator:
    """Safe math expression evaluator using simpleeval."""

    def __init__(self) -> None:
        from simpleeval import SimpleEval, DEFAULT_OPERATORS

        self._evaluator = SimpleEval()
        # Add math functions
        self._evaluator.functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "abs": abs,
            "min": min,
            "max": max,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "pow": pow,
            "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
            "lerp": lambda a, b, t: a + (b - a) * t,
            "smoothstep": lambda edge0, edge1, x: (
                lambda t: t * t * (3 - 2 * t)
            )(max(0.0, min(1.0, (x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0))),
            "where": lambda cond, a, b: a if cond else b,
            "floor": math.floor,
            "ceil": math.ceil,
        }
        self._evaluator.operators = DEFAULT_OPERATORS

    def validate(self, expression: str, available_vars: list[str]) -> str | None:
        """Validate an expression. Returns error message or None if valid."""
        try:
            # Set dummy values for all variables
            self._evaluator.names = {v: 0.5 for v in available_vars}
            self._evaluator.names.update({"t": 0, "max_f": 100, "fps": 24.0, "s": 0.0})
            self._evaluator.eval(expression)
            return None
        except Exception as e:
            return str(e)

    def evaluate(self, expression: str, variables: dict[str, float]) -> float:
        """Evaluate expression with given variables. Returns float result."""
        self._evaluator.names = variables
        result = self._evaluator.eval(expression)
        return float(result)


class ModulationEngine:
    """Computes per-frame parameter schedules from audio analysis + modulation config."""

    def __init__(self) -> None:
        self._expr_eval: ExpressionEvaluator | None = None

    def _get_evaluator(self) -> ExpressionEvaluator:
        if self._expr_eval is None:
            self._expr_eval = ExpressionEvaluator()
        return self._expr_eval

    @staticmethod
    def get_preset(name: str) -> list[ModulationSlot]:
        """Get a built-in modulation preset by name."""
        if name not in PRESETS:
            raise ValueError(f"Unknown modulation preset: {name!r}. "
                           f"Available: {list(PRESETS.keys())}")
        return [ModulationSlot(**slot) for slot in PRESETS[name]]

    @staticmethod
    def list_presets() -> list[str]:
        return list(PRESETS.keys())

    def validate_expressions(self, expressions: dict[str, str],
                           available_features: list[str]) -> dict[str, str]:
        """Validate all expressions. Returns dict of target→error for invalid ones."""
        evaluator = self._get_evaluator()
        errors = {}
        for target, expr in expressions.items():
            if target not in TARGET_RANGES:
                errors[target] = f"Unknown target parameter: {target}"
                continue
            err = evaluator.validate(expr, available_features)
            if err:
                errors[target] = err
        return errors

    def compute_schedule(self, analysis: AudioAnalysis,
                        slots: list[ModulationSlot],
                        expressions: dict[str, str] | None = None) -> ParameterSchedule:
        """Compute per-frame parameter overrides from audio features + modulation config.

        Args:
            analysis: Pre-computed audio analysis with per-frame features.
            slots: Modulation matrix slots (source→target routing).
            expressions: Optional custom expressions per target (overrides slots).
        """
        total = analysis.total_frames
        schedule = ParameterSchedule(total_frames=total)

        # Filter to enabled slots with valid sources
        active_slots = [s for s in slots
                       if s.enabled and s.source in analysis.features
                       and s.target in TARGET_RANGES]

        if not active_slots and not expressions:
            # No modulation — return empty schedule
            schedule.frame_params = [{} for _ in range(total)]
            return schedule

        log.info("Computing schedule: %d frames, %d active slots, %d expressions",
                 total, len(active_slots),
                 len(expressions) if expressions else 0)

        # Pre-build expression evaluator if needed
        evaluator = self._get_evaluator() if expressions else None

        for frame_idx in range(total):
            params: dict[str, float] = {}
            # Track per-target contributions for multi-slot aggregation
            target_values: dict[str, list[float]] = {}

            # Apply modulation matrix slots
            for slot in active_slots:
                feature_val = float(analysis.features[slot.source][frame_idx])
                # Map [0, 1] feature to [min_val, max_val]
                output = slot.min_val + (slot.max_val - slot.min_val) * feature_val
                target_values.setdefault(slot.target, []).append(output)

            # Aggregate multi-slot targets (average)
            for target, values in target_values.items():
                lo, hi = TARGET_RANGES[target]
                params[target] = max(lo, min(hi, sum(values) / len(values)))

            # Override with custom expressions (take priority over slots)
            if expressions and evaluator:
                # Build variable dict for this frame
                variables: dict[str, float] = {
                    "t": float(frame_idx),
                    "max_f": float(total),
                    "fps": analysis.fps,
                    "s": frame_idx / analysis.fps,  # seconds
                    "bpm": analysis.bpm,
                }
                # Add all audio features as variables
                for feat_name, feat_arr in analysis.features.items():
                    variables[feat_name] = float(feat_arr[frame_idx])

                for target, expr in expressions.items():
                    if target not in TARGET_RANGES:
                        continue
                    try:
                        val = evaluator.evaluate(expr, variables)
                        lo, hi = TARGET_RANGES[target]
                        params[target] = max(lo, min(hi, val))
                    except Exception as e:
                        log.warning("Expression error at frame %d for %s: %s",
                                   frame_idx, target, e)
                        # Keep slot value if expression fails

            # Convert seed_offset to integer
            if "seed_offset" in params:
                params["seed_offset"] = float(int(params["seed_offset"]))

            schedule.frame_params.append(params)

        return schedule
