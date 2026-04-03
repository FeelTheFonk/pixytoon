"""Expression presets — curated mathematical expression templates for audio reactivity.

Provides categorized, one-click expression presets that fill the custom expression
fields in the Audio tab. Includes single-target presets and multi-target composite
presets (choreographies) that coordinate camera motion across channels.
"""

from __future__ import annotations

# ─── Expression Preset Definitions ──────────────────────────────────

EXPRESSION_PRESETS: dict[str, dict] = {

    # ─── Rhythmic (BPM-synced) ──────────────────────────────────
    "bpm_pulse": {
        "category": "rhythmic",
        "targets": {"denoise_strength": "0.25 + 0.35 * abs(sin(s * 3.141592653589793 * bpm / 60))"},
        "description": "Denoise pulses at BPM — intensity rises and falls with each beat",
    },
    "half_time_pulse": {
        "category": "rhythmic",
        "targets": {"denoise_strength": "0.25 + 0.25 * abs(sin(s * 3.141592653589793 * bpm / 120))"},
        "description": "Half-time pulse — one cycle every two beats, relaxed feel",
    },
    "double_time_pulse": {
        "category": "rhythmic",
        "targets": {"denoise_strength": "0.25 + 0.25 * abs(sin(s * 3.141592653589793 * bpm / 30))"},
        "description": "Double-time pulse — twice per beat, energetic",
    },
    "beat_gate_noise": {
        "category": "rhythmic",
        "targets": {"noise_amplitude": "where(global_beat > 0.3, 0.4, 0.0)"},
        "description": "Noise injection only on detected beats — gated effect",
    },
    "alternating_bars": {
        "category": "rhythmic",
        "targets": {"cfg_scale": "4.0 + 4.0 * abs(sin(s * 3.141592653589793 * bpm / 240))"},
        "description": "CFG oscillates every 4 beats — structural variation per bar",
    },
    "bpm_zoom_pulse": {
        "category": "rhythmic",
        "targets": {"motion_zoom": "1.0 + 0.015 * abs(sin(s * 3.141592653589793 * bpm / 60))"},
        "description": "Subtle zoom pulse synchronized to BPM",
    },

    # ─── Temporal (time-based evolution) ────────────────────────
    "gradual_buildup": {
        "category": "temporal",
        "targets": {"denoise_strength": "lerp(0.20, 0.65, t / max_f)"},
        "description": "Linear denoise ramp from subtle to intense over the clip",
    },
    "slow_fade_out": {
        "category": "temporal",
        "targets": {"denoise_strength": "lerp(0.65, 0.20, t / max_f)"},
        "description": "Linear denoise decay from intense to subtle — outro feel",
    },
    "arc_denoise": {
        "category": "temporal",
        "targets": {"denoise_strength": "0.20 + 0.45 * sin(3.141592653589793 * t / max_f)"},
        "description": "Rise-and-fall arc — builds to midpoint then decays",
    },
    "breathing": {
        "category": "temporal",
        "targets": {"denoise_strength": "0.30 + 0.12 * sin(s * 0.5)"},
        "description": "Slow organic breathing rhythm independent of audio",
    },
    "exponential_onset": {
        "category": "temporal",
        "targets": {"denoise_strength": "0.20 + 0.50 * pow(global_onset, 2.0)"},
        "description": "Squared onset response — only strong transients trigger high values",
    },

    # ─── Spectral (frequency-domain driven) ─────────────────────
    "brightness_cfg": {
        "category": "spectral",
        "targets": {"cfg_scale": "max(3.0, 5.0 + 4.0 * global_centroid)"},
        "description": "CFG follows spectral brightness — brighter sound = more guidance",
    },
    "tonal_palette": {
        "category": "spectral",
        "targets": {"palette_shift": "smoothstep(0.2, 0.8, global_chroma_energy) * 0.4"},
        "description": "Hue rotation from chroma energy — tonal shifts change color",
    },
    "multi_band_noise": {
        "category": "spectral",
        "targets": {"noise_amplitude": "clamp(0.3 * global_low + 0.2 * global_high, 0.05, 0.6)"},
        "description": "Noise from combined bass and treble — frequency-aware texture",
    },
    "flux_cfg": {
        "category": "spectral",
        "targets": {"cfg_scale": "smoothstep(0.2, 0.8, global_spectral_flux) * 6.0 + 3.0"},
        "description": "CFG follows timbral change rate — more flux = stronger guidance",
    },
    "contrast_denoise": {
        "category": "spectral",
        "targets": {"denoise_strength": "0.25 + 0.40 * global_spectral_contrast"},
        "description": "Denoise driven by spectral contrast — tonality drives transformation",
    },

    # ─── Easing (motion modifiers) ──────────────────────────────
    "ease_in_zoom": {
        "category": "easing",
        "targets": {"motion_zoom": "1.0 + 0.04 * easeIn(t / max_f)"},
        "description": "Zoom accelerates over clip — slow start, fast finish",
    },
    "ease_out_zoom": {
        "category": "easing",
        "targets": {"motion_zoom": "1.0 + 0.04 * easeOut(t / max_f)"},
        "description": "Zoom decelerates over clip — fast start, gentle settle",
    },
    "ease_in_out_pan": {
        "category": "easing",
        "targets": {"motion_x": "lerp(-2.0, 2.0, easeInOut(t / max_f))"},
        "description": "Smooth S-curve horizontal pan — accelerates then decelerates",
    },
    "bounce_zoom": {
        "category": "easing",
        "targets": {"motion_zoom": "1.0 + 0.02 * bounce(t / max_f)"},
        "description": "Decaying zoom bounce — impact that settles",
    },
    "elastic_rotation": {
        "category": "easing",
        "targets": {"motion_rotation": "lerp(-1.0, 0.0, elastic(t / max_f))"},
        "description": "Elastic rotation snap — overshoots then settles",
    },

    # ─── Camera / Motion (single-target) ────────────────────────
    "audio_drift_x": {
        "category": "camera",
        "targets": {"motion_x": "global_low * 3.0 - 1.5"},
        "description": "Horizontal drift driven by bass — low frequencies push left/right",
    },
    "beat_zoom": {
        "category": "camera",
        "targets": {"motion_zoom": "1.0 + 0.02 * global_beat"},
        "description": "Zoom pulse on detected beats",
    },
    "pendulum_rotation": {
        "category": "camera",
        "targets": {"motion_rotation": "sin(s * 0.6) * 0.8"},
        "description": "Slow pendulum rotation swing — mesmerizing periodic motion",
    },
    "tilt_from_centroid": {
        "category": "camera",
        "targets": {"motion_tilt_y": "lerp(-1.0, 1.0, global_centroid)"},
        "description": "Perspective tilt follows spectral centroid — bright = right tilt",
    },
    "gentle_vertical_drift": {
        "category": "camera",
        "targets": {"motion_y": "sin(s * 0.2) * global_mid * 1.5"},
        "description": "Vertical drift modulated by mids — organic floating feel",
    },
}


# ─── Choreography Meta-Presets ──────────────────────────────────
# Combine modulation slots + multi-target expressions for coordinated camera work.

CHOREOGRAPHY_PRESETS: dict[str, dict] = {
    "orbit_journey": {
        "category": "choreography",
        "description": "Audio-reactive orbital camera — speed increases with energy",
        "slots": [
            {"source": "global_rms", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.55, "attack": 3, "release": 12, "enabled": True},
            {"source": "global_onset", "target": "cfg_scale",
             "min_val": 3.0, "max_val": 7.0, "attack": 2, "release": 8, "enabled": True},
        ],
        "expressions": {
            "motion_x": "sin(s * (0.2 + global_rms * 0.3)) * 2.0",
            "motion_y": "cos(s * (0.2 + global_rms * 0.3)) * 1.5",
            "motion_rotation": "sin(s * (0.2 + global_rms * 0.3)) * 0.4",
        },
    },
    "dolly_zoom_vertigo": {
        "category": "choreography",
        "description": "Vertigo effect — zoom in while perspective shifts, audio-modulated intensity",
        "slots": [
            {"source": "global_rms", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.55, "attack": 3, "release": 15, "enabled": True},
            {"source": "global_onset", "target": "cfg_scale",
             "min_val": 4.0, "max_val": 7.0, "attack": 2, "release": 10, "enabled": True},
        ],
        "expressions": {
            "motion_zoom": "1.0 + 0.025 * smoothstep(0.0, 1.0, t / max_f) * (0.5 + global_rms)",
            "motion_tilt_x": "lerp(1.0, -1.0, smoothstep(0.0, 1.0, t / max_f)) * 0.8",
        },
    },
    "crane_ascending": {
        "category": "choreography",
        "description": "Vertical crane sweep — speed modulated by audio energy",
        "slots": [
            {"source": "global_rms", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        ],
        "expressions": {
            "motion_y": "lerp(-1.5, 1.5, smoothstep(0.0, 1.0, t / max_f)) * (0.5 + global_rms)",
            "motion_zoom": "1.0 + 0.005 * easeIn(t / max_f)",
        },
    },
    "wandering_voyage": {
        "category": "choreography",
        "description": "Smooth multi-frequency Lissajous path — creates a travel/journey feel",
        "slots": [
            {"source": "global_rms", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.50, "attack": 4, "release": 18, "enabled": True},
            {"source": "global_centroid", "target": "cfg_scale",
             "min_val": 3.0, "max_val": 6.0, "attack": 3, "release": 15, "enabled": True},
        ],
        "expressions": {
            "motion_x": "sin(s * 0.2) * (1.0 + global_low) * 1.5",
            "motion_y": "cos(s * 0.15) * (1.0 + global_mid) * 1.0",
            "motion_zoom": "1.0 + 0.008 * sin(s * 0.1)",
            "motion_rotation": "sin(s * 0.08) * 0.25",
            "palette_shift": "smoothstep(0.3, 0.7, global_chroma_energy) * 0.2",
        },
    },
    "hypnotic_spiral": {
        "category": "choreography",
        "description": "Spiral inward with accelerating rotation — trance-like pull",
        "slots": [
            {"source": "global_rms", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.50, "attack": 4, "release": 20, "enabled": True},
        ],
        "expressions": {
            "motion_zoom": "1.0 + 0.004 * (1.0 + global_rms)",
            "motion_rotation": "lerp(0.1, 0.7, easeIn(t / max_f)) * sin(s * 0.5)",
        },
    },
    "breathing_calm": {
        "category": "choreography",
        "description": "Ultra-gentle breathing camera — stops below energy threshold",
        "slots": [
            {"source": "global_rms", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.40, "attack": 6, "release": 25, "enabled": True},
        ],
        "expressions": {
            "motion_zoom": "where(global_rms > 0.1, 1.0 + 0.005 * sin(s * 0.3), 1.0)",
            "motion_x": "where(global_rms > 0.1, sin(s * 0.15) * 0.5, 0.0)",
            "motion_y": "where(global_rms > 0.1, cos(s * 0.12) * 0.3, 0.0)",
        },
    },
    "staccato_cuts": {
        "category": "choreography",
        "description": "Sharp motion on beats, frozen between — rhythmic visual punctuation",
        "slots": [
            {"source": "global_beat", "target": "denoise_strength",
             "min_val": 0.30, "max_val": 0.65, "attack": 1, "release": 4, "enabled": True},
            {"source": "global_onset", "target": "cfg_scale",
             "min_val": 3.0, "max_val": 8.0, "attack": 1, "release": 3, "enabled": True},
        ],
        "expressions": {
            "motion_x": "where(global_beat > 0.4, sign(sin(s * 2.1)) * 1.5, 0.0)",
            "motion_zoom": "where(global_beat > 0.4, 1.0 + 0.02, 1.0)",
            "motion_rotation": "where(global_onset > 0.5, sin(s * 3.0) * 0.6, 0.0)",
        },
    },
}


# ─── API Functions ──────────────────────────────────────────────

def list_expression_presets() -> dict[str, list[dict]]:
    """Return expression presets grouped by category."""
    by_cat: dict[str, list[dict]] = {}
    for name, preset in EXPRESSION_PRESETS.items():
        cat = preset["category"]
        by_cat.setdefault(cat, []).append({
            "name": name,
            "targets": list(preset["targets"].keys()),
            "description": preset["description"],
        })
    return by_cat


def get_expression_preset(name: str) -> dict | None:
    """Return a copy of full preset details or None if not found.

    Returns a shallow copy to prevent callers from mutating the
    module-level EXPRESSION_PRESETS registry.
    """
    preset = EXPRESSION_PRESETS.get(name)
    return dict(preset) if preset is not None else None


def list_choreography_presets() -> list[dict]:
    """Return choreography presets as a list of summaries."""
    return [
        {
            "name": name,
            "description": preset["description"],
            "slot_count": len(preset.get("slots", [])),
            "expression_targets": list(preset.get("expressions", {}).keys()),
        }
        for name, preset in CHOREOGRAPHY_PRESETS.items()
    ]


def get_choreography_preset(name: str) -> dict | None:
    """Return a copy of full choreography preset or None if not found."""
    preset = CHOREOGRAPHY_PRESETS.get(name)
    return dict(preset) if preset is not None else None


def detect_conflicts(
    targets_a: list[str], targets_b: list[str]
) -> list[str]:
    """Return target names present in both lists."""
    set_a = set(targets_a)
    return [t for t in targets_b if t in set_a]
