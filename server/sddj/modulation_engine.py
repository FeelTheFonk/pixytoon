"""Modulation engine — maps audio features to inference parameters.

Implements a synth-style modulation matrix where audio feature sources
are routed to inference parameter targets with configurable range and smoothing.
Also supports custom math expressions via simpleeval (sandboxed).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numba
import numpy as np

from .audio_analyzer import AudioAnalysis

log = logging.getLogger("sddj.modulation_engine")

# Valid modulation targets and their clamping ranges
TARGET_RANGES: dict[str, tuple[float, float]] = {
    "denoise_strength": (0.20, 0.95),
    "cfg_scale": (1.0, 30.0),
    "noise_amplitude": (0.0, 1.0),
    "controlnet_scale": (0.0, 2.0),
    "seed_offset": (0.0, 1000.0),
    "palette_shift": (0.0, 1.0),  # hue rotation amount (0 = no shift, 1 = full 360°)
    "frame_cadence": (1.0, 8.0),  # frame skip cadence (1 = every frame, 4 = skip 3)
    # Motion / camera (smooth Deforum-like, image-space affine)
    "motion_x": (-5.0, 5.0),          # pixels, horizontal translation
    "motion_y": (-5.0, 5.0),          # pixels, vertical translation
    "motion_zoom": (0.92, 1.08),      # scale factor (1.0 = none)
    "motion_rotation": (-2.0, 2.0),   # degrees, planar rotation
    # Perspective tilt (faux 3D via homography)
    "motion_tilt_x": (-3.0, 3.0),     # degrees, perspective pitch
    "motion_tilt_y": (-3.0, 3.0),     # degrees, perspective yaw
    # LoRA weight (per-frame modulation, applied between pipeline calls)
    "lora_weight": (0.0, 2.0),
}

# Max allowed frame-to-frame delta per motion parameter.
# Prevents saccade/jerk from rapid audio transients.
MOTION_MAX_DELTA: dict[str, float] = {
    "motion_x": 2.0,
    "motion_y": 2.0,
    "motion_zoom": 0.015,
    "motion_rotation": 0.8,
    "motion_tilt_x": 1.0,
    "motion_tilt_y": 1.0,
}

# Built-in modulation presets — organized by category
PRESETS: dict[str, list[dict]] = {
    # ─── Genre-Specific ──────────────────────────────────────────
    "electronic_pulse": [
        {"source": "global_beat", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.65, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 7.0, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_high", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.3, "attack": 1, "release": 3, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 2, "release": 12, "enabled": True},
    ],
    "rock_energy": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.70, "attack": 2, "release": 5, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 1, "release": 8, "enabled": True},
        {"source": "global_low", "target": "seed_offset",
         "min_val": 0.0, "max_val": 200.0, "attack": 2, "release": 10, "enabled": True},
        {"source": "global_rms", "target": "motion_x",
         "min_val": -1.5, "max_val": 1.5, "attack": 3, "release": 12, "enabled": True},
    ],
    "hiphop_bounce": [
        {"source": "global_low", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.55, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_beat", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 7.0, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_onset", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.2, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_low", "target": "motion_y",
         "min_val": -1.0, "max_val": 1.0, "attack": 2, "release": 10, "enabled": True},
    ],
    "classical_flow": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 4, "release": 15, "enabled": True},
        {"source": "global_rms", "target": "motion_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 6, "release": 25, "enabled": True},
    ],
    "ambient_drift": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 8, "release": 30, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 5.0, "attack": 6, "release": 25, "enabled": True},
        {"source": "global_mid", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.15, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_mid", "target": "motion_x",
         "min_val": -1.5, "max_val": 1.5, "attack": 8, "release": 30, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 6, "release": 25, "enabled": True},
    ],
    # ─── Style-Specific ──────────────────────────────────────────
    "glitch_chaos": [
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.35, "max_val": 0.90, "attack": 1, "release": 2, "enabled": True},
        {"source": "global_high", "target": "cfg_scale",
         "min_val": 1.0, "max_val": 8.0, "attack": 1, "release": 2, "enabled": True},
        {"source": "global_beat", "target": "seed_offset",
         "min_val": 0.0, "max_val": 1000.0, "attack": 1, "release": 1, "enabled": True},
        {"source": "global_rms", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.5, "attack": 1, "release": 3, "enabled": True},
        {"source": "global_onset", "target": "motion_rotation",
         "min_val": -1.5, "max_val": 1.5, "attack": 1, "release": 3, "enabled": True},
    ],
    "smooth_morph": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 6, "release": 18, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 6.0, "attack": 5, "release": 15, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 5, "release": 20, "enabled": True},
    ],
    "rhythmic_pulse": [
        {"source": "global_beat", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.60, "attack": 1, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 1, "release": 10, "enabled": True},
    ],
    "atmospheric": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 4, "release": 25, "enabled": True},
        {"source": "global_mid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 6.5, "attack": 3, "release": 20, "enabled": True},
        {"source": "global_high", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.10, "attack": 4, "release": 15, "enabled": True},
        {"source": "global_mid", "target": "motion_x",
         "min_val": -0.8, "max_val": 0.8, "attack": 5, "release": 20, "enabled": True},
    ],
    "abstract_noise": [
        {"source": "global_rms", "target": "noise_amplitude",
         "min_val": 0.05, "max_val": 0.40, "attack": 2, "release": 5, "enabled": True},
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.35, "max_val": 0.85, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_centroid", "target": "seed_offset",
         "min_val": 0.0, "max_val": 500.0, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_high", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 8.0, "attack": 1, "release": 3, "enabled": True},
        {"source": "global_high", "target": "motion_rotation",
         "min_val": -1.0, "max_val": 1.0, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "motion_x",
         "min_val": -2.0, "max_val": 2.0, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_high", "target": "motion_tilt_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 2, "release": 8, "enabled": True},
    ],
    # ─── Complexity Levels ───────────────────────────────────────
    "one_click_easy": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 10, "enabled": True},
    ],
    "beginner_balanced": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 10, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 2, "release": 8, "enabled": True},
    ],
    "intermediate_full": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.55, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_low", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.2, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 2, "release": 12, "enabled": True},
    ],
    "advanced_max": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.65, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 7.0, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_low", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.3, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_beat", "target": "seed_offset",
         "min_val": 0.0, "max_val": 300.0, "attack": 1, "release": 10, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -2.0, "max_val": 2.0, "attack": 2, "release": 10, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 1, "release": 8, "enabled": True},
        {"source": "global_low", "target": "motion_tilt_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 3, "release": 15, "enabled": True},
    ],
    # ─── Target-Specific ─────────────────────────────────────────
    "controlnet_reactive": [
        {"source": "global_rms", "target": "controlnet_scale",
         "min_val": 0.3, "max_val": 1.5, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 2, "release": 6, "enabled": True},
    ],
    "seed_scatter": [
        {"source": "global_onset", "target": "seed_offset",
         "min_val": 0.0, "max_val": 800.0, "attack": 1, "release": 3, "enabled": True},
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 2, "release": 8, "enabled": True},
    ],
    "noise_sculpt": [
        {"source": "global_rms", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.35, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.60, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 3, "release": 10, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 3, "release": 12, "enabled": True},
    ],
    # ─── Legacy (backward-compatible) ────────────────────────────
    "energetic": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.70, "attack": 2, "release": 6, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 1, "release": 10, "enabled": True},
        {"source": "global_rms", "target": "motion_x",
         "min_val": -1.5, "max_val": 1.5, "attack": 2, "release": 8, "enabled": True},
    ],
    "ambient": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 6.0, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_centroid", "target": "motion_x",
         "min_val": -0.5, "max_val": 0.5, "attack": 5, "release": 20, "enabled": True},
    ],
    "bass_driven": [
        {"source": "global_low", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.60, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_high", "target": "cfg_scale",
         "min_val": 4.0, "max_val": 7.0, "attack": 1, "release": 5, "enabled": True},
        {"source": "global_low", "target": "motion_y",
         "min_val": -1.5, "max_val": 1.5, "attack": 2, "release": 10, "enabled": True},
    ],
    # ─── Motion / Camera ──────────────────────────────────────
    "gentle_drift": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -2.0, "max_val": 2.0, "attack": 4, "release": 20, "enabled": True},
        {"source": "global_mid", "target": "motion_y",
         "min_val": -1.5, "max_val": 1.5, "attack": 4, "release": 20, "enabled": True},
    ],
    "pulse_zoom": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 2, "release": 10, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 2, "release": 15, "enabled": True},
    ],
    "slow_rotate": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 4, "release": 18, "enabled": True},
        {"source": "global_centroid", "target": "motion_rotation",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 25, "enabled": True},
    ],
    "cinematic_sweep": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -3.0, "max_val": 3.0, "attack": 5, "release": 25, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 3, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "motion_rotation",
         "min_val": -0.5, "max_val": 0.5, "attack": 6, "release": 30, "enabled": True},
        {"source": "global_centroid", "target": "motion_tilt_y",
         "min_val": -0.8, "max_val": 0.8, "attack": 5, "release": 25, "enabled": True},
    ],
    # ─── Perspective / Advanced Camera ────────────────────────────
    "cinematic_tilt": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_low", "target": "motion_tilt_x",
         "min_val": -1.5, "max_val": 1.5, "attack": 4, "release": 25, "enabled": True},
        {"source": "global_centroid", "target": "motion_tilt_y",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 25, "enabled": True},
    ],
    "zoom_breathe": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 6, "release": 30, "enabled": True},
    ],
    "parallax_drift": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -2.0, "max_val": 2.0, "attack": 4, "release": 20, "enabled": True},
        {"source": "global_mid", "target": "motion_tilt_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 20, "enabled": True},
    ],
    "full_cinematic": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -1.5, "max_val": 1.5, "attack": 4, "release": 20, "enabled": True},
        {"source": "global_mid", "target": "motion_y",
         "min_val": -1.0, "max_val": 1.0, "attack": 4, "release": 20, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 2, "release": 15, "enabled": True},
        {"source": "global_centroid", "target": "motion_rotation",
         "min_val": -0.5, "max_val": 0.5, "attack": 5, "release": 25, "enabled": True},
        {"source": "global_low", "target": "motion_tilt_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 25, "enabled": True},
    ],
    # ─── Spectral / Pinnacle Quality ──────────────────────────────
    "spectral_sculptor": [
        {"source": "global_spectral_flatness", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.35, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_spectral_contrast", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.65, "attack": 2, "release": 10, "enabled": True},
        {"source": "global_spectral_bandwidth", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 3, "release": 12, "enabled": True},
    ],
    "tonal_drift": [
        {"source": "global_chroma_energy", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.55, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_centroid", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 7.0, "attack": 4, "release": 15, "enabled": True},
        {"source": "global_spectral_rolloff", "target": "palette_shift",
         "min_val": 0.0, "max_val": 0.5, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_spectral_bandwidth", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 4, "release": 20, "enabled": True},
    ],
    "ultra_precision": [
        {"source": "global_onset", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.70, "attack": 1, "release": 3, "enabled": True},
        {"source": "global_spectral_contrast", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 8.0, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_sub_bass", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 2, "release": 10, "enabled": True},
        {"source": "global_presence", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.25, "attack": 1, "release": 5, "enabled": True},
    ],
    "micro_reactive": [
        {"source": "global_bass", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_low_mid", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.60, "attack": 2, "release": 8, "enabled": True},
        {"source": "global_brilliance", "target": "noise_amplitude",
         "min_val": 0.0, "max_val": 0.3, "attack": 1, "release": 4, "enabled": True},
        {"source": "global_spectral_flux", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 8.0, "attack": 1, "release": 5, "enabled": True},
    ],
    # ─── Voyage / Journey ────────────────────────────────────────
    "voyage_serene": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.45, "attack": 6, "release": 25, "enabled": True},
        {"source": "global_mid", "target": "motion_x",
         "min_val": -1.5, "max_val": 1.5, "attack": 6, "release": 25, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 0.99, "max_val": 1.01, "attack": 8, "release": 30, "enabled": True},
        {"source": "global_centroid", "target": "palette_shift",
         "min_val": 0.0, "max_val": 0.15, "attack": 6, "release": 25, "enabled": True},
    ],
    "voyage_exploratory": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 4, "release": 18, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -2.0, "max_val": 2.0, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_mid", "target": "motion_y",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "motion_rotation",
         "min_val": -0.5, "max_val": 0.5, "attack": 6, "release": 25, "enabled": True},
        {"source": "global_spectral_rolloff", "target": "palette_shift",
         "min_val": 0.0, "max_val": 0.2, "attack": 5, "release": 20, "enabled": True},
    ],
    "voyage_dramatic": [
        {"source": "global_beat", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.60, "attack": 1, "release": 10, "enabled": True},
        {"source": "global_onset", "target": "cfg_scale",
         "min_val": 3.0, "max_val": 8.0, "attack": 1, "release": 6, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -2.5, "max_val": 2.5, "attack": 2, "release": 12, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 1, "release": 12, "enabled": True},
        {"source": "global_rms", "target": "motion_rotation",
         "min_val": -0.8, "max_val": 0.8, "attack": 3, "release": 15, "enabled": True},
    ],
    "voyage_psychedelic": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.55, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_spectral_contrast", "target": "cfg_scale",
         "min_val": 2.0, "max_val": 7.0, "attack": 3, "release": 12, "enabled": True},
        {"source": "global_low", "target": "motion_x",
         "min_val": -2.0, "max_val": 2.0, "attack": 4, "release": 18, "enabled": True},
        {"source": "global_mid", "target": "motion_y",
         "min_val": -1.5, "max_val": 1.5, "attack": 4, "release": 18, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 0.98, "max_val": 1.02, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_centroid", "target": "motion_rotation",
         "min_val": -1.0, "max_val": 1.0, "attack": 4, "release": 20, "enabled": True},
        {"source": "global_low", "target": "motion_tilt_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 20, "enabled": True},
        {"source": "global_chroma_energy", "target": "palette_shift",
         "min_val": 0.0, "max_val": 0.3, "attack": 4, "release": 18, "enabled": True},
    ],
    # ─── Rest-Aware (energy-gated motion) ────────────────────────
    "intelligent_drift": [
        {"source": "global_rms", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.50, "attack": 3, "release": 15, "enabled": True},
        {"source": "global_rms", "target": "motion_x",
         "min_val": -1.0, "max_val": 1.0, "attack": 5, "release": 25, "enabled": True},
        {"source": "global_rms", "target": "motion_y",
         "min_val": -0.5, "max_val": 0.5, "attack": 5, "release": 25, "enabled": True},
        {"source": "global_rms", "target": "motion_zoom",
         "min_val": 1.0, "max_val": 1.01, "attack": 6, "release": 30, "enabled": True},
    ],
    "reactive_pause": [
        {"source": "global_beat", "target": "denoise_strength",
         "min_val": 0.30, "max_val": 0.55, "attack": 1, "release": 15, "enabled": True},
        {"source": "global_beat", "target": "motion_zoom",
         "min_val": 1.0, "max_val": 1.015, "attack": 1, "release": 20, "enabled": True},
        {"source": "global_beat", "target": "motion_rotation",
         "min_val": 0.0, "max_val": 0.3, "attack": 1, "release": 20, "enabled": True},
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
    invert: bool = False   # invert source: 0→max, 1→min


@dataclass
class ParameterSchedule:
    """Pre-computed per-frame parameter overrides for audio-reactive generation."""
    total_frames: int
    frame_params: list[dict[str, float]] = field(default_factory=list)

    def get_params(self, frame_idx: int) -> dict[str, float]:
        if 0 <= frame_idx < len(self.frame_params):
            return self.frame_params[frame_idx]
        return {}

    def get_chunk_params(self, start: int, end: int) -> dict[str, float]:
        """Average parameters over a chunk of frames [start, end).

        Used by AnimateDiff audio mode to compute per-chunk averages
        since AnimateDiff applies uniform parameters across a batch.
        """
        if start >= end or not self.frame_params:
            return {}
        # Clamp range
        start = max(0, start)
        end = min(end, len(self.frame_params))
        if start >= end:
            return {}

        # Collect all keys present across chunk frames
        all_keys: set[str] = set()
        for i in range(start, end):
            all_keys.update(self.frame_params[i].keys())

        if not all_keys:
            return {}

        # Average each parameter over the chunk
        result: dict[str, float] = {}
        for key in all_keys:
            values = [
                self.frame_params[i][key]
                for i in range(start, end)
                if key in self.frame_params[i]
            ]
            if values:
                avg = sum(values) / len(values)
                # Clamp to valid range
                if key in TARGET_RANGES:
                    lo, hi = TARGET_RANGES[key]
                    avg = max(lo, min(hi, avg))
                result[key] = avg

        return result


class ExpressionEvaluator:
    """Safe math expression evaluator using simpleeval."""

    def __init__(self) -> None:
        from simpleeval import SimpleEval, DEFAULT_OPERATORS

        self._evaluator = SimpleEval()
        # Add math functions
        self._evaluator.functions = {
            # Core math
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
            "floor": math.floor,
            "ceil": math.ceil,
            "sign": lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
            "atan2": math.atan2,
            # Interpolation & mapping
            "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
            "lerp": lambda a, b, t: a + (b - a) * t,
            "mix": lambda a, b, t: a + (b - a) * t,  # GLSL alias for lerp
            "smoothstep": lambda edge0, edge1, x: (
                lambda t: t * t * (3 - 2 * t)
            )(max(0.0, min(1.0, (x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0))),
            "remap": lambda x, a, b, c, d: c + (d - c) * max(0.0, min(1.0,
                (x - a) / (b - a) if b != a else 0.0)),
            # Conditionals
            "where": lambda cond, a, b: a if cond else b,
            # Easing functions (input x is typically t/max_f in [0,1])
            "easeIn": lambda x: x * x,
            "easeOut": lambda x: 1.0 - (1.0 - x) * (1.0 - x),
            "easeInOut": lambda x: (2.0 * x * x if x < 0.5
                                    else 1.0 - (-2.0 * x + 2.0) ** 2 / 2.0),
            "easeInCubic": lambda x: x * x * x,
            "easeOutCubic": lambda x: 1.0 - (1.0 - x) ** 3,
            # Animation easing
            "bounce": lambda x: abs(math.sin(x * math.pi * 3)) * max(0.0, 1.0 - x),
            "elastic": lambda x: (0.0 if x <= 0.0 else (1.0 if x >= 1.0 else
                2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * 2.0 * math.pi / 3.0) + 1.0)),
            # Utility / procedural
            "step": lambda x, n: math.floor(x * n) / n if n > 0 else x,
            "fract": lambda x: x - math.floor(x),
            "pingpong": lambda x, length: (
                length - abs((x % (2.0 * length)) - length) if length > 0 else 0.0),
            "hash1d": lambda x: (math.sin(x * 127.1) * 43758.5453) % 1.0,
            "smoothnoise": lambda x: (
                (math.sin(math.floor(x) * 127.1) * 43758.5453) % 1.0 * (1.0 - (x - math.floor(x)))
                + (math.sin((math.floor(x) + 1) * 127.1) * 43758.5453) % 1.0 * (x - math.floor(x))),
        }
        self._evaluator.operators = DEFAULT_OPERATORS

    def validate(self, expression: str, available_vars: list[str]) -> str | None:
        """Validate an expression. Returns error message or None if valid."""
        if len(expression) > 1024:
            return "Expression too long (max 1024 characters)"
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
        if len(expression) > 1024:
            raise ValueError("Expression too long (max 1024 characters)")
        self._evaluator.names = variables
        result = self._evaluator.eval(expression)
        return float(result)


@numba.jit(nopython=True, cache=True)
def _ema_slot_vectorized(
    src: np.ndarray,
    alpha_attack: float,
    alpha_release: float,
    invert: bool,
    min_val: float,
    range_val: float,
) -> np.ndarray:
    """Asymmetric EMA + range mapping for a single modulation slot.

    Numba-accelerated: eliminates Python loop overhead per frame.
    For N frames, this is ~10-50x faster than the equivalent Python loop.
    """
    n = src.shape[0]
    out = np.empty(n, dtype=np.float64)
    prev = src[0]
    for i in range(n):
        raw = src[i]
        if raw > prev:
            prev = prev + alpha_attack * (raw - prev)
        else:
            prev = prev + alpha_release * (raw - prev)
        mapped = (1.0 - prev) if invert else prev
        out[i] = min_val + range_val * mapped
    return out


class ModulationEngine:
    """Computes per-frame parameter schedules from audio analysis + modulation config."""

    def __init__(self) -> None:
        self._expr_eval: ExpressionEvaluator | None = None

    def _get_evaluator(self) -> ExpressionEvaluator:
        if self._expr_eval is None:
            self._expr_eval = ExpressionEvaluator()
        return self._expr_eval

    _LEGACY_PRESETS = {"energetic", "ambient", "bass_driven"}

    @staticmethod
    def get_preset(name: str) -> list[ModulationSlot]:
        """Get a built-in modulation preset by name."""
        if name not in PRESETS:
            raise ValueError(f"Unknown modulation preset: {name!r}. "
                           f"Available: {list(PRESETS.keys())}")
        if name in ModulationEngine._LEGACY_PRESETS:
            log.warning("Preset %r is legacy (v0.7.0) — consider using a newer preset", name)
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

        # Use raw (un-smoothed) features when available — per-slot EMA replaces global smoothing.
        # Fall back to smoothed features for backward compat with old caches.
        feat_source = analysis.raw_features if analysis.raw_features else analysis.features

        # Filter to enabled slots with valid sources
        active_slots = [s for s in slots
                       if s.enabled and s.source in feat_source
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

        # Per-slot EMA coefficients (attack/release)
        from math import exp as _exp
        slot_alphas: list[tuple[float, float]] = []
        for slot in active_slots:
            a_att = 1.0 - _exp(-1.0 / max(1, slot.attack))
            a_rel = 1.0 - _exp(-1.0 / max(1, slot.release))
            slot_alphas.append((a_att, a_rel))

        # Pre-extract per-slot source arrays (avoid repeated dict lookup per frame)
        slot_sources = [feat_source[s.source] for s in active_slots]
        slot_inverts = [s.invert for s in active_slots]
        slot_mins = [s.min_val for s in active_slots]
        slot_ranges = [s.max_val - s.min_val for s in active_slots]
        slot_targets = [s.target for s in active_slots]

        # ── Phase 1: Vectorized per-slot EMA via Numba ──────────────
        # Each slot produces a full-length output array in one Numba call,
        # replacing the previous Python frame×slot nested loop.
        slot_outputs: list[np.ndarray] = []
        for slot_idx in range(len(active_slots)):
            src = slot_sources[slot_idx]
            src_arr = np.asarray(src[:total], dtype=np.float64)
            if len(src_arr) < total:
                # Pad short sources with last value
                pad = np.full(total - len(src_arr), src_arr[-1], dtype=np.float64)
                src_arr = np.concatenate((src_arr, pad))
            a_att, a_rel = slot_alphas[slot_idx]
            out = _ema_slot_vectorized(
                src_arr, a_att, a_rel,
                slot_inverts[slot_idx],
                slot_mins[slot_idx],
                slot_ranges[slot_idx],
            )
            slot_outputs.append(out)

        # ── Phase 2: Aggregate per-target (average multiple slots) ────
        # Group slot indices by target
        target_slot_indices: dict[str, list[int]] = {}
        for i, t in enumerate(slot_targets):
            target_slot_indices.setdefault(t, []).append(i)

        # Pre-compute per-target arrays (averaged + clamped)
        target_arrays: dict[str, np.ndarray] = {}
        for target, indices in target_slot_indices.items():
            if len(indices) == 1:
                arr = slot_outputs[indices[0]]
            else:
                arr = np.mean([slot_outputs[i] for i in indices], axis=0)
            lo, hi = TARGET_RANGES[target]
            np.clip(arr, lo, hi, out=arr)
            target_arrays[target] = arr

        # ── Phase 3: Build per-frame dicts + expression overrides ─────
        expr_variables: dict[str, float] | None = None
        if expressions and evaluator:
            expr_variables = {
                "t": 0.0, "max_f": float(total),
                "fps": analysis.fps, "s": 0.0, "bpm": analysis.bpm,
            }
            for feat_name in analysis.features:
                expr_variables[feat_name] = 0.0

        for frame_idx in range(total):
            params: dict[str, float] = {}
            for target, arr in target_arrays.items():
                params[target] = float(arr[frame_idx])

            # Override with custom expressions (take priority over slots)
            if expr_variables is not None and evaluator:
                expr_variables["t"] = float(frame_idx)
                expr_variables["s"] = frame_idx / analysis.fps
                for feat_name, feat_arr in analysis.features.items():
                    expr_variables[feat_name] = float(feat_arr[frame_idx])

                for target, expr in expressions.items():
                    if target not in TARGET_RANGES:
                        continue
                    try:
                        val = evaluator.evaluate(expr, expr_variables)
                        lo, hi = TARGET_RANGES[target]
                        params[target] = max(lo, min(hi, val))
                    except Exception as e:
                        log.warning("Expression error at frame %d for %s: %s",
                                   frame_idx, target, e)

            if "seed_offset" in params:
                params["seed_offset"] = float(int(params["seed_offset"]))
            if "frame_cadence" in params:
                params["frame_cadence"] = float(max(1, int(params["frame_cadence"])))

            schedule.frame_params.append(params)

        # ── Post-pass: motion rate limiting ──────────────────────────
        # Clamp frame-to-frame delta for motion params to prevent saccade.
        if len(schedule.frame_params) > 1:
            prev_motion: dict[str, float] = {}
            # Total motion budget: sum of |delta/max_delta| per channel ≤ 1.0
            for fidx, params in enumerate(schedule.frame_params):
                budget_sum = 0.0
                clamped: dict[str, float] = {}
                for mkey, max_d in MOTION_MAX_DELTA.items():
                    if mkey not in params:
                        continue
                    cur = params[mkey]
                    prev = prev_motion.get(mkey, cur)  # first frame: no clamping
                    delta = cur - prev
                    if abs(delta) > max_d:
                        delta = max_d if delta > 0 else -max_d
                    clamped[mkey] = prev + delta
                    budget_sum += abs(delta) / max_d if max_d > 0 else 0.0
                # Scale proportionally if total budget exceeded
                if budget_sum > 1.0 and budget_sum > 0:
                    for mkey in clamped:
                        prev = prev_motion.get(mkey, clamped[mkey])
                        delta = clamped[mkey] - prev
                        clamped[mkey] = prev + delta / budget_sum
                # Write back and track
                for mkey, val in clamped.items():
                    params[mkey] = val
                    prev_motion[mkey] = val

        return schedule
