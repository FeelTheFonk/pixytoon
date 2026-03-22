"""Auto-calibrate: recommend modulation preset from audio characteristics."""

from __future__ import annotations

import numpy as np

from .audio_analyzer import AudioAnalysis


def recommend_preset(analysis: AudioAnalysis) -> str:
    """Analyze audio features and return the best-fitting preset name.

    Uses a decision tree based on energy, variance, beat density, BPM,
    spectral centroid, and low-band dominance to pick the most expressive
    preset without any user configuration.
    """
    rms = analysis.features.get("global_rms", np.zeros(1))
    onset = analysis.features.get("global_onset", np.zeros(1))
    centroid = analysis.features.get("global_centroid", np.zeros(1))
    low = analysis.features.get("global_low", np.zeros(1))
    beat = analysis.features.get("global_beat", np.zeros(1))

    avg_rms = float(np.mean(rms))
    avg_onset = float(np.mean(onset))
    avg_centroid = float(np.mean(centroid))
    peak_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    rms_variance = float(np.var(rms))
    beat_density = float(np.sum(beat > 0.5)) / max(1, len(beat))
    low_dominance = float(np.mean(low))
    bpm = analysis.bpm

    # Very quiet / minimal dynamics -> ambient
    if avg_rms < 0.15 and rms_variance < 0.01:
        return "ambient_drift"

    # Fast BPM + clear beats -> electronic or hiphop
    if bpm > 120 and beat_density > 0.03:
        if avg_centroid > 0.5:
            return "electronic_pulse"
        return "hiphop_bounce"

    # High onset + loud peaks -> rock
    if avg_onset > 0.4 and peak_rms > 0.8:
        return "rock_energy"

    # Bass-heavy content
    if low_dominance > 0.5 and avg_rms > 0.3:
        return "bass_driven"

    # Strong dynamic variation + onsets -> rhythmic
    if rms_variance > 0.05 and avg_onset > 0.3:
        return "rhythmic_pulse"

    # Moderate BPM + low energy -> classical
    if avg_rms < 0.3 and rms_variance < 0.02:
        return "classical_flow"

    # Very percussive -> glitch
    if avg_onset > 0.5:
        return "glitch_chaos"

    # Default: balanced for unknown content
    return "beginner_balanced"
