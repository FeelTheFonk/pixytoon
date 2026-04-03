"""Auto-calibrate: recommend modulation preset from audio characteristics."""

from __future__ import annotations

import numpy as np

from .audio_analyzer import AudioAnalysis

# Pre-allocated zero array — avoids fresh np.zeros(1) allocation per feature lookup
_ZERO = np.zeros(1)


def recommend_preset(analysis: AudioAnalysis) -> str:
    """Analyze audio features and return the best-fitting preset name.

    Uses a decision tree based on energy, variance, beat density, BPM,
    spectral centroid, low-band dominance, spectral contrast/flatness,
    chroma energy, and brilliance to pick the most expressive preset.
    """
    rms = analysis.features.get("global_rms", _ZERO)
    onset = analysis.features.get("global_onset", _ZERO)
    centroid = analysis.features.get("global_centroid", _ZERO)
    low = analysis.features.get("global_low", _ZERO)
    beat = analysis.features.get("global_beat", _ZERO)
    flatness = analysis.features.get("global_spectral_flatness", _ZERO)
    contrast = analysis.features.get("global_spectral_contrast", _ZERO)
    brilliance = analysis.features.get("global_brilliance", _ZERO)
    flux = analysis.features.get("global_spectral_flux", _ZERO)
    chroma = analysis.features.get("global_chroma_energy", _ZERO)

    avg_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0
    avg_onset = float(np.mean(onset)) if len(onset) > 0 else 0.0
    avg_centroid = float(np.mean(centroid)) if len(centroid) > 0 else 0.0
    peak_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    rms_variance = float(np.var(rms)) if len(rms) > 0 else 0.0
    beat_density = float(np.sum(beat > 0.5)) / max(1, len(beat))
    low_dominance = float(np.mean(low)) if len(low) > 0 else 0.0
    avg_flatness = float(np.mean(flatness)) if len(flatness) > 0 else 0.0
    avg_contrast = float(np.mean(contrast)) if len(contrast) > 0 else 0.0
    avg_brilliance = float(np.mean(brilliance)) if len(brilliance) > 0 else 0.0
    avg_flux = float(np.mean(flux)) if len(flux) > 0 else 0.0
    avg_chroma = float(np.mean(chroma)) if len(chroma) > 0 else 0.0
    contrast_var = float(np.var(contrast)) if len(contrast) > 0 else 0.0
    bpm = analysis.bpm

    # Very quiet / minimal dynamics -> ambient
    if avg_rms < 0.15 and rms_variance < 0.01:
        return "ambient_drift"

    # High spectral flux + onset = micro_reactive (timbral transitions)
    if avg_onset > 0.3 and avg_flux > 0.4:
        return "micro_reactive"

    # High contrast variance = spectral_sculptor (dynamic timbral content)
    if contrast_var > 0.03 and avg_contrast > 0.4:
        return "spectral_sculptor"

    # High chroma + low onset = tonal_drift (melodic, harmonic content)
    if avg_chroma > 0.5 and avg_flatness < 0.2 and avg_onset < 0.35:
        return "tonal_drift"

    # Fast BPM + clear beats -> electronic or hiphop
    if bpm > 120 and beat_density > 0.03:
        if avg_centroid > 0.5 or avg_brilliance > 0.4:
            return "electronic_pulse"
        return "hiphop_bounce"

    # Noisy / textural content -> glitch
    if avg_flatness > 0.5 and avg_onset > 0.3:
        return "glitch_chaos"

    # High onset + loud peaks -> rock
    if avg_onset > 0.4 and peak_rms > 0.8:
        if avg_contrast > 0.6:
            return "rock_energy"
        return "ultra_precision"

    # Bass-heavy content
    if low_dominance > 0.5 and avg_rms > 0.3:
        return "bass_driven"

    # Strong dynamic variation + onsets -> rhythmic
    if rms_variance > 0.05 and avg_onset > 0.3:
        return "rhythmic_pulse"

    # Moderate BPM + low energy -> classical
    if avg_rms < 0.3 and rms_variance < 0.02:
        if avg_chroma > 0.4:
            return "atmospheric"
        return "classical_flow"

    # Very percussive -> glitch
    if avg_onset > 0.5:
        return "glitch_chaos"

    # Default: balanced for unknown content
    return "beginner_balanced"
