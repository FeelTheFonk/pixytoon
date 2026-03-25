"""Tests for auto_calibrate — recommend_preset decision tree."""

from __future__ import annotations

import numpy as np

from sddj.audio_analyzer import AudioAnalysis
from sddj.auto_calibrate import recommend_preset


# ─── Helpers ────────────────────────────────────────────────

def _make_analysis(
    n_frames: int = 100,
    fps: float = 24.0,
    bpm: float = 0.0,
    features: dict[str, np.ndarray] | None = None,
) -> AudioAnalysis:
    """Create a test AudioAnalysis with controllable features."""
    if features is None:
        features = {}
    return AudioAnalysis(
        fps=fps,
        duration=n_frames / fps,
        total_frames=n_frames,
        sample_rate=22050,
        audio_path="/test/audio.wav",
        features=features,
        bpm=bpm,
    )


def _const(value: float, n: int = 100) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _beat_array(density: float, n: int = 100) -> np.ndarray:
    """Create a beat array where `density` fraction of values are > 0.5."""
    arr = np.zeros(n, dtype=np.float32)
    n_beats = int(n * density)
    if n_beats > 0:
        indices = np.linspace(0, n - 1, n_beats, dtype=int)
        arr[indices] = 1.0
    return arr


def _high_variance(low: float, high: float, n: int = 100) -> np.ndarray:
    """Alternating low/high to produce high variance."""
    arr = np.empty(n, dtype=np.float32)
    arr[0::2] = low
    arr[1::2] = high
    return arr


# ─── Decision Tree Branch Tests ─────────────────────────────


class TestRecommendPreset:
    """Each test targets a specific branch of the decision tree."""

    def test_ambient_drift(self):
        """Low RMS + low variance → ambient_drift."""
        features = {
            "global_rms": _const(0.10),
            "global_onset": _const(0.1),
            "global_centroid": _const(0.3),
            "global_low": _const(0.2),
            "global_beat": _const(0.0),
        }
        result = recommend_preset(_make_analysis(features=features))
        assert result == "ambient_drift"

    def test_electronic_pulse(self):
        """Fast BPM + beats + high centroid → electronic_pulse."""
        features = {
            "global_rms": _const(0.5),
            "global_onset": _const(0.3),
            "global_centroid": _const(0.7),
            "global_low": _const(0.3),
            "global_beat": _beat_array(0.10),
        }
        result = recommend_preset(_make_analysis(bpm=140, features=features))
        assert result == "electronic_pulse"

    def test_hiphop_bounce(self):
        """Fast BPM + beats + low centroid → hiphop_bounce."""
        features = {
            "global_rms": _const(0.5),
            "global_onset": _const(0.3),
            "global_centroid": _const(0.3),
            "global_low": _const(0.5),
            "global_beat": _beat_array(0.10),
        }
        result = recommend_preset(_make_analysis(bpm=140, features=features))
        assert result == "hiphop_bounce"

    def test_rock_energy(self):
        """High onset + loud peaks + high contrast → rock_energy."""
        rms = _const(0.6)
        rms[50] = 0.9  # loud peak
        features = {
            "global_rms": rms,
            "global_onset": _const(0.5),
            "global_centroid": _const(0.4),
            "global_low": _const(0.3),
            "global_beat": _const(0.0),
            "global_spectral_contrast": _const(0.7),
            "global_spectral_flatness": _const(0.1),
            "global_spectral_flux": _const(0.2),
            "global_chroma_energy": _const(0.2),
            "global_brilliance": _const(0.2),
        }
        result = recommend_preset(_make_analysis(bpm=90, features=features))
        assert result == "rock_energy"

    def test_bass_driven(self):
        """High low-band dominance + moderate RMS → bass_driven."""
        features = {
            "global_rms": _const(0.4),
            "global_onset": _const(0.2),
            "global_centroid": _const(0.3),
            "global_low": _const(0.6),
            "global_beat": _const(0.0),
        }
        result = recommend_preset(_make_analysis(bpm=80, features=features))
        assert result == "bass_driven"

    def test_rhythmic_pulse(self):
        """High RMS variance + moderate onset → rhythmic_pulse."""
        features = {
            "global_rms": _high_variance(0.1, 0.9),
            "global_onset": _const(0.35),
            "global_centroid": _const(0.4),
            "global_low": _const(0.3),
            "global_beat": _const(0.0),
        }
        result = recommend_preset(_make_analysis(bpm=80, features=features))
        assert result == "rhythmic_pulse"

    def test_classical_flow(self):
        """Low RMS + low variance (above ambient floor) → classical_flow."""
        features = {
            "global_rms": _const(0.20),
            "global_onset": _const(0.2),
            "global_centroid": _const(0.3),
            "global_low": _const(0.2),
            "global_beat": _const(0.0),
        }
        result = recommend_preset(_make_analysis(bpm=60, features=features))
        assert result == "classical_flow"

    def test_glitch_chaos(self):
        """Very high onset → glitch_chaos."""
        features = {
            "global_rms": _const(0.5),
            "global_onset": _const(0.6),
            "global_centroid": _const(0.4),
            "global_low": _const(0.3),
            "global_beat": _const(0.0),
        }
        result = recommend_preset(_make_analysis(bpm=80, features=features))
        assert result == "glitch_chaos"

    def test_default_beginner_balanced(self):
        """Balanced features that don't trigger any specific branch."""
        features = {
            "global_rms": _const(0.35),
            "global_onset": _const(0.25),
            "global_centroid": _const(0.4),
            "global_low": _const(0.3),
            "global_beat": _const(0.0),
        }
        result = recommend_preset(_make_analysis(bpm=80, features=features))
        assert result == "beginner_balanced"

    def test_empty_features_no_crash(self):
        """Missing features should not crash — fallback to defaults."""
        result = recommend_preset(_make_analysis(features={}))
        assert isinstance(result, str)
        assert result in {
            "ambient_drift", "electronic_pulse", "hiphop_bounce",
            "rock_energy", "bass_driven", "rhythmic_pulse",
            "classical_flow", "glitch_chaos", "beginner_balanced",
            "micro_reactive", "spectral_sculptor", "tonal_drift",
            "ultra_precision",
        }
