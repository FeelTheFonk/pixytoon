"""Tests for audio_analyzer — feature extraction, smoothing, resampling."""

from __future__ import annotations

import numpy as np
import pytest

from pixytoon.audio_analyzer import (
    AudioAnalysis,
    AudioAnalyzer,
    _normalize,
    _resample_to_fps,
    smooth_features,
)


# ─── Helpers ────────────────────────────────────────────────

def _make_sine_wav(tmp_path, freq=440, duration=1.0, sr=22050):
    """Create a short WAV file with a sine wave."""
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    path = tmp_path / "test_sine.wav"
    sf.write(str(path), y, sr)
    return str(path)


def _make_click_wav(tmp_path, n_clicks=5, duration=1.0, sr=22050):
    """Create a WAV with transient clicks (onset test)."""
    import soundfile as sf

    y = np.zeros(int(sr * duration), dtype=np.float32)
    interval = len(y) // (n_clicks + 1)
    for i in range(1, n_clicks + 1):
        pos = i * interval
        y[pos:pos + 100] = 0.9
    path = tmp_path / "test_clicks.wav"
    sf.write(str(path), y, sr)
    return str(path)


# ─── _normalize ─────────────────────────────────────────────

class TestNormalize:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _normalize(arr)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_array_returns_zeros(self):
        arr = np.array([3.0, 3.0, 3.0])
        result = _normalize(arr)
        assert np.all(result == 0.0)

    def test_single_element(self):
        result = _normalize(np.array([42.0]))
        assert result[0] == pytest.approx(0.0)

    def test_output_dtype_float32(self):
        result = _normalize(np.array([1.0, 10.0]))
        assert result.dtype == np.float32


# ─── _resample_to_fps ──────────────────────────────────────

class TestResampleToFps:
    def test_same_length(self):
        arr = np.array([0.0, 0.5, 1.0])
        result = _resample_to_fps(arr, 10.0, 10.0, 3)
        np.testing.assert_allclose(result, arr, atol=1e-5)

    def test_upsampling(self):
        arr = np.array([0.0, 1.0])
        result = _resample_to_fps(arr, 1.0, 4.0, 4)
        assert len(result) == 4
        assert result[0] == pytest.approx(0.0)

    def test_downsampling(self):
        arr = np.arange(100, dtype=np.float32)
        result = _resample_to_fps(arr, 100.0, 10.0, 10)
        assert len(result) == 10

    def test_empty_input(self):
        result = _resample_to_fps(np.array([]), 10.0, 10.0, 5)
        assert len(result) == 5
        assert np.all(result == 0.0)


# ─── smooth_features ───────────────────────────────────────

class TestSmoothFeatures:
    def test_smooth_constant_signal(self):
        features = {"test": np.ones(10, dtype=np.float32)}
        result = smooth_features(features, attack_frames=2, release_frames=8)
        np.testing.assert_allclose(result["test"], 1.0, atol=1e-5)

    def test_smooth_preserves_keys(self):
        features = {"a": np.zeros(5), "b": np.ones(5)}
        result = smooth_features(features)
        assert set(result.keys()) == {"a", "b"}

    def test_attack_faster_than_release(self):
        # Step signal: 0 then 1
        arr = np.concatenate([np.zeros(5), np.ones(5)]).astype(np.float32)
        result = smooth_features({"s": arr}, attack_frames=1, release_frames=10)
        # After step up, should respond quickly (attack=1 → fast)
        assert result["s"][5] > 0.3  # fast attack
        # After step down, should be slow
        arr2 = np.concatenate([np.ones(5), np.zeros(5)]).astype(np.float32)
        result2 = smooth_features({"s": arr2}, attack_frames=1, release_frames=10)
        assert result2["s"][6] > 0.3  # slow release

    def test_empty_array(self):
        result = smooth_features({"e": np.array([], dtype=np.float32)})
        assert len(result["e"]) == 0

    def test_minimum_frames_clamped(self):
        """attack_frames and release_frames < 1 should be clamped to 1."""
        features = {"x": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
        result = smooth_features(features, attack_frames=0, release_frames=-5)
        assert len(result["x"]) == 3


# ─── AudioAnalyzer ──────────────────────────────────────────

class TestAudioAnalyzer:
    def test_analyze_sine_wav(self, tmp_path):
        path = _make_sine_wav(tmp_path, freq=440, duration=1.0)
        analyzer = AudioAnalyzer()
        analysis = analyzer.analyze(path, fps=10)

        assert isinstance(analysis, AudioAnalysis)
        assert analysis.total_frames == 10
        assert analysis.fps == 10.0
        assert analysis.duration == pytest.approx(1.0, abs=0.1)
        assert analysis.sample_rate == 22050

        # Should have 6 global features
        expected_features = {
            "global_rms", "global_onset", "global_centroid",
            "global_low", "global_mid", "global_high",
        }
        assert expected_features.issubset(set(analysis.feature_names))

    def test_features_normalized_0_1(self, tmp_path):
        path = _make_click_wav(tmp_path, n_clicks=5, duration=2.0)
        analysis = AudioAnalyzer().analyze(path, fps=24)

        for name, arr in analysis.features.items():
            assert arr.min() >= -0.01, f"{name} min below 0"
            assert arr.max() <= 1.01, f"{name} max above 1"
            assert len(arr) == analysis.total_frames

    def test_features_correct_length(self, tmp_path):
        path = _make_sine_wav(tmp_path, duration=2.0)
        analysis = AudioAnalyzer().analyze(path, fps=30)

        for arr in analysis.features.values():
            assert len(arr) == analysis.total_frames

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AudioAnalyzer().analyze("/nonexistent/file.wav", fps=24)

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported audio format"):
            AudioAnalyzer().analyze(str(bad_file), fps=24)

    def test_with_stems(self, tmp_path):
        path = _make_sine_wav(tmp_path, duration=1.0)
        stems = {
            "drums": np.random.randn(22050).astype(np.float32),
            "bass": np.random.randn(22050).astype(np.float32),
        }
        analysis = AudioAnalyzer().analyze(path, fps=10, stems=stems)
        assert "drums_rms" in analysis.features
        assert "drums_onset" in analysis.features
        assert "bass_rms" in analysis.features
        assert "bass_onset" in analysis.features

    def test_feature_names_sorted(self, tmp_path):
        path = _make_sine_wav(tmp_path)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        assert analysis.feature_names == sorted(analysis.feature_names)

    def test_extended_bands_present(self, tmp_path):
        """v0.7.3: sub_bass, upper_mid, presence bands should be extracted."""
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        for band in ("global_sub_bass", "global_upper_mid", "global_presence"):
            assert band in analysis.features, f"Missing band: {band}"
            assert len(analysis.features[band]) == analysis.total_frames

    def test_get_waveform_preview(self, tmp_path):
        """v0.7.3: waveform preview returns correct number of points."""
        path = _make_sine_wav(tmp_path, duration=2.0)
        analysis = AudioAnalyzer().analyze(path, fps=24)
        wf = analysis.get_waveform_preview(100)
        assert len(wf) == 100
        assert all(0.0 <= v <= 1.0 for v in wf)

    def test_get_waveform_preview_short_audio(self, tmp_path):
        """Waveform preview works even with fewer frames than points."""
        path = _make_sine_wav(tmp_path, duration=0.5)
        analysis = AudioAnalyzer().analyze(path, fps=4)
        wf = analysis.get_waveform_preview(100)
        assert len(wf) == 100
