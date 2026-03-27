"""Tests for audio_analyzer — feature extraction, smoothing, resampling."""

from __future__ import annotations

import numpy as np
import pytest

from sddj.audio_analyzer import (
    AudioAnalysis,
    AudioAnalyzer,
    _normalize,
    _normalize_percentile,
    _resample_to_fps,
    smooth_features,
    smooth_features_ema,
    smooth_features_savgol,
    _compute_mel_band_indices,
    _CHROMA_NAMES,
    _BAND_BOUNDARIES,
)


# ─── Helpers ────────────────────────────────────────────────

def _make_sine_wav(tmp_path, freq=440, duration=1.0, sr=44100):
    """Create a short WAV file with a sine wave."""
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    path = tmp_path / "test_sine.wav"
    sf.write(str(path), y, sr)
    return str(path)


def _make_click_wav(tmp_path, n_clicks=5, duration=1.0, sr=44100):
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


def _make_silence_wav(tmp_path, duration=1.0, sr=44100):
    """Create a silent WAV file."""
    import soundfile as sf

    y = np.zeros(int(sr * duration), dtype=np.float32)
    path = tmp_path / "test_silence.wav"
    sf.write(str(path), y, sr)
    return str(path)


def _make_stereo_wav(tmp_path, freq=440, duration=1.0, sr=44100):
    """Create a stereo WAV file."""
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    right = 0.3 * np.sin(2 * np.pi * freq * 2 * t).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "test_stereo.wav"
    sf.write(str(path), stereo, sr)
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


class TestNormalizePercentile:
    def test_basic(self):
        arr = np.array([0.0, 1.0, 2.0, 100.0])  # 100 is outlier
        result = _normalize_percentile(arr, pct=90.0)
        assert result.dtype == np.float32
        assert result.max() == pytest.approx(1.0)

    def test_clips_outlier(self):
        arr = np.concatenate([np.ones(99), [1000.0]])
        result = _normalize_percentile(arr, pct=95.0)
        # The outlier should be clipped to 1.0
        assert result[-1] == pytest.approx(1.0)


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

    def test_peak_preservation_cubic_spline(self):
        """Linear interpolation attenuates peaks falling between sample points. Cubic preserves them."""
        arr = np.zeros(100, dtype=np.float32)
        # Create a sharp transient peak at t=0.15s
        # 100 fps means dt=0.01s.
        arr[14] = 0.1
        arr[15] = 1.0
        arr[16] = 0.1
        # Resample to 60 fps (dt=0.01666s)
        # Frames: F8 = 0.133s, F9 = 0.150s (exact match if no float drift), F10 = 0.166s.
        # Wait, if it exactly matches, linear will pass. Let's make it not exactly match.
        # F9 at 60fps = 9/60 = 0.150.
        # Let's put peak at 0.155s (idx 15.5 at 100fps is not an int array index, so idx 15 is 0.15s)
        # Let's put the peak at index 15 (0.15s), but sample at 44fps.
        resampled = _resample_to_fps(arr, 100.0, 44.0, 44)
        # At 44 fps, frames are at 0, 0.0227, 0.0454, ..., 0.136, 0.159.
        # Nearest are 0.136 and 0.159. The peak is at 0.150.
        # It falls right between them. Linear interpolation will average it out significantly.
        assert resampled.max() > 0.85, f"Peak severely attenuated: {resampled.max():.3f} < 0.85"

    def test_downsampling_preserves_peak_amplitude(self):
        """Max-pooling downsampling must preserve the amplitude of transient peaks."""
        arr = np.zeros(1000, dtype=np.float32)
        arr[500] = 1.0  # single spike at t=5.0s (1000 samples / 100 fps = 10s)
        result = _resample_to_fps(arr, 100.0, 10.0, 100)  # 10s × 10 fps = 100 frames
        assert result.max() == pytest.approx(1.0), "Max-pool downsampling lost the spike"


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
        arr = np.concatenate([np.zeros(5), np.ones(5)]).astype(np.float32)
        result = smooth_features({"s": arr}, attack_frames=1, release_frames=10)
        assert result["s"][5] > 0.3  # fast attack
        arr2 = np.concatenate([np.ones(5), np.zeros(5)]).astype(np.float32)
        result2 = smooth_features({"s": arr2}, attack_frames=1, release_frames=10)
        assert result2["s"][6] > 0.3  # slow release

    def test_empty_array(self):
        result = smooth_features({"e": np.array([], dtype=np.float32)})
        assert len(result["e"]) == 0

    def test_minimum_frames_clamped(self):
        features = {"x": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
        result = smooth_features(features, attack_frames=0, release_frames=-5)
        assert len(result["x"]) == 3


class TestSmoothFeaturesSavgol:
    def test_savgol_constant_signal(self):
        features = {"test": np.ones(30, dtype=np.float32)}
        result = smooth_features_savgol(features, attack_frames=2, release_frames=8)
        np.testing.assert_allclose(result["test"], 1.0, atol=1e-3)

    def test_savgol_smoothing_runs(self):
        """SavGol should produce output different from input (smoothing occurs)."""
        arr = np.zeros(50, dtype=np.float32)
        arr[20:30] = 1.0  # 10-sample pulse (not single spike)
        ema_result = smooth_features_ema({"s": arr.copy()}, attack_frames=2, release_frames=8)
        savgol_result = smooth_features_savgol({"s": arr.copy()}, attack_frames=2, release_frames=8)
        # Both should produce valid smoothed output
        assert len(savgol_result["s"]) == 50
        assert len(ema_result["s"]) == 50
        # SavGol output should differ from raw input (smoothing happened)
        assert not np.array_equal(savgol_result["s"], arr)

    def test_savgol_short_array(self):
        """Short arrays shorter than window should be returned as-is."""
        arr = np.array([0.5, 0.7], dtype=np.float32)
        result = smooth_features_savgol({"s": arr}, attack_frames=5, release_frames=10)
        np.testing.assert_allclose(result["s"], arr)


# ─── Mel band computation ─────────────────────────────────

class TestMelBandIndices:
    def test_all_9_bands_present(self):
        bands = _compute_mel_band_indices(44100, 256)
        expected_names = [name for name, _, _ in _BAND_BOUNDARIES]
        assert set(bands.keys()) == set(expected_names)

    def test_bands_non_empty(self):
        bands = _compute_mel_band_indices(44100, 256)
        for name, (start, end) in bands.items():
            assert end > start, f"Band '{name}' has zero width: [{start}, {end})"

    def test_bands_ordered(self):
        bands = _compute_mel_band_indices(44100, 256)
        names = [name for name, _, _ in _BAND_BOUNDARIES]
        starts = [bands[n][0] for n in names]
        assert starts == sorted(starts), "Band start indices not monotonically ordered"


# ─── AudioAnalyzer ──────────────────────────────────────────

class TestAudioAnalyzer:
    def test_analyze_sine_wav(self, tmp_path):
        from sddj.config import settings
        path = _make_sine_wav(tmp_path, freq=440, duration=1.0)
        analyzer = AudioAnalyzer()
        analysis = analyzer.analyze(path, fps=10)

        assert isinstance(analysis, AudioAnalysis)
        assert analysis.total_frames == 10
        assert analysis.fps == 10.0
        assert analysis.duration == pytest.approx(1.0, abs=0.1)
        assert analysis.sample_rate == settings.audio_sample_rate

        # Should have all base global features
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
            "drums": np.random.randn(44100).astype(np.float32),
            "bass": np.random.randn(44100).astype(np.float32),
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
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        for band in ("global_sub_bass", "global_upper_mid", "global_presence"):
            assert band in analysis.features, f"Missing band: {band}"
            assert len(analysis.features[band]) == analysis.total_frames

    def test_get_waveform_preview(self, tmp_path):
        path = _make_sine_wav(tmp_path, duration=2.0)
        analysis = AudioAnalyzer().analyze(path, fps=24)
        wf = analysis.get_waveform_preview(100)
        assert len(wf) == 100
        assert all(0.0 <= v <= 1.0 for v in wf)

    def test_get_waveform_preview_short_audio(self, tmp_path):
        path = _make_sine_wav(tmp_path, duration=0.5)
        analysis = AudioAnalyzer().analyze(path, fps=4)
        wf = analysis.get_waveform_preview(100)
        assert len(wf) == 100

    # ─── New pinnacle-quality tests (v0.9.35) ──────────────

    def test_sample_rate_matches_settings(self, tmp_path):
        """Step 8: Verify analysis uses correct settings sample rate."""
        from sddj.config import settings
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        assert analysis.sample_rate == settings.audio_sample_rate

    def test_new_spectral_features_exist(self, tmp_path):
        """Step 14-18: All 5 new spectral features must be present."""
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        for feat in ("global_spectral_contrast", "global_spectral_flatness",
                      "global_spectral_bandwidth", "global_spectral_rolloff",
                      "global_spectral_flux"):
            assert feat in analysis.features, f"Missing spectral feature: {feat}"
            assert len(analysis.features[feat]) == analysis.total_frames

    def test_chroma_12_bins(self, tmp_path):
        """Step 19-20: All 12 chroma bins + aggregate must be present."""
        path = _make_sine_wav(tmp_path, freq=440, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        for pitch in _CHROMA_NAMES:
            key = f"global_chroma_{pitch}"
            assert key in analysis.features, f"Missing chroma bin: {key}"
        assert "global_chroma_energy" in analysis.features

    def test_9_band_segmentation(self, tmp_path):
        """Step 12: All 9 frequency bands must be present."""
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        for band_name, _, _ in _BAND_BOUNDARIES:
            key = f"global_{band_name}"
            assert key in analysis.features, f"Missing band: {key}"

    def test_backward_compat_aliases(self, tmp_path):
        """Step 13: global_low, global_mid, global_high still work."""
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        for alias in ("global_low", "global_mid", "global_high"):
            assert alias in analysis.features, f"Missing alias: {alias}"

    def test_lufs_field(self, tmp_path):
        """Step 7: Integrated LUFS field should be populated."""
        path = _make_sine_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        assert hasattr(analysis, "lufs")
        assert isinstance(analysis.lufs, float)
        assert analysis.lufs < 0  # A sine wave should have negative LUFS

    def test_silent_audio(self, tmp_path):
        """Edge case: silent audio should not crash, all features return zeros."""
        path = _make_silence_wav(tmp_path, duration=1.0)
        analysis = AudioAnalyzer().analyze(path, fps=10)
        assert analysis.total_frames == 10
        # At least rms should be zero
        rms = analysis.features["global_rms"]
        assert rms.max() < 0.01

    def test_mono_stereo(self, tmp_path):
        """Edge case: both mono and stereo files should work."""
        mono_path = _make_sine_wav(tmp_path, duration=1.0)
        stereo_path = _make_stereo_wav(tmp_path, duration=1.0)
        mono_analysis = AudioAnalyzer().analyze(mono_path, fps=10)
        stereo_analysis = AudioAnalyzer().analyze(stereo_path, fps=10)
        assert mono_analysis.total_frames == stereo_analysis.total_frames
        assert set(mono_analysis.features.keys()) == set(stereo_analysis.features.keys())

    def test_stems_get_all_new_features(self, tmp_path):
        """Step 28: Stems should get all new features, not just rms/onset."""
        path = _make_sine_wav(tmp_path, duration=1.0)
        stems = {"drums": np.random.randn(44100).astype(np.float32)}
        analysis = AudioAnalyzer().analyze(path, fps=10, stems=stems)
        # Stems should have spectral features too
        assert "drums_spectral_contrast" in analysis.features
        assert "drums_spectral_flatness" in analysis.features
        assert "drums_chroma_energy" in analysis.features
