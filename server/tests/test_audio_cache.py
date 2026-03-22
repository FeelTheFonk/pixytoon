"""Tests for audio_cache — disk-backed cache with SHA256 keys."""

from __future__ import annotations

import numpy as np
import pytest

from pixytoon.audio_analyzer import AudioAnalysis
from pixytoon.audio_cache import AudioCache, _cache_key


# ─── Helpers ────────────────────────────────────────────────

def _make_test_wav(tmp_path, name="test.wav"):
    """Create a minimal WAV file for cache key testing."""
    import soundfile as sf

    y = np.zeros(1000, dtype=np.float32)
    path = tmp_path / name
    sf.write(str(path), y, 22050)
    return str(path)


def _make_analysis(audio_path="test.wav", fps=24.0, n_frames=10):
    """Create a dummy AudioAnalysis for testing."""
    return AudioAnalysis(
        fps=fps,
        duration=n_frames / fps,
        total_frames=n_frames,
        sample_rate=22050,
        audio_path=audio_path,
        features={
            "global_rms": np.random.rand(n_frames).astype(np.float32),
            "global_onset": np.random.rand(n_frames).astype(np.float32),
        },
    )


# ─── _cache_key ─────────────────────────────────────────────

class TestCacheKey:
    def test_deterministic(self, tmp_path):
        path = _make_test_wav(tmp_path)
        key1 = _cache_key(path, 24.0, False)
        key2 = _cache_key(path, 24.0, False)
        assert key1 == key2

    def test_different_fps_different_key(self, tmp_path):
        path = _make_test_wav(tmp_path)
        k1 = _cache_key(path, 24.0, False)
        k2 = _cache_key(path, 30.0, False)
        assert k1 != k2

    def test_stems_flag_different_key(self, tmp_path):
        path = _make_test_wav(tmp_path)
        k1 = _cache_key(path, 24.0, False)
        k2 = _cache_key(path, 24.0, True)
        assert k1 != k2

    def test_key_length(self, tmp_path):
        path = _make_test_wav(tmp_path)
        key = _cache_key(path, 24.0, False)
        assert len(key) == 24  # truncated SHA256


# ─── AudioCache ─────────────────────────────────────────────

class TestAudioCache:
    def test_put_and_get(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        wav = _make_test_wav(tmp_path)
        analysis = _make_analysis(wav)

        cache.put(wav, 24.0, analysis)
        result = cache.get(wav, 24.0)

        assert result is not None
        assert result.fps == analysis.fps
        assert result.total_frames == analysis.total_frames
        assert result.duration == analysis.duration
        assert set(result.features.keys()) == set(analysis.features.keys())
        for key in analysis.features:
            np.testing.assert_allclose(result.features[key], analysis.features[key])

    def test_cache_miss(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        wav = _make_test_wav(tmp_path)
        assert cache.get(wav, 24.0) is None

    def test_invalidate(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        wav = _make_test_wav(tmp_path)
        analysis = _make_analysis(wav)

        cache.put(wav, 24.0, analysis)
        assert cache.get(wav, 24.0) is not None
        cache.invalidate(wav, 24.0)
        assert cache.get(wav, 24.0) is None

    def test_cleanup_removes_nothing_when_fresh(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        wav = _make_test_wav(tmp_path)
        cache.put(wav, 24.0, _make_analysis(wav))
        removed = cache.cleanup()
        assert removed == 0

    def test_default_cache_dir(self):
        cache = AudioCache()
        assert cache._dir.exists()

    def test_different_stems_flag_separate_cache(self, tmp_path):
        cache = AudioCache(cache_dir=str(tmp_path / "cache"))
        wav = _make_test_wav(tmp_path)

        a1 = _make_analysis(wav)
        a2 = _make_analysis(wav)
        a2.features["drums_rms"] = np.random.rand(10).astype(np.float32)

        cache.put(wav, 24.0, a1, enable_stems=False)
        cache.put(wav, 24.0, a2, enable_stems=True)

        r1 = cache.get(wav, 24.0, enable_stems=False)
        r2 = cache.get(wav, 24.0, enable_stems=True)

        assert "drums_rms" not in r1.features
        assert "drums_rms" in r2.features
