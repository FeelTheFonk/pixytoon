"""Audio feature extraction for audio-reactive generation.

Extracts per-frame audio features (RMS, onset, spectral, multi-band energy)
from audio files using librosa, resampled to target FPS with asymmetric EMA smoothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import exp
from pathlib import Path

import numpy as np

log = logging.getLogger("pixytoon.audio")

# Supported audio file extensions
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})

# Mel spectrogram band boundaries (128 mel bands)
_BAND_SUB_BASS_END = 4   # bands 0-3    → ~20-60 Hz
_BAND_LOW_END = 16        # bands 4-15   → ~60-300 Hz
_BAND_MID_END = 64        # bands 16-63  → ~300-2 kHz
_BAND_UPPER_MID_END = 90  # bands 64-89  → ~2-4 kHz
_BAND_PRESENCE_END = 110  # bands 90-109 → ~4-8 kHz
                           # bands 110-127 → ~8-16 kHz (high)


@dataclass
class AudioAnalysis:
    """Result of audio analysis — per-frame feature vectors normalized to [0, 1]."""

    fps: float
    duration: float
    total_frames: int
    sample_rate: int
    audio_path: str
    features: dict[str, np.ndarray] = field(default_factory=dict)
    bpm: float = 0.0

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.features.keys())

    def get_waveform_preview(self, num_points: int = 100) -> list[float]:
        """Return a downsampled RMS waveform for UI display."""
        rms = self.features.get("global_rms")
        if rms is None or len(rms) == 0:
            return [0.0] * num_points
        # Downsample by averaging into num_points bins
        indices = np.linspace(0, len(rms), num_points + 1, dtype=int)
        result = []
        for i in range(num_points):
            segment = rms[indices[i]:indices[i + 1]]
            result.append(float(segment.mean()) if len(segment) > 0 else 0.0)
        return result


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resample_to_fps(feature: np.ndarray, orig_fps: float, target_fps: float,
                     total_frames: int) -> np.ndarray:
    """Resample a feature array from its original frame rate to target FPS."""
    if len(feature) == 0:
        return np.zeros(total_frames, dtype=np.float32)
    orig_times = np.arange(len(feature)) / orig_fps
    target_times = np.arange(total_frames) / target_fps
    return np.interp(target_times, orig_times, feature).astype(np.float32)


def smooth_features(features: dict[str, np.ndarray],
                    attack_frames: int = 2,
                    release_frames: int = 8) -> dict[str, np.ndarray]:
    """Apply asymmetric EMA smoothing (fast attack, slow release) to all features.

    This prevents visual flickering by allowing fast response to transients
    while maintaining smooth decay.
    """
    attack_frames = max(1, attack_frames)
    release_frames = max(1, release_frames)
    attack_alpha = 1.0 - exp(-1.0 / attack_frames)
    release_alpha = 1.0 - exp(-1.0 / release_frames)

    result = {}
    for name, arr in features.items():
        smoothed = np.empty_like(arr)
        if len(arr) == 0:
            result[name] = smoothed
            continue
        smoothed[0] = arr[0]
        for i in range(1, len(arr)):
            if arr[i] > smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1] + attack_alpha * (arr[i] - smoothed[i - 1])
            else:
                smoothed[i] = smoothed[i - 1] + release_alpha * (arr[i] - smoothed[i - 1])
        result[name] = smoothed
    return result


class AudioAnalyzer:
    """Extracts audio features from files using librosa."""

    def analyze(self, audio_path: str, fps: float,
                stems: dict[str, np.ndarray] | None = None,
                attack_frames: int = 2,
                release_frames: int = 8) -> AudioAnalysis:
        """Analyze audio file and return per-frame features normalized to [0, 1].

        Args:
            audio_path: Path to audio file.
            fps: Target frames per second for resampling.
            stems: Optional pre-separated stems {name: mono_audio_array}.
            attack_frames: EMA attack smoothing in frames.
            release_frames: EMA release smoothing in frames.
        """
        import librosa

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")

        log.info("Analyzing audio: %s (fps=%.1f)", path.name, fps)

        # Load audio (mono, 22050 Hz default)
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        duration = len(y) / sr
        total_frames = max(1, int(duration * fps))

        log.info("Audio loaded: %.1fs, %d Hz, %d target frames", duration, sr, total_frames)

        # Compute hop length for librosa features
        hop_length = 512

        # Feature frame rate from librosa (depends on hop_length and sr)
        librosa_fps = sr / hop_length

        features: dict[str, np.ndarray] = {}

        # --- Global features ---

        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        features["global_rms"] = _normalize(_resample_to_fps(rms, librosa_fps, fps, total_frames))

        # Onset strength
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        features["global_onset"] = _normalize(_resample_to_fps(onset, librosa_fps, fps, total_frames))

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        features["global_centroid"] = _normalize(_resample_to_fps(centroid, librosa_fps, fps, total_frames))

        # Multi-band energy from mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Convert back to linear for energy summing (dB is log scale)
        mel_linear = librosa.db_to_power(mel_db)

        # Core 3-band split (backward-compatible)
        low_energy = mel_linear[:_BAND_LOW_END, :].mean(axis=0)
        mid_energy = mel_linear[_BAND_LOW_END:_BAND_MID_END, :].mean(axis=0)
        high_energy = mel_linear[_BAND_MID_END:, :].mean(axis=0)

        features["global_low"] = _normalize(_resample_to_fps(low_energy, librosa_fps, fps, total_frames))
        features["global_mid"] = _normalize(_resample_to_fps(mid_energy, librosa_fps, fps, total_frames))
        features["global_high"] = _normalize(_resample_to_fps(high_energy, librosa_fps, fps, total_frames))

        # Extended bands: sub_bass (20-60Hz), upper_mid (2-4kHz), presence (4-8kHz)
        sub_bass_energy = mel_linear[:_BAND_SUB_BASS_END, :].mean(axis=0)
        upper_mid_energy = mel_linear[_BAND_MID_END:_BAND_UPPER_MID_END, :].mean(axis=0)
        presence_energy = mel_linear[_BAND_UPPER_MID_END:_BAND_PRESENCE_END, :].mean(axis=0)

        features["global_sub_bass"] = _normalize(_resample_to_fps(sub_bass_energy, librosa_fps, fps, total_frames))
        features["global_upper_mid"] = _normalize(_resample_to_fps(upper_mid_energy, librosa_fps, fps, total_frames))
        features["global_presence"] = _normalize(_resample_to_fps(presence_energy, librosa_fps, fps, total_frames))

        # BPM detection + beat feature
        tempo, beat_frames_idx = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        bpm = float(np.atleast_1d(tempo)[0])
        beat_signal = np.zeros_like(onset)
        for bf in beat_frames_idx:
            if bf < len(beat_signal):
                beat_signal[bf] = 1.0
        features["global_beat"] = _normalize(
            _resample_to_fps(beat_signal, librosa_fps, fps, total_frames)
        )
        log.info("BPM detected: %.1f", bpm)

        # --- Per-stem features (if stems provided) ---
        if stems:
            for stem_name, stem_audio in stems.items():
                # Ensure mono and correct sample rate
                if len(stem_audio.shape) > 1:
                    stem_audio = stem_audio.mean(axis=0) if stem_audio.shape[0] <= 4 else stem_audio[:, 0]
                if len(stem_audio) == 0:
                    continue

                stem_rms = librosa.feature.rms(y=stem_audio, frame_length=2048, hop_length=hop_length)[0]
                features[f"{stem_name}_rms"] = _normalize(
                    _resample_to_fps(stem_rms, librosa_fps, fps, total_frames)
                )

                stem_onset = librosa.onset.onset_strength(y=stem_audio, sr=sr, hop_length=hop_length)
                features[f"{stem_name}_onset"] = _normalize(
                    _resample_to_fps(stem_onset, librosa_fps, fps, total_frames)
                )

        # Apply smoothing
        features = smooth_features(features, attack_frames, release_frames)

        log.info("Analysis complete: %d features, %d frames", len(features), total_frames)

        return AudioAnalysis(
            fps=fps,
            duration=duration,
            total_frames=total_frames,
            sample_rate=sr,
            audio_path=audio_path,
            features=features,
            bpm=bpm,
        )
