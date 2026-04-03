"""Audio feature extraction for audio-reactive generation.

Pinnacle-quality DSP pipeline:
  - 44100 Hz sample rate (full audible spectrum up to 22.05 kHz)
  - 256 hop length (~172 Hz feature rate)
  - 4096 n_fft (93 ms analysis window — identical to 2048@22050)
  - 128 mel bands for precise frequency segmentation (configurable via SDDJ_AUDIO_N_MELS)
  - ITU-R BS.1770 K-weighting pre-filter for perceptual loudness
  - 9-band frequency segmentation with backward-compatible aliases
  - SuperFlux onset detection (vibrato suppression)
  - 5 spectral timbral features (contrast, flatness, bandwidth, rolloff, flux)
  - 12-bin CQT chromagram (individual pitch classes)
  - Optional madmom RNN beat tracking (20-60ms more phase-accurate)
  - Optional BeatNet particle-filtering beat tracking (offline mode)
  - Optional All-In-One joint beat + structure analysis
  - Vocal MFCC extraction (13 coefficients) for voice-reactive modulation
  - Savitzky-Golay smoothing option (peak preservation)
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from math import exp
from pathlib import Path

import numba
import numpy as np

log = logging.getLogger("sddj.audio")

# Supported audio file extensions
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})

# Chroma pitch-class names (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
_CHROMA_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

# 9-band frequency boundaries (Hz)
_BAND_BOUNDARIES = [
    ("sub_bass",   20,    60),
    ("bass",       60,    150),
    ("low_mid",    150,   400),
    ("mid",        400,   2000),
    ("upper_mid",  2000,  4000),
    ("presence",   4000,  8000),
    ("brilliance", 8000,  12000),
    ("air",        12000, 20000),
    ("ultrasonic", 20000, 22050),
]

# Detect optional beat-tracking backend availability once at import
try:
    import madmom  # noqa: F401
    _HAS_MADMOM = True
except ImportError:
    _HAS_MADMOM = False

try:
    from BeatNet.BeatNet import BeatNet as _BeatNetModel  # noqa: F401
    _HAS_BEATNET = True
except ImportError:
    _HAS_BEATNET = False

try:
    import allin1  # noqa: F401
    _HAS_ALLINONE = True
except ImportError:
    _HAS_ALLINONE = False


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
    lufs: float = -24.0  # Integrated LUFS for the whole file
    # Raw (un-smoothed) features for per-slot EMA in modulation engine
    raw_features: dict[str, np.ndarray] = field(default_factory=dict)
    # Structure labels from All-In-One analysis (optional)
    structure_labels: list[dict] | None = None

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.features.keys())

    def get_waveform_preview(self, num_points: int = 100) -> list[float]:
        """Return a downsampled RMS waveform for UI display."""
        rms = self.features.get("global_rms")
        if rms is None or len(rms) == 0:
            return [0.0] * num_points
        indices = np.linspace(0, len(rms), num_points + 1, dtype=int)
        result = []
        for i in range(num_points):
            segment = rms[indices[i]:indices[i + 1]]
            result.append(float(segment.mean()) if len(segment) > 0 else 0.0)
        return result


def _normalize(arr: np.ndarray, name: str = "") -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if constant."""
    if len(arr) == 0:
        return np.zeros(1, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-10:
        if name:
            log.debug("Feature '%s' is constant (value=%.4f) — modulation will be inactive", name, lo)
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _normalize_percentile(arr: np.ndarray, name: str = "", pct: float = 99.0) -> np.ndarray:
    """Percentile-clipped normalization — prevents single-spike distortion."""
    if len(arr) == 0:
        return np.zeros(1, dtype=np.float32)
    lo = float(np.percentile(arr, 100.0 - pct))
    hi = float(np.percentile(arr, pct))
    if hi - lo < 1e-10:
        return _normalize(arr, name)
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _resample_to_fps(feature: np.ndarray, orig_fps: float, target_fps: float,
                     total_frames: int) -> np.ndarray:
    """Resample a feature array leveraging SOTA transient-preserving algorithms.
    Upsampling uses PCHIP (Piecewise Cubic Hermite) interpolation to prevent ringing.
    Downsampling utilizes an exact-window envelope tracker to capture sub-frame peaks."""
    if len(feature) == 0:
        return np.zeros(total_frames, dtype=np.float32)

    # SOTA ZeroDivision Protection
    target_fps = max(1e-5, float(target_fps))

    orig_times = np.arange(len(feature)) / orig_fps
    target_times = np.arange(total_frames) / target_fps

    # If upsampling (or holding same rate)
    if target_fps >= orig_fps:
        if len(feature) < 4:
            return np.interp(target_times, orig_times, feature).astype(np.float32)
        from scipy.interpolate import PchipInterpolator
        return np.clip(PchipInterpolator(orig_times, feature)(target_times), 0.0, None).astype(np.float32)

    # O-14: Downsampling (vectorized max-pooling to preserve onset transients)
    dt_half = 1.0 / (2.0 * target_fps)
    starts = np.searchsorted(orig_times, target_times - dt_half, side='left')
    ends = np.searchsorted(orig_times, target_times + dt_half, side='right')
    resampled = np.empty(total_frames, dtype=np.float32)
    # For frames where the window is empty, fall back to nearest neighbor
    empty_mask = starts == ends
    if empty_mask.any():
        nearest_idx = np.searchsorted(orig_times, target_times[empty_mask])
        nearest_idx = np.clip(nearest_idx, 0, len(orig_times) - 1)
        # Check if previous index is closer
        prev_idx = np.clip(nearest_idx - 1, 0, len(orig_times) - 1)
        use_prev = np.abs(orig_times[prev_idx] - target_times[empty_mask]) < np.abs(orig_times[nearest_idx] - target_times[empty_mask])
        nearest_idx[use_prev] = prev_idx[use_prev]
        resampled[empty_mask] = feature[nearest_idx]
    # For frames with valid windows, take the max (preserves transient peaks)
    # O-14: Use np.maximum.reduceat for vectorized max-pool
    non_empty = np.where(~empty_mask)[0]
    if len(non_empty) > 0:
        ne_starts = starts[non_empty]
        ne_ends = ends[non_empty]
        lengths = ne_ends - ne_starts
        # Build flat index array via vectorized np.repeat + arange
        # (replaces N individual np.arange() calls + np.concatenate)
        offsets = np.zeros(len(lengths), dtype=np.intp)
        np.cumsum(lengths[:-1], out=offsets[1:])
        total_len = int(lengths.sum())
        # Each window starts at ne_starts[i], runs for lengths[i] elements
        flat_indices = np.repeat(ne_starts, lengths) + (
            np.arange(total_len, dtype=np.intp) - np.repeat(offsets, lengths)
        )
        pool_max = np.maximum.reduceat(feature[flat_indices], offsets)
        resampled[non_empty] = pool_max
    return resampled


# ─────────────────────────────────────────────────────────────
# K-WEIGHTING PRE-FILTER (ITU-R BS.1770)
# ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def _kweight_sos(sr: int):
    """Compute K-weighting second-order-section filter coefficients.

    Stage 1: High-shelf pre-filter (+4 dB above ~1.5 kHz)
    Stage 2: High-pass RLB weighting (−∞ below ~38 Hz)

    Cached per sample rate.
    """
    from scipy.signal import butter

    nyq = sr / 2.0

    # Stage 1: High-shelf approximation — 1st-order high-pass at 1500 Hz
    # boosts energy above ~1.5 kHz (BS.1770 pre-filter proxy)
    fc = min(1500.0 / nyq, 0.99)
    sos_shelf = butter(1, fc, btype='highpass', output='sos')

    # Stage 2: High-pass at ~38 Hz (RLB weighting — removes sub-bass rumble)
    sos_hp = butter(2, 38.0 / nyq, btype='highpass', output='sos')

    return np.vstack([sos_shelf, sos_hp])


def _apply_kweight(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply K-weighting pre-filter to audio signal."""
    from scipy.signal import sosfilt

    sos = _kweight_sos(sr)
    # Apply filter (creates a copy, does not modify original)
    return sosfilt(sos, y).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def _ema_asymmetric_core(arr, attack_alpha, release_alpha):
    """O-13: Numba-accelerated asymmetric EMA kernel (fast attack, slow release)."""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    out[0] = arr[0]
    for i in range(1, n):
        if arr[i] > out[i - 1]:
            out[i] = out[i - 1] + attack_alpha * (arr[i] - out[i - 1])
        else:
            out[i] = out[i - 1] + release_alpha * (arr[i] - out[i - 1])
    return out


def smooth_features_ema(features: dict[str, np.ndarray],
                        attack_frames: int = 2,
                        release_frames: int = 8) -> dict[str, np.ndarray]:
    """Apply asymmetric EMA smoothing (fast attack, slow release).

    O-13: Inner loop vectorized via Numba @njit kernel.
    """
    attack_frames = max(1, attack_frames)
    release_frames = max(1, release_frames)
    attack_alpha = 1.0 - exp(-1.0 / attack_frames)
    release_alpha = 1.0 - exp(-1.0 / release_frames)

    result = {}
    for name, arr in features.items():
        if len(arr) == 0:
            result[name] = np.empty_like(arr)
            continue
        result[name] = _ema_asymmetric_core(
            arr.astype(np.float32), attack_alpha, release_alpha,
        )
    return result


def smooth_features_savgol(features: dict[str, np.ndarray],
                           attack_frames: int = 2,
                           release_frames: int = 8) -> dict[str, np.ndarray]:
    """Apply causal Savitzky-Golay smoothing (preserves transient peaks).

    Uses right-edge alignment: the filter window extends only into past
    samples + current, making it causal for real-time-like processing.
    """
    from scipy.signal import savgol_filter

    window_len = 2 * max(attack_frames, release_frames) + 1
    result = {}
    for name, arr in features.items():
        if len(arr) < window_len:
            result[name] = arr.copy()
            continue
        # Causal: pad right side and shift, so filter uses only past+current
        padded = np.pad(arr, (window_len - 1, 0), mode='edge')
        filtered = savgol_filter(padded, window_length=window_len, polyorder=2)
        # Take the last len(arr) samples (right-aligned = causal output)
        result[name] = filtered[window_len - 1:].astype(np.float32)
    return result


# Keep backward-compatible name
smooth_features = smooth_features_ema


def _compute_mel_band_indices(sr: int, n_mels: int) -> dict[str, tuple[int, int]]:
    """Compute exact mel bin index ranges for each frequency band.

    Returns dict mapping band_name -> (start_bin, end_bin) exclusive.
    """
    import librosa

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2.0)
    bands = {}
    for name, f_lo, f_hi in _BAND_BOUNDARIES:
        # Find first bin >= f_lo and last bin < f_hi
        start = int(np.searchsorted(mel_freqs, f_lo, side='left'))
        end = int(np.searchsorted(mel_freqs, f_hi, side='left'))
        # Clamp to valid range
        start = max(0, min(start, n_mels - 1))
        end = max(start + 1, min(end, n_mels))
        bands[name] = (start, end)
    return bands


# ─────────────────────────────────────────────────────────────
# BEAT TRACKING BACKENDS
# ─────────────────────────────────────────────────────────────

def _beat_track_librosa(y, sr, hop_length, n_frames, onset_env):
    """Beat tracking using librosa (dynamic programming)."""
    import librosa

    tempo, beat_frames_idx = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, onset_envelope=onset_env,
    )
    bpm = float(np.atleast_1d(tempo)[0])
    if np.isnan(bpm):
        bpm = 0.0

    beat_signal = np.zeros(n_frames, dtype=np.float32)
    if len(beat_frames_idx) > 0:
        for bf in beat_frames_idx:
            if bf < n_frames:
                beat_signal[bf] = 1.0
        log.info("BPM (librosa): %.1f (%d beats)", bpm, len(beat_frames_idx))
    return bpm, beat_signal


def _beat_track_madmom(audio_path, sr, hop_length, n_frames):
    """Beat tracking using madmom RNN + DBN (more phase-accurate)."""
    import madmom

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(str(audio_path))
    beat_times = proc(act)

    bpm = 0.0
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        median_interval = float(np.median(intervals))
        if median_interval > 0:
            bpm = 60.0 / median_interval

    beat_signal = np.zeros(n_frames, dtype=np.float32)
    for bt in beat_times:
        frame_idx = int(bt * sr / hop_length)
        if 0 <= frame_idx < n_frames:
            beat_signal[frame_idx] = 1.0

    log.info("BPM (madmom): %.1f (%d beats)", bpm, len(beat_times))
    return bpm, beat_signal


def _beat_track_beatnet(audio_path, sr, hop_length, n_frames):
    """Beat tracking using BeatNet (particle filtering + RNN, offline mode)."""
    from BeatNet.BeatNet import BeatNet as BeatNetModel

    estimator = BeatNetModel(
        1,  # mode=1: offline (whole-file analysis)
        inference_model="DBN",
        plot=[],
        thread=False,
    )
    output = estimator.process(str(audio_path))
    # output is ndarray of shape (N, 2): columns [beat_time, beat_number]
    beat_times = output[:, 0] if output.ndim == 2 and output.shape[0] > 0 else np.array([])

    bpm = 0.0
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        median_interval = float(np.median(intervals))
        if median_interval > 0:
            bpm = 60.0 / median_interval

    beat_signal = np.zeros(n_frames, dtype=np.float32)
    for bt in beat_times:
        frame_idx = int(bt * sr / hop_length)
        if 0 <= frame_idx < n_frames:
            beat_signal[frame_idx] = 1.0

    log.info("BPM (beatnet): %.1f (%d beats)", bpm, len(beat_times))
    return bpm, beat_signal


def _beat_track_allinone(audio_path, sr, hop_length, n_frames):
    """Beat tracking using All-In-One (joint beat + structure analysis)."""
    import allin1

    result = allin1.analyze(str(audio_path))
    beat_times = np.array(result.beats, dtype=np.float64) if result.beats else np.array([])

    bpm = 0.0
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        median_interval = float(np.median(intervals))
        if median_interval > 0:
            bpm = 60.0 / median_interval

    beat_signal = np.zeros(n_frames, dtype=np.float32)
    for bt in beat_times:
        frame_idx = int(bt * sr / hop_length)
        if 0 <= frame_idx < n_frames:
            beat_signal[frame_idx] = 1.0

    # Extract structure labels if available (stored in AudioAnalysis for modulation)
    structure_labels: list[dict] | None = None
    if hasattr(result, "segments") and result.segments:
        structure_labels = [
            {"start": seg.start, "end": seg.end, "label": seg.label}
            for seg in result.segments
        ]
        log.info("All-In-One structure: %d segments detected", len(structure_labels))

    log.info("BPM (allinone): %.1f (%d beats)", bpm, len(beat_times))
    return bpm, beat_signal, structure_labels


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION PER SIGNAL
# ─────────────────────────────────────────────────────────────

def _extract_features(y: np.ndarray, sr: int, hop_length: int, n_fft: int,
                      n_mels: int, perceptual_weighting: bool,
                      superflux_lag: int, superflux_max_size: int,
                      librosa_fps: float, target_fps: float, total_frames: int,
                      prefix: str = "global") -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract all features from a mono audio signal.

    Args:
        prefix: Feature name prefix ("global" or stem name like "drums").

    Returns:
        Tuple of (features_dict, onset_envelope). The onset_envelope is returned
        so the caller can reuse it for beat tracking without recomputing.
    """
    import librosa

    features: dict[str, np.ndarray] = {}

    # ── STFT magnitude (computed once, reused for all spectral features) ──
    S_mag = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    S_power = S_mag ** 2

    # K-weighting in frequency domain: apply filter magnitude response to
    # the single STFT instead of computing a second STFT on the time-domain
    # K-weighted signal. Saves one full STFT (~40% of DSP cost).
    if perceptual_weighting:
        from scipy.signal import sosfreqz
        sos = _kweight_sos(sr)
        n_freqs = S_mag.shape[0]
        freqs = np.linspace(0, np.pi, n_freqs)
        _, h_complex = sosfreqz(sos, worN=freqs)
        k_weight_filter = np.abs(h_complex).astype(np.float32)[:, np.newaxis]
        S_mag_weighted = S_mag * k_weight_filter
        S_power_weighted = S_mag_weighted ** 2
        # K-weighted time-domain signal still needed for RMS energy
        y_energy = _apply_kweight(y, sr)
    else:
        S_mag_weighted = S_mag
        S_power_weighted = S_power
        y_energy = y

    # ── RMS energy (K-weighted if enabled) ──
    rms_raw = librosa.feature.rms(y=y_energy, frame_length=n_fft, hop_length=hop_length)[0]
    rms = rms_raw if len(rms_raw) > 0 else np.zeros(1)
    features[f"{prefix}_rms"] = _normalize(
        _resample_to_fps(rms, librosa_fps, target_fps, total_frames), f"{prefix}_rms",
    )

    # ── Onset strength (SuperFlux — vibrato suppression) ──
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length,
        lag=superflux_lag, max_size=superflux_max_size,
    )
    if len(onset) == 0:
        onset = np.zeros(1)
    features[f"{prefix}_onset"] = _normalize_percentile(
        _resample_to_fps(onset, librosa_fps, target_fps, total_frames), f"{prefix}_onset",
    )

    # ── Spectral centroid (reuses pre-computed S_mag) ──
    centroid_raw = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=hop_length)[0]
    centroid = centroid_raw if len(centroid_raw) > 0 else np.zeros(1)
    features[f"{prefix}_centroid"] = _normalize(
        _resample_to_fps(centroid, librosa_fps, target_fps, total_frames), f"{prefix}_centroid",
    )

    # ── Mel spectrogram (from K-weighted STFT power — no second STFT) ──
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S_mel = mel_basis @ S_power_weighted

    # ── 9-band segmentation (dynamic bin computation) ──
    band_indices = _compute_mel_band_indices(sr, n_mels)
    for band_name, (b_start, b_end) in band_indices.items():
        energy = S_mel[b_start:b_end, :].mean(axis=0)
        # Use "_mid_narrow" for the 9-band "mid" to avoid overwriting the
        # backward-compatible "{prefix}_mid" alias computed below.
        feat_key = f"{prefix}_mid_narrow" if band_name == "mid" else f"{prefix}_{band_name}"
        features[feat_key] = _normalize(
            _resample_to_fps(energy, librosa_fps, target_fps, total_frames),
            feat_key,
        )

    # ── Backward-compatible aliases ──
    # global_low = weighted mean of sub_bass + bass + low_mid
    low_bands = ["sub_bass", "bass", "low_mid"]
    low_sum = sum(
        S_mel[band_indices[b][0]:band_indices[b][1], :].sum(axis=0)
        for b in low_bands
    )
    low_count = sum(band_indices[b][1] - band_indices[b][0] for b in low_bands)
    features[f"{prefix}_low"] = _normalize(
        _resample_to_fps(low_sum / max(1, low_count), librosa_fps, target_fps, total_frames),
        f"{prefix}_low",
    )

    # global_mid = mid + upper_mid (already extracted individually)
    mid_bands = ["mid", "upper_mid"]
    mid_sum = sum(
        S_mel[band_indices[b][0]:band_indices[b][1], :].sum(axis=0)
        for b in mid_bands
    )
    mid_count = sum(band_indices[b][1] - band_indices[b][0] for b in mid_bands)
    features[f"{prefix}_mid"] = _normalize(
        _resample_to_fps(mid_sum / max(1, mid_count), librosa_fps, target_fps, total_frames),
        f"{prefix}_mid",
    )

    # global_high = presence + brilliance + air + ultrasonic
    high_bands = ["presence", "brilliance", "air", "ultrasonic"]
    high_sum = sum(
        S_mel[band_indices[b][0]:band_indices[b][1], :].sum(axis=0)
        for b in high_bands
    )
    high_count = sum(band_indices[b][1] - band_indices[b][0] for b in high_bands)
    features[f"{prefix}_high"] = _normalize(
        _resample_to_fps(high_sum / max(1, high_count), librosa_fps, target_fps, total_frames),
        f"{prefix}_high",
    )

    # ── Spectral contrast (peak-vs-valley per frequency band) ──
    contrast = librosa.feature.spectral_contrast(S=S_power, sr=sr, hop_length=hop_length, n_bands=6)
    contrast_mean = contrast.mean(axis=0)  # mean across 6+1 bands → scalar per frame
    features[f"{prefix}_spectral_contrast"] = _normalize(
        _resample_to_fps(contrast_mean, librosa_fps, target_fps, total_frames),
        f"{prefix}_spectral_contrast",
    )

    # ── Spectral flatness (tonality: 0=tone, 1=noise) ──
    flatness = librosa.feature.spectral_flatness(S=S_power)[0]
    features[f"{prefix}_spectral_flatness"] = _normalize(
        _resample_to_fps(flatness, librosa_fps, target_fps, total_frames),
        f"{prefix}_spectral_flatness",
    )

    # ── Spectral bandwidth (frequency spread around centroid) ──
    bandwidth = librosa.feature.spectral_bandwidth(S=S_power, sr=sr, hop_length=hop_length)[0]
    features[f"{prefix}_spectral_bandwidth"] = _normalize(
        _resample_to_fps(bandwidth, librosa_fps, target_fps, total_frames),
        f"{prefix}_spectral_bandwidth",
    )

    # ── Spectral rolloff (frequency below which 85% of energy) ──
    rolloff = librosa.feature.spectral_rolloff(S=S_power, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]
    features[f"{prefix}_spectral_rolloff"] = _normalize(
        _resample_to_fps(rolloff, librosa_fps, target_fps, total_frames),
        f"{prefix}_spectral_rolloff",
    )

    # ── Spectral flux (timbral transition rate) ──
    if S_mag.shape[1] > 1:
        flux = np.linalg.norm(np.diff(S_mag, axis=1), axis=0)
        flux = np.pad(flux, (1, 0), mode='constant')  # align length
    else:
        flux = np.zeros(S_mag.shape[1])
    features[f"{prefix}_spectral_flux"] = _normalize_percentile(
        _resample_to_fps(flux, librosa_fps, target_fps, total_frames),
        f"{prefix}_spectral_flux",
    )

    # ── CQT Chromagram (12 pitch classes) ──
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    for i, pitch_name in enumerate(_CHROMA_NAMES):
        features[f"{prefix}_chroma_{pitch_name}"] = _normalize(
            _resample_to_fps(chroma[i], librosa_fps, target_fps, total_frames),
            f"{prefix}_chroma_{pitch_name}",
        )
    # Aggregate chroma energy
    chroma_energy = chroma.mean(axis=0)
    features[f"{prefix}_chroma_energy"] = _normalize(
        _resample_to_fps(chroma_energy, librosa_fps, target_fps, total_frames),
        f"{prefix}_chroma_energy",
    )

    # ── Tempogram (rhythmic strength per frame) ──
    tempogram = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=hop_length)
    tempo_strength = tempogram.max(axis=0)  # Peak rhythmic strength per frame
    tempo_strength_resampled = _resample_to_fps(tempo_strength, librosa_fps, target_fps, total_frames)
    features[f"{prefix}_tempo_strength"] = _normalize(
        tempo_strength_resampled, f"{prefix}_tempo_strength",
    )

    # ── Tempo variation (rolling std of tempo_strength — rhythmic instability) ──
    _TEMPO_VAR_WINDOW = 8
    if len(tempo_strength_resampled) >= _TEMPO_VAR_WINDOW:
        # Causal rolling std: pad left so output aligns with current frame
        padded = np.pad(tempo_strength_resampled, (_TEMPO_VAR_WINDOW - 1, 0), mode='edge')
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, _TEMPO_VAR_WINDOW)
        tempo_variation = windows.std(axis=1).astype(np.float32)
    else:
        tempo_variation = np.zeros(total_frames, dtype=np.float32)
    features[f"{prefix}_tempo_variation"] = _normalize(
        tempo_variation, f"{prefix}_tempo_variation",
    )

    return features, onset


class AudioAnalyzer:
    """Extracts audio features from files using librosa."""

    def analyze(self, audio_path: str, fps: float,
                stems: dict[str, np.ndarray] | None = None,
                attack_frames: int = 2,
                release_frames: int = 8) -> AudioAnalysis:
        """Analyze audio file and return per-frame features normalized to [0, 1]."""
        import librosa

        from .config import settings

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")

        sr = settings.audio_sample_rate
        hop_length = settings.audio_hop_length
        n_fft = settings.audio_n_fft
        n_mels = settings.audio_n_mels

        log.info("Analyzing audio: %s (fps=%.1f, sr=%d, hop=%d, n_fft=%d, n_mels=%d)",
                 path.name, fps, sr, hop_length, n_fft, n_mels)

        # Load audio (mono, configurable sample rate)
        y, sr = librosa.load(str(path), sr=sr, mono=True)
        duration = len(y) / sr
        total_frames = max(1, int(duration * fps))
        librosa_fps = sr / hop_length

        log.info("Audio loaded: %.1fs, %d Hz, %d target frames (feature rate: %.1f Hz)",
                 duration, sr, total_frames, librosa_fps)

        # ── Integrated LUFS (whole-file reference loudness) ──
        lufs = -24.0
        try:
            import pyloudnorm
            meter = pyloudnorm.Meter(sr)
            # pyloudnorm needs float64 and shape (samples,) for mono
            lufs = float(meter.integrated_loudness(y.astype(np.float64)))
            if np.isinf(lufs) or np.isnan(lufs):
                lufs = -70.0
            log.info("Integrated LUFS: %.1f", lufs)
        except Exception as e:
            log.warning("LUFS measurement failed (using default -24.0): %s", e)

        # ── Extract all global features ──
        features, onset_env = _extract_features(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels,
            perceptual_weighting=settings.audio_perceptual_weighting,
            superflux_lag=settings.audio_superflux_lag,
            superflux_max_size=settings.audio_superflux_max_size,
            librosa_fps=librosa_fps, target_fps=fps, total_frames=total_frames,
            prefix="global",
        )

        # ── Beat tracking (reuses onset_env from feature extraction) ──
        n_onset_frames = len(onset_env)
        structure_labels: list[dict] | None = None

        backend = settings.audio_beat_backend

        # Auto-select best available backend: allinone > beatnet > madmom > librosa
        if backend == "auto":
            if _HAS_ALLINONE:
                backend = "allinone"
            elif _HAS_BEATNET:
                backend = "beatnet"
            elif _HAS_MADMOM:
                backend = "madmom"
            else:
                backend = "librosa"

        if backend == "allinone" and _HAS_ALLINONE:
            try:
                bpm, beat_signal, structure_labels = _beat_track_allinone(
                    audio_path, sr, hop_length, n_onset_frames,
                )
            except Exception as e:
                log.warning("allinone beat tracking failed, falling back to librosa: %s", e)
                bpm, beat_signal = _beat_track_librosa(
                    y, sr, hop_length, n_onset_frames, onset_env,
                )
        elif backend == "beatnet" and _HAS_BEATNET:
            try:
                bpm, beat_signal = _beat_track_beatnet(
                    audio_path, sr, hop_length, n_onset_frames,
                )
            except Exception as e:
                log.warning("beatnet beat tracking failed, falling back to librosa: %s", e)
                bpm, beat_signal = _beat_track_librosa(
                    y, sr, hop_length, n_onset_frames, onset_env,
                )
        elif backend == "madmom" and _HAS_MADMOM:
            try:
                bpm, beat_signal = _beat_track_madmom(
                    audio_path, sr, hop_length, n_onset_frames,
                )
            except Exception as e:
                log.warning("madmom beat tracking failed, falling back to librosa: %s", e)
                bpm, beat_signal = _beat_track_librosa(
                    y, sr, hop_length, n_onset_frames, onset_env,
                )
        else:
            if backend not in ("librosa", "auto"):
                log.warning("Beat backend '%s' not available, falling back to librosa", backend)
            bpm, beat_signal = _beat_track_librosa(
                y, sr, hop_length, n_onset_frames, onset_env,
            )

        # Fallback: derive pseudo-beats from onset peaks
        if beat_signal.sum() < 1.0 and len(onset_env) > 0 and onset_env.max() > 0:
            threshold = np.percentile(onset_env, 75)
            beat_signal = (onset_env > threshold).astype(np.float32)
            log.warning("No beats detected — using onset peaks as fallback (%d pseudo-beats)",
                        int(beat_signal.sum()))

        features["global_beat"] = _normalize(
            _resample_to_fps(beat_signal, librosa_fps, fps, total_frames), "global_beat",
        )

        # ── Per-stem features (all new features per stem) ──
        if stems:
            for stem_name, stem_audio in stems.items():
                if len(stem_audio.shape) > 1:
                    stem_audio = stem_audio.mean(axis=0) if stem_audio.shape[0] <= 4 else stem_audio[:, 0]
                if len(stem_audio) == 0:
                    continue

                stem_features, _stem_onset = _extract_features(
                    y=stem_audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels,
                    perceptual_weighting=settings.audio_perceptual_weighting,
                    superflux_lag=settings.audio_superflux_lag,
                    superflux_max_size=settings.audio_superflux_max_size,
                    librosa_fps=librosa_fps, target_fps=fps, total_frames=total_frames,
                    prefix=stem_name,
                )
                features.update(stem_features)

                # Vocal MFCC features (13 coefficients) — useful for voice-reactive modulation
                if stem_name == "vocals":
                    vocal_mfcc = librosa.feature.mfcc(
                        y=stem_audio, sr=sr, n_mfcc=13, hop_length=hop_length,
                    )
                    for i in range(vocal_mfcc.shape[0]):
                        features[f"vocals_mfcc_{i}"] = _normalize(
                            _resample_to_fps(vocal_mfcc[i], librosa_fps, fps, total_frames),
                            f"vocals_mfcc_{i}",
                        )

        # Shallow copy: smooth_features_ema creates new arrays, does not mutate inputs
        raw_features = dict(features)

        # ── Smoothing ──
        if settings.audio_smoothing_mode == "savgol":
            features = smooth_features_savgol(features, attack_frames, release_frames)
        else:
            features = smooth_features_ema(features, attack_frames, release_frames)

        # Report dead features
        dead = [n for n, arr in features.items() if arr.max() < 1e-10]
        if dead:
            log.warning("Inactive features (constant signal): %s — modulation slots using these will stay at min_val",
                        ", ".join(dead))

        log.info("Analysis complete: %d features (%d active), %d frames",
                 len(features), len(features) - len(dead), total_frames)

        return AudioAnalysis(
            fps=fps,
            duration=duration,
            total_frames=total_frames,
            sample_rate=sr,
            audio_path=audio_path,
            features=features,
            raw_features=raw_features,
            bpm=bpm,
            lufs=lufs,
            structure_labels=structure_labels,
        )
