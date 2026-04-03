"""Cache for audio analysis results — avoids re-analyzing the same file."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .audio_analyzer import AudioAnalysis

log = logging.getLogger("sddj.audio_cache")


def _cache_key(audio_path: str, fps: float, enable_stems: bool) -> str:
    """Compute a cache key from file mtime+size + fps + stems + DSP config."""
    from .config import settings

    stat = os.stat(audio_path)
    hasher = hashlib.sha256()
    hasher.update(f"{audio_path}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    hasher.update(f"{fps:.2f}".encode())
    hasher.update(b"stems" if enable_stems else b"nostem")
    # DSP config — changing these invalidates the cache
    hasher.update(f"sr{settings.audio_sample_rate}".encode())
    hasher.update(f"hop{settings.audio_hop_length}".encode())
    hasher.update(f"nfft{settings.audio_n_fft}".encode())
    hasher.update(f"nmel{settings.audio_n_mels}".encode())
    hasher.update(b"kw" if settings.audio_perceptual_weighting else b"nokw")
    return hasher.hexdigest()[:24]


def _max_age_seconds() -> int:
    """Return cache TTL from config."""
    from .config import settings
    return settings.audio_cache_ttl_hours * 3600


class AudioCache:
    """Disk-backed cache for AudioAnalysis results using .npz files."""

    def __init__(self, cache_dir: str = "") -> None:
        if cache_dir:
            self._dir = Path(cache_dir)
        else:
            self._dir = Path(tempfile.gettempdir()) / "sddj_audio_cache"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._put_count = 0
        log.info("Audio cache directory: %s", self._dir)

    def get(self, audio_path: str, fps: float, enable_stems: bool = False) -> AudioAnalysis | None:
        """Return cached analysis or None if not found / expired."""
        with self._lock:
            key = _cache_key(audio_path, fps, enable_stems)
            npz_path = self._dir / f"{key}.npz"
            meta_path = self._dir / f"{key}.meta"

            if not npz_path.is_file() or not meta_path.is_file():
                return None

            # Check age
            max_age = _max_age_seconds()
            age = time.time() - npz_path.stat().st_mtime
            if age > max_age:
                log.debug("Cache expired for %s (%.0fh old)", audio_path, age / 3600)
                npz_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None

            try:
                data = np.load(str(npz_path), allow_pickle=False)
                features = {}
                raw_features = {}
                for name in data.files:
                    if name.startswith("raw_"):
                        raw_features[name[4:]] = data[name]
                    else:
                        features[name] = data[name]

                meta_dict = json.loads(meta_path.read_text())

                analysis = AudioAnalysis(
                    fps=float(meta_dict["fps"]),
                    duration=float(meta_dict["duration"]),
                    total_frames=int(meta_dict["total_frames"]),
                    sample_rate=int(meta_dict["sample_rate"]),
                    audio_path=meta_dict["audio_path"],
                    features=features,
                    raw_features=raw_features,
                    bpm=float(meta_dict.get("bpm", 0.0)),
                    lufs=float(meta_dict.get("lufs", -24.0)),
                )
                log.info("Cache hit for %s (%d features)", Path(audio_path).name, len(features))
                return analysis
            except Exception as e:
                log.warning("Cache read failed for %s: %s", audio_path, e)
                npz_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None

    def put(self, audio_path: str, fps: float, analysis: AudioAnalysis,
            enable_stems: bool = False) -> None:
        """Store analysis result in cache (auto-evicts expired entries)."""
        with self._lock:
            # Auto-evict expired entries every 10 puts to amortize I/O cost
            self._put_count += 1
            if self._put_count % 10 == 0:
                try:
                    removed = self._cleanup_unlocked()
                    if removed:
                        log.debug("Auto-evicted %d expired cache entries", removed)
                except Exception as e:
                    log.warning("Auto-evict failed (non-fatal): %s", e)
            key = _cache_key(audio_path, fps, enable_stems)
            npz_path = self._dir / f"{key}.npz"
            meta_path = self._dir / f"{key}.meta"

            try:
                save_dict = dict(analysis.features)
                for name, arr in analysis.raw_features.items():
                    save_dict[f"raw_{name}"] = arr
                # Atomic write: save to temp file then os.replace (cross-platform atomic)
                fd, tmp_npz = tempfile.mkstemp(suffix=".npz", dir=str(self._dir))
                os.close(fd)
                np.savez_compressed(tmp_npz, **save_dict)
                os.replace(tmp_npz, str(npz_path))

                fd, tmp_meta = tempfile.mkstemp(suffix=".meta", dir=str(self._dir))
                os.close(fd)
                Path(tmp_meta).write_text(json.dumps({
                    "fps": analysis.fps,
                    "duration": analysis.duration,
                    "total_frames": analysis.total_frames,
                    "sample_rate": analysis.sample_rate,
                    "audio_path": analysis.audio_path,
                    "bpm": analysis.bpm,
                    "lufs": analysis.lufs,
                }))
                os.replace(tmp_meta, str(meta_path))
                log.info("Cached analysis for %s (%d features)", Path(audio_path).name, len(analysis.features))
            except Exception as e:
                log.warning("Cache write failed: %s", e)

    def invalidate(self, audio_path: str, fps: float, enable_stems: bool = False) -> None:
        """Remove cached entry for a specific file."""
        with self._lock:
            key = _cache_key(audio_path, fps, enable_stems)
            (self._dir / f"{key}.npz").unlink(missing_ok=True)
            (self._dir / f"{key}.meta").unlink(missing_ok=True)

    def _cleanup_unlocked(self) -> int:
        """Remove expired cache entries (caller must hold self._lock)."""
        removed = 0
        now = time.time()
        max_age = _max_age_seconds()
        for f in self._dir.glob("*.npz"):
            if now - f.stat().st_mtime > max_age:
                f.unlink(missing_ok=True)
                meta = f.with_suffix(".meta")
                meta.unlink(missing_ok=True)
                removed += 1
        return removed

    def cleanup(self) -> int:
        """Remove expired cache entries. Returns number of entries removed."""
        with self._lock:
            return self._cleanup_unlocked()
