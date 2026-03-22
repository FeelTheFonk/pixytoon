"""Cache for audio analysis results — avoids re-analyzing the same file."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from .audio_analyzer import AudioAnalysis

log = logging.getLogger("sddj.audio")

# Cache entries older than this are purged on access
_MAX_AGE_SECONDS = 24 * 3600  # 24 hours

# Bytes to read for hashing (first 1 MB + file size = fast unique key)
_HASH_CHUNK = 1024 * 1024


def _cache_key(audio_path: str, fps: float, enable_stems: bool) -> str:
    """Compute a cache key from file content hash + fps + stems flag."""
    p = Path(audio_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        hasher.update(f.read(_HASH_CHUNK))
    hasher.update(str(p.stat().st_size).encode())
    hasher.update(f"{fps:.2f}".encode())
    hasher.update(b"stems" if enable_stems else b"nostem")
    return hasher.hexdigest()[:24]


class AudioCache:
    """Disk-backed cache for AudioAnalysis results using .npz files."""

    def __init__(self, cache_dir: str = "") -> None:
        if cache_dir:
            self._dir = Path(cache_dir)
        else:
            self._dir = Path(tempfile.gettempdir()) / "sddj_audio_cache"
        self._dir.mkdir(parents=True, exist_ok=True)
        log.info("Audio cache directory: %s", self._dir)

    def get(self, audio_path: str, fps: float, enable_stems: bool = False) -> AudioAnalysis | None:
        """Return cached analysis or None if not found / expired."""
        key = _cache_key(audio_path, fps, enable_stems)
        npz_path = self._dir / f"{key}.npz"
        meta_path = self._dir / f"{key}.meta"

        if not npz_path.is_file() or not meta_path.is_file():
            return None

        # Check age
        age = time.time() - npz_path.stat().st_mtime
        if age > _MAX_AGE_SECONDS:
            log.debug("Cache expired for %s (%.0fh old)", audio_path, age / 3600)
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

        try:
            data = np.load(str(npz_path), allow_pickle=False)
            features = {name: data[name] for name in data.files}

            meta = meta_path.read_text().strip().split("\n")
            meta_dict = {}
            for line in meta:
                k, v = line.split("=", 1)
                meta_dict[k] = v

            analysis = AudioAnalysis(
                fps=float(meta_dict["fps"]),
                duration=float(meta_dict["duration"]),
                total_frames=int(meta_dict["total_frames"]),
                sample_rate=int(meta_dict["sample_rate"]),
                audio_path=meta_dict["audio_path"],
                features=features,
                bpm=float(meta_dict.get("bpm", 0.0)),
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
        """Store analysis result in cache."""
        key = _cache_key(audio_path, fps, enable_stems)
        npz_path = self._dir / f"{key}.npz"
        meta_path = self._dir / f"{key}.meta"

        try:
            np.savez_compressed(str(npz_path), **analysis.features)
            meta_path.write_text(
                f"fps={analysis.fps}\n"
                f"duration={analysis.duration}\n"
                f"total_frames={analysis.total_frames}\n"
                f"sample_rate={analysis.sample_rate}\n"
                f"audio_path={analysis.audio_path}\n"
                f"bpm={analysis.bpm}\n"
            )
            log.info("Cached analysis for %s (%d features)", Path(audio_path).name, len(analysis.features))
        except Exception as e:
            log.warning("Cache write failed: %s", e)

    def invalidate(self, audio_path: str, fps: float, enable_stems: bool = False) -> None:
        """Remove cached entry for a specific file."""
        key = _cache_key(audio_path, fps, enable_stems)
        (self._dir / f"{key}.npz").unlink(missing_ok=True)
        (self._dir / f"{key}.meta").unlink(missing_ok=True)

    def cleanup(self) -> int:
        """Remove expired cache entries. Returns number of entries removed."""
        removed = 0
        now = time.time()
        for f in self._dir.glob("*.npz"):
            if now - f.stat().st_mtime > _MAX_AGE_SECONDS:
                f.unlink(missing_ok=True)
                meta = f.with_suffix(".meta")
                meta.unlink(missing_ok=True)
                removed += 1
        return removed
