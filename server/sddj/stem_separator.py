"""Audio stem separation via demucs or BS-RoFormer (optional dependencies).

Separates audio into stems on CPU to avoid competing with the GPU
inference pipeline.

Backends:
  - **demucs** (default): htdemucs — 4 stems (drums, bass, vocals, other).
  - **roformer** (F-O2): BS-RoFormer via audio-separator — 6 stems
    (drums, bass, vocals, other, guitar, piano), +3 dB SDR improvement.
"""

from __future__ import annotations

import atexit
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np

from .config import settings

log = logging.getLogger("sddj.stem_separator")

# Stem names produced by htdemucs (4 stems)
DEMUCS_STEM_NAMES = ("drums", "bass", "vocals", "other")

# Stem names produced by BS-RoFormer (6 stems)
ROFORMER_STEM_NAMES = ("drums", "bass", "vocals", "other", "guitar", "piano")

# Public alias — union of all possible stem names
STEM_NAMES = ROFORMER_STEM_NAMES

_demucs_available: bool | None = None
_roformer_available: bool | None = None


def is_available() -> bool:
    """Check if any stem separation backend is installed and importable."""
    return _is_demucs_available() or _is_roformer_available()


def _is_demucs_available() -> bool:
    """Check if demucs is installed and importable."""
    global _demucs_available
    if _demucs_available is None:
        try:
            import demucs.api  # noqa: F401
            _demucs_available = True
        except ImportError:
            _demucs_available = False
    return _demucs_available


def _is_roformer_available() -> bool:
    """Check if audio-separator (BS-RoFormer backend) is installed."""
    global _roformer_available
    if _roformer_available is None:
        try:
            from audio_separator.separator import Separator  # noqa: F401
            _roformer_available = True
        except ImportError:
            _roformer_available = False
    return _roformer_available


# ─────────────────────────────────────────────────────────────
# Demucs backend
# ─────────────────────────────────────────────────────────────

class _DemucsBackend:
    """htdemucs stem separation — 4 stems (drums, bass, vocals, other)."""

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._separator = None
        self._load_lock = threading.Lock()

    def is_available(self) -> bool:
        return _is_demucs_available()

    def _ensure_loaded(self) -> None:
        if self._separator is not None:
            return
        with self._load_lock:
            if self._separator is not None:
                return
            if not _is_demucs_available():
                raise RuntimeError(
                    "Stem separation (demucs) requires demucs. "
                    "Install with: pip install demucs>=4.0"
                )
            import demucs.api
            log.info("Loading demucs separator: model=%s, device=%s",
                     self._model_name, self._device)
            self._separator = demucs.api.Separator(
                model=self._model_name,
                device=self._device,
            )
            log.info("Demucs separator loaded")

    def separate(self, audio_path: str, target_sr: int = 44100) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Separating stems (demucs): %s", path.name)
        _, stems_tensor = self._separator.separate_audio_file(str(path))
        model_sr = self._separator.samplerate

        result: dict[str, np.ndarray] = {}
        for stem_name in DEMUCS_STEM_NAMES:
            if stem_name in stems_tensor:
                tensor = stems_tensor[stem_name]
                if tensor.dim() == 2:
                    mono = tensor.mean(dim=0)
                else:
                    mono = tensor
                audio = mono.cpu().numpy().astype(np.float32)
                if model_sr != target_sr and len(audio) > 0:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=model_sr, target_sr=target_sr)
                result[stem_name] = audio

        log.info("Demucs separation complete: %d stems (%s) resampled to %d Hz",
                 len(result), ", ".join(result.keys()), target_sr)
        return result

    def unload(self) -> None:
        if self._separator is not None:
            del self._separator
            self._separator = None
            log.info("Demucs separator unloaded")


# ─────────────────────────────────────────────────────────────
# BS-RoFormer backend (via audio-separator)
# ─────────────────────────────────────────────────────────────

class _RoFormerBackend:
    """BS-RoFormer stem separation via audio-separator.

    Produces 6 stems (drums, bass, vocals, other, guitar, piano) with
    +3 dB SDR improvement over htdemucs on standard benchmarks.
    Uses ONNX runtime for inference — no PyTorch dependency for this path.
    """

    # Default ONNX model — Kim_Vocal_2 is the most popular BS-RoFormer checkpoint.
    _DEFAULT_MODEL = "Kim_Vocal_2.onnx"

    # Mapping from audio-separator output filename patterns to canonical stem names.
    # Class constant — avoids dict recreation per separate() call.
    _STEM_MAP = {
        "vocal": "vocals",
        "drum": "drums",
        "bass": "bass",
        "other": "other",
        "guitar": "guitar",
        "piano": "piano",
        "instrum": "other",  # instrumental fallback
    }

    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        self._model_name = model_name or self._DEFAULT_MODEL
        self._device = device
        self._separator = None
        self._load_lock = threading.Lock()

    def is_available(self) -> bool:
        return _is_roformer_available()

    def _ensure_loaded(self) -> None:
        if self._separator is not None:
            return
        with self._load_lock:
            if self._separator is not None:
                return
            if not _is_roformer_available():
                raise RuntimeError(
                    "Stem separation (roformer) requires audio-separator. "
                    "Install with: pip install audio-separator[onnx]"
                )
            from audio_separator.separator import Separator
            log.info("Loading RoFormer separator: model=%s, device=%s",
                     self._model_name, self._device)
            self._output_dir = tempfile.mkdtemp(prefix="sddj_roformer_")
            atexit.register(self._cleanup_output_dir)
            self._separator = Separator(
                output_dir=self._output_dir,
                model_file_dir=str(settings.models_dir / "roformer"),
            )
            self._separator.load_model(model_filename=self._model_name)
            log.info("RoFormer separator loaded")

    def separate(self, audio_path: str, target_sr: int = 44100) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Separating stems (roformer): %s", path.name)
        output_files = self._separator.separate(str(path))

        import librosa

        result: dict[str, np.ndarray] = {}
        # audio-separator returns a list of output file paths.
        # Filenames contain the stem name (e.g., "song_(Vocals).wav").
        # Map known stem name patterns to our canonical names.
        for out_path in output_files:
            out_name = Path(out_path).stem.lower()
            matched_stem = None
            for pattern, canonical in self._STEM_MAP.items():
                if pattern in out_name:
                    # Guard against "non-vocal" matching "vocal"
                    if pattern == "vocal" and "non" in out_name:
                        continue
                    matched_stem = canonical
                    break
            if matched_stem is None:
                log.debug("Skipping unrecognised RoFormer output: %s", out_path)
                continue
            if matched_stem in result:
                continue  # first match wins

            audio, sr = librosa.load(out_path, sr=None, mono=True)
            audio = audio.astype(np.float32)
            if sr != target_sr and len(audio) > 0:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            result[matched_stem] = audio

        # Clean up temporary WAV files
        for out_path in output_files:
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                pass

        log.info("RoFormer separation complete: %d stems (%s) resampled to %d Hz",
                 len(result), ", ".join(result.keys()), target_sr)
        return result

    def _cleanup_output_dir(self) -> None:
        """Remove the temporary output directory (called by atexit and unload)."""
        if hasattr(self, '_output_dir') and self._output_dir is not None:
            import shutil
            shutil.rmtree(self._output_dir, ignore_errors=True)
            self._output_dir = None

    def unload(self) -> None:
        if self._separator is not None:
            del self._separator
            self._separator = None
            log.info("RoFormer separator unloaded")
        self._cleanup_output_dir()


# ─────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────

class StemSeparator:
    """Separates audio into stems.

    Dispatches to demucs or BS-RoFormer based on ``settings.stem_backend``.
    Falls back to demucs if roformer is requested but not installed.
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._demucs: _DemucsBackend | None = None
        self._roformer: _RoFormerBackend | None = None

    def is_available(self) -> bool:
        return is_available()

    def _get_backend(self) -> _DemucsBackend | _RoFormerBackend:
        """Return the active backend, with fallback logic."""
        backend = settings.stem_backend

        if backend == "roformer":
            if _is_roformer_available():
                if self._roformer is None:
                    self._roformer = _RoFormerBackend(device=self._device)
                return self._roformer
            else:
                log.warning(
                    "stem_backend='roformer' requested but audio-separator not installed — "
                    "falling back to demucs"
                )
                # Fall through to demucs

        # Default: demucs
        if self._demucs is None:
            self._demucs = _DemucsBackend(
                model_name=self._model_name,
                device=self._device,
            )
        return self._demucs

    def separate(self, audio_path: str, target_sr: int = 44100) -> dict[str, np.ndarray]:
        """Separate audio file into stems.

        Args:
            audio_path: Path to audio file.
            target_sr: Resample stems to this sample rate (must match analyzer SR).

        Returns:
            Dict mapping stem name to mono float32 numpy array at target_sr.
        """
        return self._get_backend().separate(audio_path, target_sr)

    def unload(self) -> None:
        """Free separator model(s) from memory."""
        if self._demucs is not None:
            self._demucs.unload()
        if self._roformer is not None:
            self._roformer.unload()
