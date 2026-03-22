"""Audio stem separation via demucs (optional dependency).

Separates audio into 4 stems (drums, bass, vocals, other) on CPU
to avoid competing with the GPU inference pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("pixytoon.audio")

# Stem names produced by htdemucs
STEM_NAMES = ("drums", "bass", "vocals", "other")

_demucs_available: bool | None = None


def is_available() -> bool:
    """Check if demucs is installed and importable."""
    global _demucs_available
    if _demucs_available is None:
        try:
            import demucs.api  # noqa: F401
            _demucs_available = True
        except ImportError:
            _demucs_available = False
    return _demucs_available


class StemSeparator:
    """Separates audio into stems using demucs on CPU."""

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._separator = None

    def is_available(self) -> bool:
        return is_available()

    def _ensure_loaded(self) -> None:
        """Lazy-load the demucs separator."""
        if self._separator is not None:
            return
        if not is_available():
            raise RuntimeError(
                "Stem separation requires demucs. "
                "Install with: pip install demucs>=4.0"
            )
        import demucs.api
        log.info("Loading stem separator: model=%s, device=%s", self._model_name, self._device)
        self._separator = demucs.api.Separator(
            model=self._model_name,
            device=self._device,
        )
        log.info("Stem separator loaded")

    def separate(self, audio_path: str, target_sr: int = 22050) -> dict[str, np.ndarray]:
        """Separate audio file into stems.

        Args:
            audio_path: Path to audio file.
            target_sr: Resample stems to this sample rate (must match analyzer SR).

        Returns:
            Dict mapping stem name to mono float32 numpy array at target_sr.
        """
        self._ensure_loaded()

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Separating stems: %s", path.name)

        import torch

        # Run separation
        _, stems_tensor = self._separator.separate_audio_file(str(path))

        # Model sample rate (htdemucs = 44100)
        model_sr = self._separator.samplerate

        # Convert to numpy arrays (mono) and resample to target_sr
        result: dict[str, np.ndarray] = {}
        for stem_name in STEM_NAMES:
            if stem_name in stems_tensor:
                tensor = stems_tensor[stem_name]
                # tensor shape: [channels, samples] → average to mono
                if tensor.dim() == 2:
                    mono = tensor.mean(dim=0)
                else:
                    mono = tensor
                audio = mono.cpu().numpy().astype(np.float32)

                # Resample to match analyzer sample rate
                if model_sr != target_sr and len(audio) > 0:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=model_sr, target_sr=target_sr)

                result[stem_name] = audio

        log.info("Separation complete: %d stems (%s) resampled to %d Hz",
                 len(result), ", ".join(result.keys()), target_sr)
        return result

    def unload(self) -> None:
        """Free separator model from memory."""
        if self._separator is not None:
            del self._separator
            self._separator = None
            log.info("Stem separator unloaded")
