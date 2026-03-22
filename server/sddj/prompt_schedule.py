"""Prompt schedule — per-segment prompt resolution for audio-reactive generation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .audio_analyzer import AudioAnalysis
    from .prompt_generator import PromptGenerator

log = logging.getLogger("sddj.prompt_schedule")


@dataclass
class PromptSegment:
    """A prompt active during a time range."""

    start_second: float
    end_second: float
    prompt: str
    weight: float = 1.0


class PromptSchedule:
    """Resolves the active prompt for a given timestamp."""

    def __init__(self, segments: list[PromptSegment], default_prompt: str) -> None:
        # Sort by start_second so overlapping segments resolve to latest-defined
        self.segments = sorted(segments, key=lambda s: s.start_second)
        self.default_prompt = default_prompt

    def get_prompt(self, time_seconds: float) -> str:
        """Return the active prompt for *time_seconds*.

        If multiple segments overlap, the one with the highest weight wins.
        Falls back to *default_prompt* when no segment covers the timestamp.
        """
        best: PromptSegment | None = None
        for seg in self.segments:
            if seg.start_second <= time_seconds < seg.end_second:
                if best is None or seg.weight > best.weight:
                    best = seg
        return best.prompt if best is not None else self.default_prompt

    @staticmethod
    def from_dicts(
        raw_segments: list[dict], default_prompt: str,
    ) -> "PromptSchedule | None":
        """Build from a list of dicts (as sent by the client).

        Returns *None* if *raw_segments* is empty or invalid.
        """
        segments: list[PromptSegment] = []
        for d in raw_segments:
            try:
                seg = PromptSegment(
                    start_second=float(d.get("start_second", 0)),
                    end_second=float(d.get("end_second", 0)),
                    prompt=str(d.get("prompt", "")),
                    weight=float(d.get("weight", 1.0)),
                )
                if seg.end_second > seg.start_second and seg.prompt:
                    segments.append(seg)
            except (TypeError, ValueError):
                continue
        return PromptSchedule(segments, default_prompt) if segments else None


# ─────────────────────────────────────────────────────────────
# Auto-generate prompt segments from audio structure
# ─────────────────────────────────────────────────────────────

def _randomness_to_segment_count(randomness: int, duration: float) -> int:
    """Map randomness level (0-20) and audio duration to a target segment count.

    Longer audio gets proportionally more segments (×1 per extra minute).
    Capped at 12 to keep generation coherent.
    """
    if randomness <= 0:
        return 0
    if randomness <= 5:
        base = 2
    elif randomness <= 10:
        base = 3
    elif randomness <= 15:
        base = max(4, min(5, 1 + randomness // 3))
    else:
        base = max(6, min(8, randomness // 2))
    duration_mult = max(1, math.ceil(duration / 60.0))
    return min(12, base * duration_mult)


def _find_onset_peaks(
    onset: np.ndarray, fps: float, n_peaks: int, min_gap: float,
) -> list[float]:
    """Find the top *n_peaks* onset peaks (as seconds) with at least *min_gap* between them."""
    if onset is None or len(onset) == 0 or n_peaks <= 0:
        return []

    # Average onset into 1-second bins to find prominent sections
    bin_size = max(1, int(fps))
    n_bins = max(1, len(onset) // bin_size)
    binned = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, len(onset))
        binned[i] = onset[start:end].mean()

    # Only consider peaks above 75th percentile
    threshold = float(np.percentile(binned, 75)) if len(binned) > 4 else 0.0
    candidates = []
    for i, val in enumerate(binned):
        if val >= threshold:
            candidates.append((float(val), float(i)))  # (strength, second)

    # Sort by strength descending, greedily pick with min_gap
    candidates.sort(key=lambda x: x[0], reverse=True)
    picked: list[float] = []
    for strength, sec in candidates:
        if all(abs(sec - p) >= min_gap for p in picked):
            picked.append(sec)
            if len(picked) >= n_peaks:
                break

    picked.sort()
    return picked


def _snap_to_beat(seconds: float, beat_interval: float) -> float:
    """Snap a timestamp to the nearest beat grid position."""
    if beat_interval <= 0:
        return seconds
    return round(seconds / beat_interval) * beat_interval


def _fill_gaps(boundaries: list[float], target_count: int, duration: float) -> list[float]:
    """Add uniformly-spaced boundaries in the largest gaps to reach *target_count*."""
    while len(boundaries) < target_count:
        # Find the largest gap
        gaps = []
        for i in range(len(boundaries) - 1):
            gaps.append((boundaries[i + 1] - boundaries[i], i))
        if not gaps:
            break
        gaps.sort(reverse=True)
        gap_size, gap_idx = gaps[0]
        if gap_size < 2.0:
            break  # Don't split gaps smaller than 2 seconds
        mid = (boundaries[gap_idx] + boundaries[gap_idx + 1]) / 2.0
        boundaries.insert(gap_idx + 1, mid)
    return boundaries


def auto_generate_segments(
    analysis: "AudioAnalysis",
    randomness: int,
    base_prompt: str,
    prompt_gen: "PromptGenerator",
) -> list[dict]:
    """Auto-generate prompt segments aligned to audio structure.

    When *randomness* > 0 and no manual segments are provided, this function
    analyzes audio onset peaks and BPM to place segment boundaries at musically
    meaningful points. Each segment receives a unique prompt variation generated
    from the base prompt's subject.

    Args:
        analysis: Result of audio analysis (features, BPM, duration).
        randomness: Diversity level 0-20.
        base_prompt: The user's prompt — subject is extracted and locked.
        prompt_gen: PromptGenerator instance for creating variations.

    Returns:
        List of segment dicts: [{"start_second", "end_second", "prompt"}, ...]
        Empty list if randomness is 0.
    """
    if randomness <= 0 or analysis.duration <= 0:
        return []

    n_segments = _randomness_to_segment_count(randomness, analysis.duration)
    if n_segments < 2:
        return []

    # ── Compute boundaries from onset peaks ──
    bpm = analysis.bpm or 0.0
    beat_interval = 60.0 / max(1.0, bpm) if bpm > 0 else 0.0
    min_gap = max(2.0, beat_interval) if beat_interval > 0 else 2.0

    onset = analysis.features.get("global_onset")
    peaks = _find_onset_peaks(onset, analysis.fps, n_segments - 1, min_gap)

    # Snap peaks to beat grid if BPM is known
    if beat_interval > 0:
        peaks = [_snap_to_beat(p, beat_interval) for p in peaks]

    # Build boundaries: [0.0, ...peaks..., duration]
    boundaries = [0.0] + peaks + [analysis.duration]
    # Deduplicate boundaries that are < 1.0s apart
    deduped = [boundaries[0]]
    for b in boundaries[1:]:
        if b - deduped[-1] >= 1.0:
            deduped.append(b)
    if deduped[-1] < analysis.duration:
        deduped[-1] = analysis.duration
    boundaries = deduped

    # Fill gaps if we don't have enough boundaries
    if len(boundaries) - 1 < n_segments:
        boundaries = _fill_gaps(boundaries, n_segments + 1, analysis.duration)

    # ── Extract subject from base prompt ──
    subject = base_prompt.strip()
    if "," in subject:
        # Take the most significant segment (usually the subject descriptor)
        parts = [p.strip() for p in subject.split(",")]
        # Skip quality tags (short segments like "masterpiece", "best quality")
        for part in parts:
            if len(part) > 10:
                subject = part
                break
        else:
            subject = parts[0]

    # ── Generate varied prompts for each segment ──
    segments: list[dict] = []
    locked = {"subject": subject}
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        try:
            prompt, _, _ = prompt_gen.generate(
                locked=locked, randomness=randomness,
            )
        except Exception:
            prompt = base_prompt  # Fallback to original
        segments.append({
            "start_second": round(start, 2),
            "end_second": round(end, 2),
            "prompt": prompt,
        })

    log.info(
        "Auto-generated %d prompt segments (randomness=%d, duration=%.1fs, bpm=%.0f)",
        len(segments), randomness, analysis.duration, bpm,
    )
    return segments
