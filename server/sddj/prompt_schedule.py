"""Prompt schedule — unified prompt resolution for all generation modes.

Supports both legacy time-range segments (audio-reactive) and keyframe-based
scheduling (animation, generation). Keyframes use frame indices with hard_cut
or blend transitions. Blend mode alternates prompts frame-by-frame within a
transition window, producing visual crossfade through the img2img chain.
"""

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


@dataclass
class PromptKeyframe:
    """A prompt keyframe at a specific frame index."""

    frame: int
    prompt: str
    negative_prompt: str = ""
    weight: float = 1.0
    transition: str = "hard_cut"  # "hard_cut" | "blend"
    transition_frames: int = 0    # blend window length (blend mode only)


_VALID_TRANSITIONS = frozenset({"hard_cut", "blend"})


class PromptSchedule:
    """Resolves the active prompt for a given timestamp or frame index."""

    def __init__(
        self,
        segments: list[PromptSegment],
        default_prompt: str,
        *,
        keyframes: list[PromptKeyframe] | None = None,
    ) -> None:
        self.segments = sorted(segments, key=lambda s: s.start_second)
        self.default_prompt = default_prompt
        # Keyframes: sorted by frame, deduplicated (last wins per frame)
        if keyframes:
            seen: dict[int, PromptKeyframe] = {}
            for kf in keyframes:
                seen[kf.frame] = kf
            self.keyframes = sorted(seen.values(), key=lambda k: k.frame)
        else:
            self.keyframes = []

    # ── Legacy time-based resolution (backward compat) ──────────

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

    # ── Frame-based keyframe resolution ─────────────────────────

    def get_prompt_for_frame(self, frame_idx: int) -> str:
        """Return the active prompt for *frame_idx*.

        Transition logic:
        - ``hard_cut``: returns the prompt of the keyframe with highest
          frame <= *frame_idx*.
        - ``blend``: within *transition_frames* after a keyframe switch,
          alternates between outgoing and incoming prompt on even/odd
          frames. The img2img chain naturally produces a visual crossfade.
        """
        if not self.keyframes:
            return self.default_prompt

        # Find active keyframe (last kf where kf.frame <= frame_idx)
        active_idx = -1
        for i, kf in enumerate(self.keyframes):
            if kf.frame <= frame_idx:
                active_idx = i
            else:
                break

        if active_idx < 0:
            return self.default_prompt

        active_kf = self.keyframes[active_idx]

        # Hard cut: simple return
        if active_kf.transition != "blend" or active_kf.transition_frames <= 0:
            return active_kf.prompt or self.default_prompt

        # Blend: check if we're within the transition window
        frames_since = frame_idx - active_kf.frame
        if frames_since >= active_kf.transition_frames or active_idx == 0:
            return active_kf.prompt or self.default_prompt

        # Within blend window: alternate between outgoing and incoming
        prev_kf = self.keyframes[active_idx - 1]
        if frame_idx % 2 == 0:
            return prev_kf.prompt or self.default_prompt
        return active_kf.prompt or self.default_prompt

    def get_negative_for_frame(self, frame_idx: int) -> str | None:
        """Return the per-keyframe negative prompt, or *None* for default."""
        if not self.keyframes:
            return None
        active_kf: PromptKeyframe | None = None
        for kf in self.keyframes:
            if kf.frame <= frame_idx:
                active_kf = kf
            else:
                break
        if active_kf is None or not active_kf.negative_prompt:
            return None
        return active_kf.negative_prompt

    def get_unique_prompts(self) -> set[str]:
        """Return all unique positive prompts (for embedding pre-cache)."""
        prompts = {self.default_prompt} if self.default_prompt else set()
        for kf in self.keyframes:
            if kf.prompt:
                prompts.add(kf.prompt)
        for seg in self.segments:
            if seg.prompt:
                prompts.add(seg.prompt)
        return prompts

    def get_unique_negatives(self) -> set[str]:
        """Return all unique per-keyframe negative prompts."""
        return {kf.negative_prompt for kf in self.keyframes if kf.negative_prompt}

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "default_prompt": self.default_prompt,
            "keyframes": [
                {
                    "frame": kf.frame,
                    "prompt": kf.prompt,
                    "negative_prompt": kf.negative_prompt,
                    "weight": kf.weight,
                    "transition": kf.transition,
                    "transition_frames": kf.transition_frames,
                }
                for kf in self.keyframes
            ],
            "segments": [
                {
                    "start_second": seg.start_second,
                    "end_second": seg.end_second,
                    "prompt": seg.prompt,
                    "weight": seg.weight,
                }
                for seg in self.segments
            ],
        }

    @staticmethod
    def from_dicts(
        raw_segments: list[dict], default_prompt: str,
    ) -> "PromptSchedule | None":
        """Build from a list of segment dicts (legacy client format).

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

    @staticmethod
    def from_keyframe_dicts(
        raw_keyframes: list[dict],
        default_prompt: str,
    ) -> "PromptSchedule | None":
        """Build from a list of keyframe dicts.

        Returns *None* if no valid keyframes.
        """
        keyframes: list[PromptKeyframe] = []
        for d in raw_keyframes:
            try:
                frame = int(d.get("frame", -1))
                prompt = str(d.get("prompt", ""))
                if frame < 0:
                    log.warning("Skipping keyframe with invalid frame: %s", d)
                    continue
                transition = str(d.get("transition", "hard_cut"))
                if transition not in _VALID_TRANSITIONS:
                    log.warning("Invalid transition %r, using hard_cut", transition)
                    transition = "hard_cut"
                kf = PromptKeyframe(
                    frame=frame,
                    prompt=prompt,
                    negative_prompt=str(d.get("negative_prompt", "")),
                    weight=float(d.get("weight", 1.0)),
                    transition=transition,
                    transition_frames=max(0, int(d.get("transition_frames", 0))),
                )
                keyframes.append(kf)
            except (TypeError, ValueError) as exc:
                log.warning("Skipping malformed keyframe dict: %s (%s)", d, exc)
                continue
        if not keyframes:
            return None
        return PromptSchedule([], default_prompt, keyframes=keyframes)

    @staticmethod
    def from_dict(data: dict, default_prompt: str = "") -> "PromptSchedule | None":
        """Deserialize from a full schedule dict (as produced by *to_dict()*)."""
        dp = data.get("default_prompt", default_prompt) or default_prompt
        kf_dicts = data.get("keyframes", [])
        seg_dicts = data.get("segments", [])
        keyframes: list[PromptKeyframe] = []
        segments: list[PromptSegment] = []
        for d in kf_dicts:
            try:
                transition = str(d.get("transition", "hard_cut"))
                if transition not in _VALID_TRANSITIONS:
                    transition = "hard_cut"
                keyframes.append(PromptKeyframe(
                    frame=int(d.get("frame", 0)),
                    prompt=str(d.get("prompt", "")),
                    negative_prompt=str(d.get("negative_prompt", "")),
                    weight=float(d.get("weight", 1.0)),
                    transition=transition,
                    transition_frames=max(0, int(d.get("transition_frames", 0))),
                ))
            except (TypeError, ValueError):
                continue
        for d in seg_dicts:
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
        if not keyframes and not segments:
            return None
        return PromptSchedule(segments, dp, keyframes=keyframes)


# ─────────────────────────────────────────────────────────────
# Auto-fill keyframe prompts via PromptGenerator
# ─────────────────────────────────────────────────────────────


def auto_fill_prompts(
    schedule: PromptSchedule,
    prompt_gen: "PromptGenerator",
    randomness: int = 5,
    locked_fields: dict[str, str] | None = None,
) -> PromptSchedule:
    """Fill empty keyframe prompts using *prompt_gen*.

    Keyframes with empty or ``{auto}`` prompts are replaced with
    generated variations. Returns a new schedule (immutable pattern).
    """
    if not schedule.keyframes:
        return schedule

    locked = dict(locked_fields) if locked_fields else {}
    filled: list[PromptKeyframe] = []
    for kf in schedule.keyframes:
        if kf.prompt and kf.prompt != "{auto}":
            filled.append(kf)
            continue
        try:
            prompt, _, meta = prompt_gen.generate(
                locked=locked, randomness=randomness,
            )
        except Exception:
            log.warning("Auto-fill failed for frame %d, keeping empty", kf.frame)
            prompt = kf.prompt
        filled.append(PromptKeyframe(
            frame=kf.frame,
            prompt=prompt,
            negative_prompt=kf.negative_prompt,
            weight=kf.weight,
            transition=kf.transition,
            transition_frames=kf.transition_frames,
        ))
    log.info("Auto-filled %d/%d keyframe prompts",
             sum(1 for f, o in zip(filled, schedule.keyframes) if f.prompt != o.prompt),
             len(filled))
    return PromptSchedule(
        schedule.segments, schedule.default_prompt, keyframes=filled,
    )


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
    locked_fields: dict[str, str] | None = None,
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
        locked_fields: Explicit locked fields from client (e.g. {"subject": "..."}).

    Returns:
        Dict defining a prompt schedule: {"keyframes": [...], "default_prompt": "..."}
        Empty dict if randomness is 0.
    """
    if randomness <= 0 or analysis.duration <= 0:
        return {}

    n_segments = _randomness_to_segment_count(randomness, analysis.duration)
    if n_segments < 2:
        return {}

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

    # ── Extract subject: explicit lock > heuristic extraction ──
    if locked_fields and locked_fields.get("subject"):
        subject = locked_fields["subject"]
    else:
        subject = base_prompt.strip()
        if "," in subject:
            parts = [p.strip() for p in subject.split(",")]
            for part in parts:
                if len(part) > 10:
                    subject = part
                    break
            else:
                subject = parts[0]

    # ── Generate varied prompts for each keyframe ──
    keyframes: list[dict] = []
    locked = {"subject": subject}
    blend_frames = max(4, int((analysis.fps or 24) * 0.5))  # Smooth half-second crossfade
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        try:
            prompt, _, _ = prompt_gen.generate(
                locked=locked, randomness=randomness,
            )
        except Exception:
            prompt = base_prompt  # Fallback to original
            
        frame_idx = int(start * (analysis.fps or 24.0))
        keyframes.append({
            "frame": frame_idx,
            "prompt": prompt,
            "transition": "blend",
            "transition_frames": blend_frames,
        })

    log.info(
        "Auto-generated %d prompt keyframes via audio analysis (randomness=%d, duration=%.1fs, bpm=%.0f)",
        len(keyframes), randomness, analysis.duration, bpm,
    )
    return {"keyframes": keyframes, "default_prompt": base_prompt}
