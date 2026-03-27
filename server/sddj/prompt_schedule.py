"""Prompt schedule — unified prompt resolution for all generation modes.

Supports both legacy time-range segments (audio-reactive) and keyframe-based
scheduling (animation, generation).  Keyframes use frame indices with
configurable transition types.

Transition modes use prompt embedding interpolation (SLERP/LERP) across a
transition window shaped by an easing curve — eliminating the previous
even/odd frame alternation which caused visual flicker.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .audio_analyzer import AudioAnalysis
    from .prompt_generator import PromptGenerator

log = logging.getLogger("sddj.prompt_schedule")


# ─────────────────────────────────────────────────────────────
# Transition Types & Easing
# ─────────────────────────────────────────────────────────────


class TransitionType(str, Enum):
    HARD_CUT = "hard_cut"
    BLEND = "blend"
    LINEAR_BLEND = "linear_blend"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    CUBIC = "cubic"
    SLERP = "slerp"

    @classmethod
    def from_str(cls, s: str) -> "TransitionType":
        try:
            return cls(s)
        except ValueError:
            log.warning("Invalid transition %r, using hard_cut", s)
            return cls.HARD_CUT


_BLENDING_TRANSITIONS = frozenset({
    TransitionType.BLEND,
    TransitionType.LINEAR_BLEND,
    TransitionType.EASE_IN,
    TransitionType.EASE_OUT,
    TransitionType.EASE_IN_OUT,
    TransitionType.CUBIC,
    TransitionType.SLERP,
})

_VALID_TRANSITION_NAMES = frozenset(t.value for t in TransitionType)


def _ease_linear(t: float) -> float:
    return t


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _ease_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


_EASING_FUNCTIONS: dict[TransitionType, callable] = {
    TransitionType.BLEND: _ease_linear,
    TransitionType.LINEAR_BLEND: _ease_linear,
    TransitionType.SLERP: _ease_linear,
    TransitionType.EASE_IN: _ease_in,
    TransitionType.EASE_OUT: _ease_out,
    TransitionType.EASE_IN_OUT: _ease_in_out,
    TransitionType.CUBIC: _ease_cubic,
}


def get_easing(transition: TransitionType) -> callable:
    """Return the easing function for a transition type."""
    return _EASING_FUNCTIONS.get(transition, _ease_linear)


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────


@dataclass
class PromptSegment:
    """A prompt active during a time range (legacy)."""

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
    weight_end: float | None = None          # animated weight: weight->weight_end
    transition: str = "hard_cut"
    transition_frames: int = 0
    # Per-keyframe parameter overrides (None = inherit base)
    denoise_strength: float | None = None
    cfg_scale: float | None = None
    steps: int | None = None


@dataclass
class PromptBlendInfo:
    """Result of prompt resolution for a single frame — provides all info
    needed for embedding-space interpolation.

    When ``blend_weight`` is 0.0, only ``prompt_a`` is active.
    When ``blend_weight`` is 1.0, only ``prompt_b`` is active.
    Between 0 and 1, the engine SLERP/LERPs embeddings of both prompts.
    """
    prompt_a: str
    prompt_b: str
    blend_weight: float         # 0.0 = fully A, 1.0 = fully B
    negative_prompt: str = ""
    weight: float = 1.0         # effective prompt embedding weight
    # Per-frame parameter overrides (None = use base)
    denoise_strength: float | None = None
    cfg_scale: float | None = None
    steps: int | None = None

    @property
    def is_blending(self) -> bool:
        return 0.0 < self.blend_weight < 1.0

    @property
    def effective_prompt(self) -> str:
        """Return the dominant prompt (for logging/display)."""
        return self.prompt_b if self.blend_weight >= 0.5 else self.prompt_a


@dataclass
class ValidationError:
    """A structured parse/validation error."""
    line: int | None
    column: int | None
    code: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationResult:
    """Result of schedule validation."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def all_issues(self) -> list[ValidationError]:
        return self.errors + self.warnings


# ─────────────────────────────────────────────────────────────
# Core Schedule Class
# ─────────────────────────────────────────────────────────────


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
        """Return the active prompt for *frame_idx* (simple string, backward compat).

        For blending support, use get_blend_info_for_frame() instead.
        """
        info = self.get_blend_info_for_frame(frame_idx)
        return info.effective_prompt

    def get_blend_info_for_frame(self, frame_idx: int) -> PromptBlendInfo:
        """Resolve prompt blending info for *frame_idx*.

        Returns a PromptBlendInfo with:
        - prompt_a, prompt_b: the two prompts involved
        - blend_weight: 0.0–1.0 shaped by the easing function
        - weight: effective prompt weight
        - negative_prompt: active negative
        - denoise_strength, cfg_scale, steps: per-keyframe overrides (or None)
        """
        if not self.keyframes:
            return PromptBlendInfo(
                prompt_a=self.default_prompt,
                prompt_b=self.default_prompt,
                blend_weight=0.0,
                weight=1.0,
            )

        # Find active keyframe (last kf where kf.frame <= frame_idx)
        active_idx = -1
        for i, kf in enumerate(self.keyframes):
            if kf.frame <= frame_idx:
                active_idx = i
            else:
                break

        if active_idx < 0:
            return PromptBlendInfo(
                prompt_a=self.default_prompt,
                prompt_b=self.default_prompt,
                blend_weight=0.0,
                weight=1.0,
            )

        active_kf = self.keyframes[active_idx]
        tr_type = TransitionType.from_str(active_kf.transition)
        prompt_active = active_kf.prompt or self.default_prompt
        neg_active = active_kf.negative_prompt or ""

        # Compute effective weight (static or animated)
        weight = self._compute_weight(active_kf, active_idx, frame_idx)

        # Compute per-keyframe parameter overrides (interpolated)
        denoise, cfg, steps = self._interpolate_params(active_idx, frame_idx)

        # Hard cut or no transition window: no blending
        if (tr_type not in _BLENDING_TRANSITIONS
                or active_kf.transition_frames <= 0
                or active_idx == 0):
            return PromptBlendInfo(
                prompt_a=prompt_active,
                prompt_b=prompt_active,
                blend_weight=0.0,
                negative_prompt=neg_active,
                weight=weight,
                denoise_strength=denoise,
                cfg_scale=cfg,
                steps=steps,
            )

        # Check if within transition window
        frames_since = frame_idx - active_kf.frame
        if frames_since >= active_kf.transition_frames:
            # Past transition window — fully on current prompt
            return PromptBlendInfo(
                prompt_a=prompt_active,
                prompt_b=prompt_active,
                blend_weight=0.0,
                negative_prompt=neg_active,
                weight=weight,
                denoise_strength=denoise,
                cfg_scale=cfg,
                steps=steps,
            )

        # Within transition window: compute blend weight via easing
        prev_kf = self.keyframes[active_idx - 1]
        prompt_outgoing = prev_kf.prompt or self.default_prompt
        neg_outgoing = prev_kf.negative_prompt or ""

        t = frames_since / active_kf.transition_frames  # [0, 1)
        easing = get_easing(tr_type)
        blend_weight = max(0.0, min(1.0, easing(t)))

        # Negative: use incoming negative prompt during blend
        effective_neg = neg_active if blend_weight >= 0.5 else neg_outgoing

        return PromptBlendInfo(
            prompt_a=prompt_outgoing,
            prompt_b=prompt_active,
            blend_weight=blend_weight,
            negative_prompt=effective_neg,
            weight=weight,
            denoise_strength=denoise,
            cfg_scale=cfg,
            steps=steps,
        )

    def _compute_weight(
        self, kf: PromptKeyframe, kf_idx: int, frame_idx: int,
    ) -> float:
        """Compute effective weight, supporting animated weight: start->end."""
        if kf.weight_end is None:
            return kf.weight

        # Animated weight: interpolate over this keyframe's active region
        next_frame: int
        if kf_idx + 1 < len(self.keyframes):
            next_frame = self.keyframes[kf_idx + 1].frame
        else:
            next_frame = kf.frame + 100  # default region length for last kf

        region = next_frame - kf.frame
        if region <= 0:
            return kf.weight
        t = min(1.0, (frame_idx - kf.frame) / region)
        return kf.weight + (kf.weight_end - kf.weight) * t

    def _interpolate_params(
        self, active_idx: int, frame_idx: int,
    ) -> tuple[float | None, float | None, int | None]:
        """Interpolate per-keyframe parameter overrides between keyframes.

        Uses linear interpolation. Each parameter is independent.
        Returns (denoise, cfg, steps) — None values mean "use base".
        """
        active_kf = self.keyframes[active_idx]

        # Find next keyframe for interpolation target
        next_kf: PromptKeyframe | None = None
        if active_idx + 1 < len(self.keyframes):
            next_kf = self.keyframes[active_idx + 1]

        def _interp(cur_val: float | None, next_val: float | None) -> float | None:
            if cur_val is None:
                return None
            if next_val is None or next_kf is None:
                return cur_val
            region = next_kf.frame - active_kf.frame
            if region <= 0:
                return cur_val
            t = min(1.0, (frame_idx - active_kf.frame) / region)
            return cur_val + (next_val - cur_val) * t

        denoise = _interp(active_kf.denoise_strength,
                          next_kf.denoise_strength if next_kf else None)
        cfg = _interp(active_kf.cfg_scale,
                      next_kf.cfg_scale if next_kf else None)
        # Steps: no interpolation (integer), use active keyframe value
        steps = active_kf.steps

        return denoise, cfg, steps

    def get_negative_for_frame(self, frame_idx: int) -> str | None:
        """Return the per-keyframe negative prompt, or *None* for default."""
        info = self.get_blend_info_for_frame(frame_idx)
        return info.negative_prompt or None

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

    # ── Validation ──────────────────────────────────────────────

    def validate(self, total_frames: int | None = None) -> ValidationResult:
        """Validate the schedule structure.

        Returns a ValidationResult with all errors and warnings.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        if not self.keyframes and not self.segments:
            warnings.append(ValidationError(
                line=None, column=None, code="W002",
                message="Empty schedule (no keyframes defined)",
                severity="warning",
            ))
            return ValidationResult(valid=True, errors=errors, warnings=warnings)

        # Validate keyframes
        if self.keyframes:
            # Check first keyframe starts at 0
            if self.keyframes[0].frame != 0:
                warnings.append(ValidationError(
                    line=None, column=None, code="W001",
                    message=f"First keyframe at frame {self.keyframes[0].frame}, "
                            "not frame 0 — implicit frame-0 keyframe will be inserted",
                    severity="warning",
                ))

            prev_frame = -1
            for i, kf in enumerate(self.keyframes):
                # Chronological order
                if kf.frame <= prev_frame:
                    errors.append(ValidationError(
                        line=None, column=None, code="E003",
                        message=f"Keyframe {i} at frame {kf.frame} is not after "
                                f"previous frame {prev_frame}",
                    ))

                # Range checks
                if total_frames is not None and kf.frame >= total_frames:
                    errors.append(ValidationError(
                        line=None, column=None, code="E001",
                        message=f"Keyframe {i} at frame {kf.frame} exceeds "
                                f"total frames {total_frames}",
                    ))

                # Transition validation
                if kf.transition not in _VALID_TRANSITION_NAMES:
                    errors.append(ValidationError(
                        line=None, column=None, code="E005",
                        message=f"Invalid transition type: {kf.transition!r}",
                    ))

                # Transition window vs gap
                if kf.transition_frames > 0 and i > 0:
                    gap = kf.frame - self.keyframes[i - 1].frame
                    if kf.transition_frames > gap:
                        errors.append(ValidationError(
                            line=None, column=None, code="E004",
                            message=f"Keyframe {i}: transition_frames "
                                    f"({kf.transition_frames}) exceeds gap to "
                                    f"previous keyframe ({gap} frames)",
                        ))

                # Weight
                if kf.weight < 0.1 or kf.weight > 5.0:
                    errors.append(ValidationError(
                        line=None, column=None, code="E006",
                        message=f"Keyframe {i}: weight {kf.weight} out of range "
                                "[0.1, 5.0]",
                    ))
                elif kf.weight > 2.0:
                    warnings.append(ValidationError(
                        line=None, column=None, code="W004",
                        message=f"Keyframe {i}: weight {kf.weight} > 2.0 may "
                                "cause artifacts",
                        severity="warning",
                    ))

                # Denoise
                if kf.denoise_strength is not None:
                    if kf.denoise_strength < 0.0 or kf.denoise_strength > 1.0:
                        errors.append(ValidationError(
                            line=None, column=None, code="E007",
                            message=f"Keyframe {i}: denoise {kf.denoise_strength} "
                                    "out of range [0.0, 1.0]",
                        ))

                # CFG
                if kf.cfg_scale is not None:
                    if kf.cfg_scale < 1.0 or kf.cfg_scale > 30.0:
                        errors.append(ValidationError(
                            line=None, column=None, code="E008",
                            message=f"Keyframe {i}: cfg {kf.cfg_scale} out of "
                                    "range [1.0, 30.0]",
                        ))

                # Steps
                if kf.steps is not None:
                    if kf.steps < 1 or kf.steps > 150:
                        errors.append(ValidationError(
                            line=None, column=None, code="E009",
                            message=f"Keyframe {i}: steps {kf.steps} out of "
                                    "range [1, 150]",
                        ))

                # Short transition warning
                if kf.transition_frames in (1, 2):
                    warnings.append(ValidationError(
                        line=None, column=None, code="W003",
                        message=f"Keyframe {i}: transition_frames "
                                f"{kf.transition_frames} is very short — may be "
                                "visually imperceptible",
                        severity="warning",
                    ))

                prev_frame = kf.frame

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

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
                    **({"weight_end": kf.weight_end} if kf.weight_end is not None else {}),
                    "transition": kf.transition,
                    "transition_frames": kf.transition_frames,
                    **({"denoise_strength": kf.denoise_strength}
                       if kf.denoise_strength is not None else {}),
                    **({"cfg_scale": kf.cfg_scale}
                       if kf.cfg_scale is not None else {}),
                    **({"steps": kf.steps}
                       if kf.steps is not None else {}),
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
    def _parse_keyframe_dict(d: dict) -> PromptKeyframe | None:
        """Parse a single keyframe dict. Returns None on failure."""
        try:
            frame = int(d.get("frame", -1))
            prompt = str(d.get("prompt", ""))
            if frame < 0:
                log.warning("Skipping keyframe with invalid frame: %s", d)
                return None
            transition = str(d.get("transition", "hard_cut"))
            if transition not in _VALID_TRANSITION_NAMES:
                log.warning("Invalid transition %r, using hard_cut", transition)
                transition = "hard_cut"

            weight = float(d.get("weight", 1.0))
            weight_end_raw = d.get("weight_end")
            weight_end = float(weight_end_raw) if weight_end_raw is not None else None

            denoise_raw = d.get("denoise_strength")
            denoise = float(denoise_raw) if denoise_raw is not None else None

            cfg_raw = d.get("cfg_scale")
            cfg = float(cfg_raw) if cfg_raw is not None else None

            steps_raw = d.get("steps")
            steps = int(steps_raw) if steps_raw is not None else None

            return PromptKeyframe(
                frame=frame,
                prompt=prompt,
                negative_prompt=str(d.get("negative_prompt", "")),
                weight=weight,
                weight_end=weight_end,
                transition=transition,
                transition_frames=max(0, int(d.get("transition_frames", 0))),
                denoise_strength=denoise,
                cfg_scale=cfg,
                steps=steps,
            )
        except (TypeError, ValueError) as exc:
            log.warning("Skipping malformed keyframe dict: %s (%s)", d, exc)
            return None

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
            kf = PromptSchedule._parse_keyframe_dict(d)
            if kf is not None:
                keyframes.append(kf)
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
            kf = PromptSchedule._parse_keyframe_dict(d)
            if kf is not None:
                keyframes.append(kf)
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
            weight_end=kf.weight_end,
            transition=kf.transition,
            transition_frames=kf.transition_frames,
            denoise_strength=kf.denoise_strength,
            cfg_scale=kf.cfg_scale,
            steps=kf.steps,
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
) -> dict:
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
            "transition": "ease_in_out",
            "transition_frames": blend_frames,
        })

    log.info(
        "Auto-generated %d prompt keyframes via audio analysis (randomness=%d, duration=%.1fs, bpm=%.0f)",
        len(keyframes), randomness, analysis.duration, bpm,
    )
    return {"keyframes": keyframes, "default_prompt": base_prompt}
