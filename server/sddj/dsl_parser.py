"""Server-side DSL parser for prompt schedules.

Mirrors the Lua parser but with proper error handling, validation,
and source-location tracking.  Used for server-side validation endpoints,
preview generation, and testing.

The parser follows the grammar defined in docs/PROMPT_SCHEDULE_DSL.md.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .prompt_schedule import (
    PromptKeyframe,
    PromptSchedule,
    ValidationError,
    ValidationResult,
    _VALID_TRANSITION_NAMES,
)

log = logging.getLogger("sddj.dsl_parser")


# ─── Token Patterns ─────────────────────────────────────────

_RE_BLANK = re.compile(r"^\s*$")
_RE_COMMENT = re.compile(r"^\s*#")
_RE_AUTO = re.compile(r"^\s*\{auto\}\s*$", re.IGNORECASE)
_RE_FILE_REF = re.compile(r"^\s*file:\s*(.+)$", re.IGNORECASE)

# Time markers: [0], [50%], [5s], [5.5s]
_RE_TIME_MARKER = re.compile(
    r"^\s*\[\s*"
    r"(?:"
    r"(\d+(?:\.\d+)?)\s*%"      # group 1: percentage
    r"|(\d+(?:\.\d+)?)\s*s"     # group 2: seconds
    r"|(\d+)"                    # group 3: absolute frame
    r")"
    r"\s*\]\s*$"
)

# Directives
_RE_TRANSITION = re.compile(r"^\s*transition:\s*(\S+)\s*$", re.IGNORECASE)
_RE_BLEND = re.compile(r"^\s*blend:\s*(\d+)\s*$", re.IGNORECASE)
_RE_WEIGHT = re.compile(
    r"^\s*weight:\s*(\d+(?:\.\d+)?)\s*(?:->\s*(\d+(?:\.\d+)?))?\s*$",
    re.IGNORECASE,
)
_RE_DENOISE = re.compile(r"^\s*denoise:\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_RE_CFG = re.compile(r"^\s*cfg:\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_RE_STEPS = re.compile(r"^\s*steps:\s*(\d+)\s*$", re.IGNORECASE)
_RE_NEGATIVE = re.compile(r"^\s*--\s*(.*)$")


@dataclass
class _KeyframeBuilder:
    """Accumulates data for a keyframe being parsed."""
    line: int
    frame: int
    prompt_lines: list[str] = field(default_factory=list)
    negative_lines: list[str] = field(default_factory=list)
    transition: str = "hard_cut"
    transition_frames: int = 0
    weight: float = 1.0
    weight_end: float | None = None
    denoise_strength: float | None = None
    cfg_scale: float | None = None
    steps: int | None = None

    def build(self, default_prompt: str) -> PromptKeyframe:
        prompt = ", ".join(self.prompt_lines).strip()
        negative = ", ".join(self.negative_lines).strip()
        return PromptKeyframe(
            frame=self.frame,
            prompt=prompt or "",
            negative_prompt=negative,
            weight=self.weight,
            weight_end=self.weight_end,
            transition=self.transition,
            transition_frames=self.transition_frames,
            denoise_strength=self.denoise_strength,
            cfg_scale=self.cfg_scale,
            steps=self.steps,
        )


@dataclass
class ParseResult:
    """Result of DSL parsing."""
    schedule: PromptSchedule | None
    validation: ValidationResult
    has_auto: bool = False
    file_ref: str | None = None


def parse(
    dsl_text: str,
    total_frames: int,
    fps: float = 24.0,
    default_prompt: str = "",
    *,
    base_dir: Path | None = None,
) -> ParseResult:
    """Parse DSL text into a PromptSchedule.

    Args:
        dsl_text: The raw DSL text.
        total_frames: Total frames in the animation.
        fps: Frames per second (for time conversion).
        default_prompt: Fallback prompt for empty keyframes.
        base_dir: Base directory for resolving ``file:`` references (sandboxed).

    Returns:
        ParseResult with the schedule and validation info.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    has_auto = False
    file_ref: str | None = None

    if not dsl_text or not dsl_text.strip():
        return ParseResult(
            schedule=None,
            validation=ValidationResult(valid=True, errors=[], warnings=[
                ValidationError(None, None, "W002", "Empty schedule", "warning"),
            ]),
        )

    # Handle file reference
    lines = dsl_text.strip().splitlines()
    if len(lines) == 1:
        m = _RE_FILE_REF.match(lines[0])
        if m:
            file_path = m.group(1).strip()
            file_ref = file_path
            resolved = _resolve_file_ref(file_path, base_dir, errors)
            if resolved is None:
                return ParseResult(
                    schedule=None,
                    validation=ValidationResult(
                        valid=False, errors=errors, warnings=warnings,
                    ),
                    file_ref=file_ref,
                )
            lines = resolved.splitlines()

    # Parse lines
    builders: list[_KeyframeBuilder] = []
    current: _KeyframeBuilder | None = None

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")

        # Blank
        if _RE_BLANK.match(line):
            continue

        # Comment
        if _RE_COMMENT.match(line):
            continue

        # Auto directive
        if _RE_AUTO.match(line):
            has_auto = True
            continue

        # File reference (only valid as first non-trivial content)
        if _RE_FILE_REF.match(line):
            if builders or current:
                errors.append(ValidationError(
                    line_num, None, "E012",
                    "file: directive must be the only content in the DSL",
                ))
            continue

        # Time marker
        m = _RE_TIME_MARKER.match(line)
        if m:
            # Finalize previous keyframe
            if current:
                builders.append(current)

            frame = _resolve_time(m, total_frames, fps, line_num, errors, warnings)
            current = _KeyframeBuilder(line=line_num, frame=frame)
            continue

        # Directives (only valid inside a keyframe)
        if current is None:
            # Prompt text before any time marker — treat as error
            stripped = line.strip()
            if stripped and not _RE_COMMENT.match(line):
                errors.append(ValidationError(
                    line_num, None, "E012",
                    f"Content before first time marker: {stripped[:50]}",
                ))
            continue

        # Transition directive
        m = _RE_TRANSITION.match(line)
        if m:
            tr = m.group(1).lower()
            if tr not in _VALID_TRANSITION_NAMES:
                errors.append(ValidationError(
                    line_num, None, "E005",
                    f"Invalid transition type: {tr!r}",
                ))
            else:
                current.transition = tr
            continue

        # Blend frames
        m = _RE_BLEND.match(line)
        if m:
            tf = int(m.group(1))
            if tf > 120:
                errors.append(ValidationError(
                    line_num, None, "E004",
                    f"Transition frames {tf} exceeds maximum (120)",
                ))
            else:
                current.transition_frames = tf
            continue

        # Weight
        m = _RE_WEIGHT.match(line)
        if m:
            w = float(m.group(1))
            w_end = float(m.group(2)) if m.group(2) else None
            if w < 0.1 or w > 5.0:
                errors.append(ValidationError(
                    line_num, None, "E006",
                    f"Weight {w} out of range [0.1, 5.0]",
                ))
            elif w > 2.0:
                warnings.append(ValidationError(
                    line_num, None, "W004",
                    f"Weight {w} > 2.0 may cause artifacts",
                    "warning",
                ))
            else:
                current.weight = w
                current.weight_end = w_end
            if w_end is not None and (w_end < 0.1 or w_end > 5.0):
                errors.append(ValidationError(
                    line_num, None, "E006",
                    f"Weight end {w_end} out of range [0.1, 5.0]",
                ))
            continue

        # Denoise
        m = _RE_DENOISE.match(line)
        if m:
            d = float(m.group(1))
            if d < 0.0 or d > 1.0:
                errors.append(ValidationError(
                    line_num, None, "E007",
                    f"Denoise {d} out of range [0.0, 1.0]",
                ))
            else:
                current.denoise_strength = d
            continue

        # CFG
        m = _RE_CFG.match(line)
        if m:
            c = float(m.group(1))
            if c < 1.0 or c > 30.0:
                errors.append(ValidationError(
                    line_num, None, "E008",
                    f"CFG {c} out of range [1.0, 30.0]",
                ))
            else:
                current.cfg_scale = c
            continue

        # Steps
        m = _RE_STEPS.match(line)
        if m:
            s = int(m.group(1))
            if s < 1 or s > 150:
                errors.append(ValidationError(
                    line_num, None, "E009",
                    f"Steps {s} out of range [1, 150]",
                ))
            else:
                current.steps = s
            continue

        # Negative prompt
        m = _RE_NEGATIVE.match(line)
        if m:
            neg_text = m.group(1).strip()
            if neg_text:
                current.negative_lines.append(neg_text)
            continue

        # Prompt text (anything else)
        stripped = line.strip()
        if stripped:
            current.prompt_lines.append(stripped)

    # Finalize last keyframe
    if current:
        builders.append(current)

    if not builders:
        warnings.append(ValidationError(
            None, None, "W002", "No keyframes defined", "warning",
        ))
        return ParseResult(
            schedule=None,
            validation=ValidationResult(valid=True, errors=errors, warnings=warnings),
            has_auto=has_auto,
            file_ref=file_ref,
        )

    # Build keyframes
    keyframes: list[PromptKeyframe] = []
    for b in builders:
        keyframes.append(b.build(default_prompt))

    # Validate chronological order
    for i in range(1, len(keyframes)):
        if keyframes[i].frame <= keyframes[i - 1].frame:
            errors.append(ValidationError(
                builders[i].line, None, "E003",
                f"Keyframe at frame {keyframes[i].frame} is not after "
                f"previous frame {keyframes[i - 1].frame}",
            ))

    # Check first keyframe
    if keyframes and keyframes[0].frame != 0:
        warnings.append(ValidationError(
            builders[0].line, None, "W001",
            f"First keyframe at frame {keyframes[0].frame}, not frame 0",
            "warning",
        ))

    # Validate transition windows
    for i, kf in enumerate(keyframes):
        if kf.transition_frames > 0 and i > 0:
            gap = kf.frame - keyframes[i - 1].frame
            if kf.transition_frames > gap:
                errors.append(ValidationError(
                    builders[i].line, None, "E004",
                    f"Transition frames ({kf.transition_frames}) exceeds "
                    f"gap to previous keyframe ({gap} frames)",
                ))

    schedule = PromptSchedule([], default_prompt, keyframes=keyframes)

    return ParseResult(
        schedule=schedule,
        validation=ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        ),
        has_auto=has_auto,
        file_ref=file_ref,
    )


def _resolve_time(
    match: re.Match,
    total_frames: int,
    fps: float,
    line_num: int,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> int:
    """Resolve a time marker match to an absolute frame index."""
    if match.group(1) is not None:
        # Percentage
        pct = float(match.group(1))
        if pct < 0 or pct > 100:
            errors.append(ValidationError(
                line_num, None, "E001",
                f"Percentage {pct}% out of range [0, 100]",
            ))
            pct = max(0, min(100, pct))
        return min(total_frames - 1, max(0, math.floor(pct / 100.0 * total_frames)))

    elif match.group(2) is not None:
        # Seconds
        secs = float(match.group(2))
        if secs < 0:
            errors.append(ValidationError(
                line_num, None, "E001",
                f"Time {secs}s is negative",
            ))
            secs = 0
        frame = min(total_frames - 1, max(0, math.floor(secs * fps)))
        return frame

    else:
        # Absolute frame
        frame = int(match.group(3))
        if frame < 0:
            errors.append(ValidationError(
                line_num, None, "E001",
                f"Frame {frame} is negative",
            ))
            frame = 0
        if frame >= total_frames:
            errors.append(ValidationError(
                line_num, None, "E001",
                f"Frame {frame} exceeds total frames {total_frames}",
            ))
            frame = min(total_frames - 1, frame)
        return frame


def _resolve_file_ref(
    file_path: str,
    base_dir: Path | None,
    errors: list[ValidationError],
) -> str | None:
    """Resolve a file: reference. Returns file contents or None on error."""
    if base_dir is None:
        errors.append(ValidationError(
            1, None, "E010",
            "file: references require a base directory context",
        ))
        return None

    # Security: reject path traversal
    if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
        errors.append(ValidationError(
            1, None, "E010",
            f"Path traversal rejected: {file_path!r}",
        ))
        return None

    # Also reject absolute Windows paths
    if len(file_path) >= 2 and file_path[1] == ":":
        errors.append(ValidationError(
            1, None, "E010",
            f"Absolute path rejected: {file_path!r}",
        ))
        return None

    resolved = (base_dir / file_path).resolve()

    # Verify resolved path is within base_dir
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        errors.append(ValidationError(
            1, None, "E010",
            f"Path escapes sandbox: {file_path!r}",
        ))
        return None

    if not resolved.is_file():
        errors.append(ValidationError(
            1, None, "E011",
            f"File not found: {file_path!r}",
        ))
        return None

    try:
        return resolved.read_text(encoding="utf-8")
    except Exception as e:
        errors.append(ValidationError(
            1, None, "E011",
            f"Failed to read file: {e}",
        ))
        return None
