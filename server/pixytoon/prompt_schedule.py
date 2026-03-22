"""Prompt schedule — per-segment prompt resolution for audio-reactive generation."""

from __future__ import annotations

from dataclasses import dataclass


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
