"""Tests for prompt_schedule — PromptSchedule + auto_generate_segments (v0.7.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np

from sddj.prompt_schedule import (
    PromptSchedule,
    PromptSegment,
    auto_generate_segments,
    _randomness_to_segment_count,
)


# ─── Helpers ──────────────────────────────────────────────────


@dataclass
class FakeAnalysis:
    """Minimal stand-in for AudioAnalysis."""

    fps: float = 24.0
    duration: float = 30.0
    total_frames: int = 720
    bpm: float = 120.0
    features: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.features.keys())


def _make_onset(duration: float, fps: float, peak_seconds: list[float]) -> np.ndarray:
    """Create a fake onset array with clear peaks at the given seconds."""
    n_frames = int(duration * fps)
    onset = np.full(n_frames, 0.1, dtype=np.float32)
    for sec in peak_seconds:
        idx = int(sec * fps)
        if 0 <= idx < n_frames:
            # Create a peak cluster around the timestamp
            for offset in range(-2, 3):
                j = idx + offset
                if 0 <= j < n_frames:
                    onset[j] = 0.9
    return onset


def _make_prompt_gen(base_prompt: str = "a cat") -> MagicMock:
    """Create a mock PromptGenerator that returns varied prompts."""
    gen = MagicMock()
    call_count = [0]

    def _generate(locked=None, randomness=0, **kwargs):
        call_count[0] += 1
        subject = locked.get("subject", "thing") if locked else "thing"
        prompt = f"{subject} variation {call_count[0]}"
        return (prompt, "bad quality", {"subject": subject})

    gen.generate = _generate
    return gen


# ─── PromptSchedule tests ────────────────────────────────────


class TestPromptSchedule:
    def test_get_prompt_basic(self):
        sched = PromptSchedule(
            [PromptSegment(0, 10, "intro"), PromptSegment(10, 20, "chorus")],
            "default",
        )
        assert sched.get_prompt(5.0) == "intro"
        assert sched.get_prompt(15.0) == "chorus"
        assert sched.get_prompt(25.0) == "default"

    def test_get_prompt_weight_priority(self):
        sched = PromptSchedule(
            [
                PromptSegment(0, 10, "low", weight=0.5),
                PromptSegment(0, 10, "high", weight=1.0),
            ],
            "default",
        )
        assert sched.get_prompt(5.0) == "high"

    def test_from_dicts_valid(self):
        raw = [
            {"start_second": 0, "end_second": 10, "prompt": "a"},
            {"start_second": 10, "end_second": 20, "prompt": "b"},
        ]
        sched = PromptSchedule.from_dicts(raw, "fallback")
        assert sched is not None
        assert sched.get_prompt(5) == "a"

    def test_from_dicts_empty(self):
        assert PromptSchedule.from_dicts([], "x") is None

    def test_from_dicts_invalid_skipped(self):
        raw = [
            {"start_second": 5, "end_second": 3, "prompt": "bad"},  # end < start
            {"start_second": 0, "end_second": 10, "prompt": "good"},
        ]
        sched = PromptSchedule.from_dicts(raw, "x")
        assert sched is not None
        assert len(sched.segments) == 1


# ─── _randomness_to_segment_count tests ──────────────────────


class TestRandomnessToSegmentCount:
    def test_zero_returns_zero(self):
        assert _randomness_to_segment_count(0, 30.0) == 0

    def test_low_randomness(self):
        count = _randomness_to_segment_count(3, 30.0)
        assert count == 2

    def test_moderate_randomness(self):
        count = _randomness_to_segment_count(8, 30.0)
        assert count == 3

    def test_high_randomness(self):
        count = _randomness_to_segment_count(15, 30.0)
        assert 4 <= count <= 5

    def test_chaos_randomness(self):
        count = _randomness_to_segment_count(20, 30.0)
        assert 6 <= count <= 8

    def test_duration_scaling(self):
        short = _randomness_to_segment_count(5, 30.0)
        long_ = _randomness_to_segment_count(5, 180.0)
        assert long_ > short

    def test_capped_at_12(self):
        count = _randomness_to_segment_count(20, 600.0)
        assert count <= 12


# ─── auto_generate_segments tests ────────────────────────────


class TestAutoGenerateSegments:
    def test_zero_randomness_returns_empty(self):
        analysis = FakeAnalysis()
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 0, "a cat", gen)
        assert result == {}

    def test_low_randomness_few_segments(self):
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [8.0, 15.0, 22.0])
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 3, "a cat in a garden", gen)
        keyframes = result.get("keyframes", [])
        assert len(keyframes) == 2
        assert keyframes[0]["frame"] == 0

    def test_high_randomness_many_segments(self):
        analysis = FakeAnalysis(duration=60.0, fps=24.0, bpm=128.0)
        peaks = [5, 12, 20, 28, 35, 42, 50, 55]
        analysis.features["global_onset"] = _make_onset(60.0, 24.0, peaks)
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 18, "abstract shapes", gen)
        keyframes = result.get("keyframes", [])
        assert len(keyframes) >= 4

    def test_segments_cover_full_duration(self):
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=100.0)
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [10.0, 20.0])
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 10, "test", gen)
        keyframes = result.get("keyframes", [])
        assert len(keyframes) >= 2
        assert keyframes[0]["frame"] == 0

    def test_segments_non_overlapping(self):
        analysis = FakeAnalysis(duration=60.0, fps=24.0, bpm=120.0)
        analysis.features["global_onset"] = _make_onset(60.0, 24.0, [10, 20, 30, 40, 50])
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 15, "test", gen)
        keyframes = result.get("keyframes", [])
        for i in range(len(keyframes) - 1):
            assert keyframes[i]["frame"] <= keyframes[i + 1]["frame"]

    def test_max_12_segments(self):
        analysis = FakeAnalysis(duration=600.0, fps=24.0, bpm=140.0)
        peaks = list(range(10, 590, 20))
        analysis.features["global_onset"] = _make_onset(600.0, 24.0, peaks)
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 20, "test", gen)
        keyframes = result.get("keyframes", [])
        assert len(keyframes) <= 12

    def test_no_onset_feature_uses_gap_fill(self):
        """Without onset data, segments should still be generated via gap filling."""
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        # No global_onset feature
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 10, "a landscape", gen)
        keyframes = result.get("keyframes", [])
        assert len(keyframes) >= 2

    def test_bpm_snapping(self):
        """Boundaries should snap to beat grid when BPM is known."""
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        # BPM=120 → beat_interval=0.5s
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [7.3, 14.7])
        gen = _make_prompt_gen()
        result = auto_generate_segments(analysis, 8, "test", gen)
        keyframes = result.get("keyframes", [])
        beat_interval = 0.5
        for kf in keyframes:
            start = kf["frame"] / analysis.fps
            if start > 0:
                # Should be on or very close to a beat
                remainder = start % beat_interval
                assert remainder < 0.05 or abs(remainder - beat_interval) < 0.05, \
                    f"start_second {start} not on beat grid (interval={beat_interval})"

    def test_prompt_gen_called_with_subject(self):
        """The prompt generator should be called with the subject locked."""
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [10.0, 20.0])
        gen = MagicMock()
        gen.generate = MagicMock(return_value=("varied prompt", "neg", {"subject": "cat"}))
        auto_generate_segments(analysis, 10, "masterpiece, a cute cat, pixel art", gen)
        # Should have been called at least once with locked containing subject
        assert gen.generate.call_count >= 2
        for call in gen.generate.call_args_list:
            locked = call.kwargs.get("locked") or call.args[0] if call.args else call.kwargs.get("locked")
            assert locked is not None
            assert "subject" in locked

    def test_prompt_gen_failure_fallback(self):
        """If prompt_gen.generate raises, fall back to base prompt."""
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [10.0, 20.0])
        gen = MagicMock()
        gen.generate = MagicMock(side_effect=RuntimeError("boom"))
        result = auto_generate_segments(analysis, 10, "fallback prompt", gen)
        keyframes = result.get("keyframes", [])
        assert len(keyframes) >= 2
        for kf in keyframes:
            assert kf["prompt"] == "fallback prompt"

    def test_locked_fields_overrides_heuristic(self):
        """Explicit locked_fields.subject should override heuristic extraction."""
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [10.0, 20.0])
        gen = MagicMock()
        gen.generate = MagicMock(return_value=("locked prompt", "neg", {"subject": "my dragon"}))
        # Prompt has misleading comma structure; locked_fields should win
        result = auto_generate_segments(
            analysis, 10, "masterpiece, pixel art, some scene", gen,
            locked_fields={"subject": "my dragon"},
        )
        keyframes = result.get("keyframes", [])
        assert len(keyframes) >= 2
        # Verify generator was called with the locked subject
        for call in gen.generate.call_args_list:
            locked = call.kwargs.get("locked") or (call.args[0] if call.args else {})
            assert locked.get("subject") == "my dragon"

    def test_short_subject_heuristic_fallback(self):
        """Without locked_fields, short subjects fall to heuristic (first part >10 chars)."""
        analysis = FakeAnalysis(duration=30.0, fps=24.0, bpm=120.0)
        analysis.features["global_onset"] = _make_onset(30.0, 24.0, [10.0, 20.0])
        gen = MagicMock()
        gen.generate = MagicMock(return_value=("heuristic prompt", "neg", {"subject": "x"}))
        # "cat" is < 10 chars, so heuristic picks "a beautiful landscape" (first >10)
        result = auto_generate_segments(
            analysis, 10, "cat, a beautiful landscape, pixel art", gen,
        )
        keyframes = result.get("keyframes", [])
        assert len(keyframes) >= 2
        for call in gen.generate.call_args_list:
            locked = call.kwargs.get("locked") or (call.args[0] if call.args else {})
            # Heuristic should pick "a beautiful landscape" (first part > 10 chars)
            assert locked.get("subject") == "a beautiful landscape"

