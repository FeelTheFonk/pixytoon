"""Tests for prompt schedule keyframe engine — Phase 2-5 features."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sddj.prompt_schedule import PromptKeyframe, PromptSchedule, auto_fill_prompts
from sddj.prompt_schedule_presets import PromptSchedulePresetsManager, _BUILTIN_PRESETS


# ─────────────────────────────────────────────────────────────
# PromptKeyframe dataclass
# ─────────────────────────────────────────────────────────────

class TestPromptKeyframe:
    def test_defaults(self):
        kf = PromptKeyframe(frame=0, prompt="test")
        assert kf.frame == 0
        assert kf.prompt == "test"
        assert kf.negative_prompt == ""
        assert kf.weight == 1.0
        assert kf.transition == "hard_cut"
        assert kf.transition_frames == 0

    def test_asdict_roundtrip(self):
        kf = PromptKeyframe(
            frame=5, prompt="hello", negative_prompt="ugly",
            weight=0.8, transition="blend", transition_frames=3,
        )
        d = asdict(kf)
        assert d["frame"] == 5
        assert d["prompt"] == "hello"
        assert d["negative_prompt"] == "ugly"
        assert d["transition"] == "blend"
        assert d["transition_frames"] == 3
        # Roundtrip
        kf2 = PromptKeyframe(**d)
        assert kf2.frame == kf.frame
        assert kf2.prompt == kf.prompt


# ─────────────────────────────────────────────────────────────
# PromptSchedule — keyframe resolution
# ─────────────────────────────────────────────────────────────

class TestPromptScheduleKeyframes:
    def _make_schedule(self):
        return PromptSchedule.from_keyframe_dicts([
            {"frame": 0, "prompt": "forest", "negative_prompt": "blur"},
            {"frame": 5, "prompt": "ocean", "transition": "hard_cut"},
            {"frame": 10, "prompt": "city", "transition": "blend", "transition_frames": 3},
        ], default_prompt="default")

    def test_hard_cut_resolution(self):
        s = self._make_schedule()
        assert s.get_prompt_for_frame(0) == "forest"
        assert s.get_prompt_for_frame(3) == "forest"
        assert s.get_prompt_for_frame(5) == "ocean"
        assert s.get_prompt_for_frame(9) == "ocean"

    def test_blend_alternation(self):
        s = self._make_schedule()
        # Frame 10 is "city" with blend, transition_frames=3
        # During blend window (frames 10,11,12), alternation between outgoing/incoming
        # Frame 13+ is fully "city" post-blend
        assert s.get_prompt_for_frame(13) == "city"
        assert s.get_prompt_for_frame(15) == "city"

    def test_negative_resolution(self):
        s = self._make_schedule()
        assert s.get_negative_for_frame(0) == "blur"
        # Frames 5+ have no negative_prompt set, so returns None
        assert s.get_negative_for_frame(5) is None
        assert s.get_negative_for_frame(10) is None

    def test_unique_prompts(self):
        s = self._make_schedule()
        unique = s.get_unique_prompts()
        assert "forest" in unique
        assert "ocean" in unique
        assert "city" in unique

    def test_unique_negatives(self):
        s = self._make_schedule()
        unique = s.get_unique_negatives()
        assert "blur" in unique

    def test_default_prompt_fallback(self):
        s = PromptSchedule.from_keyframe_dicts(
            [{"frame": 5, "prompt": "later"}],
            default_prompt="start",
        )
        # Frame 0-4: before any keyframe → default
        assert s.get_prompt_for_frame(0) == "start"
        assert s.get_prompt_for_frame(5) == "later"

    def test_empty_keyframes_returns_none(self):
        """from_keyframe_dicts returns None for empty list."""
        result = PromptSchedule.from_keyframe_dicts([], default_prompt="fallback")
        assert result is None


# ─────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────

class TestPromptScheduleSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        original = PromptSchedule.from_keyframe_dicts([
            {"frame": 0, "prompt": "A"},
            {"frame": 4, "prompt": "B", "transition": "blend", "transition_frames": 2},
        ], default_prompt="default_p")
        d = original.to_dict()
        restored = PromptSchedule.from_dict(d)
        assert restored is not None
        assert len(restored.keyframes) == len(original.keyframes)
        assert restored.get_prompt_for_frame(0) == "A"
        assert restored.get_prompt_for_frame(5) == "B"

    def test_to_dict_contains_keyframes(self):
        s = PromptSchedule.from_keyframe_dicts([
            {"frame": 0, "prompt": "X"},
        ], default_prompt="Y")
        d = s.to_dict()
        assert "keyframes" in d
        assert len(d["keyframes"]) == 1
        assert d["keyframes"][0]["prompt"] == "X"

    def test_from_dict_preserves_transitions(self):
        s = PromptSchedule.from_dict({
            "keyframes": [
                {"frame": 0, "prompt": "A", "transition": "hard_cut"},
                {"frame": 5, "prompt": "B", "transition": "blend", "transition_frames": 3},
            ],
            "default_prompt": "fallback",
        })
        assert s.keyframes[1].transition == "blend"
        assert s.keyframes[1].transition_frames == 3


# ─────────────────────────────────────────────────────────────
# PromptSchedulePresetsManager
# ─────────────────────────────────────────────────────────────

class TestPromptSchedulePresetsManager:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return tmp_path / "prompt_schedules"

    @pytest.fixture
    def mgr(self, tmp_dir):
        return PromptSchedulePresetsManager(tmp_dir)

    def test_list_includes_builtins(self, mgr):
        names = mgr.list_presets()
        for builtin in _BUILTIN_PRESETS:
            assert builtin in names

    def test_get_builtin(self, mgr):
        data = mgr.get_preset("evolving_3act")
        assert data["name"] == "evolving_3act"
        assert "keyframes" in data

    def test_save_and_get_user_preset(self, mgr):
        mgr.save_preset("my_sched", {
            "keyframes": [{"frame": 0, "prompt": "test"}],
        })
        data = mgr.get_preset("my_sched")
        assert data["name"] == "my_sched"
        assert data["keyframes"][0]["prompt"] == "test"

    def test_delete_user_preset(self, mgr):
        mgr.save_preset("to_delete", {"keyframes": []})
        mgr.delete_preset("to_delete")
        with pytest.raises(FileNotFoundError):
            mgr.get_preset("to_delete")

    def test_cannot_overwrite_builtin(self, mgr):
        with pytest.raises(ValueError, match="built-in"):
            mgr.save_preset("evolving_3act", {"keyframes": []})

    def test_cannot_delete_builtin(self, mgr):
        with pytest.raises(ValueError, match="built-in"):
            mgr.delete_preset("evolving_3act")

    def test_path_traversal_rejected(self, mgr):
        with pytest.raises(ValueError, match="Invalid"):
            mgr.save_preset("../evil", {})

    def test_invalid_name_rejected(self, mgr):
        with pytest.raises(ValueError, match="Invalid"):
            mgr.save_preset("bad name!", {})

    def test_empty_name_rejected(self, mgr):
        with pytest.raises(ValueError, match="empty"):
            mgr.get_preset("")

    def test_list_includes_user_presets(self, mgr):
        mgr.save_preset("custom1", {"keyframes": []})
        names = mgr.list_presets()
        assert "custom1" in names


# ─────────────────────────────────────────────────────────────
# Protocol integration
# ─────────────────────────────────────────────────────────────

class TestPromptScheduleProtocol:
    def test_prompt_schedule_actions_exist(self):
        from sddj.protocol import Action
        assert Action("list_prompt_schedules") == Action.LIST_PROMPT_SCHEDULES
        assert Action("get_prompt_schedule") == Action.GET_PROMPT_SCHEDULE
        assert Action("save_prompt_schedule") == Action.SAVE_PROMPT_SCHEDULE
        assert Action("delete_prompt_schedule") == Action.DELETE_PROMPT_SCHEDULE

    def test_prompt_schedule_spec_validation(self):
        from sddj.protocol import PromptScheduleSpec, PromptKeyframeSpec
        spec = PromptScheduleSpec(
            keyframes=[
                PromptKeyframeSpec(frame=0, prompt="A"),
                PromptKeyframeSpec(frame=5, prompt="B", transition="blend", transition_frames=2),
            ],
            default_prompt="fallback",
        )
        assert len(spec.keyframes) == 2
        assert spec.keyframes[1].transition == "blend"

    def test_invalid_transition_defaults_to_hard_cut(self):
        from sddj.protocol import PromptKeyframeSpec
        kf = PromptKeyframeSpec(frame=0, prompt="A", transition="invalid_type")
        assert kf.transition == "hard_cut"

    def test_generate_request_accepts_prompt_schedule(self):
        from sddj.protocol import GenerateRequest, PromptScheduleSpec, PromptKeyframeSpec
        req = GenerateRequest(
            prompt="test",
            prompt_schedule=PromptScheduleSpec(
                keyframes=[PromptKeyframeSpec(frame=0, prompt="scheduled")],
            ),
        )
        assert req.prompt_schedule is not None
        assert req.prompt_schedule.keyframes[0].prompt == "scheduled"

    def test_animation_request_accepts_prompt_schedule(self):
        from sddj.protocol import AnimationRequest, PromptScheduleSpec, PromptKeyframeSpec
        req = AnimationRequest(
            prompt="test",
            prompt_schedule=PromptScheduleSpec(
                keyframes=[PromptKeyframeSpec(frame=0, prompt="anim_sched")],
            ),
        )
        assert req.prompt_schedule is not None

    def test_audio_reactive_request_accepts_prompt_schedule(self):
        from sddj.protocol import AudioReactiveRequest, PromptScheduleSpec, PromptKeyframeSpec
        req = AudioReactiveRequest(
            audio_path="/test.wav",
            prompt_schedule=PromptScheduleSpec(
                keyframes=[PromptKeyframeSpec(frame=0, prompt="audio_sched")],
            ),
        )
        assert req.prompt_schedule is not None

    def test_request_to_generate_forwards_prompt_schedule(self):
        from sddj.protocol import Request
        req = Request(
            action="generate", prompt="test",
            prompt_schedule={
                "keyframes": [{"frame": 0, "prompt": "sched"}],
                "default_prompt": "fallback",
            },
        )
        gen = req.to_generate_request()
        assert gen.prompt_schedule is not None

    def test_request_to_animation_forwards_prompt_schedule(self):
        from sddj.protocol import Request
        req = Request(
            action="generate_animation", prompt="test",
            frame_count=8,
            prompt_schedule={
                "keyframes": [{"frame": 0, "prompt": "anim"}],
            },
        )
        anim = req.to_animation_request()
        assert anim.prompt_schedule is not None

    def test_request_to_audio_reactive_forwards_prompt_schedule(self):
        from sddj.protocol import Request
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav",
            prompt="test",
            prompt_schedule={
                "keyframes": [{"frame": 0, "prompt": "audio"}],
            },
        )
        ar = req.to_audio_reactive_request()
        assert ar.prompt_schedule is not None

    def test_crud_fields_excluded_from_generate(self):
        from sddj.protocol import Request
        req = Request(
            action="generate", prompt="test",
            prompt_schedule_name="my_sched",
            prompt_schedule_data={"keyframes": []},
        )
        gen = req.to_generate_request()
        dumped = gen.model_dump()
        assert "prompt_schedule_name" not in dumped
        assert "prompt_schedule_data" not in dumped


# ─────────────────────────────────────────────────────────────
# build_prompt_schedule helper
# ─────────────────────────────────────────────────────────────

class TestBuildPromptSchedule:
    def test_returns_none_for_no_schedule(self):
        from sddj.engine.helpers import build_prompt_schedule
        from sddj.protocol import GenerateRequest
        req = GenerateRequest(prompt="test")
        assert build_prompt_schedule(req) is None

    def test_builds_from_prompt_schedule_spec(self):
        from sddj.engine.helpers import build_prompt_schedule
        from sddj.protocol import GenerateRequest, PromptScheduleSpec, PromptKeyframeSpec
        req = GenerateRequest(
            prompt="fallback",
            prompt_schedule=PromptScheduleSpec(
                keyframes=[
                    PromptKeyframeSpec(frame=0, prompt="A"),
                    PromptKeyframeSpec(frame=5, prompt="B"),
                ],
            ),
        )
        sched = build_prompt_schedule(req)
        assert sched is not None
        assert sched.get_prompt_for_frame(0) == "A"
        assert sched.get_prompt_for_frame(5) == "B"

    def test_builds_from_raw_dict(self):
        from sddj.engine.helpers import build_prompt_schedule

        class FakeReq:
            prompt = "fallback"
            prompt_schedule = {
                "keyframes": [
                    {"frame": 0, "prompt": "X"},
                    {"frame": 3, "prompt": "Y"},
                ],
                "default_prompt": "fallback",
            }

        sched = build_prompt_schedule(FakeReq())
        assert sched is not None
        assert sched.get_prompt_for_frame(0) == "X"
        assert sched.get_prompt_for_frame(3) == "Y"


    def test_lua_dict_encoded_keyframes(self):
        """Lua json.lua may encode arrays as objects with numeric keys."""
        from sddj.engine.helpers import build_prompt_schedule

        class FakeReq:
            prompt = "fallback"
            prompt_schedule = {
                "keyframes": {
                    "1": {"frame": 0, "prompt": "A"},
                    "2": {"frame": 5, "prompt": "B"},
                },
            }

        sched = build_prompt_schedule(FakeReq())
        assert sched is not None
        assert sched.get_prompt_for_frame(0) == "A"
        assert sched.get_prompt_for_frame(5) == "B"


# ─────────────────────────────────────────────────────────────
# Auto-fill
# ─────────────────────────────────────────────────────────────

class TestAutoFillPrompts:
    def test_auto_fill_populates_empty_keyframes(self):
        from unittest.mock import MagicMock
        schedule = PromptSchedule.from_keyframe_dicts([
            {"frame": 0, "prompt": ""},
            {"frame": 5, "prompt": ""},
        ], default_prompt="")
        mock_gen = MagicMock()
        mock_gen.generate.return_value = ("generated prompt", "neg", {})
        filled = auto_fill_prompts(schedule, mock_gen, randomness=5)
        # All keyframes should now have non-empty prompts
        for kf in filled.keyframes:
            assert kf.prompt != ""

    def test_auto_fill_preserves_existing_prompts(self):
        from unittest.mock import MagicMock
        schedule = PromptSchedule.from_keyframe_dicts([
            {"frame": 0, "prompt": "keep me"},
            {"frame": 5, "prompt": ""},
        ], default_prompt="")
        mock_gen = MagicMock()
        mock_gen.generate.return_value = ("generated", "neg", {})
        filled = auto_fill_prompts(schedule, mock_gen, randomness=5)
        assert filled.keyframes[0].prompt == "keep me"
        assert filled.keyframes[1].prompt != ""


# ─────────────────────────────────────────────────────────────
# Backward compatibility (legacy segments still work)
# ─────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_from_dicts_still_works(self):
        segments = [
            {"start_second": 0.0, "end_second": 5.0, "prompt": "intro"},
            {"start_second": 5.0, "end_second": 10.0, "prompt": "climax"},
        ]
        s = PromptSchedule.from_dicts(segments, default_prompt="fallback")
        assert s is not None
        # Time-based resolution
        prompt = s.get_prompt(2.5)
        assert prompt == "intro"
        prompt2 = s.get_prompt(7.0)
        assert prompt2 == "climax"

    def test_keyframes_take_precedence(self):
        """Schedule built from keyframes resolves via get_prompt_for_frame."""
        s = PromptSchedule.from_keyframe_dicts([
            {"frame": 0, "prompt": "kf_prompt"},
        ], default_prompt="default")
        assert s.get_prompt_for_frame(0) == "kf_prompt"
        assert s.get_prompt_for_frame(100) == "kf_prompt"
