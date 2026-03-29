"""Exhaustive tests for the Python DSL parser (server/sddj/dsl_parser.py).

Covers: parsing, time markers, directives, validation errors/warnings,
edge cases, safety limits, file references, and roundtrip via schedule_to_dsl.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sddj.dsl_parser import (
    ParseResult,
    _MAX_DSL_LENGTH,
    _MAX_KEYFRAMES,
    parse,
)
from sddj.prompt_schedule import (
    PromptSchedule,
    schedule_to_dsl,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _error_codes(result: ParseResult) -> set[str]:
    return {e.code for e in result.validation.errors}


def _warning_codes(result: ParseResult) -> set[str]:
    return {w.code for w in result.validation.warnings}


# ─────────────────────────────────────────────────────────────
# Empty / minimal inputs
# ─────────────────────────────────────────────────────────────


class TestEmptyInputs:
    def test_empty_string(self):
        r = parse("", 100, 24.0)
        assert r.schedule is None
        assert r.validation.valid
        assert "W002" in _warning_codes(r)

    def test_whitespace_only(self):
        r = parse("   \n\t\n  ", 100, 24.0)
        assert r.schedule is None
        assert "W002" in _warning_codes(r)

    def test_comments_only(self):
        r = parse("# comment\n# another", 100, 24.0)
        assert r.schedule is None
        assert "W002" in _warning_codes(r)

    def test_auto_only(self):
        r = parse("{auto}", 100, 24.0)
        assert r.has_auto
        assert r.schedule is None  # no keyframes
        assert "W002" in _warning_codes(r)


# ─────────────────────────────────────────────────────────────
# Time marker formats
# ─────────────────────────────────────────────────────────────


class TestTimeMarkers:
    def test_absolute_frame(self):
        r = parse("[0]\nhello\n[10]\nworld", 100, 24.0)
        assert r.validation.valid
        assert r.schedule is not None
        assert len(r.schedule.keyframes) == 2
        assert r.schedule.keyframes[0].frame == 0
        assert r.schedule.keyframes[1].frame == 10

    def test_percentage(self):
        r = parse("[0%]\nstart\n[50%]\nmid\n[100%]\nend", 100, 24.0)
        assert r.validation.valid
        kfs = r.schedule.keyframes
        assert kfs[0].frame == 0
        assert kfs[1].frame == 50
        assert kfs[2].frame == 99  # 100% clamped to total_frames-1

    def test_seconds(self):
        r = parse("[0s]\nstart\n[2s]\ntwo\n[5.5s]\nfive_half", 200, 24.0)
        assert r.validation.valid
        kfs = r.schedule.keyframes
        assert kfs[0].frame == 0
        assert kfs[1].frame == 48  # 2 * 24
        assert kfs[2].frame == 132  # 5.5 * 24

    def test_frame_exceeding_total(self):
        r = parse("[0]\nstart\n[200]\nover", 100, 24.0)
        assert "E001" in _error_codes(r)
        # Frame gets clamped
        assert r.schedule.keyframes[1].frame == 99

    def test_percentage_over_100(self):
        r = parse("[0%]\nstart\n[150%]\nover", 100, 24.0)
        assert "E001" in _error_codes(r)

    def test_mixed_formats_chronological(self):
        r = parse("[0]\nstart\n[3s]\nthree_sec\n[50%]\nmid", 200, 24.0)
        assert r.validation.valid
        kfs = r.schedule.keyframes
        assert kfs[0].frame == 0
        assert kfs[1].frame == 72   # 3*24
        assert kfs[2].frame == 100  # 50% of 200

    def test_mixed_formats_out_of_order(self):
        # [50%]=100, then [3s]=72 → out of order
        r = parse("[0]\nstart\n[50%]\nmid\n[3s]\nthree_sec", 200, 24.0)
        assert "E003" in _error_codes(r)


# ─────────────────────────────────────────────────────────────
# Directives
# ─────────────────────────────────────────────────────────────


class TestDirectives:
    def test_transition_blend(self):
        r = parse("[0]\na\n[10]\ntransition: blend\nblend: 5\nb", 100, 24.0)
        assert r.validation.valid
        kf1 = r.schedule.keyframes[1]
        assert kf1.transition == "blend"
        assert kf1.transition_frames == 5

    def test_all_transition_types(self):
        for tr in ("hard_cut", "blend", "linear_blend", "ease_in",
                    "ease_out", "ease_in_out", "cubic", "slerp"):
            r = parse(f"[0]\na\n[10]\ntransition: {tr}\nblend: 3\nb", 100, 24.0)
            assert r.schedule.keyframes[1].transition == tr, f"Failed for {tr}"

    def test_invalid_transition(self):
        r = parse("[0]\ntransition: invalid\nhello", 100, 24.0)
        assert "E005" in _error_codes(r)

    def test_weight_simple(self):
        r = parse("[0]\nweight: 1.5\nhello", 100, 24.0)
        assert r.schedule.keyframes[0].weight == 1.5

    def test_weight_animated(self):
        r = parse("[0]\nweight: 0.5->2.0\nhello", 100, 24.0)
        kf = r.schedule.keyframes[0]
        assert kf.weight == 0.5
        assert kf.weight_end == 2.0

    def test_weight_over_2_warning(self):
        r = parse("[0]\nweight: 2.5\nhello", 100, 24.0)
        assert r.validation.valid  # not an error
        assert "W004" in _warning_codes(r)
        assert r.schedule.keyframes[0].weight == 2.5

    def test_weight_out_of_range_error(self):
        r = parse("[0]\nweight: 0.05\nhello", 100, 24.0)
        assert "E006" in _error_codes(r)

    def test_weight_end_out_of_range(self):
        r = parse("[0]\nweight: 1.0->6.0\nhello", 100, 24.0)
        assert "E006" in _error_codes(r)

    def test_weight_end_over_2_warning(self):
        r = parse("[0]\nweight: 1.0->3.0\nhello", 100, 24.0)
        assert "W004" in _warning_codes(r)

    def test_weight_valid_but_weight_end_invalid_no_leak(self):
        """weight_end out of range must NOT leak into the keyframe."""
        r = parse("[0]\nweight: 1.5->6.0\nhello", 100, 24.0)
        assert "E006" in _error_codes(r)
        # weight is valid and assigned
        assert r.schedule.keyframes[0].weight == 1.5
        # weight_end is invalid and must NOT be assigned
        assert r.schedule.keyframes[0].weight_end is None

    def test_denoise(self):
        r = parse("[0]\ndenoise: 0.55\nhello", 100, 24.0)
        assert r.schedule.keyframes[0].denoise_strength == 0.55

    def test_denoise_out_of_range(self):
        r = parse("[0]\ndenoise: 1.5\nhello", 100, 24.0)
        assert "E007" in _error_codes(r)

    def test_cfg(self):
        r = parse("[0]\ncfg: 7.5\nhello", 100, 24.0)
        assert r.schedule.keyframes[0].cfg_scale == 7.5

    def test_cfg_out_of_range(self):
        r = parse("[0]\ncfg: 35\nhello", 100, 24.0)
        assert "E008" in _error_codes(r)

    def test_steps(self):
        r = parse("[0]\nsteps: 20\nhello", 100, 24.0)
        assert r.schedule.keyframes[0].steps == 20

    def test_steps_out_of_range(self):
        r = parse("[0]\nsteps: 200\nhello", 100, 24.0)
        assert "E009" in _error_codes(r)

    def test_negative_prompt(self):
        r = parse("[0]\nbeautiful landscape\n-- ugly, blurry\n-- worst quality", 100, 24.0)
        kf = r.schedule.keyframes[0]
        assert kf.prompt == "beautiful landscape"
        assert "ugly, blurry" in kf.negative_prompt
        assert "worst quality" in kf.negative_prompt

    def test_blend_exceeds_max(self):
        r = parse("[0]\na\n[10]\nblend: 150\nb", 100, 24.0)
        assert "E004" in _error_codes(r)

    def test_all_directives_combined(self):
        dsl = (
            "[0]\n"
            "transition: ease_in_out\n"
            "blend: 5\n"
            "weight: 1.2->1.8\n"
            "denoise: 0.45\n"
            "cfg: 6.5\n"
            "steps: 15\n"
            "a glowing crystal\n"
            "-- ugly, blurry"
        )
        r = parse(dsl, 100, 24.0)
        assert r.validation.valid
        kf = r.schedule.keyframes[0]
        assert kf.transition == "ease_in_out"
        assert kf.transition_frames == 5
        assert kf.weight == 1.2
        assert kf.weight_end == 1.8
        assert kf.denoise_strength == 0.45
        assert kf.cfg_scale == 6.5
        assert kf.steps == 15
        assert kf.prompt == "a glowing crystal"
        assert kf.negative_prompt == "ugly, blurry"


# ─────────────────────────────────────────────────────────────
# Prompt text parsing
# ─────────────────────────────────────────────────────────────


class TestPromptParsing:
    def test_multi_line_prompt(self):
        dsl = "[0]\nbeautiful landscape\nwith mountains\nand rivers"
        r = parse(dsl, 100, 24.0)
        assert r.schedule.keyframes[0].prompt == "beautiful landscape, with mountains, and rivers"

    def test_prompt_after_directives(self):
        dsl = "[0]\ntransition: blend\nblend: 5\nmy prompt text"
        r = parse(dsl, 100, 24.0)
        assert r.schedule.keyframes[0].prompt == "my prompt text"

    def test_empty_prompt(self):
        dsl = "[0]\ntransition: blend\nblend: 5"
        r = parse(dsl, 100, 24.0)
        assert r.schedule.keyframes[0].prompt == ""

    def test_content_before_time_marker(self):
        r = parse("orphan text\n[0]\nhello", 100, 24.0)
        assert "E012" in _error_codes(r)


# ─────────────────────────────────────────────────────────────
# Auto directive
# ─────────────────────────────────────────────────────────────


class TestAutoDirective:
    def test_auto_with_keyframes(self):
        r = parse("{auto}\n[0]\nhello\n[50]\nworld", 100, 24.0)
        assert r.has_auto
        assert r.validation.valid
        assert len(r.schedule.keyframes) == 2

    def test_auto_case_insensitive(self):
        for variant in ("{auto}", "{AUTO}", "{Auto}", "{aUtO}"):
            r = parse(f"{variant}\n[0]\nhello", 100, 24.0)
            assert r.has_auto, f"Failed for {variant}"

    def test_auto_mid_document(self):
        r = parse("[0]\nhello\n{auto}\n[50]\nworld", 100, 24.0)
        assert r.has_auto


# ─────────────────────────────────────────────────────────────
# Validation: ordering, duplicates, transitions
# ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_duplicate_frame(self):
        r = parse("[0]\nfirst\n[0]\nsecond", 100, 24.0)
        assert "E002" in _error_codes(r)

    def test_out_of_order(self):
        r = parse("[10]\nlater\n[5]\nearlier", 100, 24.0)
        assert "E003" in _error_codes(r)

    def test_first_keyframe_not_zero_warning(self):
        r = parse("[5]\nhello", 100, 24.0)
        assert "W001" in _warning_codes(r)

    def test_blend_with_hard_cut_warning(self):
        r = parse("[0]\na\n[10]\ntransition: hard_cut\nblend: 5\nb", 100, 24.0)
        assert "W006" in _warning_codes(r)

    def test_transition_exceeds_gap(self):
        r = parse("[0]\na\n[5]\ntransition: blend\nblend: 10\nb", 100, 24.0)
        assert "E004" in _error_codes(r)

    def test_unrecognized_directive_warning(self):
        r = parse("[0]\nfoo: bar\nhello", 100, 24.0)
        assert "W007" in _warning_codes(r)
        # The directive-like line is still treated as prompt text
        assert "foo: bar" in r.schedule.keyframes[0].prompt


# ─────────────────────────────────────────────────────────────
# Safety limits
# ─────────────────────────────────────────────────────────────


class TestSafetyLimits:
    def test_oversized_dsl(self):
        big = "x" * (_MAX_DSL_LENGTH + 1)
        r = parse(big, 100, 24.0)
        assert not r.validation.valid
        assert "E013" in _error_codes(r)

    def test_too_many_keyframes(self):
        # Each keyframe needs its own line: [N]\nprompt
        lines = []
        for i in range(_MAX_KEYFRAMES + 10):
            lines.append(f"[{i}]")
            lines.append(f"prompt_{i}")
        dsl = "\n".join(lines)
        r = parse(dsl, _MAX_KEYFRAMES + 100, 24.0)
        assert "E014" in _error_codes(r)
        assert len(r.schedule.keyframes) <= _MAX_KEYFRAMES

    def test_at_limit_is_ok(self):
        lines = []
        for i in range(50):
            lines.append(f"[{i}]")
            lines.append(f"prompt_{i}")
        dsl = "\n".join(lines)
        r = parse(dsl, 1000, 24.0)
        assert "E014" not in _error_codes(r)
        assert len(r.schedule.keyframes) == 50


# ─────────────────────────────────────────────────────────────
# File reference
# ─────────────────────────────────────────────────────────────


class TestFileReference:
    def test_file_ref_no_base_dir(self):
        r = parse("file: test.txt", 100, 24.0)
        assert "E010" in _error_codes(r)

    def test_file_ref_missing_file(self, tmp_path):
        r = parse("file: nonexistent.txt", 100, 24.0, base_dir=tmp_path)
        assert "E011" in _error_codes(r)

    def test_file_ref_path_traversal(self, tmp_path):
        r = parse("file: ../../../etc/passwd", 100, 24.0, base_dir=tmp_path)
        assert "E010" in _error_codes(r)

    def test_file_ref_absolute_path(self, tmp_path):
        r = parse("file: /etc/passwd", 100, 24.0, base_dir=tmp_path)
        assert "E010" in _error_codes(r)

    def test_file_ref_windows_absolute(self, tmp_path):
        r = parse("file: C:\\Windows\\test.txt", 100, 24.0, base_dir=tmp_path)
        assert "E010" in _error_codes(r)

    def test_file_ref_valid(self, tmp_path):
        dsl_file = tmp_path / "schedule.txt"
        dsl_file.write_text("[0]\nhello from file\n[50]\nworld from file", encoding="utf-8")
        r = parse("file: schedule.txt", 100, 24.0, base_dir=tmp_path)
        assert r.validation.valid
        assert r.file_ref == "schedule.txt"
        assert r.schedule is not None
        assert r.schedule.keyframes[0].prompt == "hello from file"

    def test_file_ref_mid_document_error(self):
        # file: only valid as single line, so if there's a [0] before,
        # the file: line happens inside a keyframe context and triggers E012
        dsl = "[0]\nhello\nfile: other.txt"
        r = parse(dsl, 100, 24.0)
        assert "E012" in _error_codes(r)


# ─────────────────────────────────────────────────────────────
# schedule_to_dsl roundtrip
# ─────────────────────────────────────────────────────────────


class TestRoundtrip:
    def test_simple_roundtrip(self):
        dsl = "[0]\na beautiful scene\n\n[50]\ntransition: blend\nblend: 10\nan ocean view\n"
        r1 = parse(dsl, 100, 24.0)
        assert r1.validation.valid

        # Convert schedule back to DSL
        kf_dicts = [
            {
                "frame": kf.frame,
                "prompt": kf.prompt,
                "negative_prompt": kf.negative_prompt,
                "weight": kf.weight,
                "transition": kf.transition,
                "transition_frames": kf.transition_frames,
            }
            for kf in r1.schedule.keyframes
        ]
        dsl2 = schedule_to_dsl(kf_dicts)
        r2 = parse(dsl2, 100, 24.0)
        assert r2.validation.valid
        assert len(r2.schedule.keyframes) == len(r1.schedule.keyframes)

        # Verify fields match
        for kf1, kf2 in zip(r1.schedule.keyframes, r2.schedule.keyframes):
            assert kf1.frame == kf2.frame
            assert kf1.prompt == kf2.prompt
            assert kf1.transition == kf2.transition
            assert kf1.transition_frames == kf2.transition_frames

    def test_roundtrip_with_all_directives(self):
        dsl = (
            "[0]\n"
            "transition: ease_in_out\n"
            "blend: 5\n"
            "weight: 1.50\n"
            "denoise: 0.45\n"
            "cfg: 7.0\n"
            "steps: 20\n"
            "crystal cave\n"
            "-- ugly\n"
        )
        r1 = parse(dsl, 100, 24.0)
        assert r1.validation.valid
        kf = r1.schedule.keyframes[0]

        kf_dict = {
            "frame": kf.frame,
            "prompt": kf.prompt,
            "negative_prompt": kf.negative_prompt,
            "weight": kf.weight,
            "transition": kf.transition,
            "transition_frames": kf.transition_frames,
            "denoise_strength": kf.denoise_strength,
            "cfg_scale": kf.cfg_scale,
            "steps": kf.steps,
        }
        dsl2 = schedule_to_dsl([kf_dict])
        r2 = parse(dsl2, 100, 24.0)
        kf2 = r2.schedule.keyframes[0]

        assert kf2.frame == kf.frame
        assert kf2.prompt == kf.prompt
        assert kf2.negative_prompt == kf.negative_prompt
        assert kf2.weight == pytest.approx(kf.weight, abs=0.01)
        assert kf2.transition == kf.transition
        assert kf2.transition_frames == kf.transition_frames
        assert kf2.denoise_strength == pytest.approx(kf.denoise_strength, abs=0.01)
        assert kf2.cfg_scale == pytest.approx(kf.cfg_scale, abs=0.1)
        assert kf2.steps == kf.steps

    def test_roundtrip_animated_weight(self):
        dsl = "[0]\nweight: 0.80->1.50\ntest prompt\n"
        r1 = parse(dsl, 100, 24.0)
        kf = r1.schedule.keyframes[0]
        assert kf.weight == pytest.approx(0.8)
        assert kf.weight_end == pytest.approx(1.5)

        kf_dict = {
            "frame": kf.frame,
            "prompt": kf.prompt,
            "weight": kf.weight,
            "weight_end": kf.weight_end,
            "transition": kf.transition,
            "transition_frames": kf.transition_frames,
        }
        dsl2 = schedule_to_dsl([kf_dict])
        r2 = parse(dsl2, 100, 24.0)
        kf2 = r2.schedule.keyframes[0]
        assert kf2.weight == pytest.approx(0.8, abs=0.01)
        assert kf2.weight_end == pytest.approx(1.5, abs=0.01)

    def test_schedule_to_dsl_include_auto(self):
        dsl = schedule_to_dsl([{"frame": 0, "prompt": "hello"}], include_auto=True)
        assert dsl.startswith("{auto}")
        r = parse(dsl, 100, 24.0)
        assert r.has_auto


# ─────────────────────────────────────────────────────────────
# Comments and whitespace
# ─────────────────────────────────────────────────────────────


class TestCommentsWhitespace:
    def test_comments_ignored(self):
        dsl = "# header comment\n[0]\nhello\n# mid comment\n[10]\nworld"
        r = parse(dsl, 100, 24.0)
        assert r.validation.valid
        assert len(r.schedule.keyframes) == 2

    def test_blank_lines_ignored(self):
        dsl = "\n\n[0]\nhello\n\n\n[10]\nworld\n\n"
        r = parse(dsl, 100, 24.0)
        assert r.validation.valid
        assert len(r.schedule.keyframes) == 2

    def test_inline_whitespace_stripped(self):
        dsl = "[0]\n   hello world   "
        r = parse(dsl, 100, 24.0)
        assert r.schedule.keyframes[0].prompt == "hello world"


# ─────────────────────────────────────────────────────────────
# Blend info resolution (integration with PromptSchedule)
# ─────────────────────────────────────────────────────────────


class TestBlendResolution:
    def test_hard_cut_no_blending(self):
        dsl = "[0]\nscene A\n[10]\nscene B"
        r = parse(dsl, 100, 24.0)
        assert r.schedule is not None
        info = r.schedule.get_blend_info_for_frame(5)
        assert not info.is_blending
        assert info.effective_prompt == "scene A"

        info10 = r.schedule.get_blend_info_for_frame(10)
        assert not info10.is_blending
        assert info10.effective_prompt == "scene B"

    def test_blend_transition(self):
        dsl = "[0]\nscene A\n[10]\ntransition: blend\nblend: 4\nscene B"
        r = parse(dsl, 100, 24.0)
        assert r.schedule is not None

        # Before transition (frame 8 is still in keyframe 0 zone)
        info8 = r.schedule.get_blend_info_for_frame(8)
        assert info8.effective_prompt == "scene A"

        # During transition (frame 10-13): frame 11 is 1 frame into 4-frame blend
        info11 = r.schedule.get_blend_info_for_frame(11)
        assert info11.is_blending
        assert info11.prompt_a == "scene A"
        assert info11.prompt_b == "scene B"
        assert 0.0 < info11.blend_weight < 1.0

        # After transition (frame 14+)
        info15 = r.schedule.get_blend_info_for_frame(15)
        assert not info15.is_blending
        assert info15.effective_prompt == "scene B"

    def test_negative_prompt_in_blend(self):
        dsl = "[0]\nscene A\n-- neg A\n[10]\ntransition: blend\nblend: 4\nscene B\n-- neg B"
        r = parse(dsl, 100, 24.0)
        assert r.schedule is not None

        # Frame 11: 1 frame into blend window
        info = r.schedule.get_blend_info_for_frame(11)
        assert info.is_blending
        assert info.negative_prompt_b == "neg B"

    def test_per_keyframe_params(self):
        dsl = "[0]\ndenoise: 0.3\ncfg: 5.0\nsteps: 10\nscene A"
        r = parse(dsl, 100, 24.0)
        info = r.schedule.get_blend_info_for_frame(0)
        assert info.denoise_strength == pytest.approx(0.3)
        assert info.cfg_scale == pytest.approx(5.0)
        assert info.steps == 10

    def test_animated_weight(self):
        dsl = "[0]\nweight: 0.5->1.5\nhello"
        r = parse(dsl, 100, 24.0)
        # At frame 0, weight should be start value
        info0 = r.schedule.get_blend_info_for_frame(0)
        assert info0.weight == pytest.approx(0.5)

        # At frame ~50 (midpoint), weight should be ~1.0
        info50 = r.schedule.get_blend_info_for_frame(50)
        assert info50.weight == pytest.approx(1.0, abs=0.1)


# ─────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_keyframe(self):
        r = parse("[0]\nonly one", 100, 24.0)
        assert r.validation.valid
        assert len(r.schedule.keyframes) == 1
        assert r.schedule.get_prompt_for_frame(50) == "only one"

    def test_total_frames_1(self):
        r = parse("[0]\nsingle frame", 1, 24.0)
        assert r.validation.valid
        assert r.schedule.keyframes[0].frame == 0

    def test_very_low_fps(self):
        r = parse("[0]\nhello\n[2s]\nworld", 100, 0.5)
        assert r.schedule is not None
        # 2s * 0.5fps = frame 1
        assert r.schedule.keyframes[1].frame == 1

    def test_multiple_negative_lines(self):
        dsl = "[0]\nprompt\n-- neg line 1\n-- neg line 2\n-- neg line 3"
        r = parse(dsl, 100, 24.0)
        neg = r.schedule.keyframes[0].negative_prompt
        assert "neg line 1" in neg
        assert "neg line 2" in neg
        assert "neg line 3" in neg

    def test_default_prompt_fallback(self):
        r = parse("[5]\nlater", 100, 24.0, default_prompt="fallback")
        # Frame 0 should fall back to default
        assert r.schedule.get_prompt_for_frame(0) == "fallback"
        assert r.schedule.get_prompt_for_frame(5) == "later"

    def test_directive_like_treated_as_prompt(self):
        """A line that looks like a directive but isn't recognized should
        emit W007 but still be added as prompt text."""
        r = parse("[0]\ncolor: red\nhello", 100, 24.0)
        assert "W007" in _warning_codes(r)
        assert "color: red" in r.schedule.keyframes[0].prompt

    def test_crlf_line_endings(self):
        dsl = "[0]\r\nhello\r\n[10]\r\nworld\r\n"
        r = parse(dsl, 100, 24.0)
        assert r.validation.valid
        assert len(r.schedule.keyframes) == 2

    def test_many_keyframes_valid(self):
        """Stress test: many valid keyframes in order."""
        lines = []
        for i in range(0, 200, 2):
            lines.append(f"[{i}]")
            lines.append(f"prompt {i}")
        dsl = "\n".join(lines)
        r = parse(dsl, 500, 24.0)
        assert r.validation.valid
        assert len(r.schedule.keyframes) == 100


# ─────────────────────────────────────────────────────────────
# Presets integration (v2 ratio-based)
# ─────────────────────────────────────────────────────────────


class TestPresetsV2:
    def test_resolve_ratio_keyframes(self):
        from sddj.prompt_schedule_presets import resolve_preset_keyframes

        preset = {
            "keyframe_ratios": [
                {"ratio": 0.0, "prompt": "a", "transition": "hard_cut"},
                {"ratio": 0.5, "prompt": "b", "transition": "blend", "blend_ratio": 0.1},
            ],
        }
        kfs = resolve_preset_keyframes(preset, 100)
        assert kfs[0]["frame"] == 0
        assert kfs[1]["frame"] == 50
        assert kfs[1]["transition_frames"] == 10  # 0.1 * 100

    def test_resolve_handles_legacy_v1(self):
        from sddj.prompt_schedule_presets import resolve_preset_keyframes

        preset = {
            "keyframes": [
                {"frame": 0, "prompt": "a"},
                {"frame": 50, "prompt": "b"},
            ],
        }
        kfs = resolve_preset_keyframes(preset, 100)
        assert kfs[0]["frame"] == 0
        assert kfs[1]["frame"] == 50

    def test_builtin_presets_have_ratios(self):
        from sddj.prompt_schedule_presets import _BUILTIN_PRESETS

        for name, preset in _BUILTIN_PRESETS.items():
            assert "keyframe_ratios" in preset, f"Builtin {name} missing keyframe_ratios"
            assert preset.get("version") == 2, f"Builtin {name} not version 2"

    def test_get_preset_resolved(self, tmp_path):
        from sddj.prompt_schedule_presets import PromptSchedulePresetsManager

        mgr = PromptSchedulePresetsManager(tmp_path / "presets")
        data = mgr.get_preset_resolved("evolving_3act", 300)
        assert "keyframes" in data
        for kf in data["keyframes"]:
            assert "frame" in kf
            assert 0 <= kf["frame"] < 300
