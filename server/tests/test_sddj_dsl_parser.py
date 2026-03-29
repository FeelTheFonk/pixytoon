"""Tests for the Lua DSL parser (requires lupa).

These tests are automatically skipped when the ``lupa`` package is not
installed.  Install it with ``pip install lupa`` to enable them.
"""

from __future__ import annotations

import os

import pytest

lupa = pytest.importorskip("lupa", reason="lupa not installed — skipping Lua parser tests")


@pytest.fixture(scope="module")
def lua_parser():
    """Load the Lua DSL parser module via lupa."""
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    parser_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "extension", "scripts", "sddj_dsl_parser.lua",
    )
    parser_path = os.path.abspath(parser_path)
    with open(parser_path, "r", encoding="utf-8") as f:
        lua_code = f.read()
    return lua.execute(lua_code)


class TestLuaParserBasic:
    def test_empty_string(self, lua_parser):
        res = lua_parser.parse("", 100, 24)
        assert len(res.keyframes) == 0

    def test_whitespace_string(self, lua_parser):
        res = lua_parser.parse("   \n \t  ", 100, 24)
        assert len(res.keyframes) == 0

    def test_auto_tag(self, lua_parser):
        res = lua_parser.parse("{auto}", 100, 24)
        assert res.auto_fill
        assert len(res.keyframes) == 1

    def test_auto_tag_case_insensitive(self, lua_parser):
        res = lua_parser.parse("{AUTO}", 100, 24)
        assert res.auto_fill

    def test_auto_tag_with_keyframes(self, lua_parser):
        res = lua_parser.parse("{auto}\n[10] hello", 100, 24)
        assert res.auto_fill
        assert len(res.keyframes) == 1
        assert res.keyframes[1].frame == 10
        assert res.keyframes[1].prompt == "hello"


class TestLuaParserTimeFormats:
    def test_absolute_frame(self, lua_parser):
        res = lua_parser.parse("[10] absolute", 100, 24)
        assert res.keyframes[1].frame == 10

    def test_percentage(self, lua_parser):
        res = lua_parser.parse("[50%] percent", 100, 24)
        assert res.keyframes[1].frame == 50

    def test_seconds(self, lua_parser):
        res = lua_parser.parse("[2s] seconds", 100, 24)
        assert res.keyframes[1].frame == 48  # 2 * 24fps


class TestLuaParserDirectives:
    def test_negative_prompts(self, lua_parser):
        dsl = "[0] a beautiful cat\n-- ugly, blurry\n-- worst quality"
        res = lua_parser.parse(dsl, 100, 24)
        assert res.keyframes[1].prompt == "a beautiful cat"
        neg = res.keyframes[1].negative_prompt
        assert "ugly" in neg
        assert "worst quality" in neg

    def test_weights_and_transitions(self, lua_parser):
        dsl = "[0]\nweight: 1.5\ntransition: blend\nblend: 12\na glowing orb"
        res = lua_parser.parse(dsl, 100, 24)
        kf = res.keyframes[1]
        assert kf.weight == 1.5
        assert kf.transition == "blend"
        assert kf.transition_frames == 12
        assert kf.prompt == "a glowing orb"

    def test_file_redirect_missing(self, lua_parser):
        res = lua_parser.parse("file: missing_file.txt", 100, 24)
        assert len(res.keyframes) == 0
