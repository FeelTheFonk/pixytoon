"""Tests for the data-driven prompt generator."""

from __future__ import annotations

import json
from pathlib import Path


from sddj.prompt_generator import PromptGenerator


class TestPromptGenerator:
    def test_load_from_data_dir(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        cats = gen.list_categories()
        assert len(cats) > 0
        assert "subject" in cats

    def test_generate_returns_tuple(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        result = gen.generate()
        assert len(result) == 3
        prompt, negative, components = result
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert isinstance(negative, str)
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_locked_fields(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        locked = {"style": "pixel art"}
        _, _, components = gen.generate(locked=locked)
        assert components.get("style") == "pixel art"

    def test_custom_template(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        template = "{subject} in {style}"
        prompt, _, _ = gen.generate(template=template)
        assert " in " in prompt

    def test_empty_data_dir(self, empty_prompts_dir: Path):
        gen = PromptGenerator(empty_prompts_dir)
        assert gen.list_categories() == []
        prompt, negative, components = gen.generate()
        assert prompt == ""
        assert negative == ""
        assert components == {}

    def test_nonexistent_dir(self, tmp_path: Path):
        gen = PromptGenerator(tmp_path / "nonexistent")
        assert gen.list_categories() == []

    def test_list_templates(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        templates = gen.list_templates()
        assert isinstance(templates, dict)

    def test_get_category_items(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        items = gen.get_category_items("subject")
        assert isinstance(items, list)
        assert len(items) > 0

    def test_get_nonexistent_category(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        items = gen.get_category_items("nonexistent")
        assert items == []

    def test_multiple_generates_vary(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        prompts = set()
        for _ in range(20):
            p, _, _ = gen.generate()
            prompts.add(p)
        # With randomization, we should get multiple unique prompts
        assert len(prompts) > 1

    def test_invalid_json_file(self, tmp_path: Path):
        d = tmp_path / "bad_prompts"
        d.mkdir()
        (d / "broken.json").write_text("not json{{{")
        gen = PromptGenerator(d)
        assert gen.list_categories() == []

    def test_json_without_items_key(self, tmp_path: Path):
        d = tmp_path / "no_items"
        d.mkdir()
        (d / "test.json").write_text(json.dumps({"other": "data"}))
        gen = PromptGenerator(d)
        assert gen.list_categories() == []

    def test_template_with_missing_category(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        template = "{nonexistent_category}"
        prompt, _, _ = gen.generate(template=template)
        # Should fallback to joining components
        assert isinstance(prompt, str)


class TestRandomness:
    """v0.7.7: randomness parameter in generate()."""

    def test_zero_uses_standard_selection(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        prompt, neg, comp = gen.generate(randomness=0)
        assert isinstance(prompt, str) and len(prompt) > 0
        # Each component must be a non-empty string (standard selection, not chaos-combined)
        for cat, val in comp.items():
            assert isinstance(val, str) and len(val) > 0

    def test_chaos_mode_combines_items(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        # Run multiple times — chaos (>=16) should combine 2 items with comma
        found_combo = False
        for _ in range(30):
            _, _, comp = gen.generate(randomness=20)
            for cat, val in comp.items():
                if ", " in val:
                    found_combo = True
                    break
            if found_combo:
                break
        assert found_combo, "Chaos mode should combine multiple items in at least one category"

    def test_wild_mode_returns_valid(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        prompt, neg, comp = gen.generate(randomness=15)
        assert isinstance(prompt, str) and len(prompt) > 0
        assert len(comp) > 0

    def test_clamped_above_20(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        # Should not raise, just clamp
        prompt, _, _ = gen.generate(randomness=50)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_clamped_below_zero(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        prompt, _, _ = gen.generate(randomness=-10)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_explicit_template_overrides_randomness(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        template = "{subject}"
        prompt, _, comp = gen.generate(template=template, randomness=20)
        # The explicit template should be used, not a random one
        assert prompt == comp.get("subject", "") or ", " in prompt  # chaos may combine subject

    def test_backward_compat_no_randomness(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        # Calling without randomness should work (default=0)
        prompt, neg, comp = gen.generate()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    # ── New categories ──

    def test_colors_category_loaded(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        items = gen.get_category_items("colors")
        assert isinstance(items, list)
        assert len(items) > 0

    def test_details_category_loaded(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        items = gen.get_category_items("details")
        assert isinstance(items, list)
        assert len(items) > 0

    # ── Negative prompts ──

    def test_negative_prompt_default(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        _, negative, _ = gen.generate()
        assert isinstance(negative, str)
        # Auto-matching may pick a specialized set (pixel_art, anime, etc.)
        # — just verify a non-empty negative is always returned.
        assert len(negative) > 0

    def test_negative_prompt_named_set(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        _, negative, _ = gen.generate(negative_set="character")
        assert "bad anatomy" in negative

    def test_negative_prompt_unknown_set(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        _, negative, _ = gen.generate(negative_set="nonexistent")
        assert negative == ""

    def test_list_negative_sets(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        sets = gen.list_negative_sets()
        assert isinstance(sets, list)
        assert "universal" in sets

    def test_negatives_not_in_positive_categories(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        cats = gen.list_categories()
        assert "negatives" not in cats

    # ── Rich template ──

    def test_rich_template(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        templates = gen.list_templates()
        assert "rich" in templates
        prompt, _, components = gen.generate(template=templates["rich"])
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Rich template should use colors and details
        assert "colors" in components
        assert "details" in components


class TestRandomnessGranularity:
    """Verify the slider produces observable changes at every 5-step increment."""

    def test_each_level_produces_valid_output(self, tmp_prompts_dir: Path):
        """Every randomness level 0-20 produces a valid non-empty prompt."""
        gen = PromptGenerator(tmp_prompts_dir)
        for r in range(21):
            prompt, neg, comp = gen.generate(randomness=r)
            assert isinstance(prompt, str) and len(prompt) > 0, f"Failed at randomness={r}"
            assert isinstance(comp, dict) and len(comp) > 0, f"Empty components at randomness={r}"

    def test_high_randomness_includes_more_categories(self, tmp_prompts_dir: Path):
        """Randomness 15 should include more categories on average than randomness 1."""
        gen = PromptGenerator(tmp_prompts_dir)
        counts_1 = []
        counts_15 = []
        for _ in range(80):
            _, _, c1 = gen.generate(randomness=1)
            counts_1.append(len(c1))
            _, _, c15 = gen.generate(randomness=15)
            counts_15.append(len(c15))
        avg_1 = sum(counts_1) / len(counts_1)
        avg_15 = sum(counts_15) / len(counts_15)
        # At randomness=15, probability is ~0.925 vs ~0.575 at randomness=1
        assert avg_15 > avg_1, f"Expected more categories at r=15 ({avg_15:.1f}) vs r=1 ({avg_1:.1f})"

    def test_moderate_randomness_broader_artists(self, tmp_prompts_dir: Path):
        """Randomness 8 should pick from a wider artist pool than randomness 0."""
        gen = PromptGenerator(tmp_prompts_dir)
        artists_0 = set()
        artists_8 = set()
        for _ in range(60):
            _, _, c0 = gen.generate(randomness=0)
            if "artist" in c0:
                artists_0.add(c0["artist"])
            _, _, c8 = gen.generate(randomness=8)
            if "artist" in c8:
                artists_8.add(c8["artist"])
        # Randomness 8 uses style-match + full random; 0 uses only popular bucket
        assert len(artists_8) >= len(artists_0), \
            f"Expected broader artists at r=8 ({len(artists_8)}) vs r=0 ({len(artists_0)})"

