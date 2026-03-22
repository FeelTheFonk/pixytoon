"""Data-driven random prompt generator for Stable Diffusion."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from .config import settings

log = logging.getLogger("sddj.prompt_generator")


class PromptGenerator:
    """Generates random prompts from JSON category databases."""

    def __init__(self, data_dir: Path) -> None:
        self._data: dict[str, list[str]] = {}
        self._templates: dict[str, str] = {}
        self._negative_sets: dict[str, str] = {}
        self._default_template = "{quality}, {subject}, {style}, {lighting}, {mood}"
        self._load_data(data_dir)

    def _load_data(self, data_dir: Path) -> None:
        if not data_dir.is_dir():
            log.warning("Prompt data directory not found: %s", data_dir)
            return
        for json_file in sorted(data_dir.glob("*.json")):
            category = json_file.stem
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if category == "templates":
                    self._templates = data.get("templates", {})
                    if "default" in self._templates:
                        self._default_template = self._templates["default"]
                    log.info("Loaded %d prompt templates", len(self._templates))
                elif category == "negatives":
                    self._negative_sets = data.get("sets", {})
                    log.info("Loaded %d negative prompt sets", len(self._negative_sets))
                else:
                    items = data.get("items", [])
                    if items:
                        self._data[category] = items
                        log.info("Loaded %d items for category '%s'", len(items), category)
            except Exception as e:
                log.warning("Failed to load prompt data '%s': %s", json_file.name, e)

    def generate(
        self,
        locked: Optional[dict[str, str]] = None,
        template: Optional[str] = None,
        negative_set: Optional[str] = "universal",
        randomness: int = 0,
    ) -> tuple[str, str, dict[str, str]]:
        """Generate a random prompt.

        Args:
            locked: Category values to keep fixed (e.g. {"style": "pixel art"}).
            template: Custom template string with {category} placeholders.
            negative_set: Name of the negative prompt set to use (default: "universal").
            randomness: Diversity level 0-20 (0=standard, 5=subtle, 10=moderate, 15=wild, 20=chaos).

        Returns:
            Tuple of (prompt_string, negative_prompt_string, components_dict).
        """
        locked = locked or {}
        randomness = max(0, min(20, randomness))
        components: dict[str, str] = {}

        for category, items in self._data.items():
            if category in locked:
                components[category] = locked[category]
            elif randomness >= 16 and len(items) > 1:
                # Chaos: combine 2 items from the same category
                picks = random.sample(items, min(2, len(items)))
                components[category] = ", ".join(picks)
            elif randomness >= 11 and len(items) > 2:
                # Wild: favor rare items (latter half of the list)
                half = len(items) // 2
                pool = items[half:]
                components[category] = random.choice(pool)
            else:
                components[category] = random.choice(items)

        # Template selection influenced by randomness
        if template is None:
            if randomness >= 11 and self._templates:
                template_names = list(self._templates.keys())
                template = self._templates[random.choice(template_names)]
            else:
                template = self._default_template

        # Sanitize template: reject attribute/index access patterns (security)
        if re.search(r"\{[^}]*[.\[\]]", template):
            log.warning("Template rejected (unsafe pattern): %s", template[:80])
            template = self._default_template

        # Only include categories that exist in the template
        try:
            prompt = template.format_map(components)
        except KeyError as e:
            log.warning("Template key %s not found, using fallback join", e)
            # Fallback: join all components
            prompt = ", ".join(v for v in components.values() if v)

        negative = self._negative_sets.get(negative_set or "universal", "")

        return prompt, negative, components

    def list_categories(self) -> list[str]:
        """Return sorted list of available categories."""
        return sorted(self._data.keys())

    def list_templates(self) -> dict[str, str]:
        """Return available named templates."""
        return dict(self._templates)

    def list_negative_sets(self) -> list[str]:
        """Return sorted list of available negative prompt sets."""
        return sorted(self._negative_sets.keys())

    def get_category_items(self, category: str) -> list[str]:
        """Return items for a specific category."""
        return list(self._data.get(category, []))


# Module-level singleton (lazy — created on first import)
prompt_generator = PromptGenerator(settings.prompts_data_dir)
