"""Data-driven random prompt generator for Stable Diffusion.

Multi-phase composition pipeline with subject type awareness,
generation modes, artist coherence, exclusion filtering,
auto-negative matching, and CLIP token budgeting.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

from .config import settings

log = logging.getLogger("sddj.prompt_generator")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

CLIP_TOKEN_BUDGET = 65  # soft cap (of 75 usable CLIP tokens)
TOKEN_MULTIPLIER = 1.5  # word → CLIP token estimate (raised for technical pixel art vocabulary)

# Categories dropped first when over budget (lowest priority last)
DROP_ORDER = [
    "descriptor", "background", "details", "material",
    "accessory", "outfit", "pose", "colors",
]

# Categories never dropped
CORE_CATEGORIES = frozenset({"subject", "quality", "artist", "style"})


class SubjectType(str, Enum):
    HUMANOID = "humanoid"
    ANIMAL = "animal"
    LANDSCAPE = "landscape"
    OBJECT = "object"
    CONCEPT = "concept"
    ANY = "any"


class Mode(str, Enum):
    STANDARD = "standard"
    ART_FOCUS = "art_focus"
    CHARACTER = "character"
    CHAOS = "chaos"


# Which extra categories are eligible per subject type
_TYPE_ELIGIBLE: dict[SubjectType, frozenset[str]] = {
    SubjectType.HUMANOID: frozenset({"pose", "outfit", "accessory", "background", "material", "descriptor"}),
    SubjectType.ANIMAL: frozenset({"pose", "background", "descriptor"}),
    SubjectType.LANDSCAPE: frozenset({"background", "material", "descriptor"}),
    SubjectType.OBJECT: frozenset({"material", "background", "descriptor"}),
    SubjectType.CONCEPT: frozenset({"descriptor"}),
    SubjectType.ANY: frozenset({"pose", "outfit", "accessory", "background", "material", "descriptor"}),
}

# Base categories always eligible (from original 11 files)
_BASE_CATEGORIES = frozenset({"quality", "style", "artist", "lighting", "mood", "camera", "colors", "details"})

# Mode → category inclusion probabilities (0.0-1.0)
_MODE_PROBS: dict[Mode, dict[str, float]] = {
    Mode.STANDARD: {},  # empty = uniform (all at default prob)
    Mode.ART_FOCUS: {
        "artist": 1.0, "style": 1.0, "quality": 0.9,
        "lighting": 0.4, "mood": 0.3, "camera": 0.2,
        "colors": 0.2, "details": 0.2,
        "pose": 0.0, "outfit": 0.0, "accessory": 0.0,
        "background": 0.1, "material": 0.1, "descriptor": 0.3,
    },
    Mode.CHARACTER: {
        "pose": 1.0, "outfit": 0.8, "accessory": 0.6,
        "artist": 0.7, "style": 0.7, "quality": 0.9,
        "lighting": 0.5, "mood": 0.4, "camera": 0.3,
        "colors": 0.3, "details": 0.3,
        "background": 0.4, "material": 0.1, "descriptor": 0.4,
    },
    Mode.CHAOS: {},  # chaos overrides probability logic entirely
}

# Keyword patterns for subject type inference
_HUMANOID_RE = re.compile(
    r"\b(warrior|knight|princess|prince|queen|king|wizard|witch|mage|sorcerer|sorceress"
    r"|woman|man|girl|boy|lady|lord|maiden|monk|nun|priest|priestess"
    r"|hero|archer|assassin|bard|druid|ranger|rogue|thief|paladin|cleric"
    r"|soldier|captain|admiral|general|commander|pirate|viking|gladiator"
    r"|dancer|musician|painter|baker|chef|blacksmith|alchemist|merchant"
    r"|child|elder|grandmother|grandfather|old man|old woman"
    r"|angel|demon|devil|god|goddess|titan|vampire|zombie|werewolf"
    r"|samurai|ronin|ninja|geisha|shogun"
    r"|explorer|traveler|nomad|wanderer|pilgrim"
    r"|fisherman|farmer|shepherd|hunter|conductor|worker|craftsman|artisan"
    r"|astronaut|diver|couple|person|figure|portrait|jester"
    r"|golem|colossus|automaton|scarecrow|troll|ogre|dwarf|elf|fairy|gnome"
    r"|sphinx|centaur|minotaur|cyclops|harpy|mermaid|siren|nymph|satyr"
    r"|necromancer|warlock|shaman|oracle|seer|prophet"
    r"|dervish|griot|nonna|keeper|weaver|player)\b",
    re.IGNORECASE,
)

_ANIMAL_RE = re.compile(
    r"\b(wolf|fox|bear|deer|stag|horse|cat(?!\w)|dog|lion|tiger|eagle|hawk|falcon|owl"
    r"|dragon|phoenix|griffin|wyvern|serpent|basilisk|hydra|kraken|leviathan"
    r"|whale|shark|fish|octopus|jellyfish|seahorse|turtle|tortoise"
    r"|butterfly|moth|bee|spider|beetle|dragonfly|firefly"
    r"|raven|crow|swan|heron|crane|peacock|parrot|flamingo"
    r"|unicorn|chimera|manticore|cerberus"
    r"|snake|cobra|lizard|chameleon|salamander|frog|toad|axolotl"
    r"|yeti|armadillo|hedgehog|badger|otter|rabbit|hare"
    r"|elephant|rhino|giraffe|gorilla|bat|mice|mouse|rat"
    r"|anglerfish|crab|starfish|coral)\b",
    re.IGNORECASE,
)

_LANDSCAPE_RE = re.compile(
    r"\b(landscape|vista|panorama|forest|jungle|woodland|grove"
    r"|mountain|peak|cliff|canyon|valley|ocean|sea|coast|beach|island"
    r"|desert|dune|oasis|wasteland|tundra|savanna|steppe"
    r"|city|town|village|metropolis|cityscape|skyline|alley|street|plaza"
    r"|castle|fortress|palace|cathedral|temple|ruins|tower"
    r"|cave|cavern|grotto|underground|tunnel"
    r"|garden|meadow|field|vineyard|orchard"
    r"|river|lake|waterfall|stream|swamp|marsh"
    r"|volcano|crater|geyser|aurora|sunset|sunrise|moonlit|eclipse)\b",
    re.IGNORECASE,
)

_OBJECT_RE = re.compile(
    r"\b(sword|dagger|axe|hammer|bow|shield|spear|staff|wand"
    r"|book|tome|scroll|map|manuscript|clock|watch|hourglass|compass|astrolabe"
    r"|potion|flask|bottle|vial|cauldron|chalice|goblet"
    r"|gem|crystal|diamond|ruby|emerald|crown|throne|scepter|medallion|amulet"
    r"|ship|boat|vessel|gondola|telescope|lantern|lamp|candle|chandelier"
    r"|violin|piano|guitar|flute|drum|harp|motorcycle|locomotive"
    r"|painting|statue|sculpture|tapestry|still life|bento)\b",
    re.IGNORECASE,
)

# Pre-compiled regexes for prompt cleanup (used in _assemble / _trim_to_budget)
_DOUBLE_COMMA_RE = re.compile(r",\s*,")
_TRAILING_COMMA_RE = re.compile(r",\s*$")
_LEADING_COMMA_RE = re.compile(r"^\s*,\s*")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_UNSAFE_TEMPLATE_RE = re.compile(r"\{[^}]*[.\[\]!:]")

# Auto-negative matching patterns
_NEG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pixel\s*art", re.IGNORECASE), "pixel_art"),
    (re.compile(r"\banime\b", re.IGNORECASE), "anime"),
    (re.compile(r"\bportrait\b|\bface\b|\bcharacter\b", re.IGNORECASE), "portrait"),
    (re.compile(r"photorealis|photograph|DSLR|RAW photo", re.IGNORECASE), "realistic"),
]


# ─────────────────────────────────────────────────────────────
# PROMPT GENERATOR
# ─────────────────────────────────────────────────────────────

class PromptGenerator:
    """Generates random prompts from JSON category databases."""

    def __init__(self, data_dir: Path) -> None:
        self._data: dict[str, list[str]] = {}
        self._templates: dict[str, str] = {}
        self._negative_sets: dict[str, str] = {}
        self._default_template = "{quality}, {subject}, {style}, {lighting}, {mood}"
        # Subject type pools (from "typed" metadata in subject.json)
        self._typed_subjects: dict[str, list[str]] = {}
        # Artist tag buckets (tag → list of artist names)
        self._artist_buckets: dict[str, list[str]] = defaultdict(list)
        # Active exclusion set (per-call, cleared between generates)
        self._active_exclude: set[str] = set()
        # Cached template key list (rebuilt when _templates changes)
        self._template_keys: list[str] = []
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
                    self._template_keys = list(self._templates.keys())
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
                    # Subject type metadata
                    if category == "subject":
                        typed = data.get("typed", {})
                        for stype, sitems in typed.items():
                            if sitems:
                                self._typed_subjects[stype] = sitems
                        if self._typed_subjects:
                            log.info(
                                "Loaded typed subjects: %s",
                                {k: len(v) for k, v in self._typed_subjects.items()},
                            )
                    # Artist tag metadata
                    if category == "artist":
                        tagged = data.get("tagged", {})
                        for artist_name, tags in tagged.items():
                            for tag in tags:
                                self._artist_buckets[tag].append(artist_name)
                        if self._artist_buckets:
                            log.info(
                                "Loaded artist tags: %d artists across %d categories",
                                len(tagged), len(self._artist_buckets),
                            )
            except Exception as e:
                log.warning("Failed to load prompt data '%s': %s", json_file.name, e)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        locked: Optional[dict[str, str]] = None,
        template: Optional[str] = None,
        negative_set: Optional[str] = "universal",
        randomness: int = 0,
        # ── New optional params (backward compat: all default None) ──
        subject_type: Optional[str] = None,
        mode: Optional[str] = None,
        exclude: Optional[list[str]] = None,
    ) -> tuple[str, str, dict[str, str]]:
        """Generate a random prompt.

        Args:
            locked: Category values to keep fixed (e.g. {"style": "pixel art"}).
            template: Custom template string with {category} placeholders.
            negative_set: Name of the negative prompt set to use (default: "universal").
            randomness: Diversity level 0-20 (0=standard, 5=subtle, 10=moderate, 15=wild, 20=chaos).
            subject_type: Filter by subject type ("humanoid"/"animal"/"landscape"/"object"/"concept"/None).
            mode: Generation mode ("standard"/"art_focus"/"character"/"chaos"/None).
            exclude: Terms to exclude from all category pools.

        Returns:
            Tuple of (prompt_string, negative_prompt_string, components_dict).
        """
        locked = locked or {}
        randomness = max(0, min(20, randomness))
        self._active_exclude = set()  # reset per-call state

        # Resolve mode
        gen_mode = self._resolve_mode(mode, randomness)

        # Character mode forces humanoid subject type
        if gen_mode == Mode.CHARACTER and subject_type is None:
            subject_type = SubjectType.HUMANOID.value

        # Phase 1: Apply exclusion filter
        data = self._apply_exclusions(exclude)

        # Phase 2: Select subject
        subject_text = self._pick_subject(locked, data, subject_type, randomness)

        # Phase 3: Infer subject type
        stype = self._infer_type(subject_text, subject_type)

        # Phase 4: Determine eligible categories
        eligible = self._eligible_categories(stype)

        # Phase 5: Roll and pick components
        components = self._pick_components(
            locked, data, eligible, randomness, gen_mode, subject_text,
        )

        # Phase 6: Pick artist (coherence-aware)
        if "artist" not in locked:
            artist = self._pick_artist(
                data, components, randomness, gen_mode,
            )
            if artist:
                components["artist"] = artist

        # Phase 7: Template assembly
        prompt = self._assemble(components, template, randomness, gen_mode)

        # Phase 8: Token budget trimming
        prompt = self._trim_to_budget(prompt, components)

        # Phase 9: Auto-negative matching
        negative = self._resolve_negative(negative_set, prompt, components)

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

    # ─────────────────────────────────────────────────────────
    # PRIVATE: Pipeline phases
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_mode(mode: Optional[str], randomness: int) -> Mode:
        """Resolve generation mode from string or randomness level."""
        if mode:
            try:
                return Mode(mode)
            except ValueError:
                pass
        # Auto-select chaos at very high randomness
        if randomness >= 18:
            return Mode.CHAOS
        return Mode.STANDARD

    def _apply_exclusions(self, exclude: Optional[list[str]]) -> dict[str, list[str]]:
        """Return (possibly filtered) category data. Never mutates self._data."""
        if not exclude:
            return self._data
        self._active_exclude = {e.strip().lower() for e in exclude if e.strip()}
        if not self._active_exclude:
            return self._data
        return {
            cat: [item for item in items if not any(ex in item.lower() for ex in self._active_exclude)]
            for cat, items in self._data.items()
        }

    def _pick_subject(
        self,
        locked: dict[str, str],
        data: dict[str, list[str]],
        subject_type: Optional[str],
        randomness: int,
    ) -> str:
        """Select a subject, respecting locks and type filters."""
        if "subject" in locked:
            return locked["subject"]

        # Try typed pool first
        if subject_type and subject_type != "any" and subject_type in self._typed_subjects:
            pool = self._typed_subjects[subject_type]
            # Apply active exclusions to typed pool too
            if self._active_exclude:
                pool = [s for s in pool if not any(ex in s.lower() for ex in self._active_exclude)]
            if pool:
                return self._pick_from_pool(pool, randomness)

        # Fall back to flat subject list
        pool = data.get("subject", [])
        return self._pick_from_pool(pool, randomness) if pool else ""

    @staticmethod
    def _pick_from_pool(pool: list[str], randomness: int) -> str:
        """Pick from a pool using randomness-influenced selection.

        Gradient: 0=uniform, 1-5=favor front 2/3, 6-10=full uniform,
        11-15=favor latter half (rarer), 16-20=combine 2 items (chaos).
        """
        if not pool:
            return ""
        if randomness >= 16 and len(pool) > 1:
            # Chaos: combine 2 items
            picks = random.sample(pool, min(2, len(pool)))
            return ", ".join(picks)
        if randomness >= 11 and len(pool) > 2:
            # Wild: favor latter half (rarer items)
            half = len(pool) // 2
            return random.choice(pool[half:])
        if 1 <= randomness <= 5 and len(pool) > 3:
            # Low: favor front 2/3 (common items) — subtly narrower than full uniform
            window = max(2, len(pool) * 2 // 3)
            return random.choice(pool[:window])
        # 0 or 6-10: full uniform
        return random.choice(pool)

    @staticmethod
    def _infer_type(subject_text: str, explicit_type: Optional[str]) -> SubjectType:
        """Infer subject type from text or explicit type."""
        if explicit_type and explicit_type != "any":
            try:
                return SubjectType(explicit_type)
            except ValueError:
                pass

        if not subject_text:
            return SubjectType.ANY

        # Concept patterns checked first (abstract/meta)
        lower = subject_text.lower()
        if any(kw in lower for kw in (
            "the duality", "the passage", "allegory", "personified",
            "abstract", "visualized", "magnified", "neurons",
            "atoms", "infinity", "reality dissolving",
        )):
            return SubjectType.CONCEPT

        if _HUMANOID_RE.search(subject_text):
            return SubjectType.HUMANOID
        if _ANIMAL_RE.search(subject_text):
            return SubjectType.ANIMAL
        if _LANDSCAPE_RE.search(subject_text):
            return SubjectType.LANDSCAPE
        if _OBJECT_RE.search(subject_text):
            return SubjectType.OBJECT

        return SubjectType.CONCEPT  # unclassified → minimal scaffolding

    @staticmethod
    def _eligible_categories(stype: SubjectType) -> frozenset[str]:
        """Return the set of eligible categories for a subject type."""
        extra = _TYPE_ELIGIBLE.get(stype, frozenset())
        return _BASE_CATEGORIES | extra

    def _pick_components(
        self,
        locked: dict[str, str],
        data: dict[str, list[str]],
        eligible: frozenset[str],
        randomness: int,
        gen_mode: Mode,
        subject_text: str,
    ) -> dict[str, str]:
        """Pick items for each eligible category."""
        components: dict[str, str] = {}
        if subject_text:
            components["subject"] = subject_text
        mode_probs = _MODE_PROBS.get(gen_mode, {})

        for category, items in data.items():
            if category == "subject":
                continue  # already handled
            if category == "artist":
                continue  # handled separately

            if category in locked:
                components[category] = locked[category]
                continue

            # Check eligibility
            if category not in eligible and category in _TYPE_ELIGIBLE.get(SubjectType.ANY, frozenset()):
                continue  # type-specific category not eligible

            # Mode probability check — randomness boosts inclusion in standard mode
            if gen_mode != Mode.CHAOS:
                if mode_probs:
                    prob = mode_probs.get(category, 0.5)
                    if prob <= 0.0:
                        continue
                    if prob < 1.0 and random.random() > prob:
                        continue
                elif randomness > 0:
                    # Standard mode with randomness: base 0.55, scaled up
                    # 1→0.575, 5→0.675, 10→0.80, 15→0.925, 20→1.0 (always)
                    prob = 0.55 + randomness * 0.025
                    if prob < 1.0 and random.random() > prob:
                        continue
                # else: standard mode, randomness=0 → include all (original behavior)

            # Randomness-based selection
            components[category] = self._pick_from_pool(items, randomness)

        return components

    def _pick_artist(
        self,
        data: dict[str, list[str]],
        components: dict[str, str],
        randomness: int,
        gen_mode: Mode,
    ) -> str:
        """Pick artist(s) with coherence awareness."""
        all_artists = data.get("artist", [])
        if not all_artists:
            return ""

        # Determine artist count based on randomness
        if randomness >= 16:
            count = random.choice([1, 2, 2, 3])  # 1-3, weighted toward 2
        elif randomness >= 11:
            count = random.choice([1, 1, 2])  # 1-2
        else:
            count = 1

        picked: list[str] = []
        for _ in range(count):
            artist = self._pick_single_artist(all_artists, components, randomness, gen_mode)
            if artist and artist not in picked:
                picked.append(artist)

        return ", ".join(picked)

    def _pick_single_artist(
        self,
        all_artists: list[str],
        components: dict[str, str],
        randomness: int,
        gen_mode: Mode,
    ) -> str:
        """Pick a single artist with tag-based coherence."""
        # Low randomness: prefer popular artists, with gradual broadening
        if randomness <= 5 and "popular" in self._artist_buckets:
            pool = self._artist_buckets["popular"]
            if pool:
                # At 0: always popular. At 1-5: increasing chance to skip popular
                # and fall through to style-match or fully random below.
                if randomness == 0 or random.random() > randomness * 0.1:
                    return random.choice(pool)

        # Medium randomness: try to match style/mood tags
        if randomness <= 15 and self._artist_buckets:
            style_words = set(components.get("style", "").lower().split())
            mood_words = set(components.get("mood", "").lower().split())
            check_words = style_words | mood_words

            # Try to find artists matching style/mood keywords
            matching_buckets: list[list[str]] = []
            for tag, bucket in self._artist_buckets.items():
                if tag == "popular":
                    continue
                if tag in check_words and bucket:
                    matching_buckets.append(bucket)
            if matching_buckets:
                # Pick from a random matching bucket
                return random.choice(random.choice(matching_buckets))

        # High randomness or no tag match: fully random
        return random.choice(all_artists) if all_artists else ""

    def _assemble(
        self,
        components: dict[str, str],
        template: Optional[str],
        randomness: int,
        gen_mode: Mode,
    ) -> str:
        """Assemble prompt from components using template."""
        if template is None:
            if gen_mode == Mode.CHAOS and self._templates:
                # Chaos: random template
                template = self._templates[random.choice(self._template_keys)]
            elif gen_mode == Mode.CHARACTER and "character" in self._templates:
                template = self._templates["character"]
            elif gen_mode == Mode.ART_FOCUS and "surrealist" in self._templates:
                template = self._templates["surrealist"]
            elif randomness >= 8 and self._templates:
                # 8-10: occasional random template (~15-45% chance)
                # 11+: always random template
                if randomness >= 11 or random.random() < (randomness - 7) * 0.15:
                    template = self._templates[random.choice(self._template_keys)]
                else:
                    template = self._default_template
            else:
                template = self._default_template

        # Security: reject attribute/index access, excessively long templates,
        # and format spec abuse (e.g. {foo!r}, {foo:>100})
        _MAX_TEMPLATE_LEN = 2000
        if len(template) > _MAX_TEMPLATE_LEN:
            log.warning("Template too long (%d chars), using default", len(template))
            template = self._default_template
        if _UNSAFE_TEMPLATE_RE.search(template):
            log.warning("Template rejected (unsafe pattern): %s", template[:80])
            template = self._default_template

        # Use defaultdict to handle missing categories gracefully
        safe_components: dict[str, str] = defaultdict(str, components)

        try:
            prompt = template.format_map(safe_components)
        except (KeyError, ValueError) as e:
            log.warning("Template error %s, using fallback join", e)
            prompt = ", ".join(v for v in components.values() if v)

        # Clean up empty placeholders (", , " → ", ")
        prompt = _DOUBLE_COMMA_RE.sub(",", prompt)
        prompt = _TRAILING_COMMA_RE.sub("", prompt)
        prompt = _LEADING_COMMA_RE.sub("", prompt)
        prompt = _MULTI_SPACE_RE.sub(" ", prompt).strip()

        return prompt

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate CLIP token count from text."""
        return max(1, int(len(text.split()) * TOKEN_MULTIPLIER))

    def _trim_to_budget(self, prompt: str, components: dict[str, str]) -> str:
        """Trim prompt if it exceeds CLIP token budget."""
        if self._estimate_tokens(prompt) <= CLIP_TOKEN_BUDGET:
            return prompt

        # Rebuild prompt by dropping low-priority categories
        for drop_cat in DROP_ORDER:
            if drop_cat in components and drop_cat not in CORE_CATEGORIES:
                val = components[drop_cat]
                if val:
                    # Use regex-escaped value to avoid regex special char issues
                    escaped = re.escape(val)
                    # Remove the value with surrounding commas/spaces
                    prompt = re.sub(r",?\s*" + escaped + r"\s*,?", ",", prompt)
                    prompt = _DOUBLE_COMMA_RE.sub(",", prompt)
                    prompt = _TRAILING_COMMA_RE.sub("", prompt)
                    prompt = _LEADING_COMMA_RE.sub("", prompt).strip()
                    if self._estimate_tokens(prompt) <= CLIP_TOKEN_BUDGET:
                        break

        return prompt

    def _resolve_negative(
        self,
        negative_set: Optional[str],
        prompt: str,
        components: dict[str, str],
    ) -> str:
        """Resolve negative prompt, with auto-matching if set is 'universal'."""
        # Explicit set requested (not default)
        if negative_set and negative_set != "universal":
            return self._negative_sets.get(negative_set, "")

        # Auto-match based on style/prompt content
        style = components.get("style", "")
        check_text = f"{style} {prompt}"

        for pattern, neg_name in _NEG_PATTERNS:
            if pattern.search(check_text):
                neg = self._negative_sets.get(neg_name)
                if neg:
                    return neg

        # Fallback: universal
        return self._negative_sets.get("universal", "")


_prompt_generator: PromptGenerator | None = None


def get_prompt_generator() -> PromptGenerator:
    global _prompt_generator
    if _prompt_generator is None:
        _prompt_generator = PromptGenerator(settings.prompts_data_dir)
    return _prompt_generator


def __getattr__(name: str):
    """Module-level lazy accessor.

    Allows ``from .prompt_generator import prompt_generator`` to work without
    eagerly instantiating the PromptGenerator at import time.  For explicit
    access, prefer ``get_prompt_generator()`` which is discoverable via
    standard tooling (IDE autocomplete, help(), dir()).
    """
    if name == "prompt_generator":
        return get_prompt_generator()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
