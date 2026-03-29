"""Prompt schedule presets — CRUD manager for saved prompt schedules."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("sddj.prompt_schedule_presets")

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_PRESETS = 50

# ─── Built-in factory presets (structural, no hardcoded prompts) ───

_BUILTIN_PRESETS: dict[str, dict] = {
    "evolving_3act": {
        "name": "evolving_3act",
        "description": "3-act structure: intro, development, climax",
        "version": 2,
        "keyframe_ratios": [
            {"ratio": 0.0, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.33, "prompt": "", "transition": "ease_in_out",
             "blend_ratio": 0.08},
            {"ratio": 0.66, "prompt": "", "transition": "ease_in_out",
             "blend_ratio": 0.08},
        ],
        "auto_fill": True,
        "auto_fill_randomness": 5,
    },
    "style_morph_4": {
        "name": "style_morph_4",
        "description": "4-phase style evolution with blend transitions",
        "version": 2,
        "keyframe_ratios": [
            {"ratio": 0.0, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.25, "prompt": "", "transition": "blend",
             "blend_ratio": 0.10},
            {"ratio": 0.50, "prompt": "", "transition": "blend",
             "blend_ratio": 0.10},
            {"ratio": 0.75, "prompt": "", "transition": "blend",
             "blend_ratio": 0.10},
        ],
        "auto_fill": True,
        "auto_fill_randomness": 8,
    },
    "beat_alternating": {
        "name": "beat_alternating",
        "description": "Rapid A-B alternation (ideal for audio beat sync)",
        "version": 2,
        "keyframe_ratios": [
            {"ratio": 0.0, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.50, "prompt": "", "transition": "hard_cut"},
        ],
        "auto_fill": True,
        "auto_fill_randomness": 10,
    },
    "slow_drift": {
        "name": "slow_drift",
        "description": "Gentle prompt evolution with long blend window",
        "version": 2,
        "keyframe_ratios": [
            {"ratio": 0.0, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.50, "prompt": "", "transition": "blend",
             "blend_ratio": 0.25},
        ],
        "auto_fill": True,
        "auto_fill_randomness": 3,
    },
    "rapid_cuts_6": {
        "name": "rapid_cuts_6",
        "description": "6 rapid hard-cut scene changes",
        "version": 2,
        "keyframe_ratios": [
            {"ratio": 0.0, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.17, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.33, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.50, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.67, "prompt": "", "transition": "hard_cut"},
            {"ratio": 0.83, "prompt": "", "transition": "hard_cut"},
        ],
        "auto_fill": True,
        "auto_fill_randomness": 15,
    },
}


def resolve_preset_keyframes(
    preset: dict, total_frames: int,
) -> list[dict]:
    """Resolve ratio-based preset keyframes to absolute frame indices.

    Handles both v2 (keyframe_ratios) and legacy v1 (keyframes) formats.
    """
    if "keyframe_ratios" in preset:
        keyframes = []
        for kr in preset["keyframe_ratios"]:
            kf = dict(kr)
            ratio = kf.pop("ratio", 0.0)
            blend_ratio = kf.pop("blend_ratio", None)
            kf["frame"] = min(total_frames - 1, max(0, round(ratio * total_frames)))
            if blend_ratio is not None:
                kf["transition_frames"] = max(1, round(blend_ratio * total_frames))
            keyframes.append(kf)
        return keyframes
    # Legacy v1: return as-is
    return list(preset.get("keyframes", []))


class PromptSchedulePresetsManager:
    """CRUD manager for prompt schedule presets (JSON files)."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name:
            raise ValueError("Preset name cannot be empty")
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Invalid preset name: {name!r} "
                "(only alphanumeric, underscore, hyphen)"
            )
        if ".." in name or "/" in name or "\\" in name:
            raise ValueError(f"Path traversal rejected: {name!r}")

    def list_presets(self) -> list[str]:
        """Return sorted list of all preset names (builtins + user)."""
        names = set(_BUILTIN_PRESETS.keys())
        if self._dir.is_dir():
            for f in self._dir.glob("*.json"):
                names.add(f.stem)
        return sorted(names)

    def get_preset(self, name: str) -> dict:
        """Load a preset by name. Builtins are returned from memory."""
        self._validate_name(name)
        if name in _BUILTIN_PRESETS:
            return dict(_BUILTIN_PRESETS[name])
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Preset not found: {name!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_preset_resolved(self, name: str, total_frames: int) -> dict:
        """Load a preset and resolve ratio-based keyframes to absolute frames."""
        preset = self.get_preset(name)
        preset["keyframes"] = resolve_preset_keyframes(preset, total_frames)
        return preset

    def save_preset(self, name: str, data: dict) -> None:
        """Save a user preset. Cannot overwrite builtins."""
        self._validate_name(name)
        if name in _BUILTIN_PRESETS:
            raise ValueError(f"Cannot overwrite built-in preset: {name!r}")
        # Count user presets
        user_count = sum(1 for f in self._dir.glob("*.json"))
        path = self._dir / f"{name}.json"
        if not path.exists() and user_count >= _MAX_PRESETS:
            raise ValueError(
                f"Maximum {_MAX_PRESETS} user presets reached"
            )
        data["name"] = name
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Saved prompt schedule preset: %s", name)

    def delete_preset(self, name: str) -> None:
        """Delete a user preset. Cannot delete builtins."""
        self._validate_name(name)
        if name in _BUILTIN_PRESETS:
            raise ValueError(f"Cannot delete built-in preset: {name!r}")
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Preset not found: {name!r}")
        path.unlink()
        log.info("Deleted prompt schedule preset: %s", name)
