"""Shared test fixtures for PixyToon server tests."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import pytest


@pytest.fixture
def tmp_presets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "presets"
    d.mkdir()
    # Copy default presets for realistic testing
    src = Path(__file__).resolve().parent.parent / "presets"
    if src.is_dir():
        for f in src.glob("*.json"):
            shutil.copy(f, d / f.name)
    else:
        warnings.warn("presets/ directory not found — tests may use empty fixtures")
    return d


@pytest.fixture
def tmp_prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    # Copy actual prompt data
    src = Path(__file__).resolve().parent.parent / "data" / "prompts"
    if src.is_dir():
        for f in src.glob("*.json"):
            shutil.copy(f, d / f.name)
    else:
        warnings.warn("data/prompts/ directory not found — tests may use empty fixtures")
    return d


@pytest.fixture
def empty_presets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "empty_presets"
    d.mkdir()
    return d


@pytest.fixture
def empty_prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "empty_prompts"
    d.mkdir()
    return d


@pytest.fixture
def sample_preset_data() -> dict:
    return {
        "prompt_prefix": "pixel art, sharp pixels",
        "negative_prompt": "blurry, smooth",
        "mode": "txt2img",
        "width": 512,
        "height": 512,
        "steps": 8,
        "cfg_scale": 5.0,
        "clip_skip": 2,
        "denoise_strength": 1.0,
        "post_process": {
            "pixelate": {"enabled": True, "target_size": 128},
            "quantize_method": "kmeans",
            "quantize_colors": 32,
            "dither": "none",
            "palette": {"mode": "auto"},
            "remove_bg": False,
        },
    }


@pytest.fixture
def tmp_palettes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "palettes"
    d.mkdir()
    # Write a test palette
    pico8 = {
        "colors": [
            "#000000", "#1D2B53", "#7E2553", "#008751",
            "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
            "#FF004D", "#FFA300", "#FFEC27", "#00E436",
            "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA",
        ]
    }
    (d / "pico8.json").write_text(json.dumps(pico8))
    return d
