from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestSettingsDefaults:

    def _make_settings(self, **overrides):
        # Import inside to avoid module-level singleton side-effects
        from sddj.config import Settings

        # Clear any SDDJ_ env vars that could leak into tests
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        with patch.dict(os.environ, env, clear=True):
            return Settings(_env_file=None, **overrides)

    def test_network_defaults(self):
        s = self._make_settings()
        assert s.host == "127.0.0.1"
        assert s.port == 9876

    def test_port_bounds(self):
        self._make_settings(port=1)
        self._make_settings(port=65535)
        with pytest.raises(ValidationError):
            self._make_settings(port=0)
        with pytest.raises(ValidationError):
            self._make_settings(port=65536)

    def test_path_defaults_are_paths(self):
        s = self._make_settings()
        for attr in ("models_dir", "checkpoints_dir", "loras_dir",
                     "embeddings_dir", "palettes_dir", "presets_dir",
                     "prompts_data_dir"):
            assert isinstance(getattr(s, attr), Path)

    def test_default_checkpoint(self):
        s = self._make_settings()
        assert s.default_checkpoint == "models/checkpoints/liberteRedmond_v10.safetensors"

    def test_hyper_sd_defaults(self):
        s = self._make_settings()
        assert s.hyper_sd_repo == "ByteDance/Hyper-SD"
        assert s.hyper_sd_fuse_scale == 0.8

    def test_hyper_sd_fuse_scale_bounds(self):
        self._make_settings(hyper_sd_fuse_scale=0.0)
        self._make_settings(hyper_sd_fuse_scale=2.0)
        with pytest.raises(ValidationError):
            self._make_settings(hyper_sd_fuse_scale=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(hyper_sd_fuse_scale=2.1)

    def test_deepcache_bounds(self):
        self._make_settings(deepcache_interval=1)
        self._make_settings(deepcache_interval=10)
        with pytest.raises(ValidationError):
            self._make_settings(deepcache_interval=0)
        with pytest.raises(ValidationError):
            self._make_settings(deepcache_interval=11)

    def test_performance_defaults(self):
        s = self._make_settings()
        assert s.enable_torch_compile is True
        assert s.compile_mode == "default"
        assert s.enable_deepcache is True
        assert s.enable_attention_slicing is True
        assert s.enable_vae_tiling is True
        assert s.enable_warmup is True

    def test_freeu_defaults(self):
        s = self._make_settings()
        assert s.enable_freeu is True
        assert s.freeu_s1 == 0.9
        assert s.freeu_s2 == 0.2
        assert s.freeu_b1 == 1.5
        assert s.freeu_b2 == 1.6

    def test_freeu_bounds(self):
        with pytest.raises(ValidationError):
            self._make_settings(freeu_b1=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(freeu_b1=3.1)

    def test_generation_timeout(self):
        s = self._make_settings()
        assert s.generation_timeout == 600.0
        with pytest.raises(ValidationError):
            self._make_settings(generation_timeout=0.0)
        with pytest.raises(ValidationError):
            self._make_settings(generation_timeout=-1.0)

    def test_animation_defaults(self):
        s = self._make_settings()
        assert s.max_animation_frames == 256
        assert s.animatediff_model == "ByteDance/AnimateDiff-Lightning"
        assert s.enable_freeinit is False
        assert s.freeinit_iterations == 2

    def test_audio_defaults(self):
        s = self._make_settings()
        assert s.audio_max_file_size_mb == 500
        assert s.audio_max_frames == 10800
        assert s.audio_default_attack == 2
        assert s.audio_default_release == 8
        assert s.stem_model == "htdemucs"
        assert s.stem_device == "cpu"

    def test_rembg_defaults(self):
        s = self._make_settings()
        assert s.rembg_model == "u2net"
        assert s.rembg_on_cpu is True

    def test_default_style_lora(self):
        s = self._make_settings()
        assert s.default_style_lora == "auto"
        assert s.default_style_lora_weight == 1.0

    def test_style_lora_weight_bounds(self):
        self._make_settings(default_style_lora_weight=0.0)
        self._make_settings(default_style_lora_weight=2.0)
        with pytest.raises(ValidationError):
            self._make_settings(default_style_lora_weight=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(default_style_lora_weight=2.1)


class TestSettingsEnvPrefix:

    def test_env_prefix_overrides_port(self):
        from sddj.config import Settings

        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        env["SDDJ_PORT"] = "8888"
        with patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)
        assert s.port == 8888

    def test_env_prefix_overrides_host(self):
        from sddj.config import Settings

        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        env["SDDJ_HOST"] = "0.0.0.0"
        with patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)
        assert s.host == "0.0.0.0"


class TestSettingsValidator:

    def test_missing_dirs_logs_warning(self, tmp_path, caplog):
        from sddj.config import Settings

        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        with patch.dict(os.environ, env, clear=True):
            import logging
            with caplog.at_level(logging.WARNING, logger="sddj.config"):
                Settings(
                    _env_file=None,
                    models_dir=tmp_path / "nonexistent",
                    checkpoints_dir=tmp_path / "nonexistent",
                    loras_dir=tmp_path / "nonexistent",
                    embeddings_dir=tmp_path / "nonexistent",
                    palettes_dir=tmp_path / "nonexistent",
                    presets_dir=tmp_path / "nonexistent",
                    prompts_data_dir=tmp_path / "nonexistent",
                )
        assert "does not exist" in caplog.text

    def test_existing_dirs_no_warning(self, tmp_path, caplog):
        from sddj.config import Settings

        for d in ("models", "checkpoints", "loras", "embeddings", "palettes", "presets", "prompts"):
            (tmp_path / d).mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        with patch.dict(os.environ, env, clear=True):
            import logging
            with caplog.at_level(logging.WARNING, logger="sddj.config"):
                Settings(
                    _env_file=None,
                    models_dir=tmp_path / "models",
                    checkpoints_dir=tmp_path / "checkpoints",
                    loras_dir=tmp_path / "loras",
                    embeddings_dir=tmp_path / "embeddings",
                    palettes_dir=tmp_path / "palettes",
                    presets_dir=tmp_path / "presets",
                    prompts_data_dir=tmp_path / "prompts",
                )
        assert "does not exist" not in caplog.text


class TestCompileMode:

    def _make_settings(self, **overrides):
        from sddj.config import Settings
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        with patch.dict(os.environ, env, clear=True):
            return Settings(_env_file=None, **overrides)

    def test_valid_modes(self):
        for mode in ("default", "max-autotune", "reduce-overhead"):
            s = self._make_settings(compile_mode=mode)
            assert s.compile_mode == mode

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            self._make_settings(compile_mode="turbo")
