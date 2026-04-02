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
        assert s.rembg_model == "birefnet-general"
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
                    prompt_schedules_dir=tmp_path / "nonexistent",
                    prompts_data_dir=tmp_path / "nonexistent",
                )
        assert "does not exist" in caplog.text

    def test_existing_dirs_no_warning(self, tmp_path, caplog):
        from sddj.config import Settings

        for d in ("models", "checkpoints", "loras", "embeddings", "palettes", "presets", "schedules", "prompts"):
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
                    prompt_schedules_dir=tmp_path / "schedules",
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
        for mode in ("default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"):
            s = self._make_settings(compile_mode=mode)
            assert s.compile_mode == mode

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            self._make_settings(compile_mode="turbo")


class TestAnimateDiffLightning:

    def _make_settings(self, **overrides):
        from sddj.config import Settings
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        with patch.dict(os.environ, env, clear=True):
            return Settings(_env_file=None, **overrides)

    def test_default_model_is_lightning(self):
        s = self._make_settings()
        assert s.is_animatediff_lightning is True

    def test_non_lightning_model(self):
        s = self._make_settings(animatediff_model="guoyww/animatediff-motion-adapter-v1-5-3")
        assert s.is_animatediff_lightning is False

    def test_lightning_max_frames_default(self):
        s = self._make_settings()
        assert s.animatediff_max_frames_lightning == 32

    def test_lightning_max_frames_bounds(self):
        self._make_settings(animatediff_max_frames_lightning=8)
        self._make_settings(animatediff_max_frames_lightning=64)
        with pytest.raises(ValidationError):
            self._make_settings(animatediff_max_frames_lightning=7)
        with pytest.raises(ValidationError):
            self._make_settings(animatediff_max_frames_lightning=65)


class TestNewFieldDefaults:

    def _make_settings(self, **overrides):
        from sddj.config import Settings
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDDJ_")}
        with patch.dict(os.environ, env, clear=True):
            return Settings(_env_file=None, **overrides)

    def test_field_constraint_defaults(self):
        """Verify all new Field()-constrained fields have sensible defaults."""
        s = self._make_settings()
        assert s.default_steps == 8
        assert s.default_cfg == 5.0
        assert s.default_width == 512
        assert s.default_height == 512
        assert s.default_clip_skip == 2
        assert s.max_animation_frames == 256
        assert s.freeinit_iterations == 2
        assert s.audio_max_file_size_mb == 500
        assert s.audio_max_frames == 10800
        assert s.audio_default_attack == 2
        assert s.audio_default_release == 8

    def test_default_steps_bounds(self):
        self._make_settings(default_steps=1)
        self._make_settings(default_steps=100)
        with pytest.raises(ValidationError):
            self._make_settings(default_steps=0)
        with pytest.raises(ValidationError):
            self._make_settings(default_steps=101)

    def test_default_cfg_bounds(self):
        self._make_settings(default_cfg=0.0)
        self._make_settings(default_cfg=30.0)
        with pytest.raises(ValidationError):
            self._make_settings(default_cfg=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(default_cfg=30.1)

    def test_default_dimensions_bounds(self):
        self._make_settings(default_width=64, default_height=64)
        self._make_settings(default_width=2048, default_height=2048)
        with pytest.raises(ValidationError):
            self._make_settings(default_width=63)
        with pytest.raises(ValidationError):
            self._make_settings(default_height=2049)

    def test_clip_skip_bounds(self):
        self._make_settings(default_clip_skip=1)
        self._make_settings(default_clip_skip=12)
        with pytest.raises(ValidationError):
            self._make_settings(default_clip_skip=0)
        with pytest.raises(ValidationError):
            self._make_settings(default_clip_skip=13)

    def test_max_animation_frames_bounds(self):
        self._make_settings(max_animation_frames=1)
        self._make_settings(max_animation_frames=10000)
        with pytest.raises(ValidationError):
            self._make_settings(max_animation_frames=0)
        with pytest.raises(ValidationError):
            self._make_settings(max_animation_frames=10001)

    def test_freeinit_iterations_bounds(self):
        self._make_settings(freeinit_iterations=1)
        self._make_settings(freeinit_iterations=10)
        with pytest.raises(ValidationError):
            self._make_settings(freeinit_iterations=0)
        with pytest.raises(ValidationError):
            self._make_settings(freeinit_iterations=11)

    def test_audio_file_size_bounds(self):
        self._make_settings(audio_max_file_size_mb=1)
        with pytest.raises(ValidationError):
            self._make_settings(audio_max_file_size_mb=0)

    def test_audio_max_frames_bounds(self):
        self._make_settings(audio_max_frames=1)
        with pytest.raises(ValidationError):
            self._make_settings(audio_max_frames=0)

    def test_audio_attack_release_bounds(self):
        self._make_settings(audio_default_attack=1, audio_default_release=1)
        self._make_settings(audio_default_attack=100, audio_default_release=100)
        with pytest.raises(ValidationError):
            self._make_settings(audio_default_attack=0)
        with pytest.raises(ValidationError):
            self._make_settings(audio_default_release=0)

    def test_new_feature_flag_defaults(self):
        """Verify all new boolean/Literal feature flag defaults."""
        s = self._make_settings()
        assert s.enable_unet_quantization is False
        assert s.unet_quantization_dtype == "auto"
        assert s.attention_backend == "auto"
        assert s.enable_tome is False
        assert s.tome_ratio == 0.3
        assert s.enable_taesd_preview is False
        assert s.enable_frame_compression is False
        assert s.queue_wait_timeout == 120.0
        assert s.color_coherence_strength == 0.5
        assert s.auto_noise_coupling is True
        assert s.optical_flow_blend == 0.0
        assert s.equivdm_noise is True
        assert s.equivdm_residual == 0.08
        assert s.distilled_step_scale_cap == 2
        assert s.stem_backend == "demucs"
        assert s.enable_tf32 is True
        assert s.enable_lora_hotswap is True
        assert s.compile_dynamic is False

    def test_tome_ratio_bounds(self):
        self._make_settings(tome_ratio=0.0)
        self._make_settings(tome_ratio=0.75)
        with pytest.raises(ValidationError):
            self._make_settings(tome_ratio=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(tome_ratio=0.76)

    def test_queue_wait_timeout_bounds(self):
        self._make_settings(queue_wait_timeout=0.1)
        with pytest.raises(ValidationError):
            self._make_settings(queue_wait_timeout=0.0)
        with pytest.raises(ValidationError):
            self._make_settings(queue_wait_timeout=-1.0)

    def test_color_coherence_bounds(self):
        self._make_settings(color_coherence_strength=0.0)
        self._make_settings(color_coherence_strength=1.0)
        with pytest.raises(ValidationError):
            self._make_settings(color_coherence_strength=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(color_coherence_strength=1.1)

    def test_equivdm_residual_bounds(self):
        self._make_settings(equivdm_residual=0.0)
        self._make_settings(equivdm_residual=0.5)
        with pytest.raises(ValidationError):
            self._make_settings(equivdm_residual=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(equivdm_residual=0.51)

    def test_optical_flow_blend_bounds(self):
        self._make_settings(optical_flow_blend=0.0)
        self._make_settings(optical_flow_blend=0.5)
        with pytest.raises(ValidationError):
            self._make_settings(optical_flow_blend=-0.1)
        with pytest.raises(ValidationError):
            self._make_settings(optical_flow_blend=0.51)

    def test_distilled_step_scale_cap_bounds(self):
        self._make_settings(distilled_step_scale_cap=1)
        self._make_settings(distilled_step_scale_cap=10)
        with pytest.raises(ValidationError):
            self._make_settings(distilled_step_scale_cap=0)
        with pytest.raises(ValidationError):
            self._make_settings(distilled_step_scale_cap=11)

    def test_stem_backend_values(self):
        self._make_settings(stem_backend="demucs")
        self._make_settings(stem_backend="roformer")
        with pytest.raises(ValidationError):
            self._make_settings(stem_backend="invalid")

    def test_attention_backend_values(self):
        for backend in ("auto", "sdp", "sage", "xformers"):
            s = self._make_settings(attention_backend=backend)
            assert s.attention_backend == backend
        with pytest.raises(ValidationError):
            self._make_settings(attention_backend="invalid")

    def test_unet_quantization_dtype_values(self):
        for dtype in ("int8dq", "fp8dq", "int8wo", "fp8wo", "auto"):
            s = self._make_settings(unet_quantization_dtype=dtype)
            assert s.unet_quantization_dtype == dtype
        with pytest.raises(ValidationError):
            self._make_settings(unet_quantization_dtype="bf16")

    def test_audio_dsp_defaults(self):
        s = self._make_settings()
        assert s.audio_sample_rate == 44100
        assert s.audio_hop_length == 256
        assert s.audio_n_fft == 4096
        assert s.audio_n_mels == 128
        assert s.audio_perceptual_weighting is True
        assert s.audio_smoothing_mode == "ema"
        assert s.audio_beat_backend == "auto"

    def test_audio_beat_backend_values(self):
        for backend in ("auto", "librosa", "madmom", "beatnet", "allinone"):
            s = self._make_settings(audio_beat_backend=backend)
            assert s.audio_beat_backend == backend
        with pytest.raises(ValidationError):
            self._make_settings(audio_beat_backend="invalid")
