"""Tests for protocol models — validation, enums, conversions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sddj.protocol import (
    Action,
    AnalyzeAudioRequest,
    AnimationRequest,
    AudioAnalysisResponse,
    AudioReactiveCompleteResponse,
    AudioReactiveFrameResponse,
    AudioReactiveRequest,
    CleanupResponse,
    DitherMode,
    ErrorResponse,
    GenerateRequest,
    GenerationMode,
    ListResponse,
    ModulationPresetsResponse,
    ModulationSlotSpec,
    PaletteMode,
    PongResponse,
    PostProcessSpec,
    PresetDeletedResponse,
    PresetResponse,
    PresetSavedResponse,
    ProgressResponse,
    PromptResultResponse,
    Request,
    ResultResponse,
    SeedStrategy,
    StemsAvailableResponse,
    PaletteSavedResponse,
    PaletteDeletedResponse,
)


class TestAction:
    def test_all_actions_exist(self):
        expected = {
            "generate", "generate_animation", "cancel",
            "list_loras", "list_palettes", "list_controlnets", "list_embeddings",
            "ping",
            "generate_prompt", "list_presets", "get_preset", "save_preset", "delete_preset",
            "save_palette", "delete_palette",
            "cleanup",
            "analyze_audio", "generate_audio_reactive", "check_stems",
            "list_modulation_presets", "get_modulation_preset",
            "list_expression_presets", "get_expression_preset",
            "list_choreography_presets", "get_choreography_preset",
            "list_prompt_schedules", "get_prompt_schedule",
            "save_prompt_schedule", "delete_prompt_schedule",
            "validate_dsl",
            "export_mp4",
            "shutdown",
        }
        actual = {a.value for a in Action}
        assert expected == actual

    def test_action_from_string(self):
        assert Action("generate") == Action.GENERATE
        assert Action("cleanup") == Action.CLEANUP


class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest(prompt="test")
        assert req.width == 512
        assert req.height == 512
        assert req.steps == 8
        assert req.seed == -1
        assert req.mode == GenerationMode.TXT2IMG

    def test_img2img_requires_source(self):
        with pytest.raises(ValidationError, match="source_image"):
            GenerateRequest(prompt="test", mode="img2img")

    def test_inpaint_requires_mask(self):
        with pytest.raises(ValidationError, match="mask_image"):
            GenerateRequest(prompt="test", mode="inpaint", source_image="base64data")

    def test_controlnet_requires_control_image(self):
        with pytest.raises(ValidationError, match="control_image"):
            GenerateRequest(prompt="test", mode="controlnet_canny")
        # QR Code Monster also requires control_image (user-provided)
        with pytest.raises(ValidationError, match="control_image"):
            GenerateRequest(prompt="test", mode="controlnet_qrcode")

    def test_valid_img2img(self):
        req = GenerateRequest(prompt="test", mode="img2img", source_image="data")
        assert req.source_image == "data"

    def test_size_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", width=32)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", width=4096)

    def test_steps_bounds(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", steps=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", steps=101)

    def test_cfg_bounds(self):
        req = GenerateRequest(prompt="test", cfg_scale=0.0)
        assert req.cfg_scale == 0.0
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", cfg_scale=31.0)


class TestAnimationRequest:
    def test_defaults(self):
        req = AnimationRequest(prompt="test")
        assert req.frame_count == 8
        assert req.frame_duration_ms == 100
        assert req.denoise_strength == 0.30

    def test_seed_strategies(self):
        for s in ("fixed", "increment", "random"):
            req = AnimationRequest(prompt="test", seed_strategy=s)
            assert req.seed_strategy == SeedStrategy(s)


class TestPostProcessSpec:
    def test_defaults(self):
        pp = PostProcessSpec()
        assert pp.pixelate.enabled is False
        assert pp.quantize_enabled is False
        assert pp.quantize_colors == 32
        assert pp.dither == DitherMode.NONE
        assert pp.palette.mode == PaletteMode.AUTO

    def test_custom_palette(self):
        pp = PostProcessSpec(palette={"mode": "custom", "colors": ["#FF0000"]})
        assert pp.palette.mode == PaletteMode.CUSTOM
        assert pp.palette.colors == ["#FF0000"]


class TestRequestConversions:
    def test_to_generate_request(self):
        req = Request(
            action="generate", prompt="hello",
            width=768, height=512, steps=12,
        )
        gen = req.to_generate_request()
        assert isinstance(gen, GenerateRequest)
        assert gen.prompt == "hello"
        assert gen.width == 768

    def test_to_animation_request(self):
        req = Request(
            action="generate_animation", prompt="anim",
            frame_count=16, seed_strategy="random",
        )
        anim = req.to_animation_request()
        assert isinstance(anim, AnimationRequest)
        assert anim.frame_count == 16

    def test_to_generate_excludes_audio_fields(self):
        req = Request(
            action="generate", prompt="hello",
            audio_path="/tmp/test.wav", fps=24.0,
            enable_stems=True,
        )
        gen = req.to_generate_request()
        assert isinstance(gen, GenerateRequest)
        dumped = gen.model_dump()
        assert "audio_path" not in dumped
        assert "fps" not in dumped
        assert "enable_stems" not in dumped

    def test_to_animation_excludes_audio_fields(self):
        req = Request(
            action="generate_animation", prompt="anim",
            frame_count=16, audio_path="/tmp/test.wav",
            modulation_preset="energetic",
        )
        anim = req.to_animation_request()
        assert isinstance(anim, AnimationRequest)
        dumped = anim.model_dump()
        assert "audio_path" not in dumped
        assert "modulation_preset" not in dumped


class TestResponseModels:
    def test_progress(self):
        r = ProgressResponse(step=3, total=8)
        assert r.type == "progress"

    def test_result(self):
        r = ResultResponse(image="b64", seed=42, time_ms=1000, width=512, height=512)
        assert r.type == "result"

    def test_error(self):
        r = ErrorResponse(code="CANCELLED", message="User cancelled")
        assert r.type == "error"

    def test_pong(self):
        assert PongResponse().type == "pong"

    def test_list(self):
        r = ListResponse(list_type="loras", items=["lora1", "lora2"])
        assert len(r.items) == 2

    def test_prompt_result(self):
        r = PromptResultResponse(prompt="test prompt", components={"style": "pixel art"})
        assert r.type == "prompt_result"
        assert r.components["style"] == "pixel art"

    def test_preset_response(self):
        r = PresetResponse(name="test", data={"steps": 8})
        assert r.type == "preset"

    def test_preset_saved(self):
        r = PresetSavedResponse(name="test")
        assert r.type == "preset_saved"

    def test_preset_deleted(self):
        r = PresetDeletedResponse(name="test")
        assert r.type == "preset_deleted"

    def test_cleanup_response(self):
        r = CleanupResponse(message="Done", freed_mb=128.5)
        assert r.type == "cleanup_done"
        assert r.freed_mb == 128.5

    def test_audio_analysis_response(self):
        r = AudioAnalysisResponse(
            duration=5.0, total_frames=120,
            features=["global_rms", "global_onset"],
            stems_available=True, stems=["drums", "bass"],
        )
        assert r.type == "audio_analysis"
        assert len(r.features) == 2

    def test_audio_reactive_frame_response(self):
        r = AudioReactiveFrameResponse(
            frame_index=0, total_frames=10, image="b64",
            seed=42, time_ms=500, width=512, height=512,
            params_used={"denoise_strength": 0.5},
        )
        assert r.type == "audio_reactive_frame"
        assert r.params_used["denoise_strength"] == 0.5

    def test_audio_reactive_complete_response(self):
        r = AudioReactiveCompleteResponse(
            total_frames=120, total_time_ms=60000, tag_name="audio_anim",
        )
        assert r.type == "audio_reactive_complete"

    def test_stems_available_response(self):
        r = StemsAvailableResponse(available=True, message="Ready")
        assert r.type == "stems_available"

    def test_modulation_presets_response(self):
        r = ModulationPresetsResponse(presets=["energetic", "ambient"])
        assert r.type == "modulation_presets"


class TestAudioRequestModels:
    def test_analyze_audio_defaults(self):
        req = AnalyzeAudioRequest(audio_path="/test.wav")
        assert req.fps == 24.0
        assert req.enable_stems is False

    def test_analyze_audio_fps_bounds(self):
        with pytest.raises(ValidationError):
            AnalyzeAudioRequest(audio_path="/test.wav", fps=0.5)
        with pytest.raises(ValidationError):
            AnalyzeAudioRequest(audio_path="/test.wav", fps=200.0)

    def test_audio_reactive_defaults(self):
        req = AudioReactiveRequest(audio_path="/test.wav")
        assert req.fps == 24.0
        assert req.denoise_strength == 0.30
        assert req.modulation_slots == []
        assert req.expressions is None
        assert req.modulation_preset is None
        assert req.method.value == "chain"

    def test_audio_reactive_animatediff_method(self):
        from sddj.protocol import AnimationMethod
        req = AudioReactiveRequest(
            audio_path="/test.wav",
            method="animatediff_audio",
            enable_freeinit=True,
        )
        assert req.method == AnimationMethod.ANIMATEDIFF_AUDIO
        assert req.enable_freeinit is True

    def test_request_to_audio_reactive_with_method(self):
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav",
            prompt="pixel art",
            method="animatediff_audio",
            enable_freeinit=True,
        )
        ar = req.to_audio_reactive_request()
        assert isinstance(ar, AudioReactiveRequest)
        assert ar.method.value == "animatediff_audio"
        assert ar.enable_freeinit is True

    def test_modulation_slot_spec(self):
        slot = ModulationSlotSpec(
            source="global_rms", target="denoise_strength",
            min_val=0.2, max_val=0.8,
        )
        assert slot.attack == 2
        assert slot.release == 8
        assert slot.enabled is True
        assert slot.invert is False

    def test_modulation_slot_spec_invert(self):
        slot = ModulationSlotSpec(
            source="global_rms", target="denoise_strength",
            invert=True,
        )
        assert slot.invert is True

    def test_modulation_slot_bounds(self):
        with pytest.raises(ValidationError):
            ModulationSlotSpec(source="", target="x")
        with pytest.raises(ValidationError):
            ModulationSlotSpec(source="x", target="y", attack=0)
        with pytest.raises(ValidationError):
            ModulationSlotSpec(source="x", target="y", release=100)

    def test_audio_reactive_with_slots(self):
        req = AudioReactiveRequest(
            audio_path="/test.wav",
            modulation_slots=[
                {"source": "global_rms", "target": "denoise_strength",
                 "min_val": 0.2, "max_val": 0.8},
            ],
        )
        assert len(req.modulation_slots) == 1
        assert req.modulation_slots[0].source == "global_rms"

    def test_request_to_analyze_audio(self):
        req = Request(
            action="analyze_audio",
            audio_path="/test.wav", fps=30.0, enable_stems=True,
        )
        ar = req.to_analyze_audio_request()
        assert isinstance(ar, AnalyzeAudioRequest)
        assert ar.fps == 30.0
        assert ar.enable_stems is True

    def test_request_to_audio_reactive(self):
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav",
            prompt="pixel art",
            modulation_slots=[
                {"source": "global_rms", "target": "denoise_strength"},
            ],
        )
        ar = req.to_audio_reactive_request()
        assert isinstance(ar, AudioReactiveRequest)
        assert ar.prompt == "pixel art"
        assert len(ar.modulation_slots) == 1

    def test_audio_reactive_max_frames_default_none(self):
        """v0.7.4: max_frames defaults to None (no limit)."""
        req = AudioReactiveRequest(audio_path="/test.wav")
        assert req.max_frames is None


class TestRandomnessField:
    """v0.7.7: randomness field on Request model."""

    def test_default_is_zero(self):
        req = Request(action="generate_prompt")
        assert req.randomness == 0

    def test_valid_range(self):
        for v in (0, 5, 10, 15, 20):
            req = Request(action="generate_prompt", randomness=v)
            assert req.randomness == v

    def test_below_min_rejected(self):
        with pytest.raises(ValidationError):
            Request(action="generate_prompt", randomness=-1)

    def test_above_max_rejected(self):
        with pytest.raises(ValidationError):
            Request(action="generate_prompt", randomness=21)

    def test_excluded_from_generate_request(self):
        req = Request(action="generate", prompt="test", randomness=15)
        gen = req.to_generate_request()
        assert isinstance(gen, GenerateRequest)
        dumped = gen.model_dump()
        assert "randomness" not in dumped

    def test_excluded_from_animation_request(self):
        req = Request(action="generate_animation", prompt="test", randomness=10, frame_count=8)
        anim = req.to_animation_request()
        assert isinstance(anim, AnimationRequest)
        dumped = anim.model_dump()
        assert "randomness" not in dumped

    def test_audio_reactive_max_frames_valid(self):
        """v0.7.4: max_frames accepts valid range 1-10800."""
        req = AudioReactiveRequest(audio_path="/test.wav", max_frames=100)
        assert req.max_frames == 100
        req2 = AudioReactiveRequest(audio_path="/test.wav", max_frames=10800)
        assert req2.max_frames == 10800

    def test_audio_reactive_max_frames_bounds(self):
        """v0.7.4: max_frames rejects out-of-range values."""
        with pytest.raises(ValidationError):
            AudioReactiveRequest(audio_path="/test.wav", max_frames=0)
        with pytest.raises(ValidationError):
            AudioReactiveRequest(audio_path="/test.wav", max_frames=20000)

    def test_request_max_frames_forwarded(self):
        """v0.7.4: max_frames passes from Request to AudioReactiveRequest."""
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav", prompt="test",
            max_frames=50,
        )
        ar = req.to_audio_reactive_request()
        assert ar.max_frames == 50

    def test_request_max_frames_none_not_forwarded(self):
        """v0.7.4: max_frames=None is excluded by exclude_none."""
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav", prompt="test",
        )
        ar = req.to_audio_reactive_request()
        assert ar.max_frames is None


class TestAudioReactiveRandomness:
    """v0.7.7: randomness forwarded to AudioReactiveRequest."""

    def test_audio_reactive_randomness_default(self):
        req = AudioReactiveRequest(audio_path="/test.wav")
        assert req.randomness == 0

    def test_audio_reactive_randomness_valid(self):
        req = AudioReactiveRequest(audio_path="/test.wav", randomness=15)
        assert req.randomness == 15

    def test_audio_reactive_randomness_bounds(self):
        with pytest.raises(ValidationError):
            AudioReactiveRequest(audio_path="/test.wav", randomness=-1)
        with pytest.raises(ValidationError):
            AudioReactiveRequest(audio_path="/test.wav", randomness=21)

    def test_forwarded_to_audio_reactive_request(self):
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav", prompt="test",
            randomness=12,
        )
        ar = req.to_audio_reactive_request()
        assert ar.randomness == 12

    def test_zero_randomness_forwarded(self):
        """randomness=0 should still be forwarded (it has a default)."""
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav", prompt="test",
            randomness=0,
        )
        ar = req.to_audio_reactive_request()
        assert ar.randomness == 0


class TestPaletteCrudProtocol:
    """v0.7.9: palette save/delete actions and responses."""

    def test_save_palette_action_exists(self):
        assert Action.SAVE_PALETTE == "save_palette"

    def test_delete_palette_action_exists(self):
        assert Action.DELETE_PALETTE == "delete_palette"

    def test_palette_saved_response(self):
        resp = PaletteSavedResponse(name="my_pal")
        assert resp.type == "palette_saved"
        assert resp.name == "my_pal"

    def test_palette_deleted_response(self):
        resp = PaletteDeletedResponse(name="my_pal")
        assert resp.type == "palette_deleted"
        assert resp.name == "my_pal"

    def test_request_palette_fields(self):
        req = Request(
            action="save_palette",
            palette_save_name="test_pal",
            palette_save_colors=["#FF0000", "#00FF00"],
        )
        assert req.palette_save_name == "test_pal"
        assert req.palette_save_colors == ["#FF0000", "#00FF00"]


class TestLockedFieldsPropagation:
    """Lock Subject: locked_fields forwarded to AudioReactiveRequest."""

    def test_audio_reactive_locked_fields_default_none(self):
        req = AudioReactiveRequest(audio_path="/test.wav")
        assert req.locked_fields is None

    def test_audio_reactive_locked_fields_explicit(self):
        req = AudioReactiveRequest(
            audio_path="/test.wav",
            locked_fields={"subject": "a dragon"},
        )
        assert req.locked_fields == {"subject": "a dragon"}

    def test_forwarded_from_request(self):
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav", prompt="test",
            locked_fields={"subject": "robot knight"},
        )
        ar = req.to_audio_reactive_request()
        assert isinstance(ar, AudioReactiveRequest)
        assert ar.locked_fields == {"subject": "robot knight"}

    def test_none_not_forwarded(self):
        req = Request(
            action="generate_audio_reactive",
            audio_path="/test.wav", prompt="test",
        )
        ar = req.to_audio_reactive_request()
        assert ar.locked_fields is None

