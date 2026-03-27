"""WebSocket JSON protocol — request and response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class Action(str, Enum):
    GENERATE = "generate"
    GENERATE_ANIMATION = "generate_animation"
    CANCEL = "cancel"
    LIST_LORAS = "list_loras"
    LIST_PALETTES = "list_palettes"
    LIST_CONTROLNETS = "list_controlnets"
    LIST_EMBEDDINGS = "list_embeddings"
    PING = "ping"
    # Auto-prompt & presets
    GENERATE_PROMPT = "generate_prompt"
    LIST_PRESETS = "list_presets"
    GET_PRESET = "get_preset"
    SAVE_PRESET = "save_preset"
    DELETE_PRESET = "delete_preset"
    # Palette management
    SAVE_PALETTE = "save_palette"
    DELETE_PALETTE = "delete_palette"
    # Resource management
    CLEANUP = "cleanup"
    # Audio reactivity
    ANALYZE_AUDIO = "analyze_audio"
    GENERATE_AUDIO_REACTIVE = "generate_audio_reactive"
    CHECK_STEMS = "check_stems"
    LIST_MODULATION_PRESETS = "list_modulation_presets"
    GET_MODULATION_PRESET = "get_modulation_preset"
    LIST_EXPRESSION_PRESETS = "list_expression_presets"
    GET_EXPRESSION_PRESET = "get_expression_preset"
    LIST_CHOREOGRAPHY_PRESETS = "list_choreography_presets"
    GET_CHOREOGRAPHY_PRESET = "get_choreography_preset"
    # Prompt schedule presets
    LIST_PROMPT_SCHEDULES = "list_prompt_schedules"
    GET_PROMPT_SCHEDULE = "get_prompt_schedule"
    SAVE_PROMPT_SCHEDULE = "save_prompt_schedule"
    DELETE_PROMPT_SCHEDULE = "delete_prompt_schedule"
    # Video export
    EXPORT_MP4 = "export_mp4"
    # Server lifecycle
    SHUTDOWN = "shutdown"


class GenerationMode(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    CONTROLNET_OPENPOSE = "controlnet_openpose"
    CONTROLNET_CANNY = "controlnet_canny"
    CONTROLNET_SCRIBBLE = "controlnet_scribble"
    CONTROLNET_LINEART = "controlnet_lineart"
    CONTROLNET_QRCODE = "controlnet_qrcode"


class QuantizeMethod(str, Enum):
    KMEANS = "kmeans"
    OCTREE = "octree"
    MEDIAN_CUT = "median_cut"


class DitherMode(str, Enum):
    NONE = "none"
    FLOYD_STEINBERG = "floyd_steinberg"
    BAYER_2X2 = "bayer_2x2"
    BAYER_4X4 = "bayer_4x4"
    BAYER_8X8 = "bayer_8x8"


class PaletteMode(str, Enum):
    AUTO = "auto"
    CUSTOM = "custom"
    PRESET = "preset"


class AnimationMethod(str, Enum):
    CHAIN = "chain"
    ANIMATEDIFF = "animatediff"
    ANIMATEDIFF_AUDIO = "animatediff_audio"


class SeedStrategy(str, Enum):
    FIXED = "fixed"
    INCREMENT = "increment"
    RANDOM = "random"


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class LoRASpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    weight: float = Field(1.0, ge=-2.0, le=2.0)


class EmbeddingSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    weight: float = Field(1.0, ge=-2.0, le=2.0)


class PixelateSpec(BaseModel):
    enabled: bool = False
    target_size: int = Field(128, ge=8, le=512)


class PaletteSpec(BaseModel):
    mode: PaletteMode = PaletteMode.AUTO
    name: Optional[str] = None       # preset name
    colors: Optional[list[str]] = None  # hex codes for custom


class PostProcessSpec(BaseModel):
    pixelate: PixelateSpec = Field(default_factory=PixelateSpec)
    quantize_enabled: bool = False
    quantize_method: QuantizeMethod = QuantizeMethod.KMEANS
    quantize_colors: int = Field(32, ge=2, le=256)
    dither: DitherMode = DitherMode.NONE
    palette: PaletteSpec = Field(default_factory=PaletteSpec)
    remove_bg: bool = False


class PromptKeyframeSpec(BaseModel):
    frame: int = Field(0, ge=0)
    prompt: str = ""
    negative_prompt: str = ""
    weight: float = Field(1.0, ge=0.0, le=5.0)
    transition: str = "hard_cut"
    transition_frames: int = Field(0, ge=0, le=120)

    @field_validator("transition")
    @classmethod
    def _valid_transition(cls, v: str) -> str:
        if v not in ("hard_cut", "blend"):
            return "hard_cut"
        return v


class PromptScheduleSpec(BaseModel):
    keyframes: list[PromptKeyframeSpec] = Field(default_factory=list)
    default_prompt: str = ""
    auto_fill: bool = False


_DEFAULT_NEGATIVE = (
    "blurry, antialiased, smooth gradient, photorealistic, 3d render, "
    "soft edges, anti-aliasing, bokeh, depth of field, "
    "low quality, worst quality, bad quality, jpeg artifacts, watermark, text, logo, "
    "deformed, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, "
    "extra fingers, fused fingers, poorly drawn hands, poorly drawn face, ugly, "
    "realistic, photo, high resolution, complex shading"
)


def _check_generation_mode_images(model: BaseModel, context: str = "") -> BaseModel:
    """Shared validator: check source/mask/control images align with mode."""
    pfx = f"{context} " if context else ""
    if model.mode == GenerationMode.IMG2IMG and model.source_image is None:
        raise ValueError(f"{pfx}img2img mode requires source_image")
    if model.mode == GenerationMode.INPAINT:
        if model.source_image is None or model.mask_image is None:
            raise ValueError(f"{pfx}inpaint mode requires source_image and mask_image")
    if model.mode.value.startswith("controlnet_") and model.control_image is None:
        raise ValueError(f"{pfx}{model.mode.value} requires control_image")
    return model


class BaseGenerationParams(BaseModel):
    """Shared generation parameters — inherited by Generate, Animation, AudioReactive."""
    prompt: str = ""
    negative_prompt: str = _DEFAULT_NEGATIVE
    mode: GenerationMode = GenerationMode.TXT2IMG
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    source_image: Optional[str] = None
    mask_image: Optional[str] = None
    control_image: Optional[str] = None
    seed: int = -1
    steps: int = Field(8, ge=1, le=100)
    cfg_scale: float = Field(5.0, ge=0.0, le=30.0)
    denoise_strength: float = Field(0.30, ge=0.0, le=1.0)
    clip_skip: int = Field(2, ge=1, le=12)
    lora: Optional[LoRASpec] = None
    negative_ti: Optional[list[EmbeddingSpec]] = None
    post_process: PostProcessSpec = Field(default_factory=PostProcessSpec)


class GenerateRequest(BaseGenerationParams):
    action: Action = Action.GENERATE
    denoise_strength: float = Field(1.0, ge=0.0, le=1.0)
    # ── ControlNet conditioning overrides (QR Code Monster, etc.) ──
    controlnet_conditioning_scale: float = Field(1.5, ge=0.0, le=3.0)
    control_guidance_start: float = Field(0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(1.0, ge=0.0, le=1.0)
    # ── Prompt scheduling ──
    prompt_schedule: Optional[PromptScheduleSpec] = None

    @model_validator(mode='after')
    def _check_mode_images(self):
        return _check_generation_mode_images(self)


class AnimationRequest(BaseGenerationParams):
    action: Action = Action.GENERATE_ANIMATION
    method: AnimationMethod = AnimationMethod.CHAIN
    # Animation-specific
    frame_count: int = Field(8, ge=2, le=256)
    frame_duration_ms: int = Field(100, ge=30, le=2000)
    seed_strategy: SeedStrategy = SeedStrategy.INCREMENT
    tag_name: Optional[str] = Field(None, max_length=64)
    # AnimateDiff-specific
    enable_freeinit: bool = False
    freeinit_iterations: int = Field(2, ge=1, le=3)
    # ── Prompt scheduling ──
    prompt_schedule: Optional[PromptScheduleSpec] = None

    @model_validator(mode='after')
    def _check_mode_images(self):
        return _check_generation_mode_images(self, "animation")



class Request(BaseModel):
    action: Action
    # Shared generation fields (optional — only required for generate/generate_animation)
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    mode: Optional[GenerationMode] = None
    width: Optional[int] = None
    height: Optional[int] = None
    source_image: Optional[str] = None
    mask_image: Optional[str] = None
    control_image: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    denoise_strength: Optional[float] = None
    clip_skip: Optional[int] = None
    lora: Optional[LoRASpec] = None
    negative_ti: Optional[list[EmbeddingSpec]] = None
    post_process: Optional[PostProcessSpec] = None
    # Animation fields
    method: Optional[AnimationMethod] = None
    frame_count: Optional[int] = None
    frame_duration_ms: Optional[int] = None
    seed_strategy: Optional[SeedStrategy] = None
    tag_name: Optional[str] = None
    enable_freeinit: Optional[bool] = None
    freeinit_iterations: Optional[int] = None
    # Auto-prompt fields
    locked_fields: Optional[dict[str, str]] = None
    prompt_template: Optional[str] = None
    randomness: int = Field(0, ge=0, le=20)
    subject_type: Optional[str] = None
    prompt_mode: Optional[str] = None
    exclude_terms: Optional[list[str]] = None
    # Preset fields
    preset_name: Optional[str] = None
    preset_data: Optional[dict] = None
    # Palette CRUD fields
    palette_save_name: Optional[str] = None
    palette_save_colors: Optional[list[str]] = None
    # Audio reactivity fields
    audio_path: Optional[str] = None
    fps: Optional[float] = None
    enable_stems: Optional[bool] = None
    max_frames: Optional[int] = None
    modulation_slots: Optional[list[dict]] = None
    expressions: Optional[dict[str, str]] = None
    modulation_preset: Optional[str] = None
    # Prompt scheduling
    prompt_schedule: Optional[dict] = None
    prompt_schedule_name: Optional[str] = None
    prompt_schedule_data: Optional[dict] = None
    # Video export fields
    output_dir: Optional[str] = None
    scale_factor: Optional[int] = None
    # ControlNet conditioning overrides
    controlnet_conditioning_scale: Optional[float] = None
    control_guidance_start: Optional[float] = None
    control_guidance_end: Optional[float] = None
    quality: Optional[str] = None

    @field_validator("modulation_slots", "palette_save_colors", mode="before")
    @classmethod
    def _empty_dict_to_list(cls, v: Any) -> Any:
        """Lua json.lua encodes empty tables as {} (object) instead of [] (array).
        Normalise empty dict to empty list so Pydantic validation doesn't reject it."""
        if isinstance(v, dict) and len(v) == 0:
            return []
        return v

    @field_validator("negative_ti", mode="before")
    @classmethod
    def _empty_dict_to_list_nullable(cls, v: Any) -> Any:
        """Same normalisation for nullable list fields."""
        if isinstance(v, dict) and len(v) == 0:
            return []
        return v

    def to_generate_request(self) -> GenerateRequest:
        _exclude = {
            "action", "method", "frame_count", "frame_duration_ms",
            "seed_strategy", "tag_name", "enable_freeinit", "freeinit_iterations",
            # Auto-prompt fields
            "locked_fields", "prompt_template", "randomness",
            "subject_type", "prompt_mode", "exclude_terms",
            # Audio reactivity fields
            "audio_path", "fps", "enable_stems",
            "modulation_slots", "expressions", "modulation_preset",
            "prompt_segments",
            # Resource / export fields
            "preset_name", "preset_data", "palette_save_name", "palette_save_colors",
            "max_frames", "output_dir", "scale_factor", "quality",
            # Prompt schedule CRUD fields (not generation)
            "prompt_schedule_name", "prompt_schedule_data",
        }
        data = self.model_dump(exclude_none=True, exclude=_exclude)
        return GenerateRequest(**data)

    def to_animation_request(self) -> AnimationRequest:
        _exclude = {
            "action",
            # Auto-prompt fields
            "locked_fields", "prompt_template", "randomness",
            "subject_type", "prompt_mode", "exclude_terms",
            # Audio reactivity fields
            "audio_path", "fps", "enable_stems",
            "modulation_slots", "expressions", "modulation_preset",
            "prompt_segments",
            # Resource / export fields
            "preset_name", "preset_data", "palette_save_name", "palette_save_colors",
            "max_frames", "output_dir", "scale_factor", "quality",
            # ControlNet conditioning fields (generate-only)
            "controlnet_conditioning_scale", "control_guidance_start", "control_guidance_end",
            # Prompt schedule CRUD fields (not generation)
            "prompt_schedule_name", "prompt_schedule_data",
        }
        data = self.model_dump(exclude_none=True, exclude=_exclude)
        return AnimationRequest(**data)

    def to_analyze_audio_request(self) -> AnalyzeAudioRequest:
        return AnalyzeAudioRequest(
            audio_path=self.audio_path or "",
            fps=self.fps or 24.0,
            enable_stems=self.enable_stems or False,
        )

    def to_audio_reactive_request(self) -> AudioReactiveRequest:
        _exclude = {
            "action", "frame_count", "frame_duration_ms", "seed_strategy",
            "prompt_template",
            "subject_type", "prompt_mode", "exclude_terms",
            # Resource / export fields
            "preset_name", "preset_data", "palette_save_name", "palette_save_colors",
            "output_dir", "scale_factor", "quality",
            # ControlNet conditioning fields (generate-only)
            "controlnet_conditioning_scale", "control_guidance_start", "control_guidance_end",
            # Prompt schedule CRUD fields (not generation)
            "prompt_schedule_name", "prompt_schedule_data",
        }
        data = self.model_dump(exclude_none=True, exclude=_exclude)
        return AudioReactiveRequest(**data)


# ─────────────────────────────────────────────────────────────
# AUDIO REACTIVITY REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class ModulationSlotSpec(BaseModel):
    source: str = Field(..., min_length=1, max_length=64)
    target: str = Field(..., min_length=1, max_length=64)
    min_val: float = 0.0
    max_val: float = 1.0
    attack: int = Field(2, ge=1, le=30)
    release: int = Field(8, ge=1, le=60)
    enabled: bool = True
    invert: bool = False


class AnalyzeAudioRequest(BaseModel):
    action: Action = Action.ANALYZE_AUDIO
    audio_path: str = ""
    fps: float = Field(24.0, ge=1.0, le=120.0)
    enable_stems: bool = False


class AudioReactiveRequest(BaseGenerationParams):
    action: Action = Action.GENERATE_AUDIO_REACTIVE
    audio_path: str = ""
    fps: float = Field(24.0, ge=1.0, le=120.0)
    enable_stems: bool = False
    modulation_slots: list[ModulationSlotSpec] = Field(default_factory=list)
    expressions: Optional[dict[str, str]] = None
    modulation_preset: Optional[str] = None
    randomness: int = Field(0, ge=0, le=20)
    locked_fields: Optional[dict[str, str]] = None
    max_frames: Optional[int] = Field(None, ge=1, le=10800)
    # Animation method: chain (default) or animatediff_audio
    method: AnimationMethod = AnimationMethod.CHAIN
    # AnimateDiff-specific
    enable_freeinit: bool = False
    freeinit_iterations: int = Field(2, ge=1, le=3)
    # Audio-reactive uses fps instead; retained for metadata backward compat only.
    frame_duration_ms: Optional[int] = Field(None, ge=30, le=2000)
    tag_name: Optional[str] = Field(None, max_length=64)
    # ── Prompt scheduling (takes precedence over prompt_segments) ──
    prompt_schedule: Optional[PromptScheduleSpec] = None

    @field_validator("modulation_slots", mode="before")
    @classmethod
    def _empty_dict_to_list(cls, v: Any) -> Any:
        if isinstance(v, dict) and len(v) == 0:
            return []
        return v


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class ProgressResponse(BaseModel):
    type: Literal["progress"] = "progress"
    step: int
    total: int
    frame_index: Optional[int] = None
    total_frames: Optional[int] = None


class ResultResponse(BaseModel):
    type: Literal["result"] = "result"
    image: str          # base64 PNG RGBA
    seed: int
    time_ms: int
    width: int
    height: int


class AnimationFrameResponse(BaseModel):
    type: Literal["animation_frame"] = "animation_frame"
    frame_index: int
    total_frames: int
    image: str          # base64 PNG or raw RGBA
    seed: int
    time_ms: int
    width: int
    height: int
    encoding: Optional[str] = None  # None = PNG, "raw_rgba" = raw RGBA bytes


class AnimationCompleteResponse(BaseModel):
    type: Literal["animation_complete"] = "animation_complete"
    total_frames: int
    total_time_ms: int
    tag_name: Optional[str] = None


class ErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    code: str = "UNKNOWN"
    message: str


class ListResponse(BaseModel):
    type: Literal["list"] = "list"
    list_type: str  # "loras" | "palettes" | "controlnets" | "embeddings" | "presets"
    items: list[str]


class PongResponse(BaseModel):
    type: Literal["pong"] = "pong"


# ─────────────────────────────────────────────────────────────
# AUTO-PROMPT & PRESETS RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class PromptResultResponse(BaseModel):
    type: Literal["prompt_result"] = "prompt_result"
    prompt: str
    negative_prompt: str = ""
    components: dict[str, str]


class PresetResponse(BaseModel):
    type: Literal["preset"] = "preset"
    name: str
    data: dict


class PresetSavedResponse(BaseModel):
    type: Literal["preset_saved"] = "preset_saved"
    name: str


class PresetDeletedResponse(BaseModel):
    type: Literal["preset_deleted"] = "preset_deleted"
    name: str


class PaletteSavedResponse(BaseModel):
    type: Literal["palette_saved"] = "palette_saved"
    name: str


class PaletteDeletedResponse(BaseModel):
    type: Literal["palette_deleted"] = "palette_deleted"
    name: str


class CleanupResponse(BaseModel):
    type: Literal["cleanup_done"] = "cleanup_done"
    message: str
    freed_mb: float


# ─────────────────────────────────────────────────────────────
# AUDIO REACTIVITY RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class AudioAnalysisResponse(BaseModel):
    type: Literal["audio_analysis"] = "audio_analysis"
    duration: float
    total_frames: int
    features: list[str]
    bpm: float = 0.0
    lufs: float = -24.0
    sample_rate: int = 44100
    hop_length: int = 256
    recommended_preset: str = ""
    stems_available: bool = False
    stems: Optional[list[str]] = None
    waveform: Optional[list[float]] = None  # mini RMS waveform (100 points, [0,1])


class AudioReactiveFrameResponse(BaseModel):
    type: Literal["audio_reactive_frame"] = "audio_reactive_frame"
    frame_index: int
    total_frames: int
    image: str          # base64 PNG or raw RGBA
    seed: int
    time_ms: int
    width: int
    height: int
    encoding: Optional[str] = None  # None = PNG, "raw_rgba" = raw RGBA bytes
    params_used: dict[str, float] = Field(default_factory=dict)


class AudioReactiveCompleteResponse(BaseModel):
    type: Literal["audio_reactive_complete"] = "audio_reactive_complete"
    total_frames: int
    total_time_ms: int
    tag_name: Optional[str] = None


class StemsAvailableResponse(BaseModel):
    type: Literal["stems_available"] = "stems_available"
    available: bool
    message: str


class ModulationPresetsResponse(BaseModel):
    type: Literal["modulation_presets"] = "modulation_presets"
    presets: list[str]


class ModulationPresetDetailResponse(BaseModel):
    type: Literal["modulation_preset_detail"] = "modulation_preset_detail"
    name: str
    slots: list[dict]


class ExpressionPresetsListResponse(BaseModel):
    type: Literal["expression_presets_list"] = "expression_presets_list"
    presets: dict  # category -> list of {name, targets, description}


class ExpressionPresetDetailResponse(BaseModel):
    type: Literal["expression_preset_detail"] = "expression_preset_detail"
    name: str
    targets: dict  # target -> expression string
    description: str
    category: str


class ChoreographyPresetsListResponse(BaseModel):
    type: Literal["choreography_presets_list"] = "choreography_presets_list"
    presets: list[dict]  # [{name, description, slot_count, expression_targets}]


class ChoreographyPresetDetailResponse(BaseModel):
    type: Literal["choreography_preset_detail"] = "choreography_preset_detail"
    name: str
    description: str
    slots: list[dict]
    expressions: dict


# ─────────────────────────────────────────────────────────────
# SERVER LIFECYCLE RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class ExportMp4Response(BaseModel):
    type: Literal["export_mp4_complete"] = "export_mp4_complete"
    path: str
    size_mb: float
    duration_s: float = 0.0


class ExportMp4ErrorResponse(BaseModel):
    type: Literal["export_mp4_error"] = "export_mp4_error"
    message: str


class ShutdownResponse(BaseModel):
    type: Literal["shutdown_ack"] = "shutdown_ack"
    message: str = "Server shutting down"
