"""WebSocket JSON protocol — request and response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    # Real-time paint mode
    REALTIME_START = "realtime_start"
    REALTIME_FRAME = "realtime_frame"
    REALTIME_UPDATE = "realtime_update"
    REALTIME_STOP = "realtime_stop"
    # Auto-prompt & presets
    GENERATE_PROMPT = "generate_prompt"
    LIST_PRESETS = "list_presets"
    GET_PRESET = "get_preset"
    SAVE_PRESET = "save_preset"
    DELETE_PRESET = "delete_preset"
    # Resource management
    CLEANUP = "cleanup"
    # Audio reactivity
    ANALYZE_AUDIO = "analyze_audio"
    GENERATE_AUDIO_REACTIVE = "generate_audio_reactive"
    CHECK_STEMS = "check_stems"
    LIST_MODULATION_PRESETS = "list_modulation_presets"
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
    enabled: bool = True
    target_size: int = Field(128, ge=8, le=512)


class PaletteSpec(BaseModel):
    mode: PaletteMode = PaletteMode.AUTO
    name: Optional[str] = None       # preset name
    colors: Optional[list[str]] = None  # hex codes for custom


class PostProcessSpec(BaseModel):
    pixelate: PixelateSpec = Field(default_factory=PixelateSpec)
    quantize_method: QuantizeMethod = QuantizeMethod.KMEANS
    quantize_colors: int = Field(32, ge=2, le=256)
    dither: DitherMode = DitherMode.NONE
    palette: PaletteSpec = Field(default_factory=PaletteSpec)
    remove_bg: bool = False


_DEFAULT_NEGATIVE = (
    "blurry, antialiased, smooth gradient, photorealistic, 3d render, "
    "soft edges, anti-aliasing, bokeh, depth of field, "
    "low quality, worst quality, bad quality, jpeg artifacts, watermark, text, logo, "
    "deformed, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, "
    "extra fingers, fused fingers, poorly drawn hands, poorly drawn face, ugly, "
    "realistic, photo, high resolution, complex shading"
)


class GenerateRequest(BaseModel):
    action: Action = Action.GENERATE
    prompt: str = ""
    negative_prompt: str = _DEFAULT_NEGATIVE
    mode: GenerationMode = GenerationMode.TXT2IMG
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    source_image: Optional[str] = None      # base64 PNG (img2img / inpaint)
    mask_image: Optional[str] = None        # base64 PNG (inpaint) — white=repaint, black=keep
    control_image: Optional[str] = None     # base64 PNG (ControlNet)
    seed: int = -1
    steps: int = Field(8, ge=1, le=100)
    cfg_scale: float = Field(5.0, ge=0.0, le=30.0)
    denoise_strength: float = Field(1.0, ge=0.0, le=1.0)
    clip_skip: int = Field(2, ge=1, le=12)
    lora: Optional[LoRASpec] = None
    negative_ti: Optional[list[EmbeddingSpec]] = None
    post_process: PostProcessSpec = Field(default_factory=PostProcessSpec)

    @model_validator(mode='after')
    def _check_mode_images(self):
        if self.mode == GenerationMode.IMG2IMG and self.source_image is None:
            raise ValueError("img2img mode requires source_image")
        if self.mode == GenerationMode.INPAINT:
            if self.source_image is None or self.mask_image is None:
                raise ValueError("inpaint mode requires source_image and mask_image")
        if self.mode.value.startswith("controlnet_") and self.control_image is None:
            raise ValueError(f"{self.mode.value} requires control_image")
        return self


class AnimationRequest(BaseModel):
    action: Action = Action.GENERATE_ANIMATION
    method: AnimationMethod = AnimationMethod.CHAIN
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
    # Animation-specific
    frame_count: int = Field(8, ge=2, le=120)
    frame_duration_ms: int = Field(100, ge=50, le=2000)
    seed_strategy: SeedStrategy = SeedStrategy.INCREMENT
    tag_name: Optional[str] = Field(None, max_length=64)
    # AnimateDiff-specific
    enable_freeinit: bool = False
    freeinit_iterations: int = Field(2, ge=1, le=3)

    @model_validator(mode='after')
    def _check_mode_images(self):
        if self.mode == GenerationMode.IMG2IMG and self.source_image is None:
            raise ValueError("img2img animation requires source_image")
        if self.mode == GenerationMode.INPAINT:
            if self.source_image is None or self.mask_image is None:
                raise ValueError("inpaint animation requires source_image and mask_image")
        if self.mode.value.startswith("controlnet_") and self.control_image is None:
            raise ValueError(f"{self.mode.value} animation requires control_image")
        return self


# ─────────────────────────────────────────────────────────────
# REAL-TIME PAINT REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class RealtimeStartRequest(BaseModel):
    action: Action = Action.REALTIME_START
    prompt: str = ""
    negative_prompt: str = _DEFAULT_NEGATIVE
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    seed: int = -1
    steps: int = Field(4, ge=2, le=8)
    cfg_scale: float = Field(2.5, ge=1.0, le=10.0)
    denoise_strength: float = Field(0.5, ge=0.05, le=0.95)
    clip_skip: int = Field(2, ge=1, le=12)
    lora: Optional[LoRASpec] = None
    negative_ti: Optional[list[EmbeddingSpec]] = None
    post_process: PostProcessSpec = Field(default_factory=PostProcessSpec)


class RealtimeFrameRequest(BaseModel):
    action: Action = Action.REALTIME_FRAME
    image: str              # base64 PNG — current canvas
    frame_id: int = 0       # monotonic frame counter (for latest-wins)
    prompt: Optional[str] = None  # override prompt mid-session
    # ROI (Region of Interest) for partial regeneration
    mask: Optional[str] = None    # base64 mask of dirty region
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_w: Optional[int] = None
    roi_h: Optional[int] = None


class RealtimeUpdateRequest(BaseModel):
    action: Action = Action.REALTIME_UPDATE
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    denoise_strength: Optional[float] = Field(None, ge=0.05, le=0.95)
    steps: Optional[int] = Field(None, ge=2, le=8)
    cfg_scale: Optional[float] = Field(None, ge=1.0, le=10.0)
    clip_skip: Optional[int] = Field(None, ge=1, le=12)
    seed: Optional[int] = None


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
    # Realtime fields
    image: Optional[str] = None
    frame_id: Optional[int] = None
    mask: Optional[str] = None      # ROI mask for realtime
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_w: Optional[int] = None
    roi_h: Optional[int] = None
    # Auto-prompt fields
    locked_fields: Optional[dict[str, str]] = None
    prompt_template: Optional[str] = None
    # Preset fields
    preset_name: Optional[str] = None
    preset_data: Optional[dict] = None
    # Audio reactivity fields
    audio_path: Optional[str] = None
    fps: Optional[float] = None
    enable_stems: Optional[bool] = None
    max_frames: Optional[int] = None
    modulation_slots: Optional[list[dict]] = None
    expressions: Optional[dict[str, str]] = None
    modulation_preset: Optional[str] = None
    prompt_segments: Optional[list[dict]] = None
    # Video export fields
    output_dir: Optional[str] = None
    scale_factor: Optional[int] = None
    quality: Optional[str] = None

    def to_generate_request(self) -> GenerateRequest:
        _exclude = {
            "action", "method", "frame_count", "frame_duration_ms",
            "seed_strategy", "tag_name", "enable_freeinit", "freeinit_iterations",
            # Audio reactivity fields
            "audio_path", "fps", "enable_stems",
            "modulation_slots", "expressions", "modulation_preset",
            "prompt_segments",
        }
        data = self.model_dump(exclude_none=True, exclude=_exclude)
        return GenerateRequest(**data)

    def to_animation_request(self) -> AnimationRequest:
        _exclude = {
            "action",
            # Audio reactivity fields
            "audio_path", "fps", "enable_stems",
            "modulation_slots", "expressions", "modulation_preset",
            "prompt_segments",
        }
        data = self.model_dump(exclude_none=True, exclude=_exclude)
        return AnimationRequest(**data)

    def to_realtime_start(self) -> RealtimeStartRequest:
        _rt_fields = {
            "action", "mode", "source_image", "mask_image", "control_image",
            "method", "frame_count", "frame_duration_ms", "seed_strategy",
            "tag_name", "enable_freeinit", "freeinit_iterations",
            "image", "frame_id",
        }
        data = self.model_dump(exclude_none=True, exclude=_rt_fields)
        return RealtimeStartRequest(**data)

    def to_realtime_frame(self) -> RealtimeFrameRequest:
        data: dict = {"action": "realtime_frame"}
        if self.image is not None:
            data["image"] = self.image
        if self.frame_id is not None:
            data["frame_id"] = self.frame_id
        if self.prompt is not None:
            data["prompt"] = self.prompt
        for k in ("mask", "roi_x", "roi_y", "roi_w", "roi_h"):
            v = getattr(self, k, None)
            if v is not None:
                data[k] = v
        return RealtimeFrameRequest(**data)

    def to_realtime_update(self) -> RealtimeUpdateRequest:
        _update_keys = {
            "prompt", "negative_prompt", "denoise_strength",
            "steps", "cfg_scale", "clip_skip", "seed",
        }
        data = {k: v for k, v in self.model_dump(exclude_none=True).items()
                if k in _update_keys}
        return RealtimeUpdateRequest(**data)

    def to_analyze_audio_request(self) -> AnalyzeAudioRequest:
        return AnalyzeAudioRequest(
            audio_path=self.audio_path or "",
            fps=self.fps or 24.0,
            enable_stems=self.enable_stems or False,
        )

    def to_audio_reactive_request(self) -> AudioReactiveRequest:
        _exclude = {
            "action", "frame_count", "seed_strategy",
            "image", "frame_id", "mask", "roi_x", "roi_y", "roi_w", "roi_h",
            "locked_fields", "prompt_template", "preset_name", "preset_data",
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


class AnalyzeAudioRequest(BaseModel):
    action: Action = Action.ANALYZE_AUDIO
    audio_path: str = ""
    fps: float = Field(24.0, ge=1.0, le=120.0)
    enable_stems: bool = False


class AudioReactiveRequest(BaseModel):
    action: Action = Action.GENERATE_AUDIO_REACTIVE
    audio_path: str = ""
    fps: float = Field(24.0, ge=1.0, le=120.0)
    enable_stems: bool = False
    modulation_slots: list[ModulationSlotSpec] = Field(default_factory=list)
    expressions: Optional[dict[str, str]] = None
    modulation_preset: Optional[str] = None
    prompt_segments: list[dict] = Field(default_factory=list)
    max_frames: Optional[int] = Field(None, ge=1, le=3600)
    # Animation method: chain (default) or animatediff_audio
    method: AnimationMethod = AnimationMethod.CHAIN
    # AnimateDiff-specific
    enable_freeinit: bool = False
    freeinit_iterations: int = Field(2, ge=1, le=3)
    # Generation parameters (same as AnimationRequest)
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
    frame_duration_ms: int = Field(100, ge=30, le=2000)
    tag_name: Optional[str] = Field(None, max_length=64)


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
    image: str          # base64 PNG
    seed: int
    time_ms: int
    width: int
    height: int


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


# ─────────────────────────────────────────────────────────────
# REAL-TIME PAINT RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class RealtimeReadyResponse(BaseModel):
    type: Literal["realtime_ready"] = "realtime_ready"
    message: str = "Real-time mode activated"


class RealtimeResultResponse(BaseModel):
    type: Literal["realtime_result"] = "realtime_result"
    image: str          # base64 PNG
    latency_ms: int
    frame_id: int
    width: int
    height: int
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None


class RealtimeStoppedResponse(BaseModel):
    type: Literal["realtime_stopped"] = "realtime_stopped"
    message: str = "Real-time mode stopped"


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
    recommended_preset: str = ""
    stems_available: bool = False
    stems: Optional[list[str]] = None
    waveform: Optional[list[float]] = None  # mini RMS waveform (100 points, [0,1])


class AudioReactiveFrameResponse(BaseModel):
    type: Literal["audio_reactive_frame"] = "audio_reactive_frame"
    frame_index: int
    total_frames: int
    image: str          # base64 PNG
    seed: int
    time_ms: int
    width: int
    height: int
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
