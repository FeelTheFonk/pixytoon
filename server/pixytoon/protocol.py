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


class SeedStrategy(str, Enum):
    FIXED = "fixed"
    INCREMENT = "increment"
    RANDOM = "random"


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class LoRASpec(BaseModel):
    name: str
    weight: float = Field(1.0, ge=-2.0, le=2.0)


class EmbeddingSpec(BaseModel):
    name: str
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
    tag_name: Optional[str] = None
    # AnimateDiff-specific
    enable_freeinit: bool = False
    freeinit_iterations: int = Field(2, ge=1, le=3)


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

    def to_generate_request(self) -> GenerateRequest:
        _anim_fields = {
            "action", "method", "frame_count", "frame_duration_ms",
            "seed_strategy", "tag_name", "enable_freeinit", "freeinit_iterations",
        }
        data = self.model_dump(exclude_none=True, exclude=_anim_fields)
        return GenerateRequest(**data)

    def to_animation_request(self) -> AnimationRequest:
        data = self.model_dump(exclude_none=True, exclude={"action"})
        return AnimationRequest(**data)


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
    list_type: str  # "loras" | "palettes" | "controlnets" | "embeddings"
    items: list[str]


class PongResponse(BaseModel):
    type: Literal["pong"] = "pong"
