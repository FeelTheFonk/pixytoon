"""WebSocket JSON protocol — request and response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class Action(str, Enum):
    GENERATE = "generate"
    LIST_LORAS = "list_loras"
    LIST_PALETTES = "list_palettes"
    LIST_CONTROLNETS = "list_controlnets"
    LIST_EMBEDDINGS = "list_embeddings"
    PING = "ping"


class GenerationMode(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
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


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class LoRASpec(BaseModel):
    name: str
    weight: float = Field(1.0, ge=-2.0, le=2.0)


class EmbeddingSpec(BaseModel):
    name: str
    weight: float = Field(1.0, ge=0.0, le=2.0)


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


class GenerateRequest(BaseModel):
    action: Action = Action.GENERATE
    prompt: str = ""
    negative_prompt: str = (
        "blurry, antialiased, smooth gradient, photorealistic, 3d render, "
        "soft edges, anti-aliasing, bokeh, depth of field, "
        "low quality, worst quality, bad quality, jpeg artifacts, watermark, text, logo, "
        "deformed, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, "
        "extra fingers, fused fingers, poorly drawn hands, poorly drawn face, ugly, "
        "realistic, photo, high resolution, complex shading"
    )
    mode: GenerationMode = GenerationMode.TXT2IMG
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    source_image: Optional[str] = None      # base64 PNG (img2img)
    control_image: Optional[str] = None     # base64 PNG (ControlNet)
    seed: int = -1
    steps: int = Field(8, ge=1, le=100)
    cfg_scale: float = Field(5.0, ge=0.0, le=30.0)
    denoise_strength: float = Field(0.75, ge=0.0, le=1.0)
    clip_skip: int = Field(2, ge=1, le=12)
    lora: Optional[LoRASpec] = None
    negative_ti: Optional[list[EmbeddingSpec]] = None
    post_process: PostProcessSpec = Field(default_factory=PostProcessSpec)


class Request(BaseModel):
    action: Action
    # GenerateRequest fields are optional — only required for "generate"
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    mode: Optional[GenerationMode] = None
    width: Optional[int] = None
    height: Optional[int] = None
    source_image: Optional[str] = None
    control_image: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    denoise_strength: Optional[float] = None
    clip_skip: Optional[int] = None
    lora: Optional[LoRASpec] = None
    negative_ti: Optional[list[EmbeddingSpec]] = None
    post_process: Optional[PostProcessSpec] = None

    def to_generate_request(self) -> GenerateRequest:
        data = self.model_dump(exclude_none=True, exclude={"action"})
        return GenerateRequest(**data)


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class ProgressResponse(BaseModel):
    type: str = "progress"
    step: int
    total: int


class ResultResponse(BaseModel):
    type: str = "result"
    image: str          # base64 PNG RGBA
    seed: int
    time_ms: int
    width: int
    height: int


class ErrorResponse(BaseModel):
    type: str = "error"
    code: str = "UNKNOWN"
    message: str


class ListResponse(BaseModel):
    type: str = "list"
    list_type: str  # "loras" | "palettes" | "controlnets" | "embeddings"
    items: list[str]


class PongResponse(BaseModel):
    type: str = "pong"
