"""Image encode/decode/resize utilities for the diffusion engine."""

from __future__ import annotations

from base64 import b64decode, b64encode
from io import BytesIO

from PIL import Image


def round8(v: int) -> int:
    """Round to nearest multiple of 8 (SD1.5 VAE requirement), clamped to 2048."""
    return min(2048, ((v + 4) // 8) * 8)


_MAX_IMAGE_PIXELS = 2048 * 2048  # 4M pixels max


def decode_b64_image(data: str) -> Image.Image:
    """Decode a base64-encoded PNG into a PIL Image (preserves alpha if present)."""
    try:
        raw = b64decode(data)
        img = Image.open(BytesIO(raw))
        w, h = img.size
        if w * h > _MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image too large: {w}x{h} ({w * h} pixels, max {_MAX_IMAGE_PIXELS})"
            )
        # Convert non-standard modes to RGB/RGBA
        if img.mode in ("P", "PA", "LA"):
            img = img.convert("RGBA")
        elif img.mode in ("L", "I", "F"):
            img = img.convert("RGB")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
        return img
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e


def encode_image_b64(image: Image.Image) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return b64encode(buf.getvalue()).decode("ascii")


def resize_to_target(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to target dimensions if sizes differ (LANCZOS)."""
    if image.size != (width, height):
        return image.resize((width, height), Image.LANCZOS)
    return image


def decode_b64_mask(data: str) -> Image.Image:
    """Decode a base64-encoded PNG mask into a grayscale PIL Image.

    White (255) = repaint area, Black (0) = keep area.
    """
    try:
        raw = b64decode(data)
        img = Image.open(BytesIO(raw))
        w, h = img.size
        if w * h > _MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Mask too large: {w}x{h} ({w * h} pixels, max {_MAX_IMAGE_PIXELS})"
            )
        return img.convert("L")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid base64 mask data: {e}") from e


def composite_with_mask(
    original: Image.Image,
    inpainted: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Composite inpainted result onto original using mask.

    White pixels in mask take from inpainted, black from original.
    Applies binary threshold (128) to avoid soft edges in pixel art.
    Both images must be same size. Mask is converted to binary L mode.
    """
    if original.size != inpainted.size:
        raise ValueError(f"Size mismatch: original {original.size} vs inpainted {inpainted.size}")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)

    # Ensure all images are same mode for compositing
    if original.mode != inpainted.mode:
        inpainted = inpainted.convert(original.mode)

    # Binary threshold — no anti-aliasing for pixel art
    mask_binary = mask.point(lambda p: 255 if p >= 128 else 0)

    return Image.composite(inpainted, original, mask_binary)
