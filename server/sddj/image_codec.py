"""Image encode/decode/resize utilities for the diffusion engine."""

from __future__ import annotations

import math
from base64 import b64decode, b64encode
from io import BytesIO

import cv2
import numpy as np
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


def encode_image_b64(image: Image.Image, compress_level: int = 1) -> str:
    """Encode a PIL Image to base64 PNG string.

    Args:
        compress_level: 0 = fastest (no compression), 1 = fast, 9 = smallest.
    """
    buf = BytesIO()
    image.save(buf, format="PNG", compress_level=compress_level)
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
    Applies binary threshold (128) to avoid soft edges.
    Both images must be same size. Mask is converted to binary L mode.
    """
    if original.size != inpainted.size:
        raise ValueError(f"Size mismatch: original {original.size} vs inpainted {inpainted.size}")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)

    # Ensure all images are same mode for compositing
    if original.mode != inpainted.mode:
        inpainted = inpainted.convert(original.mode)

    # Binary threshold — no anti-aliasing
    mask_binary = mask.point(lambda p: 255 if p >= 128 else 0)

    return Image.composite(inpainted, original, mask_binary)


def apply_motion_warp(
    image: Image.Image,
    tx: float = 0.0,
    ty: float = 0.0,
    zoom: float = 1.0,
    rotation: float = 0.0,
    denoise_strength: float = 0.5,
) -> Image.Image:
    """Apply smooth 2D affine warp with denoise-correlated motion scaling.

    Motion amplitude is scaled by denoise_strength to prevent spaghetti:
    higher denoise = model can absorb more change, lower = motion dampened.

    Args:
        image: Source PIL image (RGB).
        tx: Horizontal translation in pixels (negative=left, positive=right).
        ty: Vertical translation in pixels (negative=up, positive=down).
        zoom: Scale factor (1.0=none, >1=zoom in, <1=zoom out).
        rotation: Planar rotation in degrees (positive=counter-clockwise).
        denoise_strength: Current denoise value — scales effective motion.

    Returns:
        Warped PIL image (same size and mode).
    """
    # Correlation: scale motion by denoise_strength (clamped 0.1-0.8)
    scale = max(0.1, min(0.8, denoise_strength))
    eff_tx = tx * scale
    eff_ty = ty * scale
    eff_zoom = 1.0 + (zoom - 1.0) * scale
    eff_rot = rotation * scale

    # Skip if negligible motion (saves CPU)
    total_motion = abs(eff_tx) + abs(eff_ty) + abs(eff_zoom - 1.0) * 100.0 + abs(eff_rot)
    if total_motion < 0.05:
        return image

    w, h = image.size
    cx, cy = w / 2.0, h / 2.0

    # Build combined affine matrix: translate to center, rotate+scale, translate back + pan
    cos_a = math.cos(math.radians(eff_rot)) * eff_zoom
    sin_a = math.sin(math.radians(eff_rot)) * eff_zoom

    # 2x3 affine matrix
    M = np.array([
        [cos_a, -sin_a, (1.0 - cos_a) * cx + sin_a * cy + eff_tx],
        [sin_a,  cos_a, -sin_a * cx + (1.0 - cos_a) * cy + eff_ty],
    ], dtype=np.float64)

    # Convert PIL -> numpy (preserve alpha if present)
    has_alpha = image.mode == "RGBA"
    arr = np.array(image)

    # Apply warp with border reflection (no black edges)
    warped = cv2.warpAffine(
        arr, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return Image.fromarray(warped)


def match_color_lab(
    image: Image.Image,
    reference: Image.Image,
    strength: float = 0.5,
) -> Image.Image:
    """Match the LAB color distribution of *image* to *reference*.

    Transfers per-channel mean and standard deviation in CIELAB space,
    then blends with the original based on *strength* (0 = no change,
    1 = full transfer).  Prevents color drift in frame chains.
    """
    if strength <= 0.0:
        return image

    img_arr = np.array(image, dtype=np.uint8)
    ref_arr = np.array(reference, dtype=np.uint8)

    # Ensure both are 3-channel (strip alpha if present)
    if img_arr.ndim == 2:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    elif img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]
    if ref_arr.ndim == 2:
        ref_arr = cv2.cvtColor(ref_arr, cv2.COLOR_GRAY2RGB)
    elif ref_arr.shape[2] == 4:
        ref_arr = ref_arr[:, :, :3]

    img_lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

    for ch in range(3):
        img_mean = img_lab[:, :, ch].mean()
        img_std = img_lab[:, :, ch].std()
        ref_mean = ref_lab[:, :, ch].mean()
        ref_std = ref_lab[:, :, ch].std()
        if img_std < 1e-6:
            continue
        img_lab[:, :, ch] = (
            (img_lab[:, :, ch] - img_mean) * (ref_std / img_std) + ref_mean
        )

    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    if strength < 1.0:
        matched = cv2.addWeighted(
            img_arr, 1.0 - strength, matched, strength, 0,
        )

    return Image.fromarray(matched)


def apply_optical_flow_blend(
    current: Image.Image,
    previous: Image.Image,
    strength: float = 0.3,
) -> Image.Image:
    """Blend *current* frame with an optical-flow-warped *previous* frame.

    Reduces inter-frame jitter by estimating dense flow (Farneback) from
    *previous* → *current*, warping *previous* to align, then blending.
    """
    if strength <= 0.0:
        return current

    curr_arr = np.array(current, dtype=np.uint8)
    prev_arr = np.array(previous, dtype=np.uint8)

    # Ensure 3-channel
    if curr_arr.ndim == 2:
        curr_arr = cv2.cvtColor(curr_arr, cv2.COLOR_GRAY2RGB)
    elif curr_arr.shape[2] == 4:
        curr_arr = curr_arr[:, :, :3]
    if prev_arr.ndim == 2:
        prev_arr = cv2.cvtColor(prev_arr, cv2.COLOR_GRAY2RGB)
    elif prev_arr.shape[2] == 4:
        prev_arr = prev_arr[:, :, :3]

    # Resize previous to match current if dimensions differ
    if prev_arr.shape[:2] != curr_arr.shape[:2]:
        prev_arr = cv2.resize(
            prev_arr, (curr_arr.shape[1], curr_arr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    curr_gray = cv2.cvtColor(curr_arr, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_arr, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    h, w = curr_gray.shape
    map_y, map_x = np.mgrid[:h, :w].astype(np.float32)
    map_x += flow[..., 0]
    map_y += flow[..., 1]

    warped = cv2.remap(
        prev_arr, map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    result = cv2.addWeighted(curr_arr, 1.0 - strength, warped, strength, 0)
    return Image.fromarray(result)
