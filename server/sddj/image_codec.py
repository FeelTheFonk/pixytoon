"""Image encode/decode/resize utilities for the diffusion engine."""

from __future__ import annotations

import functools
import hashlib
import math
import threading
from base64 import b64decode, b64encode
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# ── Thread lock for mutable module-level caches ───────────────
_codec_lock = threading.Lock()

# ── Cached matrices (computed once, reused across frames) ─────
# C-17: _K_INV_CACHE and _FLOW_GRID_CACHE have deterministic keys → lru_cache (see functions).
# _REF_LAB_CACHE uses content-based key (C-16) and is guarded by _codec_lock (C-17).
_REF_LAB_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _img_cache_key(img) -> str:
    """Content-based cache key for images (C-16).

    Stride-samples ~1024 elements across the full pixel range to avoid
    collisions from images sharing identical top-left corners.
    """
    arr = np.asarray(img)
    n = arr.size
    if n <= 1024:
        sample = arr.flat[:].tobytes()
    else:
        stride = n // 1024
        sample = arr.flat[::stride][:1024].tobytes()
    return hashlib.md5(sample).hexdigest()


@functools.lru_cache(maxsize=16)
def _get_k_inv(w: int, h: int, fv: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache camera intrinsic matrix K and its inverse K_inv (C-17)."""
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fv, 0, cx], [0, fv, cy], [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    return K, K_inv


@functools.lru_cache(maxsize=8)
def _get_flow_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache meshgrid for optical flow remapping (C-17)."""
    return np.mgrid[:h, :w].astype(np.float32)


def round8(v: int) -> int:
    """Round to nearest multiple of 8 (SD1.5 VAE requirement), clamped 8–2048."""
    return max(8, min(2048, ((v + 4) // 8) * 8))


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


def encode_image_b64(image: Image.Image, compress_level: int = 0) -> str:
    """Encode a PIL Image to base64 PNG string.

    .. deprecated::
        M-35: PNG encode path is deprecated for frame transport. Use
        ``encode_image_raw_bytes`` or ``encode_image_raw_b64`` for
        zero-copy binary WS frames instead. PNG encode remains for
        single-image export and debug only.

    Args:
        compress_level: 0 = fastest (no compression), 1 = fast, 9 = smallest.
    """
    buf = BytesIO()
    image.save(buf, format="PNG", compress_level=compress_level)
    return b64encode(buf.getvalue()).decode("ascii")


def encode_image_raw_bytes(image: Image.Image) -> bytes:
    """Return raw RGBA pixel bytes for binary frame transport.

    No base64, no PNG compression — raw pixels for zero-copy binary WS frames.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image.tobytes()


def encode_image_raw_b64(image: Image.Image) -> str:
    """Encode PIL Image as raw RGBA bytes → base64.

    No PNG compression — raw pixel data for maximum throughput in local
    inter-process transport. The client reconstructs the Image directly
    from the raw bytes using Image.bytes (Aseprite API), bypassing
    temp file I/O and PNG decode entirely.

    Payload size: width × height × 4 bytes (before base64).
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    raw = image.tobytes()
    return b64encode(raw).decode("ascii")


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
    # Safety guard: skip motion when denoise is too low for the model
    # to absorb warp artifacts (< 4 effective steps at 8-step/cap-2).
    if denoise_strength < 0.25:
        return image

    # Correlation: scale motion by denoise_strength (clamped 0.15-0.8)
    scale = max(0.15, min(0.8, denoise_strength))
    eff_tx = tx * scale
    eff_ty = ty * scale
    eff_zoom = 1.0 + (zoom - 1.0) * scale
    eff_rot = rotation * scale

    # warpAffine uses inverse mapping (dst→src): scale > 1 in the matrix
    # means each dst pixel samples from MORE src pixels = zoom OUT.
    # Invert so zoom > 1 = zoom IN (Deforum convention).
    inv_zoom = 1.0 / eff_zoom if abs(eff_zoom) > 1e-6 else 1.0

    # Skip if negligible motion (saves CPU)
    total_motion = abs(eff_tx) + abs(eff_ty) + abs(inv_zoom - 1.0) * 100.0 + abs(eff_rot)
    if total_motion < 0.05:
        return image

    w, h = image.size
    cx, cy = w / 2.0, h / 2.0

    # Build combined affine matrix: translate to center, rotate+scale, translate back + pan
    cos_a = math.cos(math.radians(eff_rot)) * inv_zoom
    sin_a = math.sin(math.radians(eff_rot)) * inv_zoom

    # 2x3 affine matrix (float32 sufficient for affine precision)
    M = np.array([
        [cos_a, -sin_a, (1.0 - cos_a) * cx + sin_a * cy + eff_tx],
        [sin_a,  cos_a, -sin_a * cx + (1.0 - cos_a) * cy + eff_ty],
    ], dtype=np.float32)

    # Convert PIL -> numpy (preserve alpha if present)
    arr = np.array(image)

    # Apply warp with edge replication (Deforum pattern — avoids
    # mirror-image artifacts from REFLECT that compound when
    # denoising cannot fully absorb the warp).
    warped = cv2.warpAffine(
        arr, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return Image.fromarray(warped)


def apply_perspective_tilt(
    image: Image.Image,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    denoise_strength: float = 0.5,
) -> Image.Image:
    """Apply faux-3D perspective tilt via homography warp.

    Inspired by Deforum's perspective_flip_phi/theta. Uses a rotation
    matrix projected through a virtual camera to produce a 3×3
    homography, giving the illusion of 3D pitch/yaw on a 2D canvas.

    Args:
        tilt_x: Pitch angle in degrees (positive = tilt top away).
        tilt_y: Yaw angle in degrees (positive = tilt right side away).
        denoise_strength: Current denoise value — scales effective tilt.

    Returns:
        Warped PIL image (same size and mode).
    """
    if denoise_strength < 0.25:
        return image

    scale = max(0.15, min(0.8, denoise_strength))
    eff_tx = tilt_x * scale
    eff_ty = tilt_y * scale

    if abs(eff_tx) + abs(eff_ty) < 0.01:
        return image

    w, h = image.size
    cx, cy = w / 2.0, h / 2.0
    # Focal length in pixels — larger = subtler perspective
    fv = float(max(w, h))

    # Build 3D rotation matrix (pitch around X, yaw around Y)
    ax = math.radians(eff_tx)
    ay = math.radians(eff_ty)

    # Rotation around X axis (pitch)
    Rx = np.array([
        [1, 0,            0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax),  math.cos(ax)],
    ], dtype=np.float32)

    # Rotation around Y axis (yaw)
    Ry = np.array([
        [ math.cos(ay), 0, math.sin(ay)],
        [ 0,            1, 0],
        [-math.sin(ay), 0, math.cos(ay)],
    ], dtype=np.float32)

    R = Ry @ Rx

    # Camera intrinsic matrix + cached inverse (C-17: lru_cache, thread-safe)
    K, K_inv = _get_k_inv(w, h, fv)

    # Homography: H = K · R · K⁻¹
    H = K @ R @ K_inv

    arr = np.array(image)
    warped = cv2.warpPerspective(
        arr, H, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
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

    img_lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    img_means, img_stds = cv2.meanStdDev(img_lab)
    img_means = img_means.flatten()
    img_stds = img_stds.flatten()

    # Cache reference frame LAB stats (same reference across animation frames)
    # C-16: content-based key (not id()), C-17: thread-safe access
    ref_key = _img_cache_key(reference)
    with _codec_lock:
        if ref_key in _REF_LAB_CACHE:
            ref_means, ref_stds = _REF_LAB_CACHE[ref_key]
        else:
            ref_lab = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2LAB)
            ref_means, ref_stds = cv2.meanStdDev(ref_lab)
            ref_means = ref_means.flatten()
            ref_stds = ref_stds.flatten()
            _REF_LAB_CACHE[ref_key] = (ref_means, ref_stds)
            if len(_REF_LAB_CACHE) > 8:
                oldest = next(iter(_REF_LAB_CACHE))
                del _REF_LAB_CACHE[oldest]

    img_lab_f = img_lab.astype(np.float32)
    for ch in range(3):
        if img_stds[ch] < 1e-6 or ref_stds[ch] < 1e-6:
            continue
        # Use inplace operations to minimize further allocation
        view = img_lab_f[:, :, ch]
        view -= img_means[ch]
        view *= (ref_stds[ch] / img_stds[ch])
        view += ref_means[ch]

    img_lab = np.clip(img_lab_f, 0, 255).astype(np.uint8)
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

    # O-15: DIS optical flow — faster and more robust than Farneback
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(prev_gray, curr_gray, None)

    h, w = curr_gray.shape
    # C-17: thread-safe lru_cache grid
    grid_y, grid_x = _get_flow_grid(h, w)
    
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    warped = cv2.remap(
        prev_arr, map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    result = cv2.addWeighted(curr_arr, 1.0 - strength, warped, strength, 0)
    return Image.fromarray(result)
