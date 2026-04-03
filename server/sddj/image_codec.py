"""Image encode/decode/resize utilities for the diffusion engine."""

from __future__ import annotations

import functools
import hashlib
import math
import threading
from base64 import b64decode
from collections import OrderedDict
from io import BytesIO

# Prefer xxhash for cache key hashing (2-3x faster than MD5 on short inputs).
# Falls back to MD5 when xxhash is not installed.
try:
    import xxhash
    def _fast_hash(data: bytes) -> str:
        return xxhash.xxh64(data).hexdigest()
except ImportError:
    def _fast_hash(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

import cv2
import numpy as np
from PIL import Image

# ── Thread lock for mutable module-level caches ───────────────
_codec_lock = threading.Lock()

# ── Module-level constants (L13b: no magic numbers) ───────────
_MIN_MOTION_THRESHOLD = 0.05
_MIN_DENOISE_FOR_MOTION = 0.25
_MAX_REF_OKLAB_CACHE = 8

# ── Cached matrices (computed once, reused across frames) ─────
# C-17: _K_INV_CACHE and _FLOW_GRID_CACHE have deterministic keys → lru_cache (see functions).
# _REF_OKLAB_CACHE uses content-based key (C-16) and is guarded by _codec_lock (C-17).
# L14: OrderedDict for LRU eviction (move_to_end on access, popitem(last=False) on evict).
_REF_OKLAB_CACHE: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()


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
    return _fast_hash(sample)


_dis_instance = None


def _get_dis_instance():
    """Return a cached DISOpticalFlow instance (avoids re-creating per frame)."""
    global _dis_instance
    if _dis_instance is None:
        _dis_instance = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis_instance


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


def encode_image_raw_bytes(image: Image.Image) -> bytes:
    """Return raw RGBA pixel bytes for binary frame transport.

    No base64, no PNG compression — raw pixels for zero-copy binary WS frames.
    Avoids unnecessary PIL .convert("RGBA") + .tobytes() double-copy when
    possible by working directly with the numpy array.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        # Grayscale → RGBA
        h, w = arr.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = arr
        rgba[:, :, 1] = arr
        rgba[:, :, 2] = arr
        rgba[:, :, 3] = 255
        return rgba.tobytes()
    c = arr.shape[2]
    if c == 4:
        # Already RGBA — single tobytes(), no copy via PIL
        return arr.tobytes()
    if c == 3:
        # RGB → RGBA: add opaque alpha without PIL.convert round-trip
        h, w, _ = arr.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = 255
        return rgba.tobytes()
    # Fallback for exotic modes
    return np.asarray(image.convert("RGBA")).tobytes()


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


def _ensure_rgb3(arr: np.ndarray) -> np.ndarray:
    """Ensure *arr* is a 3-channel RGB uint8 array.

    Strips alpha if 4-channel, converts grayscale to RGB if 2D.
    Returns the original array unchanged when already 3-channel.
    """
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    return arr


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
    if denoise_strength < _MIN_DENOISE_FOR_MOTION:
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
    if total_motion < _MIN_MOTION_THRESHOLD:
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
    if denoise_strength < _MIN_DENOISE_FOR_MOTION:
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


def apply_frame_transforms(
    image: Image.Image,
    warp_params: dict | None = None,
    tilt_params: dict | None = None,
) -> Image.Image:
    """Apply motion warp and perspective tilt with a single PIL↔numpy round-trip.

    Composes affine warp and homographic tilt into one (or two sequential)
    cv2 warp calls, avoiding redundant PIL→numpy→PIL conversions that the
    individual functions perform.

    *warp_params* keys: tx, ty, zoom, rotation, denoise_strength.
    *tilt_params* keys: tilt_x, tilt_y, denoise_strength.

    When only one transform is requested, delegates to the corresponding
    single-transform function for simplicity.  When both are requested,
    the numpy array is shared across both warps and reconverted to PIL
    only once at the end.
    """
    has_warp = warp_params is not None
    has_tilt = tilt_params is not None

    if not has_warp and not has_tilt:
        return image

    # Single transform — delegate to the original function (no savings from
    # fusing when there is only one operation).
    if has_warp and not has_tilt:
        return apply_motion_warp(image, **warp_params)
    if has_tilt and not has_warp:
        return apply_perspective_tilt(image, **tilt_params)

    # ── Both transforms requested: fuse the PIL↔numpy conversions ──

    # --- Affine warp parameters ---
    w_tx = warp_params.get("tx", 0.0)
    w_ty = warp_params.get("ty", 0.0)
    w_zoom = warp_params.get("zoom", 1.0)
    w_rot = warp_params.get("rotation", 0.0)
    w_ds = warp_params.get("denoise_strength", 0.5)

    # --- Perspective tilt parameters ---
    t_tx = tilt_params.get("tilt_x", 0.0)
    t_ty = tilt_params.get("tilt_y", 0.0)
    t_ds = tilt_params.get("denoise_strength", 0.5)

    # Convert PIL → numpy ONCE
    arr = np.array(image)
    w, h = image.size

    # --- Apply affine warp (if significant) ---
    warp_applied = False
    if w_ds >= _MIN_DENOISE_FOR_MOTION:
        scale = max(0.15, min(0.8, w_ds))
        eff_tx = w_tx * scale
        eff_ty = w_ty * scale
        eff_zoom = 1.0 + (w_zoom - 1.0) * scale
        eff_rot = w_rot * scale
        inv_zoom = 1.0 / eff_zoom if abs(eff_zoom) > 1e-6 else 1.0
        total_motion = abs(eff_tx) + abs(eff_ty) + abs(inv_zoom - 1.0) * 100.0 + abs(eff_rot)
        if total_motion >= _MIN_MOTION_THRESHOLD:
            cx, cy = w / 2.0, h / 2.0
            cos_a = math.cos(math.radians(eff_rot)) * inv_zoom
            sin_a = math.sin(math.radians(eff_rot)) * inv_zoom
            M = np.array([
                [cos_a, -sin_a, (1.0 - cos_a) * cx + sin_a * cy + eff_tx],
                [sin_a,  cos_a, -sin_a * cx + (1.0 - cos_a) * cy + eff_ty],
            ], dtype=np.float32)
            arr = cv2.warpAffine(
                arr, M, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warp_applied = True

    # --- Apply perspective tilt (if significant) ---
    tilt_applied = False
    if t_ds >= _MIN_DENOISE_FOR_MOTION:
        scale_t = max(0.15, min(0.8, t_ds))
        eff_t_tx = t_tx * scale_t
        eff_t_ty = t_ty * scale_t
        if abs(eff_t_tx) + abs(eff_t_ty) >= 0.01:
            fv = float(max(w, h))
            ax = math.radians(eff_t_tx)
            ay = math.radians(eff_t_ty)
            Rx = np.array([
                [1, 0,            0],
                [0, math.cos(ax), -math.sin(ax)],
                [0, math.sin(ax),  math.cos(ax)],
            ], dtype=np.float32)
            Ry = np.array([
                [ math.cos(ay), 0, math.sin(ay)],
                [ 0,            1, 0],
                [-math.sin(ay), 0, math.cos(ay)],
            ], dtype=np.float32)
            R = Ry @ Rx
            K, K_inv = _get_k_inv(w, h, fv)
            H = K @ R @ K_inv
            arr = cv2.warpPerspective(
                arr, H, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            tilt_applied = True

    if not warp_applied and not tilt_applied:
        return image

    # Convert numpy → PIL ONCE
    return Image.fromarray(arr)


def match_color_lab(
    image: Image.Image,
    reference: Image.Image,
    strength: float = 0.5,
    frame_id: int | None = None,
    work_buf_f32: np.ndarray | None = None,
    out_buf_u8: np.ndarray | None = None,
) -> Image.Image:
    """Match the OKLAB color distribution of *image* to *reference*.

    Transfers per-channel mean and standard deviation in OKLAB space,
    then blends with the original based on *strength* (0 = no change,
    1 = full transfer).  Prevents color drift in frame chains.

    Migrated from OpenCV CIELAB (uint8 scaled) to OKLAB (float32).
    Function name preserved for backward compatibility with callers.

    Args:
        frame_id: When provided, used as cache key for *reference* OKLAB stats
            instead of computing an MD5 digest (M10 — avoids ~0% hit-rate
            hashing in animation where the reference changes every frame).
        work_buf_f32: Pre-allocated float32 buffer (H, W, 3) for the OKLAB
            working copy.  When *None* a new array is allocated (backward compat).
        out_buf_u8: Pre-allocated uint8 buffer (H, W, 3) for the final sRGB
            output.  When *None* a new array is allocated (backward compat).
    """
    from .oklab import rgb_to_oklab, oklab_to_rgb

    if strength <= 0.0:
        return image

    img_arr = np.array(image, dtype=np.uint8)

    # L10: deduplicated channel normalisation
    img_arr = _ensure_rgb3(img_arr)

    # Convert image to OKLAB (float32 throughout)
    img_ok = rgb_to_oklab(img_arr.astype(np.float32) / 255.0)

    # Compute per-channel mean and std for image
    img_flat = img_ok.reshape(-1, 3)
    img_means = img_flat.mean(axis=0)
    img_stds = img_flat.std(axis=0)

    # Cache reference frame OKLAB stats (same reference across animation frames)
    # M10: use frame_id when available (avoids MD5 with ~0% hit-rate).
    # C-16: content-based key (not id()), C-17: thread-safe access
    ref_key: str = str(frame_id) if frame_id is not None else _img_cache_key(reference)
    with _codec_lock:
        if ref_key in _REF_OKLAB_CACHE:
            # L14: promote to most-recently-used
            _REF_OKLAB_CACHE.move_to_end(ref_key)
            ref_means, ref_stds = _REF_OKLAB_CACHE[ref_key]
        else:
            # Convert reference only on cache miss (avoids np.array + rgb_to_oklab on hits)
            ref_arr = np.array(reference, dtype=np.uint8)
            ref_arr = _ensure_rgb3(ref_arr)
            ref_ok = rgb_to_oklab(ref_arr.astype(np.float32) / 255.0)
            ref_flat = ref_ok.reshape(-1, 3)
            ref_means = ref_flat.mean(axis=0)
            ref_stds = ref_flat.std(axis=0)
            _REF_OKLAB_CACHE[ref_key] = (ref_means, ref_stds)
            if len(_REF_OKLAB_CACHE) > _MAX_REF_OKLAB_CACHE:
                _REF_OKLAB_CACHE.popitem(last=False)

    # M2: reuse pre-allocated float32 buffer when provided
    if work_buf_f32 is not None and work_buf_f32.shape == img_ok.shape:
        np.copyto(work_buf_f32, img_ok)
        img_ok_f = work_buf_f32
    else:
        img_ok_f = img_ok.copy()

    for ch in range(3):
        if img_stds[ch] < 1e-6 or ref_stds[ch] < 1e-6:
            continue
        # Use inplace operations to minimize further allocation
        view = img_ok_f[:, :, ch]
        view -= img_means[ch]
        view *= (ref_stds[ch] / img_stds[ch])
        view += ref_means[ch]

    # Convert back to sRGB uint8
    matched_float = oklab_to_rgb(img_ok_f)
    # M2: reuse pre-allocated uint8 buffer when provided (avoids per-frame heap alloc)
    if out_buf_u8 is not None and out_buf_u8.shape == img_arr.shape and out_buf_u8.dtype == np.uint8:
        np.multiply(matched_float, 255, out=work_buf_f32 if work_buf_f32 is not None and work_buf_f32.shape == matched_float.shape else matched_float)
        src = work_buf_f32 if work_buf_f32 is not None and work_buf_f32.shape == matched_float.shape else matched_float
        np.clip(src, 0, 255, out=src)
        np.copyto(out_buf_u8, src.astype(np.uint8))
        matched = out_buf_u8
    else:
        matched = np.clip(matched_float * 255, 0, 255).astype(np.uint8)

    if strength < 1.0:
        matched = cv2.addWeighted(
            img_arr, 1.0 - strength, matched, strength, 0,
        )

    return Image.fromarray(matched)


def apply_optical_flow_blend(
    current: Image.Image,
    previous: Image.Image,
    strength: float = 0.3,
    map_x_buf: np.ndarray | None = None,
    map_y_buf: np.ndarray | None = None,
) -> Image.Image:
    """Blend *current* frame with an optical-flow-warped *previous* frame.

    Reduces inter-frame jitter by estimating dense flow (Farneback) from
    *previous* → *current*, warping *previous* to align, then blending.

    Args:
        map_x_buf: Pre-allocated float32 buffer (H, W) for the x remap grid.
            When *None* a new array is allocated (backward compat).
        map_y_buf: Pre-allocated float32 buffer (H, W) for the y remap grid.
            When *None* a new array is allocated (backward compat).
    """
    if strength <= 0.0:
        return current

    curr_arr = np.array(current, dtype=np.uint8)
    prev_arr = np.array(previous, dtype=np.uint8)

    # L10: deduplicated channel normalisation
    curr_arr = _ensure_rgb3(curr_arr)
    prev_arr = _ensure_rgb3(prev_arr)

    # Resize previous to match current if dimensions differ
    if prev_arr.shape[:2] != curr_arr.shape[:2]:
        prev_arr = cv2.resize(
            prev_arr, (curr_arr.shape[1], curr_arr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    curr_gray = cv2.cvtColor(curr_arr, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_arr, cv2.COLOR_RGB2GRAY)

    # O-15: DIS optical flow — faster and more robust than Farneback.
    # Cached instance: avoids re-creating internal pyramids each frame.
    flow = _get_dis_instance().calc(prev_gray, curr_gray, None)

    h, w = curr_gray.shape
    # C-17: thread-safe lru_cache grid
    grid_y, grid_x = _get_flow_grid(h, w)

    # M11: reuse pre-allocated map buffers when provided
    if map_x_buf is not None and map_x_buf.shape == (h, w):
        np.add(grid_x, flow[..., 0], out=map_x_buf)
        map_x = map_x_buf
    else:
        map_x = grid_x + flow[..., 0]

    if map_y_buf is not None and map_y_buf.shape == (h, w):
        np.add(grid_y, flow[..., 1], out=map_y_buf)
        map_y = map_y_buf
    else:
        map_y = grid_y + flow[..., 1]

    warped = cv2.remap(
        prev_arr, map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    result = cv2.addWeighted(curr_arr, 1.0 - strength, warped, strength, 0)
    return Image.fromarray(result)
