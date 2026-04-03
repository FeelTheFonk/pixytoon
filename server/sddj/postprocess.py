"""Post-processing pipeline.

Order of operations (non-negotiable):
  1. Background removal  (rembg)
  2. Pixelation           (NEAREST, BOX, or PixelOE downscale)
  3. Color quantization   (MiniBatchKMeans / PIL / Octree-OKLAB)
  4. Palette enforcement  (OKLAB nearest)
  5. Dithering            (Floyd-Steinberg / Bayer, OKLAB palette-aware)
  6. Alpha cleanup        (binary threshold)

Color space: OKLAB (Björn Ottosson, 2020) — replaces CIELAB.
OKLAB is float32 throughout (no float64 intermediaries), has better
perceptual uniformity for saturated colors, and identical Euclidean
distance semantics for nearest-neighbor / KMeans / error diffusion.
L in [0,1], a/b in ~[-0.5, 0.5].
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np
import numba
from PIL import Image

from .protocol import (
    DitherMode,
    PaletteMode,
    PixelateMethod,
    PostProcessSpec,
    QuantizeMethod,
)
from . import palette_manager
from .config import settings
from .oklab import rgb_to_oklab, oklab_to_rgb

log = logging.getLogger("sddj.postprocess")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def _any_processing_active(spec: PostProcessSpec) -> bool:
    """Check if any post-processing stage is enabled."""
    if spec.remove_bg:
        return True
    if spec.upscale_enabled:
        return True
    if spec.pixelate.enabled:
        return True
    if spec.quantize_enabled:
        return True
    if spec.palette.mode != PaletteMode.AUTO:
        return True
    if spec.dither != DitherMode.NONE:
        return True
    return False


# Public alias for use by animation loops (pre-compute outside frame loop)
is_processing_active = _any_processing_active


def apply(image: Image.Image, spec: PostProcessSpec) -> Image.Image:
    """Apply the full post-processing pipeline.

    Returns the image untouched if no processing flags are active.

    Unified ndarray pipeline: converts PIL→ndarray ONCE at entry, passes
    ndarray through all stages, converts back to PIL ONCE at exit.
    Alpha is extracted once and reattached at exit to avoid redundant
    splits across stages.
    """
    # Fast bypass: no processing enabled → return raw image
    if not _any_processing_active(spec):
        return image

    # ── Convert to ndarray ONCE at entry ──────────────────────
    arr = np.array(image.convert("RGBA"))  # H, W, 4 (RGBA uint8)
    alpha = arr[:, :, 3].copy()            # Extract alpha ONCE
    rgb = arr[:, :, :3]                    # RGB view (H, W, 3)

    # 1. Background removal (requires PIL — convert only for this call)
    if spec.remove_bg:
        pil_tmp = Image.fromarray(arr)
        pil_tmp = _remove_background(pil_tmp)
        arr = np.array(pil_tmp)
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3]

    # 1b. Upscale (requires PIL for Real-ESRGAN — convert only for this call)
    if spec.upscale_enabled and settings.enable_upscaler:
        pil_tmp = Image.fromarray(np.dstack([rgb, alpha]))
        pil_tmp = _upscale(pil_tmp, spec.upscale_factor)
        arr = np.array(pil_tmp.convert("RGBA"))
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3]

    # 2. Pixelation (ndarray-native)
    if spec.pixelate.enabled:
        rgb, alpha = _pixelate_ndarray(rgb, alpha, spec.pixelate.target_size, spec.pixelate.method)

    # 3. Color quantization (ndarray-native, only if explicitly enabled)
    kmeans_centers = None
    if spec.quantize_enabled:
        rgb, alpha, kmeans_centers = _quantize_ndarray(rgb, alpha, spec.quantize_method, spec.quantize_colors)

    # 4. Palette enforcement (ndarray-native)
    palette_rgb: Optional[list[tuple[int, int, int]]] = None
    if spec.palette.mode != PaletteMode.AUTO:
        palette_rgb = _resolve_palette(spec.palette)
        if palette_rgb:
            # Skip explicit enforcement if dithering will handle it —
            # Bayer dither calls _enforce_palette internally.
            if spec.dither == DitherMode.NONE:
                rgb = _enforce_palette_ndarray(rgb, palette_rgb)

    # 5. Dithering (OKLAB palette-aware, ndarray-native)
    if spec.dither != DitherMode.NONE:
        # Reuse KMeans centers if available, avoiding a redundant second KMeans run
        if palette_rgb is None:
            if kmeans_centers is not None:
                palette_rgb = kmeans_centers
            else:
                palette_rgb = _extract_palette_from_ndarray(rgb, spec.quantize_colors)
        rgb = _apply_dither_ndarray(rgb, alpha, spec.dither, palette_rgb, has_bg_removal=spec.remove_bg)

    # 6. Alpha cleanup (ndarray-native)
    if spec.remove_bg:
        alpha = _cleanup_alpha_ndarray(alpha)

    # ── Convert to PIL ONCE at exit ───────────────────────────
    return Image.fromarray(np.dstack([rgb, alpha]))


# ─────────────────────────────────────────────────────────────
# STEP 1: BACKGROUND REMOVAL
# ─────────────────────────────────────────────────────────────

def _remove_background(img: Image.Image) -> Image.Image:
    from .rembg_wrapper import remove_bg
    return remove_bg(img)


# ─────────────────────────────────────────────────────────────
# STEP 1b: UPSCALE (Real-ESRGAN)
# ─────────────────────────────────────────────────────────────

_upscaler = None


def _ensure_upscaler():
    global _upscaler
    if _upscaler is not None:
        return _upscaler
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from huggingface_hub import hf_hub_download
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        # Download model weights if not cached
        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4plus_anime_6B.pth",
            local_files_only=False,
        )
        _upscaler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        return _upscaler
    except ImportError:
        log.warning("realesrgan not installed — upscaling unavailable")
        return None
    except Exception as e:
        log.warning("Upscaler initialization failed: %s", e)
        return None


def _upscale(img: Image.Image, factor: int = 4) -> Image.Image:
    """Upscale image using Real-ESRGAN (lazy-loaded)."""
    upscaler = _ensure_upscaler()
    if upscaler is None:
        return img
    try:
        has_alpha = img.mode == "RGBA"
        if has_alpha:
            alpha = img.getchannel("A")
            rgb = img.convert("RGB")
        else:
            rgb = img if img.mode == "RGB" else img.convert("RGB")
        img_array = np.array(rgb)
        output, _ = upscaler.enhance(img_array, outscale=factor)
        result = Image.fromarray(output)
        if has_alpha:
            # Upscale alpha channel to match
            alpha_resized = alpha.resize(result.size, Image.NEAREST)
            result = result.convert("RGBA")
            result.putalpha(alpha_resized)
        return result
    except Exception as e:
        log.warning("Upscale failed: %s — returning original", e)
        return img


# ─────────────────────────────────────────────────────────────
# STEP 2: PIXELATION
# ─────────────────────────────────────────────────────────────


def _pixelate_ndarray(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_size: int,
    method: PixelateMethod = PixelateMethod.NEAREST,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixelate operating on ndarray (rgb H,W,3 uint8 + alpha H,W uint8).

    Returns (rgb, alpha) at the new resolution.
    """
    h, w = rgb.shape[:2]
    # Compute proportional target maintaining aspect ratio
    if w >= h:
        new_w = target_size
        new_h = max(4, round(h * target_size / w))
    else:
        new_h = target_size
        new_w = max(4, round(w * target_size / h))

    if method == PixelateMethod.PIXELOE:
        # F-O10: PixelOE contrast-aware downscaling — produces sharper
        # pixel art by analysing local contrast to preserve edge detail.
        try:
            from pixeloe.pixelize import pixelize
            # PixelOE expects RGB numpy array, returns RGB numpy array.
            result = pixelize(rgb, target_size=min(new_w, new_h))
            # Downscale alpha to match PixelOE output dimensions
            out_h, out_w = result.shape[:2]
            alpha_pil = Image.fromarray(alpha).resize((out_w, out_h), Image.NEAREST)
            return result, np.array(alpha_pil)
        except ImportError:
            log.warning("pixeloe not installed, falling back to box downscale")
            method = PixelateMethod.BOX

    # For NEAREST / BOX: use PIL resize (fast C implementation) on ndarray
    if method == PixelateMethod.BOX:
        resample = Image.BOX
    else:
        resample = Image.NEAREST

    rgb_pil = Image.fromarray(rgb).resize((new_w, new_h), resample)
    alpha_pil = Image.fromarray(alpha).resize((new_w, new_h), Image.NEAREST)
    return np.array(rgb_pil), np.array(alpha_pil)


# ─────────────────────────────────────────────────────────────
# STEP 3: COLOR QUANTIZATION
# ─────────────────────────────────────────────────────────────

def _quantize_ndarray(
    rgb: np.ndarray,
    alpha: np.ndarray,
    method: QuantizeMethod,
    n_colors: int,
) -> tuple[np.ndarray, np.ndarray, Optional[list[tuple[int, int, int]]]]:
    """Quantize operating on ndarray. Returns (rgb, alpha, palette_centers)."""
    if method == QuantizeMethod.KMEANS:
        rgb_out, centers = _quantize_kmeans_ndarray(rgb, n_colors)
        return rgb_out, alpha, centers
    elif method == QuantizeMethod.MEDIAN_CUT:
        rgb_out, alpha_out = _quantize_pil_ndarray(rgb, alpha, n_colors, method=Image.Quantize.MEDIANCUT)
        palette = _extract_palette_fast_ndarray(rgb_out)
        return rgb_out, alpha_out, palette
    elif method == QuantizeMethod.OCTREE:
        rgb_out, alpha_out = _quantize_pil_ndarray(rgb, alpha, n_colors, method=Image.Quantize.MAXCOVERAGE)
        palette = _extract_palette_fast_ndarray(rgb_out)
        return rgb_out, alpha_out, palette
    elif method == QuantizeMethod.OCTREE_LAB:
        rgb_out, centers = _quantize_octree_lab_ndarray(rgb, n_colors)
        return rgb_out, alpha, centers
    raise ValueError(f"Unknown quantize method: {method}")


def _quantize_kmeans_ndarray(
    rgb: np.ndarray,
    n_colors: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """KMeans quantization on ndarray (H,W,3 uint8). Returns (rgb, palette)."""
    from sklearn.cluster import MiniBatchKMeans

    h, w, _ = rgb.shape
    # O-18: Defer float32 copy until after fast-path check
    pixels_u8 = rgb.reshape(-1, 3)
    n_pixels = len(pixels_u8)
    n_colors = min(n_colors, n_pixels)

    # Fast approximate unique color count (sample-based to avoid full sort)
    sample_size = min(4096, n_pixels)
    if sample_size < n_pixels:
        indices = np.random.default_rng(42).choice(n_pixels, sample_size, replace=False)
        sample = pixels_u8[indices]
    else:
        sample = pixels_u8
    n_approx_unique = len(np.unique(sample, axis=0))
    if n_approx_unique <= n_colors:
        # Full check only if sample suggests few unique colors
        unique_colors = np.unique(pixels_u8, axis=0)
        if len(unique_colors) <= n_colors:
            centers = [tuple(int(x) for x in c) for c in unique_colors]
            return rgb, centers

    # O-18: float32 conversion only when KMeans is actually needed
    pixels = pixels_u8.astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        batch_size=min(4096, n_pixels),
        n_init=3,
        max_iter=100,
        random_state=42,
    )
    labels = kmeans.fit_predict(pixels)
    # Consistent rounding: round THEN cast to uint8 (avoids truncation mismatch)
    centers_rounded = np.round(kmeans.cluster_centers_).astype(np.uint8)
    quantized = centers_rounded[labels].reshape(h, w, 3)

    # Extract palette from rounded centers for downstream use
    palette = [tuple(int(x) for x in c) for c in centers_rounded]
    return quantized, palette


def _quantize_octree_lab_ndarray(
    rgb: np.ndarray,
    n_colors: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Octree quantization in OKLAB space on ndarray (H,W,3 uint8).

    Returns (rgb, palette). Perceptually uniform palettes via
    MiniBatchKMeans in OKLAB space with deterministic init.

    Migrated from CIELAB (skimage float64) to OKLAB (float32).
    """
    from sklearn.cluster import MiniBatchKMeans

    h, w, _ = rgb.shape

    # Convert to OKLAB for perceptually uniform clustering (float32 throughout)
    img_ok = rgb_to_oklab(rgb.astype(np.float32) / 255.0)
    pixels_ok = img_ok.reshape(-1, 3)
    n_pixels = len(pixels_ok)
    n_colors = min(n_colors, n_pixels)

    # M-34: Fast-path: if image has fewer unique colors than requested (single np.unique)
    unique_ok = np.unique(pixels_ok.round(4), axis=0)
    n_unique = len(unique_ok)
    if n_unique <= n_colors:
        # Convert unique OKLAB centers back to RGB
        centers_rgb_float = oklab_to_rgb(unique_ok.reshape(1, -1, 3))
        centers_rgb = np.clip(centers_rgb_float * 255, 0, 255).astype(np.uint8).reshape(-1, 3)
        palette = [tuple(int(x) for x in c) for c in centers_rgb]
        # Map each pixel to nearest center
        from scipy.spatial import cKDTree
        tree = cKDTree(unique_ok)
        _, nearest_idx = tree.query(pixels_ok)
        quantized = centers_rgb[nearest_idx].reshape(h, w, 3)
        return quantized, palette

    # MiniBatchKMeans in OKLAB space
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        batch_size=min(4096, n_pixels),
        n_init=3,
        max_iter=100,
        random_state=42,
    )
    labels = kmeans.fit_predict(pixels_ok)
    centers_ok = kmeans.cluster_centers_

    # Convert OKLAB centers back to RGB
    centers_rgb_float = oklab_to_rgb(centers_ok.reshape(1, -1, 3))
    centers_rgb = np.clip(centers_rgb_float * 255, 0, 255).astype(np.uint8).reshape(-1, 3)
    quantized = centers_rgb[labels].reshape(h, w, 3)

    palette = [tuple(int(x) for x in c) for c in centers_rgb]
    return quantized, palette


def _quantize_pil_ndarray(
    rgb: np.ndarray,
    alpha: np.ndarray,
    n_colors: int,
    method: Image.Quantize,
) -> tuple[np.ndarray, np.ndarray]:
    """PIL-based quantization operating on ndarray. Returns (rgb, alpha).

    Converts to PIL only for the quantize() call, then back to ndarray.
    Alpha is binarized before quantization to avoid edge artifacts.
    """
    # Binarize alpha to avoid edge artifacts
    alpha_out = np.where(alpha >= 128, np.uint8(255), np.uint8(0))

    # PIL quantize needs a PIL RGB image — convert only for this call
    rgb_pil = Image.fromarray(rgb)
    quantized = rgb_pil.quantize(colors=n_colors, method=method, dither=0)
    result_pil = quantized.convert("RGB")
    return np.array(result_pil), alpha_out


# ─────────────────────────────────────────────────────────────
# STEP 4: PALETTE ENFORCEMENT (OKLAB NEAREST NEIGHBOR)
# ─────────────────────────────────────────────────────────────

def _resolve_palette(spec) -> list[tuple[int, int, int]] | None:
    if spec.mode == PaletteMode.PRESET and spec.name:
        return palette_manager.load_palette(spec.name)
    if spec.mode == PaletteMode.CUSTOM and spec.colors:
        return palette_manager.hex_list_to_rgb(spec.colors)
    return None


@functools.lru_cache(maxsize=32)
def _palette_to_oklab(palette_key: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    """Convert palette RGB tuples to OKLAB array (cached).

    Migrated from CIELAB (_palette_to_lab) to OKLAB — float32 throughout.
    """
    palette_arr = np.array(palette_key, dtype=np.float32).reshape(1, -1, 3) / 255.0
    return rgb_to_oklab(palette_arr).reshape(-1, 3)


@functools.lru_cache(maxsize=32)
def _build_palette_tree(palette_key: tuple[tuple[int, int, int], ...]):
    """Build and cache cKDTree for palette OKLAB colors."""
    from scipy.spatial import cKDTree
    palette_ok = _palette_to_oklab(palette_key)
    return cKDTree(palette_ok)


def _enforce_palette_ndarray(
    rgb: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
) -> np.ndarray:
    """Snap every pixel to the nearest palette color in OKLAB space.

    Operates on ndarray (H,W,3 uint8), returns ndarray (H,W,3 uint8).
    Migrated from CIELAB to OKLAB — Euclidean distance in OKLAB is
    perceptually uniform, same semantics as CIEDE76 but float32.
    """
    if not palette_rgb:
        return rgb

    h, w, _ = rgb.shape

    # Convert image to OKLAB
    img_ok = rgb_to_oklab(rgb.astype(np.float32) / 255.0)

    # Get cached palette KD-Tree — O(n log k) nearest neighbor
    palette_key = tuple(tuple(c) for c in palette_rgb)
    tree = _build_palette_tree(palette_key)

    # Flatten image to (N, 3) OKLAB pixels
    pixels_ok = img_ok.reshape(-1, 3)
    _, nearest_idx = tree.query(pixels_ok)

    palette_uint8 = np.array(palette_rgb, dtype=np.uint8)
    return palette_uint8[nearest_idx].reshape(h, w, 3)


# ─────────────────────────────────────────────────────────────
# STEP 5: DITHERING (OKLAB PALETTE-AWARE)
# ─────────────────────────────────────────────────────────────

def _apply_dither_ndarray(
    rgb: np.ndarray,
    alpha: np.ndarray,
    mode: DitherMode,
    palette_rgb: list[tuple[int, int, int]],
    *,
    has_bg_removal: bool = False,
) -> np.ndarray:
    """Dither dispatch operating on ndarray. Returns rgb (H,W,3 uint8)."""
    # Dithering with <= 1 color is a no-op
    if len(palette_rgb) <= 1:
        return rgb
    if mode == DitherMode.FLOYD_STEINBERG:
        return _floyd_steinberg_ndarray(rgb, alpha, palette_rgb, alpha_aware=has_bg_removal)
    elif mode in (DitherMode.BAYER_2X2, DitherMode.BAYER_4X4, DitherMode.BAYER_8X8):
        size = {
            DitherMode.BAYER_2X2: 2,
            DitherMode.BAYER_4X4: 4,
            DitherMode.BAYER_8X8: 8,
        }[mode]
        return _bayer_dither_ndarray(rgb, alpha, palette_rgb, size, alpha_aware=has_bg_removal)
    return rgb


def _extract_palette_fast_ndarray(rgb: np.ndarray) -> list[tuple[int, int, int]]:
    """Extract unique colors from a quantized ndarray (H,W,3 uint8)."""
    unique = np.unique(rgb.reshape(-1, 3), axis=0)
    return [tuple(int(x) for x in c) for c in unique]


def _extract_palette_from_ndarray(rgb: np.ndarray, n_colors: int) -> list[tuple[int, int, int]]:
    """Extract n_colors dominant colors from ndarray (H,W,3 uint8).

    Uses fast unique-color extraction when image has ≤n_colors unique
    colors (common after pixelation), falls back to KMeans only for
    high-color-count images.
    """
    pixels = rgb.reshape(-1, 3)
    unique = np.unique(pixels, axis=0)

    # Fast path: image already has few enough unique colors
    if len(unique) <= n_colors:
        return [tuple(int(x) for x in c) for c in unique]

    # Fallback: KMeans extraction
    from sklearn.cluster import MiniBatchKMeans

    pixels_f = pixels.astype(np.float32)
    k = min(n_colors, len(unique))

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(4096, len(pixels_f)),
        n_init=3,
        random_state=42,
    )
    kmeans.fit(pixels_f)
    return [tuple(int(x) for x in np.round(c)) for c in kmeans.cluster_centers_]


# ── Floyd-Steinberg: OKLAB error diffusion ───────────────────

@numba.jit(nopython=True, cache=True, parallel=True)
def _bayer_snap_oklab(ok_img, pal_ok, pal_rgb, alpha_mask):
    """Bayer dither palette snap — embarrassingly parallel nearest-neighbor.

    Each pixel independently finds its nearest palette color in OKLAB space.
    Uses numba parallel=True + prange for multi-core acceleration.

    Args:
        ok_img: float32 (H, W, 3) image in OKLAB space (with Bayer offset applied).
        pal_ok: float32 (N, 3) palette in OKLAB space.
        pal_rgb: uint8 (N, 3) palette in RGB space (output mapping).
        alpha_mask: bool (H, W) — True for opaque pixels, False for transparent.

    Returns:
        uint8 (H, W, 3) RGB result image.
    """
    h, w, _ = ok_img.shape
    n_pal = len(pal_ok)
    result = np.empty((h, w, 3), dtype=numba.uint8)

    for y in numba.prange(h):
        for x in range(w):
            if not alpha_mask[y, x]:
                result[y, x, 0] = 0
                result[y, x, 1] = 0
                result[y, x, 2] = 0
                continue

            pL = ok_img[y, x, 0]
            pa = ok_img[y, x, 1]
            pb = ok_img[y, x, 2]

            min_dist = 1e18
            best = 0
            for i in range(n_pal):
                dL = pal_ok[i, 0] - pL
                da = pal_ok[i, 1] - pa
                db = pal_ok[i, 2] - pb
                d = dL * dL + da * da + db * db
                if d < min_dist:
                    min_dist = d
                    best = i

            result[y, x, 0] = pal_rgb[best, 0]
            result[y, x, 1] = pal_rgb[best, 1]
            result[y, x, 2] = pal_rgb[best, 2]

    return result


@numba.jit(nopython=True, cache=True)
def _fs_core_oklab(ok_img, pal_ok, pal_rgb, alpha_mask):
    """Floyd-Steinberg error-diffusion kernel in OKLAB space (Numba-accelerated).

    Operates entirely in OKLAB for perceptually uniform error diffusion:
    - Distance computation uses Euclidean L2 in OKLAB, which is perceptually
      uniform and sufficient for the 8-32 color palettes typical in pixel art.
    - Error propagation in perceptual space produces natural dithering
      where equal error magnitude = equal perceived difference.

    Migrated from CIELAB (_fs_core_lab) to OKLAB. Algorithm is identical;
    only the value ranges differ: L in [0,1], a/b in ~[-0.5, 0.5].

    Args:
        ok_img: float32 (H, W, 3) image in OKLAB space.
        pal_ok: float32 (N, 3) palette in OKLAB space.
        pal_rgb: uint8 (N, 3) palette in RGB space (output mapping).
        alpha_mask: bool (H, W) — True for opaque pixels, False for transparent.

    Returns:
        uint8 (H, W, 3) RGB result image.
    """
    h, w, _ = ok_img.shape
    n_pal = len(pal_ok)
    result = np.empty((h, w, 3), dtype=numba.uint8)

    for y in range(h):
        for x in range(w):
            # Skip transparent pixels — no error diffusion to/from them
            if not alpha_mask[y, x]:
                result[y, x, 0] = 0
                result[y, x, 1] = 0
                result[y, x, 2] = 0
                continue

            old_L = ok_img[y, x, 0]
            old_a = ok_img[y, x, 1]
            old_b = ok_img[y, x, 2]

            # Find nearest palette color in OKLAB (Euclidean distance)
            min_dist = 1e18
            best = 0
            for i in range(n_pal):
                dL = pal_ok[i, 0] - old_L
                da = pal_ok[i, 1] - old_a
                db = pal_ok[i, 2] - old_b
                d = dL * dL + da * da + db * db
                if d < min_dist:
                    min_dist = d
                    best = i

            # Map to RGB output
            result[y, x, 0] = pal_rgb[best, 0]
            result[y, x, 1] = pal_rgb[best, 1]
            result[y, x, 2] = pal_rgb[best, 2]

            # Compute error in OKLAB space
            err_L = old_L - pal_ok[best, 0]
            err_a = old_a - pal_ok[best, 1]
            err_b = old_b - pal_ok[best, 2]

            # C-18: Redistribute error among opaque neighbors only.
            # When a neighbor is transparent, its weight is zeroed and the
            # remaining weights are renormalized so total error is preserved.
            w_r = 7.0 if (x + 1 < w and alpha_mask[y, x + 1]) else 0.0
            w_bl = 3.0 if (y + 1 < h and x - 1 >= 0 and alpha_mask[y + 1, x - 1]) else 0.0
            w_b = 5.0 if (y + 1 < h and alpha_mask[y + 1, x]) else 0.0
            w_br = 1.0 if (y + 1 < h and x + 1 < w and alpha_mask[y + 1, x + 1]) else 0.0
            total_w = w_r + w_bl + w_b + w_br
            if total_w > 0.0:
                inv_w = 1.0 / total_w
                if w_r > 0.0:
                    frac = w_r * inv_w
                    ok_img[y, x + 1, 0] += err_L * frac
                    ok_img[y, x + 1, 1] += err_a * frac
                    ok_img[y, x + 1, 2] += err_b * frac
                if w_bl > 0.0:
                    frac = w_bl * inv_w
                    ok_img[y + 1, x - 1, 0] += err_L * frac
                    ok_img[y + 1, x - 1, 1] += err_a * frac
                    ok_img[y + 1, x - 1, 2] += err_b * frac
                if w_b > 0.0:
                    frac = w_b * inv_w
                    ok_img[y + 1, x, 0] += err_L * frac
                    ok_img[y + 1, x, 1] += err_a * frac
                    ok_img[y + 1, x, 2] += err_b * frac
                if w_br > 0.0:
                    frac = w_br * inv_w
                    ok_img[y + 1, x + 1, 0] += err_L * frac
                    ok_img[y + 1, x + 1, 1] += err_a * frac
                    ok_img[y + 1, x + 1, 2] += err_b * frac
    return result


def _floyd_steinberg_ndarray(
    rgb: np.ndarray,
    alpha: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    *,
    alpha_aware: bool = False,
) -> np.ndarray:
    """Floyd-Steinberg error-diffusion dithering on ndarray.

    Operates on rgb (H,W,3 uint8) + alpha (H,W uint8).
    Returns rgb (H,W,3 uint8). Alpha is not modified.

    OKLAB, palette-aware, Numba-accelerated. Float32 throughout.
    """
    h, w, _ = rgb.shape

    # Convert image to OKLAB (float32 — no float64 intermediate)
    img_ok = rgb_to_oklab(rgb.astype(np.float32) / 255.0)

    # Convert palette to OKLAB (cached)
    palette_key = tuple(tuple(c) for c in palette_rgb)
    pal_ok = _palette_to_oklab(palette_key)
    pal_rgb = np.array(palette_rgb, dtype=np.uint8)

    # Build alpha mask: True for opaque pixels
    if alpha_aware:
        alpha_mask = alpha >= 128
    else:
        alpha_mask = np.ones((h, w), dtype=np.bool_)

    return _fs_core_oklab(img_ok, pal_ok, pal_rgb, alpha_mask)


# ── Bayer: OKLAB ordered dithering ───────────────────────────

@functools.lru_cache(maxsize=4)  # M-32: only 3 sizes used (2, 4, 8)
def _bayer_matrix(n: int) -> np.ndarray:
    """Generate normalized Bayer threshold matrix of size n×n (cached)."""
    return _bayer_matrix_unnorm(n) / (n * n)


def _bayer_matrix_unnorm(n: int) -> np.ndarray:
    """Generate unnormalized Bayer matrix recursively."""
    if n == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.float32)
    smaller = _bayer_matrix_unnorm(n // 2)
    return np.block([
        [4 * smaller + 0, 4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1],
    ])


def _bayer_dither_ndarray(
    rgb: np.ndarray,
    alpha: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    matrix_size: int,
    *,
    alpha_aware: bool = False,
) -> np.ndarray:
    """Ordered (Bayer) dithering on ndarray with OKLAB palette snap.

    Operates on rgb (H,W,3 uint8) + alpha (H,W uint8).
    Returns rgb (H,W,3 uint8). Alpha is not modified.

    Eliminates double-quantization: applies threshold offset in OKLAB L
    channel then snaps directly to nearest palette color via Numba kernel.
    """
    h, w, _ = rgb.shape

    # Convert to OKLAB (float32 throughout — no float64 intermediate)
    img_ok = rgb_to_oklab(rgb.astype(np.float32) / 255.0)

    # Bayer threshold in OKLAB L channel (perceptual lightness)
    threshold = _bayer_matrix(matrix_size)
    th_tiled = np.tile(threshold, (h // matrix_size + 1, w // matrix_size + 1))
    th_tiled = th_tiled[:h, :w]

    # Apply threshold as L offset (OKLAB L range is [0, 1], was [0, 100] in CIELAB)
    n_colors = len(palette_rgb)
    l_step = 1.0 / max(1, n_colors - 1)
    offset = (th_tiled - 0.5) * l_step

    # Alpha-aware: zero out offset for transparent pixels before addition
    if alpha_aware:
        alpha_mask = alpha < 128
        offset[alpha_mask] = 0

    img_ok[:, :, 0] += offset

    # Snap to nearest palette color via parallel Numba kernel
    palette_key = tuple(tuple(c) for c in palette_rgb)
    pal_ok = _palette_to_oklab(palette_key)
    pal_rgb_arr = np.array(palette_rgb, dtype=np.uint8)

    # Build alpha mask for the kernel
    if alpha_aware:
        bayer_alpha_mask = alpha >= 128
    else:
        bayer_alpha_mask = np.ones((h, w), dtype=np.bool_)

    return _bayer_snap_oklab(img_ok, pal_ok, pal_rgb_arr, bayer_alpha_mask)


# ─────────────────────────────────────────────────────────────
# NUMBA JIT WARMUP
# ─────────────────────────────────────────────────────────────

def warmup_numba() -> None:
    """Pre-compile Floyd-Steinberg and Bayer JIT kernels with float32 data."""
    _dummy_img = np.zeros((2, 2, 3), dtype=np.float32)
    _dummy_pal_ok = np.zeros((2, 3), dtype=np.float32)
    _dummy_pal_rgb = np.zeros((2, 3), dtype=np.uint8)
    _dummy_mask = np.ones((2, 2), dtype=np.bool_)
    _fs_core_oklab(_dummy_img, _dummy_pal_ok, _dummy_pal_rgb, _dummy_mask)
    _bayer_snap_oklab(
        _dummy_img.copy(), _dummy_pal_ok, _dummy_pal_rgb, _dummy_mask,
    )


# ─────────────────────────────────────────────────────────────
# STEP 6: ALPHA CLEANUP
# ─────────────────────────────────────────────────────────────

def _cleanup_alpha_ndarray(alpha: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Binarize alpha channel — no semi-transparency.

    Operates on alpha (H,W uint8), returns alpha (H,W uint8).
    """
    return np.where(alpha >= threshold, np.uint8(255), np.uint8(0))
