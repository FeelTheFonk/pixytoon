"""Post-processing pipeline.

Order of operations (non-negotiable):
  1. Background removal  (rembg)
  2. Pixelation           (NEAREST downscale)
  3. Color quantization   (MiniBatchKMeans / PIL)
  4. Palette enforcement  (CIELAB nearest)
  5. Dithering            (Floyd-Steinberg / Bayer, palette-aware)
  6. Alpha cleanup        (binary threshold)
"""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np
import numba
from PIL import Image

from .protocol import (
    DitherMode,
    PaletteMode,
    PostProcessSpec,
    QuantizeMethod,
)
from . import palette_manager


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def _any_processing_active(spec: PostProcessSpec) -> bool:
    """Check if any post-processing stage is enabled."""
    if spec.remove_bg:
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
    """
    # Fast bypass: no processing enabled → return raw image
    if not _any_processing_active(spec):
        return image

    img = image.convert("RGBA")

    # 1. Background removal
    if spec.remove_bg:
        img = _remove_background(img)

    # 2. Pixelation
    if spec.pixelate.enabled:
        img = _pixelate(img, spec.pixelate.target_size)

    # 3. Color quantization (only if explicitly enabled)
    kmeans_centers = None
    if spec.quantize_enabled:
        img, kmeans_centers = _quantize(img, spec.quantize_method, spec.quantize_colors)

    # 4. Palette enforcement
    palette_rgb: Optional[list[tuple[int, int, int]]] = None
    if spec.palette.mode != PaletteMode.AUTO:
        palette_rgb = _resolve_palette(spec.palette)
        if palette_rgb:
            # Skip explicit enforcement if dithering will handle it —
            # Bayer dither calls _enforce_palette internally.
            if spec.dither == DitherMode.NONE:
                img = _enforce_palette(img, palette_rgb)

    # 5. Dithering (palette-aware — uses resolved palette or extracts from quantized image)
    if spec.dither != DitherMode.NONE:
        # Reuse KMeans centers if available, avoiding a redundant second KMeans run
        if palette_rgb is None:
            if kmeans_centers is not None:
                palette_rgb = kmeans_centers
            else:
                palette_rgb = _extract_palette(img, spec.quantize_colors)
        img = _apply_dither(img, spec.dither, palette_rgb)

    # 6. Alpha cleanup
    if spec.remove_bg:
        img = _cleanup_alpha(img)

    return img


# ─────────────────────────────────────────────────────────────
# STEP 1: BACKGROUND REMOVAL
# ─────────────────────────────────────────────────────────────

def _remove_background(img: Image.Image) -> Image.Image:
    from .rembg_wrapper import remove_bg
    return remove_bg(img)


# ─────────────────────────────────────────────────────────────
# STEP 2: PIXELATION
# ─────────────────────────────────────────────────────────────

def _pixelate(img: Image.Image, target_size: int) -> Image.Image:
    w, h = img.size
    # Compute proportional target maintaining aspect ratio
    if w >= h:
        new_w = target_size
        new_h = max(4, round(h * target_size / w))
    else:
        new_h = target_size
        new_w = max(4, round(w * target_size / h))
    # NEAREST is mandatory — any other interpolation introduces anti-aliasing
    return img.resize((new_w, new_h), Image.NEAREST)


# ─────────────────────────────────────────────────────────────
# STEP 3: COLOR QUANTIZATION
# ─────────────────────────────────────────────────────────────

def _quantize(
    img: Image.Image,
    method: QuantizeMethod,
    n_colors: int,
) -> tuple[Image.Image, Optional[list[tuple[int, int, int]]]]:
    """Quantize and optionally return extracted palette centers."""
    if method == QuantizeMethod.KMEANS:
        return _quantize_kmeans(img, n_colors)
    elif method == QuantizeMethod.MEDIAN_CUT:
        img = _quantize_pil(img, n_colors, method=Image.Quantize.MEDIANCUT)
        return img, _extract_palette_fast(img)
    elif method == QuantizeMethod.OCTREE:
        img = _quantize_pil(img, n_colors, method=Image.Quantize.MAXCOVERAGE)
        return img, _extract_palette_fast(img)
    raise ValueError(f"Unknown quantize method: {method}")


def _quantize_kmeans(
    img: Image.Image,
    n_colors: int,
) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    from sklearn.cluster import MiniBatchKMeans

    arr = np.array(img)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    n_pixels = len(pixels)
    n_colors = min(n_colors, n_pixels)

    # Fast approximate unique color count (sample-based to avoid full sort)
    sample_size = min(4096, n_pixels)
    if sample_size < n_pixels:
        indices = np.random.default_rng(42).choice(n_pixels, sample_size, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels
    n_approx_unique = len(np.unique(sample, axis=0))
    if n_approx_unique <= n_colors:
        # Full check only if sample suggests few unique colors
        n_unique = len(np.unique(pixels, axis=0))
        if n_unique <= n_colors:
            centers = [tuple(int(x) for x in c) for c in np.unique(pixels, axis=0).astype(int)]
            if has_alpha:
                return Image.fromarray(arr), centers
            return Image.fromarray(rgb), centers

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

    if has_alpha:
        result = np.dstack([quantized, alpha])
        return Image.fromarray(result), palette
    return Image.fromarray(quantized), palette


def _quantize_pil(
    img: Image.Image,
    n_colors: int,
    method: Image.Quantize,
) -> Image.Image:
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        alpha = img.getchannel("A")
        # Binarize alpha before quantization to avoid edge artifacts
        alpha_arr = np.array(alpha)
        alpha_arr = np.where(alpha_arr >= 128, 255, 0).astype(np.uint8)
        alpha = Image.fromarray(alpha_arr)

    rgb = img.convert("RGB")
    quantized = rgb.quantize(colors=n_colors, method=method, dither=0)
    result = quantized.convert("RGB")

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result


# ─────────────────────────────────────────────────────────────
# STEP 4: PALETTE ENFORCEMENT (CIELAB NEAREST NEIGHBOR)
# ─────────────────────────────────────────────────────────────

def _resolve_palette(spec) -> list[tuple[int, int, int]] | None:
    if spec.mode == PaletteMode.PRESET and spec.name:
        return palette_manager.load_palette(spec.name)
    if spec.mode == PaletteMode.CUSTOM and spec.colors:
        return palette_manager.hex_list_to_rgb(spec.colors)
    return None


@functools.lru_cache(maxsize=32)
def _palette_to_lab(palette_key: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    """Convert palette RGB tuples to CIELAB array (cached)."""
    from skimage.color import rgb2lab
    palette_arr = np.array(palette_key, dtype=np.float32).reshape(1, -1, 3) / 255.0
    return rgb2lab(palette_arr).reshape(-1, 3)


@functools.lru_cache(maxsize=32)
def _build_palette_tree(palette_key: tuple[tuple[int, int, int], ...]):
    """Build and cache cKDTree for palette LAB colors."""
    from scipy.spatial import cKDTree
    palette_lab = _palette_to_lab(palette_key)
    return cKDTree(palette_lab)


def _enforce_palette(
    img: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
) -> Image.Image:
    if not palette_rgb:
        return img

    from skimage.color import rgb2lab

    arr = np.array(img)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    h, w, _ = rgb.shape

    # Convert image to CIELAB
    img_lab = rgb2lab(rgb.astype(np.float64) / 255.0)

    # Get cached palette KD-Tree — O(n log k) nearest neighbor
    palette_key = tuple(tuple(c) for c in palette_rgb)
    tree = _build_palette_tree(palette_key)

    # Flatten image to (N, 3) LAB pixels
    pixels_lab = img_lab.reshape(-1, 3)
    _, nearest_idx = tree.query(pixels_lab)

    palette_uint8 = np.array(palette_rgb, dtype=np.uint8)
    result = palette_uint8[nearest_idx].reshape(h, w, 3)

    if has_alpha:
        return Image.fromarray(np.dstack([result, alpha]))
    return Image.fromarray(result)


# ─────────────────────────────────────────────────────────────
# STEP 5: DITHERING (PALETTE-AWARE)
# ─────────────────────────────────────────────────────────────

def _apply_dither(
    img: Image.Image,
    mode: DitherMode,
    palette_rgb: list[tuple[int, int, int]],
) -> Image.Image:
    # Dithering with <= 1 color is a no-op
    if len(palette_rgb) <= 1:
        return img
    if mode == DitherMode.FLOYD_STEINBERG:
        return _floyd_steinberg(img, palette_rgb)
    elif mode in (DitherMode.BAYER_2X2, DitherMode.BAYER_4X4, DitherMode.BAYER_8X8):
        size = {
            DitherMode.BAYER_2X2: 2,
            DitherMode.BAYER_4X4: 4,
            DitherMode.BAYER_8X8: 8,
        }[mode]
        return _bayer_dither(img, palette_rgb, size)
    return img


def _extract_palette_fast(img: Image.Image) -> list[tuple[int, int, int]]:
    """Extract unique colors from a quantized image (fast, no KMeans)."""
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
    elif arr.ndim == 3:
        rgb = arr
    else:
        rgb = arr.reshape(-1, 1).repeat(3, axis=1)
    unique = np.unique(rgb.reshape(-1, 3), axis=0)
    return [tuple(int(x) for x in c) for c in unique]


def _extract_palette(img: Image.Image, n_colors: int) -> list[tuple[int, int, int]]:
    """Extract n_colors dominant colors from image via KMeans."""
    from sklearn.cluster import MiniBatchKMeans

    arr = np.array(img)
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3]
    else:
        rgb = arr

    pixels = rgb.reshape(-1, 3).astype(np.float32)
    n_unique = len(np.unique(pixels, axis=0))
    k = min(n_colors, n_unique)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(4096, len(pixels)),
        n_init=3,
        random_state=42,
    )
    kmeans.fit(pixels)
    return [tuple(int(x) for x in np.round(c)) for c in kmeans.cluster_centers_]


@numba.jit(nopython=True, cache=True)
def _fs_core(rgb, pal):
    """Floyd-Steinberg error-diffusion kernel (Numba-accelerated)."""
    h, w, _ = rgb.shape
    n_pal = len(pal)
    for y in range(h):
        for x in range(w):
            old_r = rgb[y, x, 0]
            old_g = rgb[y, x, 1]
            old_b = rgb[y, x, 2]
            # Find nearest palette color
            min_dist = 1e18
            best = 0
            for i in range(n_pal):
                dr = pal[i, 0] - old_r
                dg = pal[i, 1] - old_g
                db = pal[i, 2] - old_b
                d = dr * dr + dg * dg + db * db
                if d < min_dist:
                    min_dist = d
                    best = i
            rgb[y, x, 0] = pal[best, 0]
            rgb[y, x, 1] = pal[best, 1]
            rgb[y, x, 2] = pal[best, 2]
            err_r = old_r - pal[best, 0]
            err_g = old_g - pal[best, 1]
            err_b = old_b - pal[best, 2]
            # Distribute error to neighbors (Floyd-Steinberg weights)
            if x + 1 < w:
                rgb[y, x + 1, 0] += err_r * 0.4375
                rgb[y, x + 1, 1] += err_g * 0.4375
                rgb[y, x + 1, 2] += err_b * 0.4375
            if y + 1 < h:
                if x - 1 >= 0:
                    rgb[y + 1, x - 1, 0] += err_r * 0.1875
                    rgb[y + 1, x - 1, 1] += err_g * 0.1875
                    rgb[y + 1, x - 1, 2] += err_b * 0.1875
                rgb[y + 1, x, 0] += err_r * 0.3125
                rgb[y + 1, x, 1] += err_g * 0.3125
                rgb[y + 1, x, 2] += err_b * 0.3125
                if x + 1 < w:
                    rgb[y + 1, x + 1, 0] += err_r * 0.0625
                    rgb[y + 1, x + 1, 1] += err_g * 0.0625
                    rgb[y + 1, x + 1, 2] += err_b * 0.0625
    return rgb


def _floyd_steinberg(
    img: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
) -> Image.Image:
    """Floyd-Steinberg error-diffusion dithering (palette-aware, Numba-accelerated)."""
    arr = np.array(img, dtype=np.float32)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3].copy()
    else:
        alpha = None
        rgb = arr.copy()

    pal = np.array(palette_rgb, dtype=np.float32)

    rgb = _fs_core(rgb, pal)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if has_alpha:
        return Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8)]))
    return Image.fromarray(rgb)


@functools.lru_cache(maxsize=8)
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


def _bayer_dither(
    img: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    matrix_size: int,
) -> Image.Image:
    """Ordered (Bayer) dithering with palette snap."""
    arr = np.array(img, dtype=np.float32)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3].copy()
    else:
        alpha = None
        rgb = arr.copy()

    h, w, _ = rgb.shape
    threshold = _bayer_matrix(matrix_size)

    # Tile threshold matrix across image
    th_tiled = np.tile(threshold, (h // matrix_size + 1, w // matrix_size + 1))
    th_tiled = th_tiled[:h, :w]

    # Apply threshold offset (vectorized across all 3 channels)
    n_colors = len(palette_rgb)
    step = 255.0 / max(1, n_colors - 1)
    offset = (th_tiled - 0.5) * step
    rgb += offset[:, :, np.newaxis]
    rgb = np.round(rgb / step) * step
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Snap to actual palette colors
    if has_alpha:
        result_img = Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8)]))
    else:
        result_img = Image.fromarray(rgb)
    return _enforce_palette(result_img, palette_rgb)


# ─────────────────────────────────────────────────────────────
# NUMBA JIT WARMUP
# ─────────────────────────────────────────────────────────────

def warmup_numba() -> None:
    """Pre-compile Floyd-Steinberg JIT kernel with float32 data."""
    _fs_core(np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32))


# ─────────────────────────────────────────────────────────────
# STEP 6: ALPHA CLEANUP
# ─────────────────────────────────────────────────────────────

def _cleanup_alpha(img: Image.Image, threshold: int = 128) -> Image.Image:
    """Binarize alpha channel — no semi-transparency."""
    if img.mode != "RGBA":
        return img
    arr = np.array(img)
    arr[:, :, 3] = np.where(arr[:, :, 3] >= threshold, 255, 0)
    return Image.fromarray(arr)
