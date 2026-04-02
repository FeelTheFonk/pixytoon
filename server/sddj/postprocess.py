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
    """
    # Fast bypass: no processing enabled → return raw image
    if not _any_processing_active(spec):
        return image

    img = image.convert("RGBA")

    # 1. Background removal
    if spec.remove_bg:
        img = _remove_background(img)

    # 1b. Upscale (before pixelation — higher resolution input improves pixel art quality)
    if spec.upscale_enabled and settings.enable_upscaler:
        img = _upscale(img, spec.upscale_factor)

    # 2. Pixelation
    if spec.pixelate.enabled:
        img = _pixelate(img, spec.pixelate.target_size, spec.pixelate.method)

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

    # 5. Dithering (OKLAB palette-aware — uses resolved palette or extracts from quantized image)
    if spec.dither != DitherMode.NONE:
        # Reuse KMeans centers if available, avoiding a redundant second KMeans run
        if palette_rgb is None:
            if kmeans_centers is not None:
                palette_rgb = kmeans_centers
            else:
                palette_rgb = _extract_palette(img, spec.quantize_colors)
        img = _apply_dither(img, spec.dither, palette_rgb, has_bg_removal=spec.remove_bg)

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


def _pixelate(
    img: Image.Image,
    target_size: int,
    method: PixelateMethod = PixelateMethod.NEAREST,
) -> Image.Image:
    w, h = img.size
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
            # Convert RGBA -> RGB for processing, restore alpha after.
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.getchannel("A")
                rgb = img.convert("RGB")
            else:
                rgb = img if img.mode == "RGB" else img.convert("RGB")
            arr = np.asarray(rgb)
            result = pixelize(arr, target_size=min(new_w, new_h))
            out = Image.fromarray(result)
            if has_alpha:
                # Downscale alpha to match PixelOE output dimensions
                alpha_resized = alpha.resize(out.size, Image.NEAREST)
                out = out.convert("RGBA")
                out.putalpha(alpha_resized)
            return out
        except ImportError:
            log.warning("pixeloe not installed, falling back to box downscale")
            method = PixelateMethod.BOX
    if method == PixelateMethod.BOX:
        # BOX (area averaging) preserves thin features as averaged colors
        # rather than losing them to point sampling. Post-snap to palette
        # restores hard pixel-art edges.
        return img.resize((new_w, new_h), Image.BOX)
    # NEAREST is the pixel-art default — no interpolation artifacts
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
    elif method == QuantizeMethod.OCTREE_LAB:
        return _quantize_octree_lab(img, n_colors)
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
            if has_alpha:
                return Image.fromarray(arr), centers
            return Image.fromarray(rgb), centers

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

    if has_alpha:
        result = np.dstack([quantized, alpha])
        return Image.fromarray(result), palette
    return Image.fromarray(quantized), palette


def _quantize_octree_lab(
    img: Image.Image,
    n_colors: int,
) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    """Octree quantization in OKLAB space — perceptually uniform palettes.

    Produces palette entries that are equidistant in human perception,
    eliminating over-representation of greens and under-representation
    of blues that plague RGB-space quantization.

    Implementation: KMeans in OKLAB space with deterministic init.
    This is semantically equivalent to an octree in perceptual space,
    using MiniBatchKMeans as the spatial partitioner for numerical
    stability and Numba/BLAS acceleration.

    Migrated from CIELAB (skimage float64) to OKLAB (float32).
    """
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
        if has_alpha:
            return Image.fromarray(np.dstack([quantized, alpha])), palette
        return Image.fromarray(quantized), palette

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


def _enforce_palette(
    img: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
) -> Image.Image:
    """Snap every pixel to the nearest palette color in OKLAB space.

    Migrated from CIELAB to OKLAB — Euclidean distance in OKLAB is
    perceptually uniform, same semantics as CIEDE76 but float32.
    """
    if not palette_rgb:
        return img

    arr = np.array(img)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

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
    result = palette_uint8[nearest_idx].reshape(h, w, 3)

    if has_alpha:
        return Image.fromarray(np.dstack([result, alpha]))
    return Image.fromarray(result)


# ─────────────────────────────────────────────────────────────
# STEP 5: DITHERING (OKLAB PALETTE-AWARE)
# ─────────────────────────────────────────────────────────────

def _apply_dither(
    img: Image.Image,
    mode: DitherMode,
    palette_rgb: list[tuple[int, int, int]],
    *,
    has_bg_removal: bool = False,
) -> Image.Image:
    # Dithering with <= 1 color is a no-op
    if len(palette_rgb) <= 1:
        return img
    if mode == DitherMode.FLOYD_STEINBERG:
        return _floyd_steinberg(img, palette_rgb, alpha_aware=has_bg_removal)
    elif mode in (DitherMode.BAYER_2X2, DitherMode.BAYER_4X4, DitherMode.BAYER_8X8):
        size = {
            DitherMode.BAYER_2X2: 2,
            DitherMode.BAYER_4X4: 4,
            DitherMode.BAYER_8X8: 8,
        }[mode]
        return _bayer_dither(img, palette_rgb, size, alpha_aware=has_bg_removal)
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
    """Extract n_colors dominant colors from image.

    Optimized: uses fast unique-color extraction when image has
    ≤256 unique colors (common after pixelation), falls back to
    KMeans only for high-color-count images.
    """
    arr = np.array(img)
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3]
    else:
        rgb = arr

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


def _floyd_steinberg(
    img: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    *,
    alpha_aware: bool = False,
) -> Image.Image:
    """Floyd-Steinberg error-diffusion dithering (OKLAB, palette-aware, Numba-accelerated).

    Migrated from CIELAB to OKLAB — float32 throughout, no skimage dependency.
    """
    arr = np.array(img)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    h, w, _ = rgb.shape

    # Convert image to OKLAB (float32 — no float64 intermediate)
    img_ok = rgb_to_oklab(rgb.astype(np.float32) / 255.0)

    # Convert palette to OKLAB (cached)
    palette_key = tuple(tuple(c) for c in palette_rgb)
    pal_ok = _palette_to_oklab(palette_key)
    pal_rgb = np.array(palette_rgb, dtype=np.uint8)

    # Build alpha mask: True for opaque pixels
    if alpha_aware and has_alpha:
        alpha_mask = alpha >= 128
    else:
        alpha_mask = np.ones((h, w), dtype=np.bool_)

    result = _fs_core_oklab(img_ok, pal_ok, pal_rgb, alpha_mask)

    if has_alpha:
        return Image.fromarray(np.dstack([result, alpha]))
    return Image.fromarray(result)


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


def _bayer_dither(
    img: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    matrix_size: int,
    *,
    alpha_aware: bool = False,
) -> Image.Image:
    """Ordered (Bayer) dithering with OKLAB palette snap.

    Eliminates double-quantization: applies threshold offset in OKLAB L
    channel then snaps directly to nearest palette color via cKDTree.

    Migrated from CIELAB to OKLAB. Key scaling change: CIELAB L* is [0,100],
    OKLAB L is [0,1], so the dither offset is scaled by 1/100.
    """
    # M-33: Only convert RGB channels to float32, keep alpha as uint8
    arr = np.array(img)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        alpha = arr[:, :, 3].copy()  # uint8, no float32 conversion needed
        rgb = arr[:, :, :3].astype(np.float32)
    else:
        alpha = None
        rgb = arr.astype(np.float32)

    h, w, _ = rgb.shape

    # Convert to OKLAB (float32 throughout — no float64 intermediate)
    img_ok = rgb_to_oklab(rgb / 255.0)

    # Bayer threshold in OKLAB L channel (perceptual lightness)
    threshold = _bayer_matrix(matrix_size)
    th_tiled = np.tile(threshold, (h // matrix_size + 1, w // matrix_size + 1))
    th_tiled = th_tiled[:h, :w]

    # Apply threshold as L offset (OKLAB L range is [0, 1], was [0, 100] in CIELAB)
    n_colors = len(palette_rgb)
    l_step = 1.0 / max(1, n_colors - 1)
    offset = (th_tiled - 0.5) * l_step

    # Alpha-aware: zero out offset for transparent pixels before addition
    if alpha_aware and has_alpha:
        alpha_mask = alpha < 128
        offset[alpha_mask] = 0

    img_ok[:, :, 0] += offset

    # Snap directly to nearest palette color in OKLAB via cKDTree
    palette_key = tuple(tuple(c) for c in palette_rgb)
    tree = _build_palette_tree(palette_key)
    pixels_ok = img_ok.reshape(-1, 3)
    _, nearest_idx = tree.query(pixels_ok)

    palette_uint8 = np.array(palette_rgb, dtype=np.uint8)
    result = palette_uint8[nearest_idx].reshape(h, w, 3)

    if has_alpha:
        return Image.fromarray(np.dstack([result, alpha]))
    return Image.fromarray(result)


# ─────────────────────────────────────────────────────────────
# NUMBA JIT WARMUP
# ─────────────────────────────────────────────────────────────

def warmup_numba() -> None:
    """Pre-compile Floyd-Steinberg JIT kernel with float32 data."""
    _fs_core_oklab(
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.bool_),
    )


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
