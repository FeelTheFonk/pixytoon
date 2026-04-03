"""OKLAB color space conversion utilities (float32 vectorized).

Replaces skimage.color CIELAB conversions in the pixel art pipeline with
OKLAB (Björn Ottosson, 2020). Advantages over CIELAB:

- float32 throughout (CIELAB via skimage requires float64 intermediaries)
- Better perceptual uniformity for saturated colors (blue hue linearity)
- Simpler math (two matrix multiplies + cbrt, no D65 whitepoint division)
- L in [0, 1], a/b in roughly [-0.5, 0.5]

All functions accept and return float32 numpy arrays.  Input sRGB values
must be in [0, 1] range (not [0, 255]).
"""

from __future__ import annotations

import numpy as np
import numba

# ── Forward transform matrices (sRGB linear → LMS → OKLAB) ──────

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.7136952004, 0.0743013014],
    [0.1793662670, 0.0720334730, 0.7485982600],
], dtype=np.float32)

_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)

# ── Inverse transform matrices (OKLAB → LMS_ → linear RGB) ──────

_M2_INV = np.linalg.inv(_M2.astype(np.float64)).astype(np.float32)
_M1_INV = np.linalg.inv(_M1.astype(np.float64)).astype(np.float32)


@numba.vectorize(['float32(float32)'], nopython=True, cache=True)
def _srgb_to_linear(x):
    """sRGB gamma → linear RGB, Numba ufunc (single-pass, no mask allocation).

    Applies the standard sRGB EOTF:
      - linear = x / 12.92              where x <= 0.04045
      - linear = ((x + 0.055) / 1.055) ^ 2.4   otherwise
    """
    if x <= 0.04045:
        return x / 12.92
    else:
        return ((x + 0.055) / 1.055) ** 2.4


@numba.vectorize(['float32(float32)'], nopython=True, cache=True)
def _linear_to_srgb(x):
    """Linear RGB → sRGB gamma, Numba ufunc (single-pass, no mask allocation).

    Applies the standard sRGB OETF with fused clamp:
      - srgb = x * 12.92              where x <= 0.0031308
      - srgb = 1.055 * x^(1/2.4) - 0.055   otherwise
    Output is clamped to [0, 1] (fused into the ufunc, no separate np.clip).
    """
    if x <= 0.0031308:
        return x * 12.92
    else:
        v = x if x > 0.0 else 0.0
        r = 1.055 * (v ** (1.0 / 2.4)) - 0.055
        # Fused clip: avoids separate np.clip allocation
        if r < 0.0:
            return 0.0
        elif r > 1.0:
            return 1.0
        else:
            return r


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] float32 array to OKLAB float32.

    Args:
        rgb: float32 array of shape (..., 3) with sRGB values in [0, 1].
             Accepts (H, W, 3) images or (N, 3) pixel arrays.

    Returns:
        float32 array of same shape with OKLAB values.
        L in [0, 1], a/b in approximately [-0.5, 0.5].
    """
    shape = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32, copy=False)

    # sRGB gamma → linear
    linear = _srgb_to_linear(pixels)

    # Linear RGB → LMS (matrix multiply: pixels @ M1^T)
    lms = linear @ _M1.T

    # LMS → LMS_ (cube root, handling negatives from out-of-gamut)
    lms_ = np.cbrt(lms)

    # LMS_ → OKLAB
    oklab = lms_ @ _M2.T

    return oklab.reshape(shape)


def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """Convert OKLAB float32 array to sRGB [0,1] float32.

    Args:
        oklab: float32 array of shape (..., 3) with OKLAB values.
               Accepts (H, W, 3) images or (N, 3) pixel arrays.

    Returns:
        float32 array of same shape with sRGB values in [0, 1].
        Values are clipped to [0, 1] to handle out-of-gamut colors.
    """
    shape = oklab.shape
    pixels = oklab.reshape(-1, 3).astype(np.float32, copy=False)

    # OKLAB → LMS_
    lms_ = pixels @ _M2_INV.T

    # LMS_ → LMS (cube)
    lms = lms_ * lms_ * lms_

    # LMS → linear RGB
    linear = lms @ _M1_INV.T

    # Linear → sRGB gamma (ufunc handles negative inputs and output clamp internally)
    srgb = _linear_to_srgb(linear)

    return srgb.reshape(shape)
