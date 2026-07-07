from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from tmem_align.register import apply_shift, register_translation


def _gaussian_blob(size=128, center=None, sigma=12.0):
    """Sharp-enough blob for registration tests."""
    if center is None:
        center = (size // 2, size // 2)
    y, x = np.mgrid[:size, :size]
    return np.exp(-((y - center[0]) ** 2 + (x - center[1]) ** 2) / (2 * sigma**2))


def _textured_image(size=128, seed=42):
    """Image with enough spatial structure for phase correlation."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    for _ in range(25):
        cy, cx = rng.integers(20, size - 20, size=2)
        r = rng.integers(2, 5)
        img[cy - r : cy + r, cx - r : cx + r] = rng.uniform(0.5, 1.0)
    return img


# ── apply_shift: interpolation order ──


def test_apply_shift_uses_cubic_interpolation():
    img = _gaussian_blob().astype(np.float64)
    dy, dx = 0.37, -0.83
    ref_order5 = ndi_shift(img, (dy, dx), order=5, mode="constant", cval=0)
    ref_order1 = ndi_shift(img, (dy, dx), order=1, mode="constant", cval=0)
    result = apply_shift(img, dy, dx)
    err_cubic = np.max(np.abs(result - ref_order5))
    err_linear = np.max(np.abs(ref_order1 - ref_order5))
    assert err_cubic < err_linear, "apply_shift should be closer to order=5 than bilinear"


def test_apply_shift_no_ringing_on_step_edge():
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 1.0
    result = apply_shift(img, 0.0, 0.5)
    # Cubic spline can ring ~10% at discontinuities; allow a small margin.
    assert result.min() >= -0.12, f"undershoot {result.min():.3f} exceeds 12%"
    assert result.max() <= 1.12, f"overshoot {result.max():.3f} exceeds 12%"


# ── apply_shift: round-trip / accuracy ──


def test_apply_shift_subpixel_accuracy():
    img = _gaussian_blob().astype(np.float64)
    dy, dx = 2.37, -4.63
    shifted = apply_shift(img, dy, dx)
    back = apply_shift(shifted, -dy, -dx)
    # Interior crop avoids border zeros.
    sl = np.s_[15:-15, 15:-15]
    np.testing.assert_allclose(back[sl], img[sl], atol=0.05)


def test_apply_shift_integer_shift_exact():
    img = _gaussian_blob().astype(np.float64)
    dy, dx = 3.0, -5.0
    result = apply_shift(img, dy, dx)
    # result[y,x] = img[y-dy, x-dx] = img[y-3, x+5].
    # Use a generous margin to avoid spline boundary artifacts from mode='constant'.
    m = 10
    idy, idx = int(dy), int(dx)
    sl_out = np.s_[idy + m : -m, m : -abs(idx) - m]
    sl_in = np.s_[m : -idy - m, abs(idx) + m : -m]
    np.testing.assert_allclose(result[sl_out], img[sl_in], atol=1e-12)


def test_apply_shift_zero_shift():
    img = _gaussian_blob().astype(np.float64)
    result = apply_shift(img, 0.0, 0.0)
    np.testing.assert_allclose(result, img, atol=1e-12)


# ── apply_shift: dtype preservation ──


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_apply_shift_preserves_dtype(dtype):
    img = (_gaussian_blob() * (65535 if np.issubdtype(dtype, np.integer) else 1.0)).astype(dtype)
    result = apply_shift(img, 1.5, -2.3)
    assert result.dtype == dtype


# ── apply_shift: dimensionality ──


def test_apply_shift_2d():
    img = _gaussian_blob(size=32).astype(np.float32)
    result = apply_shift(img, 1.0, -1.0)
    assert result.shape == img.shape


def test_apply_shift_3d():
    ch = _gaussian_blob(size=32).astype(np.float32)
    img = np.stack([ch, ch * 0.5], axis=0)  # (2, 32, 32)
    result = apply_shift(img, 2.0, -1.0)
    assert result.shape == img.shape
    # Channel axis untouched: ratios preserved in interior.
    mask = result[0][8:-8, 8:-8] > 0.01
    ratio = result[1, 8:-8, 8:-8][mask] / result[0, 8:-8, 8:-8][mask]
    np.testing.assert_allclose(ratio, 0.5, atol=0.05)


def test_apply_shift_4d():
    ch = _gaussian_blob(size=32).astype(np.float32)
    img = np.stack([ch, ch], axis=0)[np.newaxis, ...]  # (1, 2, 32, 32)
    result = apply_shift(img, 1.0, -1.0)
    assert result.shape == img.shape


# ── apply_shift: border fill ──


def test_apply_shift_fills_borders_with_zero():
    img = np.ones((32, 32), dtype=np.float64)
    result = apply_shift(img, 10.0, 0.0)
    # Top 10 rows should be zero (shifted down).
    np.testing.assert_allclose(result[:9, :], 0.0, atol=1e-6)


# ── register_translation ──


def test_register_translation_recovers_known_shift():
    img = _textured_image(size=128)
    dy, dx = 3.0, -5.0
    moved = ndi_shift(img, (dy, dx), order=3, mode="constant", cval=0)
    _, (rec_dy, rec_dx) = register_translation(img, moved, upsample_factor=20)
    np.testing.assert_allclose(rec_dy, -dy, atol=0.25)
    np.testing.assert_allclose(rec_dx, -dx, atol=0.25)


def test_register_translation_subpixel_recovery():
    img = _textured_image(size=128)
    dy, dx = 0.37, -0.83
    moved = ndi_shift(img, (dy, dx), order=5, mode="constant", cval=0)
    _, (rec_dy, rec_dx) = register_translation(img, moved, upsample_factor=100)
    np.testing.assert_allclose(rec_dy, -dy, atol=0.15)
    np.testing.assert_allclose(rec_dx, -dx, atol=0.15)


def test_register_translation_max_shift_guard():
    img = _textured_image(size=128)
    moved = ndi_shift(img, (10.0, 10.0), order=3, mode="constant", cval=0)
    with pytest.raises(ValueError, match="exceeds max_shift_pixels"):
        register_translation(img, moved, max_shift_pixels=5.0)
