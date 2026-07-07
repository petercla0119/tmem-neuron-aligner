"""Tests for tmem_align.preprocess module."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from tmem_align.preprocess import (
    _extract_channel,
    _load_image,
    _rescale_field,
    apply_ic_field,
    calculate_ic_field,
    calculate_ic_field_for_plate,
    calculate_ic_field_for_well,
    preprocess_image,
    subtract_background,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_images():
    """Ten identical 64x64 uint16 images with a gradient."""
    rng = np.random.default_rng(42)
    # Smooth gradient + small noise so median filter is nearly identity
    y, x = np.mgrid[0:64, 0:64]
    base = (100 + y * 5 + x * 3).astype(np.uint16)
    return [base + rng.integers(0, 5, size=(64, 64), dtype=np.uint16) for _ in range(10)]


@pytest.fixture
def constant_images():
    """Five constant 32x32 uint16 images."""
    return [np.full((32, 32), 500, dtype=np.uint16) for _ in range(5)]


@pytest.fixture
def multichannel_images():
    """Four 3-channel (CYX) 32x32 images."""
    rng = np.random.default_rng(7)
    return [rng.integers(100, 600, size=(3, 32, 32), dtype=np.uint16) for _ in range(4)]


# ---------------------------------------------------------------------------
# _rescale_field
# ---------------------------------------------------------------------------


class TestRescaleField:
    def test_min_becomes_one(self):
        field = np.array([[2.0, 4.0], [6.0, 8.0]])
        result = _rescale_field(field)
        assert result.min() >= 1.0

    def test_all_zeros(self):
        field = np.zeros((4, 4))
        result = _rescale_field(field)
        # robust_min=0 → falls back to 1, then clamp to 1
        assert np.all(result >= 1.0)

    def test_preserves_relative_ratios(self):
        field = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = _rescale_field(field)
        # 40/10 ratio should be preserved (min is already at 2nd percentile)
        assert result.max() > result.min()


# ---------------------------------------------------------------------------
# _load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_array_passthrough(self):
        arr = np.ones((10, 10), dtype=np.uint16)
        result = _load_image(arr)
        np.testing.assert_array_equal(result, arr)

    def test_list_converted_to_array(self):
        result = _load_image([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)

    def test_load_from_path(self, tmp_path):
        arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), arr)
        result = _load_image(p)
        np.testing.assert_array_equal(result, arr)

    def test_load_from_str_path(self, tmp_path):
        arr = np.full((4, 4), 42, dtype=np.uint16)
        p = tmp_path / "img2.tif"
        tifffile.imwrite(str(p), arr)
        result = _load_image(str(p))
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# _extract_channel
# ---------------------------------------------------------------------------


class TestExtractChannel:
    def test_extract_from_3d(self):
        img = np.zeros((3, 8, 8), dtype=np.uint16)
        img[1] = 99
        result = _extract_channel(img, 1)
        assert result.shape == (8, 8)
        assert result[0, 0] == 99

    def test_none_channel_normalizes_to_2d(self):
        img = np.ones((1, 8, 8), dtype=np.uint16) * 5
        result = _extract_channel(img, None)
        assert result.ndim == 2

    def test_2d_passthrough(self):
        img = np.ones((8, 8), dtype=np.uint16)
        result = _extract_channel(img, None)
        assert result.ndim == 2
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# calculate_ic_field
# ---------------------------------------------------------------------------


class TestCalculateICField:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No images"):
            calculate_ic_field([])

    def test_constant_images(self, constant_images):
        field = calculate_ic_field(constant_images, rescale_field=False)
        assert field.ndim == 2
        assert field.shape == (32, 32)

    def test_rescale_field_min_ge_one(self, flat_images):
        field = calculate_ic_field(flat_images, rescale_field=True)
        assert field.min() >= 1.0

    def test_no_rescale(self, constant_images):
        field = calculate_ic_field(constant_images, rescale_field=False)
        assert field.ndim == 2

    def test_multichannel_returns_cyx(self, multichannel_images):
        field = calculate_ic_field(multichannel_images, rescale_field=False)
        assert field.ndim == 3
        assert field.shape[0] == 3
        assert field.shape[1:] == (32, 32)

    def test_multichannel_with_channel_returns_2d(self, multichannel_images):
        field = calculate_ic_field(multichannel_images, channel=0, rescale_field=False)
        assert field.ndim == 2
        assert field.shape == (32, 32)

    def test_sample_fraction(self, flat_images):
        # Just check it runs without error and produces valid shape
        field = calculate_ic_field(flat_images, sample_fraction=0.5, rescale_field=False)
        assert field.shape == (64, 64)

    def test_custom_smooth(self, constant_images):
        field = calculate_ic_field(constant_images, smooth=3, rescale_field=False)
        assert field.shape == (32, 32)

    def test_path_based(self, tmp_path):
        rng = np.random.default_rng(0)
        paths = []
        for i in range(3):
            arr = rng.integers(100, 300, size=(16, 16), dtype=np.uint16)
            p = tmp_path / f"img_{i}.tif"
            tifffile.imwrite(str(p), arr)
            paths.append(p)
        field = calculate_ic_field(paths, rescale_field=False)
        assert field.shape == (16, 16)


# ---------------------------------------------------------------------------
# apply_ic_field
# ---------------------------------------------------------------------------


class TestApplyICField:
    def test_none_field_passthrough(self):
        img = np.ones((8, 8), dtype=np.uint16) * 100
        result = apply_ic_field(img, None)
        np.testing.assert_array_equal(result, img)

    def test_2d_2d(self):
        img = np.full((8, 8), 200, dtype=np.uint16)
        field = np.full((8, 8), 2.0)
        result = apply_ic_field(img, field)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, 100)

    def test_2d_field_3d_image(self):
        img = np.full((2, 8, 8), 300, dtype=np.uint16)
        field = np.full((8, 8), 3.0)
        result = apply_ic_field(img, field)
        assert result.shape == (2, 8, 8)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, 100)

    def test_3d_field_3d_image(self):
        img = np.full((2, 8, 8), 400, dtype=np.uint16)
        field = np.full((2, 8, 8), 4.0)
        result = apply_ic_field(img, field)
        assert result.shape == (2, 8, 8)
        np.testing.assert_array_equal(result, 100)

    def test_2d_field_4d_image(self):
        img = np.full((3, 2, 8, 8), 500, dtype=np.uint16)
        field = np.full((8, 8), 5.0)
        result = apply_ic_field(img, field)
        assert result.shape == (3, 2, 8, 8)
        np.testing.assert_array_equal(result, 100)

    def test_3d_field_4d_image(self):
        img = np.full((3, 2, 8, 8), 600, dtype=np.uint16)
        field = np.full((2, 8, 8), 6.0)
        result = apply_ic_field(img, field)
        assert result.shape == (3, 2, 8, 8)
        np.testing.assert_array_equal(result, 100)

    def test_zeros_in_field_no_crash(self):
        img = np.full((8, 8), 100, dtype=np.uint16)
        field = np.zeros((8, 8))
        result = apply_ic_field(img, field)
        # zeros replaced with 1, so result = 100/1 = 100
        np.testing.assert_array_equal(result, 100)

    def test_returns_uint16(self):
        img = np.full((8, 8), 1000, dtype=np.uint16)
        field = np.full((8, 8), 1.5)
        result = apply_ic_field(img, field)
        assert result.dtype == np.uint16


# ---------------------------------------------------------------------------
# subtract_background
# ---------------------------------------------------------------------------


class TestSubtractBackground:
    def test_basic(self):
        img = np.full((64, 64), 500, dtype=np.uint16)
        result = subtract_background(img, radius=10)
        assert result.dtype == np.uint16
        assert result.shape == (64, 64)

    def test_small_radius(self):
        img = np.full((32, 32), 300, dtype=np.uint16)
        result = subtract_background(img, radius=5)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint16

    def test_large_radius(self):
        rng = np.random.default_rng(1)
        img = rng.integers(100, 1000, size=(64, 64), dtype=np.uint16)
        result = subtract_background(img, radius=50)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint16

    def test_no_negative_values(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 100, size=(64, 64), dtype=np.uint16)
        result = subtract_background(img, radius=10)
        assert result.min() >= 0

    def test_custom_shrink_factor(self):
        img = np.full((64, 64), 400, dtype=np.uint16)
        result = subtract_background(img, radius=20, shrink_factor=1)
        assert result.shape == (64, 64)


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_ic_only(self):
        img = np.full((32, 32), 200, dtype=np.uint16)
        field = np.full((32, 32), 2.0)
        result = preprocess_image(img, ic_field=field)
        np.testing.assert_array_equal(result, 100)

    def test_bg_only(self):
        img = np.full((64, 64), 500, dtype=np.uint16)
        result = preprocess_image(img, background_radius=10)
        assert result.dtype == np.uint16
        assert result.shape == (64, 64)

    def test_both(self):
        img = np.full((64, 64), 400, dtype=np.uint16)
        field = np.full((64, 64), 2.0)
        result = preprocess_image(img, ic_field=field, background_radius=10)
        assert result.dtype == np.uint16
        assert result.shape == (64, 64)

    def test_neither_passthrough(self):
        img = np.full((32, 32), 123, dtype=np.uint16)
        result = preprocess_image(img)
        np.testing.assert_array_equal(result, img)

    def test_3d_with_bg(self):
        img = np.full((2, 64, 64), 300, dtype=np.uint16)
        result = preprocess_image(img, background_radius=10)
        assert result.shape == (2, 64, 64)
        assert result.dtype == np.uint16

    def test_4d_with_bg(self):
        img = np.full((2, 2, 64, 64), 300, dtype=np.uint16)
        result = preprocess_image(img, background_radius=10)
        assert result.shape == (2, 2, 64, 64)
        assert result.dtype == np.uint16

    def test_3d_with_ic_and_bg(self):
        img = np.full((2, 64, 64), 400, dtype=np.uint16)
        field = np.full((64, 64), 2.0)
        result = preprocess_image(img, ic_field=field, background_radius=10)
        assert result.shape == (2, 64, 64)
        assert result.dtype == np.uint16


# ---------------------------------------------------------------------------
# calculate_ic_field_for_well / plate
# ---------------------------------------------------------------------------


class TestWellAndPlate:
    def test_well(self, tmp_path):
        rng = np.random.default_rng(3)
        for i in range(3):
            arr = rng.integers(100, 500, size=(16, 16), dtype=np.uint16)
            tifffile.imwrite(str(tmp_path / f"img_{i}.tif"), arr)
        field = calculate_ic_field_for_well(tmp_path)
        assert field.shape == (16, 16)

    def test_well_empty_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            calculate_ic_field_for_well(tmp_path)

    def test_plate(self, tmp_path):
        rng = np.random.default_rng(4)
        for well in ["A01", "A02"]:
            well_dir = tmp_path / well
            well_dir.mkdir()
            for i in range(3):
                arr = rng.integers(100, 500, size=(16, 16), dtype=np.uint16)
                tifffile.imwrite(str(well_dir / f"img_{i}.tif"), arr)
        field = calculate_ic_field_for_plate(tmp_path, sample_fraction=1.0)
        assert field.shape == (16, 16)

    def test_plate_empty_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            calculate_ic_field_for_plate(tmp_path)
