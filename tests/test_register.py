from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from tmem_align.register import apply_shift, register_translation


def test_register_translation_recovers_known_shift():
    image = np.zeros((96, 96), dtype=np.float32)
    image[30:42, 40:52] = 1.0
    image[60:68, 18:30] = 0.8
    known_shift = (5.0, -7.0)
    moving = ndi_shift(image, shift=known_shift, order=1, mode="constant", cval=0)

    _, recovered, error = register_translation(image, moving, upsample_factor=10)

    assert abs(recovered[0] + known_shift[0]) < 0.25
    assert abs(recovered[1] + known_shift[1]) < 0.25
    assert error >= 0


def test_register_translation_with_robust_preprocess():
    image = np.zeros((96, 96), dtype=np.float32)
    image[30:42, 40:52] = 1.0
    # Add outlier pixels that would confuse naive correlation
    image[10, 10] = 50.0
    image[80, 80] = 50.0
    known_shift = (3.0, -4.0)
    moving = ndi_shift(image, shift=known_shift, order=1, mode="constant", cval=0)
    moving[15, 15] = 50.0

    _, recovered, _ = register_translation(
        image, moving, upsample_factor=10, robust_preprocess=True,
    )

    assert abs(recovered[0] + known_shift[0]) < 0.5
    assert abs(recovered[1] + known_shift[1]) < 0.5


def test_max_shift_pixels_guard():
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:30, 20:30] = 1.0
    moving = ndi_shift(image, shift=(20.0, 0.0), order=1, mode="constant", cval=0)

    with pytest.raises(ValueError, match="max_shift_pixels"):
        register_translation(image, moving, max_shift_pixels=5.0)


def test_apply_shift_round_trip():
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:30, 20:30] = 1.0
    shifted = apply_shift(image, 3.0, -5.0)
    restored = apply_shift(shifted, -3.0, 5.0)
    # Interior region should survive the round trip
    assert np.allclose(image[25:28, 25:28], restored[25:28, 25:28], atol=0.05)
