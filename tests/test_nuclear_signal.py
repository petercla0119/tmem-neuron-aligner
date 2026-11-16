import numpy as np
import pytest


def test_nuclear_signal_weights_nuclei_equally_and_preserves_units():
    from tmem_align.analysis.nuclear_signal import quantify_nuclear_mng

    green = np.array([[10, 999], [30, 30], [30, 999]], dtype=np.float32)
    labels = np.array([[1, 0], [7, 7], [7, 0]], dtype=np.uint32)
    before = green.copy()
    result = quantify_nuclear_mng(green, labels, np.ones_like(labels, dtype=bool), scale=2)
    assert result["mng_nuclei_count_check"] == 2
    assert result["mng_mean_per_nucleus_raw"] == 20
    assert result["mng_median_per_nucleus_raw"] == 20
    assert result["mng_mean_nuclear_pixels_raw"] == 25
    assert result["mng_integrated_nuclear_raw"] == 400
    assert result["nuclear_pixels_original"] == 16
    np.testing.assert_array_equal(green, before)


def test_nuclear_signal_no_nuclei():
    from tmem_align.analysis.nuclear_signal import quantify_nuclear_mng

    result = quantify_nuclear_mng(
        np.ones((2, 2)), np.zeros((2, 2), dtype=np.uint32), np.ones((2, 2), dtype=bool), 2
    )
    assert result["mng_nuclei_count_check"] == 0
    assert np.isnan(result["mng_mean_per_nucleus_raw"])
    assert result["mng_integrated_nuclear_raw"] == 0


def test_nuclear_signal_rejects_masks_outside_valid_area():
    from tmem_align.analysis.nuclear_signal import quantify_nuclear_mng

    with pytest.raises(ValueError, match="invalid"):
        quantify_nuclear_mng(
            np.ones((2, 2)), np.ones((2, 2), dtype=np.uint32), np.zeros((2, 2), dtype=bool), 2
        )
