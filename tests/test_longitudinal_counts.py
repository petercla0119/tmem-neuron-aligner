import numpy as np
import pandas as pd
import pytest

from tmem_align.analysis.longitudinal_counts import (
    add_baseline_changes,
    prepare_channels,
    quantify_frame,
)


def test_preprocessing_preserves_red_units_and_does_not_mutate_input():
    raw = np.full((2, 8, 8), 100, dtype=np.uint16)
    raw[0, 2:4, 2:4] = 500
    raw[1, 2:4, 2:4] = 700
    before = raw.copy()
    green, red, valid = prepare_channels(raw, 2)
    np.testing.assert_array_equal(raw, before)
    assert green.shape == (4, 4) and red[1, 1] == 700 and valid.all()


def test_counts_unique_labels_not_max_and_retains_negative_correction():
    red = np.full((6, 6), 100.0, dtype=np.float32)
    red[2:4, 2:4] = 200
    labels = np.zeros((6, 6), np.uint32)
    labels[2:4, 2:4] = 7
    metrics, cells = quantify_frame(
        red, labels, np.ones(red.shape, bool), pixel_um=1, scale=2, background=150
    )
    assert metrics["nuclei_count"] == 1
    assert metrics["mcherry_integrated_bg_corrected"] == pytest.approx((red.sum() - 150 * 36) * 4)
    assert metrics["mcherry_mean_bg_corrected"] < 0
    assert cells[0]["nuclear_mcherry_mean_raw"] == 200
    assert metrics["area_mm2"] == pytest.approx(144 / 1e6)


def test_empty_masks_missing_valid_pixels_and_excluded_padding():
    red = np.ones((4, 4)) * 100
    valid = np.zeros((4, 4), bool)
    valid[1:3, 1:3] = True
    met, cells = quantify_frame(red, np.zeros((4, 4), int), valid, 1, 1, 0)
    assert met["nuclei_count"] == 0 and np.isnan(met["mcherry_integrated_per_nucleus"])
    assert met["mcherry_integrated_raw"] == 400 and cells == []
    with pytest.raises(ValueError, match="valid"):
        quantify_frame(red, np.zeros((4, 4), int), np.zeros((4, 4), bool), 1, 1, 0)


def test_baseline_is_per_well_and_zero_baseline_is_nan():
    df = pd.DataFrame(
        {
            "well": ["A", "B", "A", "B"],
            "day": [6, 5, 5, 6],
            "nuclei_count": [10, 0, 5, 2],
            "mcherry_mean_raw": [200, 100, 100, 150],
            "mcherry_mean_bg_corrected": [100, 0, 50, 5],
            "mcherry_integrated_per_nucleus": [500, 0, 250, 200],
        }
    )
    out = add_baseline_changes(df, baseline_day=5)
    assert out.loc[(out.well == "A") & (out.day == 6), "nuclei_count_fold_baseline"].iloc[0] == 2
    assert np.isnan(
        out.loc[(out.well == "B") & (out.day == 6), "nuclei_count_fold_baseline"].iloc[0]
    )
    assert (
        out.loc[(out.well == "A") & (out.day == 6), "mcherry_mean_raw_change_baseline"].iloc[0]
        == 100
    )
    out6 = add_baseline_changes(df)  # default baseline is day 6
    assert (out6.baseline_day == 6).all()
    assert (
        out6.loc[(out6.well == "A") & (out6.day == 5), "nuclei_count_fold_baseline"].iloc[0] == 0.5
    )
    assert out6.loc[(out6.well == "A") & (out6.day == 6), "nuclei_count_fold_baseline"].iloc[0] == 1
    partial = add_baseline_changes(df[df.day == 5])  # baseline day not reached yet
    assert (partial.baseline_day == 5).all()


def test_model_normalization_cannot_mutate_saved_green():
    from tmem_align.analysis.longitudinal_counts import segment_nuclei

    class InPlaceModel:
        def eval(self, image, **kwargs):
            image[:] = 0.5
            return np.ones(image.shape, dtype=np.uint32), None, None

    green = np.full((4, 4), 123.0, dtype=np.float32)
    labels = segment_nuclei(InPlaceModel(), green, {})
    assert np.all(green == 123) and np.all(labels == 1)


def test_block_mean_integrated_signal_matches_original_and_centroid_center():
    raw = np.arange(1, 129, dtype=np.uint16).reshape(2, 8, 8)
    _green, red, valid = prepare_channels(raw, 4)
    labels = np.zeros((2, 2), dtype=np.uint32)
    labels[0, 0] = 1
    met, cells = quantify_frame(red, labels, valid, 1, 4, 0)
    assert met["mcherry_integrated_raw"] == raw[1].sum()
    assert cells[0]["centroid_y_px"] == 1.5


@pytest.mark.parametrize(
    "scenario, message",
    [
        ("duplicate", "Duplicate"),
        ("missing", "Missing well/day"),
        ("exposure", "Exposure settings changed"),
        ("channels", "Unexpected channels"),
    ],
)
def test_inventory_rejects_ambiguous_or_incomparable_inputs(
    tmp_path, monkeypatch, scenario, message
):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace as NS

    import nd2

    class FakeND2:
        dtype = np.dtype("uint16")
        attributes = NS(bitsPerComponentSignificant=16)
        metadata = NS(
            channels=[
                NS(channel=NS(name=c))
                for c in (["488nm", "wrong"] if scenario == "channels" else ["488nm", "561nm"])
            ]
        )

        def __init__(self, path):
            self.path = path
            self.sizes = {"C": 2, "Y": 8, "X": 8}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def voxel_size(self):
            return NS(x=0.3)

        def frame_metadata(self, index):
            return NS(channels=[NS(time=NS(absoluteJulianDayNumber=2460000.5))])

        def events(self):
            return [
                {
                    "Exposure Time(488nm) [ms]": 100,
                    "Exposure Time(561nm) [ms]": 200
                    if scenario == "exposure" and "_d6_" in self.path.name
                    else 100,
                }
            ]

    monkeypatch.setattr(nd2, "ND2File", FakeND2)
    (tmp_path / "x_d5_a_WellE04_a.nd2").touch()
    if scenario == "duplicate":
        (tmp_path / "x_d5_b_WellE04_b.nd2").touch()
    if scenario == "exposure":
        (tmp_path / "x_d6_a_WellE04_a.nd2").touch()
    days = [5, 6] if scenario in ("missing", "exposure") else [5]
    runner = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/run_longitudinal_counts.py")
    )
    with pytest.raises(ValueError, match=message):
        runner["inventory"](tmp_path, ["E04"], days)
