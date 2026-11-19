import numpy as np
import pandas as pd

from tmem_align.analysis.spatial_comparisons import relative_cv, spatial_tiles


def test_spatial_tiles_assign_whole_objects_once_and_conserve_measurements():
    green = np.full((6, 6), 999.0, dtype=np.float32)
    red = np.arange(36, dtype=np.float32).reshape(6, 6) + 100
    valid = np.ones((6, 6), bool)
    valid[0, 0] = False
    labels = np.zeros((6, 6), np.uint32)
    labels[1, 1:3] = 7
    green[1, 1:3] = [10, 30]
    labels[4, 4] = 2
    green[4, 4] = 60
    cells = pd.DataFrame(
        {"label": [7, 2], "centroid_y_px": [2.0, 8.0], "centroid_x_px": [3.0, 8.0]}
    )
    tiles = spatial_tiles(green, red, valid, labels, cells, pixel_um=1, scale=2)
    assert len(tiles) == 9
    assert tiles.nuclei_count.sum() == 2
    assert np.isclose(tiles.valid_area_mm2.sum(), 35 * 4 / 1e6)
    assert np.isclose(
        np.sum(tiles.mcherry_mean_raw * tiles.valid_pixels) / tiles.valid_pixels.sum(),
        red[valid].mean(),
    )
    measured = tiles[tiles.nuclei_count > 0]
    assert np.isclose(
        np.average(measured.mng_mean_per_nucleus_raw, weights=measured.nuclei_count), 40
    )
    assert tiles.iloc[0].mng_mean_per_nucleus_raw == 20
    assert tiles.iloc[8].mng_mean_per_nucleus_raw == 60
    assert np.isnan(tiles.iloc[1].mng_mean_per_nucleus_raw)
    assert green[0, 1] == 999


def test_empty_area_and_zero_mean_are_missing_not_infinite():
    a = np.zeros((6, 6), np.float32)
    t = spatial_tiles(
        a,
        a,
        np.zeros((6, 6), bool),
        a.astype(np.uint32),
        pd.DataFrame(columns=["label", "centroid_y_px", "centroid_x_px"]),
        1,
        2,
    )
    assert t.mcherry_mean_raw.isna().all()
    assert t.nuclei_per_mm2.isna().all()
    assert np.isnan(relative_cv([0, 0, 0]))
    assert np.isnan(relative_cv([1, np.nan]))
    assert np.isclose(relative_cv([1, 2, 3]), 0.5)


def test_examples_are_unique_and_selection_uses_all_readouts():
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "comparisons", scripts / "build_well_comparisons.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for i in range(5):
        for day in range(5, 18):
            fold = 1 if day == 6 else 0.8 + (i - 2) * 0.1
            rows.append(
                {
                    "condition": "WT61",
                    "well": f"C{i:02d}",
                    "day": day,
                    "nuclei_count_day6_fold": fold,
                    "mcherry_mean_raw_day6_fold": fold,
                    "mng_mean_per_nucleus_raw_day6_fold": fold,
                }
            )
    selected = mod.select_examples(pd.DataFrame(rows), 6)
    assert selected.well.tolist() == ["C02", "C00", "C04"]
    assert selected.well.nunique() == 3
    assert np.isclose(selected.iloc[0].trajectory_distance, 0)
