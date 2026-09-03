from __future__ import annotations

from pathlib import Path

import numpy as np

from tmem_align.analysis.nuclear_health import nuclear_health_stats


def test_flags_healthy_and_low_signal():
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[10:25, 10:25] = 1  # bright → healthy
    masks[40:55, 40:55] = 2  # dim    → low_signal
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:25, 10:25] = 1000.0
    img[40:55, 40:55] = 50.0

    df = nuclear_health_stats(masks, img, min_mean_intensity=300.0, min_nucleus_area=10)

    assert set(df["nucleus_label"]) == {1, 2}
    indexed = df.set_index("nucleus_label")
    assert indexed.loc[1, "health_flag"] == "healthy"
    assert indexed.loc[2, "health_flag"] == "low_signal"
    assert indexed.loc[1, "is_healthy"]
    assert not indexed.loc[2, "is_healthy"]
    assert {"mean_intensity", "skewness", "is_healthy", "area_px"} <= set(df.columns)


def test_flags_skewed():
    # Highly skewed: a few very bright pixels inside an otherwise dim nucleus
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[5:25, 5:25] = 1
    img = np.zeros((64, 64), dtype=np.float32)
    img[5:25, 5:25] = 500.0  # mean above min_mean_intensity
    img[5:7, 5:7] = 20000.0  # a few extreme outlier pixels → large positive skew
    df = nuclear_health_stats(
        masks, img, min_mean_intensity=300.0, max_skewness=1.5, min_nucleus_area=10
    )
    assert df.iloc[0]["health_flag"] == "skewed"


def test_debris_dropped():
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[10:12, 10:12] = 1  # 4 px — below default min_nucleus_area
    img = np.full((64, 64), 1000.0, dtype=np.float32)
    df = nuclear_health_stats(masks, img, min_nucleus_area=200)
    assert len(df) == 0


def test_shape_mismatch_raises():
    import pytest

    masks = np.zeros((64, 64), dtype=np.int32)
    img = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        nuclear_health_stats(masks, img)


_REPORT_DIR = Path(__file__).parent.parent / "reports" / "nuclear_health_qc"


def test_plot_writes_png():
    from tmem_align.analysis.nuclear_health import plot_nuclear_health

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[10:25, 10:25] = 1
    masks[40:55, 40:55] = 2
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:25, 10:25] = 1000.0
    img[40:55, 40:55] = 50.0
    df = nuclear_health_stats(masks, img, min_mean_intensity=300.0, min_nucleus_area=10)

    out = plot_nuclear_health(
        df,
        _REPORT_DIR / "synthetic_nuclear_health.png",
        title="Synthetic QC — healthy vs low_signal",
    )

    assert out.exists()
    assert out.stat().st_size > 0
