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


def test_segment_nuclei_cache(tmp_path):
    """Cache hit returns saved masks without calling Cellpose."""
    from unittest.mock import patch

    import numpy as np

    from tmem_align.analysis.if_spatial import segment_nuclei

    fake_masks = np.array([[0, 1], [1, 0]], dtype=np.int32)
    cache = tmp_path / "nuclei.npy"

    # Prime the cache manually — no Cellpose call needed.
    np.save(cache, fake_masks)

    # Patch cellpose so any accidental call raises.
    with patch.dict("sys.modules", {"cellpose": None, "cellpose.models": None}):
        result = segment_nuclei(np.zeros((2, 2), dtype=np.float32), cache_path=cache)

    np.testing.assert_array_equal(result, fake_masks)


def test_segment_nuclei_cache_writes(tmp_path, monkeypatch):
    """On cache miss, result is saved to disk."""
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from tmem_align.analysis import if_spatial

    fake_masks = np.array([[0, 2], [2, 0]], dtype=np.int32)

    # Stub out CellposeModel so we never hit real Cellpose.
    mock_model = MagicMock()
    mock_model.eval.return_value = (fake_masks, None, None)
    mock_cellpose = SimpleNamespace(models=SimpleNamespace(CellposeModel=lambda **kw: mock_model))
    monkeypatch.setattr(if_spatial, "__builtins__", __builtins__)  # no-op; needed for scope
    import builtins, importlib, sys
    sys.modules["cellpose"] = mock_cellpose
    sys.modules["cellpose.models"] = mock_cellpose.models

    cache = tmp_path / "sub" / "nuclei.npy"
    result = if_spatial.segment_nuclei(
        np.zeros((2, 2), dtype=np.float32), gpu=False, cache_path=cache
    )

    assert cache.exists()
    np.testing.assert_array_equal(np.load(cache), fake_masks)
    np.testing.assert_array_equal(result, fake_masks)

    del sys.modules["cellpose"]
    del sys.modules["cellpose.models"]


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
