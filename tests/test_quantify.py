import numpy as np
import tifffile as tif

from tmem_align.quantify import quantify_puncta_vs_diffuse, quantify_puncta_vs_diffuse_roi


def test_quantify_puncta_vs_diffuse(tmp_path):
    arr = np.zeros((3, 64, 64), dtype=np.uint16)
    arr[:, 20:25, 20:25] = 1000
    path = tmp_path / "stack.ome.tif"
    tif.imwrite(path, arr, metadata={"axes": "TYX"}, ome=True)
    df = quantify_puncta_vs_diffuse(path)
    assert len(df) == 3
    assert "rupture_like_score" in df.columns


def test_quantify_single_frame_yx(tmp_path):
    arr = np.zeros((64, 64), dtype=np.uint16)
    arr[20:25, 20:25] = 1000
    path = tmp_path / "frame.ome.tif"
    tif.imwrite(path, arr, metadata={"axes": "YX"}, ome=True)
    df = quantify_puncta_vs_diffuse(path)
    assert len(df) == 1
    assert df.loc[0, "puncta_count"] >= 1


def test_quantify_puncta_vs_diffuse_roi(tmp_path):
    arr = np.zeros((2, 64, 64), dtype=np.uint16)
    arr[:, 18:46, 18:46] = 80
    arr[:, 28:34, 28:34] = 1000
    path = tmp_path / "roi_stack.ome.tif"
    tif.imwrite(path, arr, metadata={"axes": "TYX"}, ome=True)

    df = quantify_puncta_vs_diffuse_roi(path, roi_min_size_pixels=16, foreground_percentile=50)

    assert len(df) == 2
    assert (df["roi_area_pixels"] > 0).all()
    assert (df["roi_fraction"] < 1).all()
    assert "rupture_like_score" in df.columns
