import numpy as np
import tifffile as tif

from tmem_align.quantify import quantify_puncta_vs_diffuse


def test_quantify_puncta_vs_diffuse(tmp_path):
    arr = np.zeros((3, 64, 64), dtype=np.uint16)
    arr[:, 20:25, 20:25] = 1000
    path = tmp_path / "stack.ome.tif"
    tif.imwrite(path, arr, metadata={"axes": "TYX"}, ome=True)
    df = quantify_puncta_vs_diffuse(path)
    assert len(df) == 3
    assert "rupture_like_score" in df.columns
