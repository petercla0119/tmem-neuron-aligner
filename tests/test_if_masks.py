"""Self-checks for napari-ready mask export (no cellpose/nd2 needed)."""
import numpy as np

from tmem_align.analysis.if_features import (
    MASK_KINDS,
    _fov_mask_paths,
    load_fov_masks,
    save_fov_masks,
)

# a filename analyze_fov would see; parse_fov_metadata pulls d7 / KO / G8_F4 from it
ND2 = "/data/d7/TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_G8_F4.nd2"


def _masks():
    nuclei = np.zeros((32, 32), np.int32)
    nuclei[4:10, 4:10] = 1
    cells = np.zeros((32, 32), np.int32)
    cells[2:14, 2:14] = 1
    lyso = np.zeros((32, 32), np.int32)
    lyso[5, 5] = 1
    lyso[20, 20] = 2  # two distinct puncta labels
    return nuclei, cells, lyso


def test_paths_foldered_by_timepoint_condition(tmp_path):
    paths = _fov_mask_paths(tmp_path, ND2)
    assert set(paths) == set(MASK_KINDS)
    p = paths["cells"]
    assert p.parent.name == "KO" and p.parent.parent.name == "d7", "foldered by condition/timepoint"
    assert p.name.endswith("__cells.tif")


def test_save_then_load_round_trips_labels(tmp_path):
    nuclei, cells, lyso = _masks()
    save_fov_masks(tmp_path, ND2, nuclei, cells, lyso)

    loaded = load_fov_masks(tmp_path, ND2)
    assert set(loaded) == set(MASK_KINDS)
    np.testing.assert_array_equal(loaded["nuclei"], nuclei)
    np.testing.assert_array_equal(loaded["cells"], cells)
    np.testing.assert_array_equal(loaded["lysosomes"], lyso)
    # labels must survive as integers (napari Labels needs int, not float)
    assert np.issubdtype(loaded["lysosomes"].dtype, np.integer)
    assert set(np.unique(loaded["lysosomes"])) == {0, 1, 2}


def test_load_skips_missing_masks(tmp_path):
    assert load_fov_masks(tmp_path, ND2) == {}, "nothing saved yet -> empty, no crash"
