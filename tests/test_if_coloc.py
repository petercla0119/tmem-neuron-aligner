"""Synthetic checks for the 3D TMEM-LAMP1 coloc method.

Puncta placed ON a LAMP1 blob must read as enriched (obs fraction >> null);
puncta scattered far from it must read at ~the null floor. If the method can't
tell those apart it is broken.
"""

import numpy as np

from tmem_align.analysis.if_coloc import (
    ColocConfig,
    coloc_enrichment,
    detect_tmem_puncta_3d,
    lamp1_mask_3d,
)

SAMPLING = (1.0, 1.0, 1.0)  # isotropic µm/voxel for a clean synthetic


def _blob(shape, center, radius):
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = center
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _label_puncta(shape, centers, half=1):
    """Labelled volume of small cubic puncta (2*half per side, >= 2 voxels)."""
    lab = np.zeros(shape, dtype=np.int32)
    for i, (cz, cy, cx) in enumerate(centers, start=1):
        lab[cz - half : cz + half, cy - half : cy + half, cx - half : cx + half] = i
    return lab


def test_detect_puncta_finds_bright_voxels():
    stack = np.zeros((8, 40, 40), dtype=np.float32)
    stack[4, 10:13, 10:13] = 5000  # one bright cluster well above the 99.9 pct
    stack[2, 30, 30] = 5000
    stack[2, 31, 30] = 5000
    _, centroids = detect_tmem_puncta_3d(stack, top_percentile=99.0, min_size_vox=2)
    assert centroids.shape[0] == 2  # two clusters, single stray voxel dropped


def test_enrichment_distinguishes_coloc_from_random():
    shape = (20, 60, 60)
    lamp1 = _blob(shape, center=(10, 15, 15), radius=5)
    cell_volume = np.ones(shape, dtype=bool)  # whole crop is one cell
    rng = np.random.default_rng(0)

    # colocalized: puncta shells sitting inside the LAMP1 blob -> min shell dist 0
    on = _label_puncta(shape, [(10, 15, 15), (10, 13, 15), (9, 15, 14), (11, 16, 15)])
    coloc = coloc_enrichment(on, lamp1, cell_volume, SAMPLING, 1.5, 200, rng)

    # far: puncta in the opposite corner, nowhere near the blob
    far = _label_puncta(shape, [(15, 50, 50), (15, 48, 50), (14, 50, 48)])
    rand = coloc_enrichment(far, lamp1, cell_volume, SAMPLING, 1.5, 200, rng)

    assert coloc["n_puncta"] == 4
    assert coloc["obs_frac_within"] == 1.0
    assert coloc["obs_min_dist_um"] == 0.0  # a shell voxel touches the blob
    assert coloc["obs_max_dist_um"] >= coloc["obs_min_dist_um"]
    assert coloc["enrichment"] > 0.5
    assert coloc["enrichment"] > rand["enrichment"]
    assert rand["obs_frac_within"] < coloc["obs_frac_within"]
    assert coloc["empirical_p"] < 0.05  # observed clears the null


def test_lamp1_mask_thresholds():
    img = np.full((5, 20, 20), 100.0, dtype=np.float32)
    img[2, 5:10, 5:10] = 3000.0
    mask = lamp1_mask_3d(img, bg_percentile=50.0)
    assert mask[2, 5:10, 5:10].all()
    assert not mask[0].any()


def test_config_roundtrip(tmp_path):
    p = tmp_path / "coloc.yaml"
    p.write_text("data_root: /d\ntimepoint: d7\noutput_dir: /o\nthreshold_um: 0.4\n")
    cfg = ColocConfig.from_yaml(p)
    assert cfg.timepoint == "d7" and cfg.threshold_um == 0.4
