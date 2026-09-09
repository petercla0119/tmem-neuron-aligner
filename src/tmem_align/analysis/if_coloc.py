"""3D cleaved-TMEM106B (561) vs LAMP1 (640) colocalization for fixed-IF FOVs.

The designed method (not the 2D `tmem_manders_m1` proxy in if_features):

  1. Detect TMEM puncta in the z-stack (3D), not the MIP. 561 is sparse/dim
     (median ~= background; only the top ~0.1% of pixels carry signal), so the
     default threshold is a high percentile -- CALIBRATE per dataset.
  2. Build the LAMP1 lysosome volume (3D mask) and its surface.
  3. For each TMEM punctum, measure centroid-to-nearest-LAMP1-surface distance.
  4. Compare the observed distances to a position-randomized null: scatter the
     same number of points uniformly inside the cell volume, recompute distances.
  5. Enrichment readout: fraction of puncta within `threshold_um` of a LAMP1
     surface, observed vs null (and median distance observed vs null).

TMEM_KO is the empirical false-positive floor: KO carries no real cleaved-TMEM,
so its enrichment (observed - null) is the noise floor other conditions clear.
That comparison is done downstream on the per-cell CSV (this module just tags
each row with the cell's condition via the caller); see scripts/run_if_coloc.py.

Cell volume = the 2D cell-body mask (from MAP2 segmentation) extruded across z.
ponytail: we have no 3D cell segmentation; lateral mask x full depth is the
defensible cell volume and is what the null randomizes within.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ColocConfig:
    """Tiny config for a coloc run. Load from YAML with ColocConfig.from_yaml."""

    data_root: str
    timepoint: str  # e.g. "d7"
    output_dir: str
    tmem_channel: str = "561nm"
    lamp1_channel: str = "640nm"
    map2_channel: str = "488nm"
    dapi_channel: str = "405nm"
    threshold_um: float = 0.5  # "within" distance for the enrichment fraction
    n_null_iterations: int = 200
    tmem_top_percentile: float = 99.9  # CALIBRATE: 561 is sparse; top ~0.1% is signal
    tmem_threshold: float | None = None  # absolute DN overrides the percentile
    lamp1_bg_percentile: float = 50.0
    lamp1_threshold: float | None = None  # absolute DN overrides Otsu
    min_puncta_size_vox: int = 2
    mask_cache_dir: str | None = None  # if set, read 'cells' masks from <stem>.npz here
    max_fovs: int | None = None  # smoke cap
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ColocConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


def detect_tmem_puncta_3d(
    tmem_zyx: np.ndarray,
    top_percentile: float = 99.9,
    threshold: float | None = None,
    min_size_vox: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels_zyx, centroids Nx3 float (z,y,x)) for 561 puncta in 3D.

    Sparse-channel detection: threshold at a high percentile of the whole stack
    (top ~0.1% by default), 3D-connected-component label, drop specks < min_size_vox.
    Pass an absolute DN `threshold` to override the percentile for cross-FOV comparability.
    """
    from scipy import ndimage as ndi

    img = np.asarray(tmem_zyx, dtype=np.float32)
    thr = float(np.percentile(img, top_percentile)) if threshold is None else float(threshold)
    labels, n = ndi.label(img > thr)
    if n == 0:
        return labels.astype(np.int32), np.empty((0, 3), dtype=float)
    counts = np.bincount(labels.ravel())
    small = np.nonzero(counts < min_size_vox)[0]
    if small.size:
        labels[np.isin(labels, small)] = 0
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return labels.astype(np.int32), np.empty((0, 3), dtype=float)
    centroids = np.array(ndi.center_of_mass(np.ones_like(labels), labels, ids), dtype=float)
    return labels.astype(np.int32), centroids


def lamp1_mask_3d(
    lamp1_zyx: np.ndarray,
    bg_percentile: float = 50.0,
    threshold: float | None = None,
) -> np.ndarray:
    """Boolean 3D LAMP1 lysosome volume: background-subtract then Otsu (or fixed DN)."""
    from skimage.filters import threshold_otsu

    img = np.asarray(lamp1_zyx, dtype=np.float32)
    corrected = np.clip(img - np.percentile(img, bg_percentile), 0, None)
    thr = threshold_otsu(corrected) if threshold is None else float(threshold)
    return corrected > thr


def _sample_distances(dt: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Look up the distance transform at (possibly fractional) z,y,x coords."""
    if coords.shape[0] == 0:
        return np.empty(0, dtype=float)
    idx = np.rint(coords).astype(int)
    for ax in range(3):
        idx[:, ax] = np.clip(idx[:, ax], 0, dt.shape[ax] - 1)
    return dt[idx[:, 0], idx[:, 1], idx[:, 2]]


def coloc_enrichment(
    puncta_centroids: np.ndarray,
    lamp1_mask: np.ndarray,
    cell_volume: np.ndarray,
    sampling: tuple[float, float, float],
    threshold_um: float,
    n_null: int,
    rng: np.random.Generator,
) -> dict:
    """Observed-vs-null nearest-LAMP1-surface enrichment for one cell.

    distances are µm to the nearest LAMP1 voxel (0 inside a lysosome), via an
    anisotropic Euclidean distance transform. Null = n_null draws of the same
    number of points, uniform over cell_volume voxels.
    """
    from scipy import ndimage as ndi

    n = int(puncta_centroids.shape[0])
    out = {
        "n_puncta": n,
        "obs_frac_within": np.nan,
        "null_frac_within": np.nan,
        "null_frac_std": np.nan,
        "enrichment": np.nan,  # obs - null fraction
        "obs_median_dist_um": np.nan,
        "null_median_dist_um": np.nan,
        "empirical_p": np.nan,  # P(null frac >= obs frac)
    }
    if n == 0 or not lamp1_mask.any() or not cell_volume.any():
        return out

    dt = ndi.distance_transform_edt(~lamp1_mask, sampling=sampling)
    obs = _sample_distances(dt, puncta_centroids)
    obs_frac = float(np.mean(obs <= threshold_um))
    out["obs_frac_within"] = obs_frac
    out["obs_median_dist_um"] = float(np.median(obs))

    candidates = np.argwhere(cell_volume)  # Kx3 voxel coords in the cropped frame
    null_fracs = np.empty(n_null, dtype=float)
    null_meds = np.empty(n_null, dtype=float)
    for i in range(n_null):
        pick = candidates[rng.integers(0, candidates.shape[0], size=n)]
        d = _sample_distances(dt, pick.astype(float))
        null_fracs[i] = np.mean(d <= threshold_um)
        null_meds[i] = np.median(d)
    out["null_frac_within"] = float(null_fracs.mean())
    out["null_frac_std"] = float(null_fracs.std())
    out["null_median_dist_um"] = float(null_meds.mean())
    out["enrichment"] = obs_frac - float(null_fracs.mean())
    out["empirical_p"] = float((np.count_nonzero(null_fracs >= obs_frac) + 1) / (n_null + 1))
    return out


def analyze_fov_coloc(
    channels_3d: dict[str, np.ndarray],
    sampling: tuple[float, float, float],
    cell_masks_2d: np.ndarray,
    cfg: ColocConfig,
    seed: int = 0,
) -> list[dict]:
    """Per-cell 3D coloc for one FOV. cell_masks_2d is the MAP2 cell-body labels (YX).

    Returns a list of per-cell dicts (cell_id + coloc_enrichment fields). The caller
    tags condition/well/fov/timepoint and writes the CSV.
    """
    tmem = np.asarray(channels_3d[cfg.tmem_channel], dtype=np.float32)  # ZYX
    lamp1 = np.asarray(channels_3d[cfg.lamp1_channel], dtype=np.float32)
    nz = tmem.shape[0]

    # Detect once over the whole stack; a per-FOV threshold keeps detection uniform.
    _, all_centroids = detect_tmem_puncta_3d(
        tmem,
        top_percentile=cfg.tmem_top_percentile,
        threshold=cfg.tmem_threshold,
        min_size_vox=cfg.min_puncta_size_vox,
    )
    lamp1_full = lamp1_mask_3d(
        lamp1, bg_percentile=cfg.lamp1_bg_percentile, threshold=cfg.lamp1_threshold
    )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    labels = np.unique(cell_masks_2d)
    labels = labels[labels != 0]
    for lab in labels:
        cell2d = cell_masks_2d == lab
        ys, xs = np.where(cell2d)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cell2d_crop = cell2d[y0:y1, x0:x1]
        cell_vol = np.broadcast_to(cell2d_crop, (nz, *cell2d_crop.shape))  # extrude z
        lamp1_crop = lamp1_full[:, y0:y1, x0:x1] & cell_vol

        # puncta whose (y,x) fall inside this cell; shift into the crop frame
        if all_centroids.shape[0]:
            iy = np.rint(all_centroids[:, 1]).astype(int)
            ix = np.rint(all_centroids[:, 2]).astype(int)
            inb = (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
            inside = inb.copy()
            inside[inb] = cell2d[iy[inb], ix[inb]]
            pc = all_centroids[inside].copy()
            pc[:, 1] -= y0
            pc[:, 2] -= x0
        else:
            pc = np.empty((0, 3), dtype=float)

        res = coloc_enrichment(
            pc, lamp1_crop, cell_vol, sampling, cfg.threshold_um, cfg.n_null_iterations, rng
        )
        res["cell_id"] = int(lab)
        res["n_z_slices"] = int(nz)
        rows.append(res)
    return rows
