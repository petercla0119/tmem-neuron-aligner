"""3D cleaved-TMEM106B (561) vs LAMP1 (640) colocalization for fixed-IF FOVs.

The designed method (not the 2D `tmem_manders_m1` proxy in if_features):

  1. Detect TMEM puncta in the z-stack (3D), not the MIP. 561 is sparse/dim
     (median ~= background; only the top ~0.1% of pixels carry signal), so the
     default threshold is a high percentile -- CALIBRATE per dataset.
  2. Build the LAMP1 lysosome volume (3D mask) and its surface.
  3. For each TMEM punctum, measure SURFACE-to-nearest-LAMP1-surface distance:
     the punctum is a voxel SHELL (its thresholded connected-component boundary,
     find_boundaries mode="inner"), not a centroid point. The punctum's distance
     is the MIN over its shell voxels of the anisotropic EDT to LAMP1 (nearest
     approach). We also record mean/median/max/std across the cell's puncta.
  4. Compare the observed distances to a SHAPE-PRESERVING position-randomized
     null: drop each real punctum's shell (same voxel geometry) at a random
     location inside the cell volume, recompute its min-shell distance. A point
     null would make surfaces look artefactually close; preserving shape keeps
     the observed-vs-null comparison honest.
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
    def from_yaml(cls, path: str | Path) -> ColocConfig:
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


def _shell_offsets(
    puncta_labels: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per-punctum voxel shells for a labelled crop.

    Returns (ids, shells) where shells[i] is a Kx3 int array of (z,y,x) shell
    voxel coords for punctum ids[i]. The shell is the inner boundary of each
    thresholded connected component (find_boundaries mode="inner") -- the
    punctum's iso-DN surface, one voxel thick. Two-voxel puncta => both voxels.
    """
    from skimage.segmentation import find_boundaries

    ids = np.unique(puncta_labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return ids, []
    shell = find_boundaries(puncta_labels, mode="inner")
    sc = np.argwhere(shell)
    sl = puncta_labels[shell]
    # A component fully interior to nothing still yields a shell; guard empties.
    shells = []
    keep_ids = []
    for pid in ids:
        pts = sc[sl == pid]
        if pts.shape[0] == 0:  # e.g. degenerate label; fall back to all voxels
            pts = np.argwhere(puncta_labels == pid)
        shells.append(pts)
        keep_ids.append(pid)
    return np.asarray(keep_ids), shells


def _summary(vals: np.ndarray, prefix: str) -> dict:
    """min/mean/median/max/std of a per-punctum distance array, NaN-safe."""
    if vals.size == 0:
        return {f"{prefix}_{k}_dist_um": np.nan for k in ("min", "mean", "median", "max", "std")}
    return {
        f"{prefix}_min_dist_um": float(vals.min()),
        f"{prefix}_mean_dist_um": float(vals.mean()),
        f"{prefix}_median_dist_um": float(np.median(vals)),
        f"{prefix}_max_dist_um": float(vals.max()),
        f"{prefix}_std_dist_um": float(vals.std()),
    }


def coloc_enrichment(
    puncta_labels: np.ndarray,
    lamp1_mask: np.ndarray,
    cell_volume: np.ndarray,
    sampling: tuple[float, float, float],
    threshold_um: float,
    n_null: int,
    rng: np.random.Generator,
) -> dict:
    """Observed-vs-null nearest-LAMP1-SURFACE enrichment for one cell.

    Each TMEM punctum is a voxel shell (see _shell_offsets); its distance is the
    MIN over its shell voxels of the anisotropic EDT to the LAMP1 surface (µm; 0
    inside a lysosome). Null = n_null draws that re-drop each punctum's shell,
    shape intact, at a random location inside cell_volume (shape-preserving).

    Per-punctum distances are summarised across the cell's puncta as
    min/mean/median/max/std (obs_*_dist_um; null_*_dist_um = mean over draws).
    """
    from scipy import ndimage as ndi

    ids, shells = _shell_offsets(puncta_labels)
    n = int(ids.size)
    out = {
        "n_puncta": n,
        "obs_frac_within": np.nan,
        "null_frac_within": np.nan,
        "null_frac_std": np.nan,
        "enrichment": np.nan,  # obs - null fraction
        "empirical_p": np.nan,  # P(null frac >= obs frac)
    }
    out.update(_summary(np.empty(0), "obs"))
    out.update(_summary(np.empty(0), "null"))
    if n == 0 or not lamp1_mask.any() or not cell_volume.any():
        return out

    dt = ndi.distance_transform_edt(~lamp1_mask, sampling=sampling)

    # Observed: min EDT over each punctum's own shell voxels.
    obs = np.array([_sample_distances(dt, s.astype(float)).min() for s in shells])
    obs_frac = float(np.mean(obs <= threshold_um))
    out["obs_frac_within"] = obs_frac
    out.update(_summary(obs, "obs"))

    # Shape-preserving null: translate each shell to a random cell voxel.
    candidates = np.argwhere(cell_volume)  # Kx3 voxel coords in the cropped frame
    centers = [np.rint(s.mean(axis=0)).astype(int) for s in shells]  # shell centroid
    offsets = [s - c for s, c in zip(shells, centers)]
    null_fracs = np.empty(n_null, dtype=float)
    null_meds = np.empty(n_null, dtype=float)
    null_means = np.empty(n_null, dtype=float)
    for i in range(n_null):
        picks = candidates[rng.integers(0, candidates.shape[0], size=n)]
        d = np.array(
            [
                _sample_distances(dt, (off + picks[j]).astype(float)).min()
                for j, off in enumerate(offsets)
            ]
        )
        null_fracs[i] = np.mean(d <= threshold_um)
        null_meds[i] = np.median(d)
        null_means[i] = d.mean()
    out["null_frac_within"] = float(null_fracs.mean())
    out["null_frac_std"] = float(null_fracs.std())
    out["null_mean_dist_um"] = float(null_means.mean())
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
    labels_full, all_centroids = detect_tmem_puncta_3d(
        tmem,
        top_percentile=cfg.tmem_top_percentile,
        threshold=cfg.tmem_threshold,
        min_size_vox=cfg.min_puncta_size_vox,
    )
    # centroid i <-> ids_all[i] (center_of_mass was called on sorted unique ids)
    ids_all = np.unique(labels_full)
    ids_all = ids_all[ids_all != 0]
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

        # puncta whose (y,x) centroid falls inside this cell; crop labels, keep only theirs
        if all_centroids.shape[0]:
            iy = np.rint(all_centroids[:, 1]).astype(int)
            ix = np.rint(all_centroids[:, 2]).astype(int)
            inb = (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
            inside = inb.copy()
            inside[inb] = cell2d[iy[inb], ix[inb]]
            inside_ids = ids_all[inside]
            lab_crop = labels_full[:, y0:y1, x0:x1].copy()
            lab_crop[~np.isin(lab_crop, inside_ids)] = 0
        else:
            lab_crop = np.zeros((nz, *cell2d_crop.shape), dtype=labels_full.dtype)

        res = coloc_enrichment(
            lab_crop, lamp1_crop, cell_vol, sampling, cfg.threshold_um, cfg.n_null_iterations, rng
        )
        res["cell_id"] = int(lab)
        res["n_z_slices"] = int(nz)
        rows.append(res)
    return rows
