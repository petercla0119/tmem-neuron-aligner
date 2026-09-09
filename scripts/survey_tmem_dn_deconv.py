"""Re-derive the TMEM (561) absolute DN threshold on the IC->RL deconvolved stacks.

Deconv-domain mirror of survey_tmem_dn.py (Step 3 of the 2026-09-09 plan).
Segment-on-raw / measure-on-deconv: MAP2 cell-body masks come from the raw ND2
(HITL cpsam model, cached to reports/if_segmentation_pilot/deconv_mask_cache/),
561 puncta voxels come from the deconv OME-TIFF. KO is the null.

Two passes, method unchanged from the raw survey:
  1. per-FOV intra-cell 561 percentiles + raw-stack Otsu -> candidate DNs from KO.
  2. re-run detect_tmem_puncta_3d at each candidate, report puncta/cell per condition;
     pick the smallest DN where KO floors out while Control/KI keep puncta.

Writes reports/if_segmentation_pilot/<tp>_tmem_dn_survey_deconv.csv per timepoint.

    PYTHONPATH=src python scripts/survey_tmem_dn_deconv.py d7 d14 d28 [max_per_cond]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tmem_align.analysis.if_coloc import ColocConfig, detect_tmem_puncta_3d  # noqa: E402
from tmem_align.analysis.if_spatial import (  # noqa: E402
    CH_MAP2,
    apply_display_lut,
    load_fov,
)

from deconv_io import condition_from_stem, deconv_fovs, load_deconv_fov_3d, raw_for_deconv  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "if_segmentation_pilot"
MASK_CACHE = OUT / "deconv_mask_cache"
_MODEL = None  # loaded once (segment_cell_bodies reinstantiates per call -> too slow at scale)


def _get_model():
    global _MODEL
    if _MODEL is None:
        from cellpose import models

        from tmem_align.analysis.if_spatial import _CELLBODY_MODEL

        _MODEL = models.CellposeModel(gpu=True, pretrained_model=str(_CELLBODY_MODEL))
    return _MODEL


def cell_masks(tif_path: Path, map2_channel: str) -> np.ndarray:
    """MAP2 cell-body labels from the RAW ND2 (cached). Matches segment_cell_bodies."""
    MASK_CACHE.mkdir(parents=True, exist_ok=True)
    npz = MASK_CACHE / f"{tif_path.name.replace('.ome.tif', '')}.npz"
    if npz.exists():
        return np.load(npz)["cells"]
    raw = raw_for_deconv(tif_path)
    map2 = load_fov(raw)[map2_channel]  # raw uint16 MIP
    img_u8 = (apply_display_lut(map2, CH_MAP2) * 255).astype(np.uint8)
    masks, _, _ = _get_model().eval(img_u8)
    masks = masks.astype(np.int32)
    np.savez_compressed(npz, cells=masks)
    return masks


def _puncta_per_cell(tmem_zyx, cells_yx, thr, min_vox) -> float:
    _, centroids = detect_tmem_puncta_3d(tmem_zyx, threshold=thr, min_size_vox=min_vox)
    n_cells = int(np.unique(cells_yx).size - 1)
    if n_cells == 0:
        return float("nan")
    if centroids.shape[0] == 0:
        return 0.0
    iy = np.clip(np.rint(centroids[:, 1]).astype(int), 0, cells_yx.shape[0] - 1)
    ix = np.clip(np.rint(centroids[:, 2]).astype(int), 0, cells_yx.shape[1] - 1)
    return float(np.count_nonzero(cells_yx[iy, ix] > 0)) / n_cells


def survey(timepoint: str, cfg: ColocConfig, max_per_cond: int | None) -> None:
    fovs = deconv_fovs(timepoint)
    if max_per_cond is not None:
        seen: dict[str, int] = defaultdict(int)
        capped = []
        for f in fovs:
            c = condition_from_stem(f.name.replace(".ome.tif", ""))
            if seen[c] < max_per_cond:
                capped.append(f)
                seen[c] += 1
        fovs = capped
    print(f"\n########## {timepoint}: {len(fovs)} FOVs ##########", flush=True)

    # ---- pass 1: per-FOV intra-cell 561 percentiles + raw-stack Otsu ----
    stats, ok = [], []
    for i, f in enumerate(fovs):
        stem = f.name.replace(".ome.tif", "")
        try:
            cells = cell_masks(f, cfg.map2_channel)
            tmem = load_deconv_fov_3d(f)[cfg.tmem_channel]  # ZYX (deconv)
            mask3d = np.broadcast_to(cells > 0, tmem.shape)
            vox = tmem[mask3d]
            if vox.size == 0:
                print(f"  SKIP {stem}: no in-cell voxels", flush=True)
                continue
            stats.append(
                {
                    "condition": condition_from_stem(stem),
                    "file": stem,
                    "p99": float(np.percentile(vox, 99)),
                    "p99.9": float(np.percentile(vox, 99.9)),
                    "p99.99": float(np.percentile(vox, 99.99)),
                    "otsu_raw": float(threshold_otsu(vox)),
                    "n_cells": int(np.unique(cells).size - 1),
                }
            )
            ok.append(f)
            print(f"  [{i + 1}/{len(fovs)}] {stem} ({stats[-1]['condition']})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {stem}: {exc}", flush=True)

    if not stats:
        print(f"{timepoint}: no FOVs produced stats")
        return
    df = pd.DataFrame(stats)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{timepoint}_tmem_dn_survey_deconv.csv", index=False)

    print(f"\n=== {timepoint} per-condition intra-cell 561 percentiles (median across FOVs) ===")
    print(df.groupby("condition")[["p99", "p99.9", "p99.99", "otsu_raw"]].median().round(1).to_string())

    ko = df[df["condition"] == "KO"]
    if ko.empty:
        print(f"{timepoint}: NO KO FOVs -- cannot set a KO-referenced floor.")
        return
    candidates = {
        "KO_median_p99.9": float(ko["p99.9"].median()),
        "KO_median_p99.99": float(ko["p99.99"].median()),
        "KO_median_otsu": float(ko["otsu_raw"].median()),
    }
    print(f"\n=== {timepoint} candidate DNs (from KO null) ===")
    for k, v in candidates.items():
        print(f"  {k}: {v:.1f} DN")

    # ---- pass 2: puncta/cell per condition at each candidate DN ----
    per_cand = {n: defaultdict(list) for n in candidates}
    for f in ok:
        try:
            cells = cell_masks(f, cfg.map2_channel)  # cached
            tmem = load_deconv_fov_3d(f)[cfg.tmem_channel]
            cond = condition_from_stem(f.name.replace(".ome.tif", ""))
            for name, thr in candidates.items():
                per_cand[name][cond].append(
                    _puncta_per_cell(tmem, cells, thr, cfg.min_puncta_size_vox)
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP (pass2) {f.name}: {exc}", flush=True)
    rows = []
    for name, thr in candidates.items():
        row = {"candidate": name, "DN": round(thr, 1)}
        for cond, vals in per_cand[name].items():
            row[cond] = round(float(np.nanmean(vals)), 2)
        rows.append(row)
    print(f"\n=== {timepoint} mean puncta/cell by condition (want KO ~0, Control/KI kept) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = sys.argv[1:]
    max_per_cond = None
    if args and args[-1].isdigit():
        max_per_cond = int(args[-1])
        args = args[:-1]
    tps = args or ["d7", "d14", "d28"]
    cfg = ColocConfig.from_yaml(str(Path(__file__).resolve().parent.parent / "configs/if_coloc_d7.yaml"))
    for tp in tps:
        survey(tp, cfg, max_per_cond)
    print("\nDONE. raw-domain reference: TMEM 1763.5 DN (KO floored 0.31/cell, KI 2.56).")


if __name__ == "__main__":
    main()
