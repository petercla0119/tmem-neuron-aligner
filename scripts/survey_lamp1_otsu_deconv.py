"""Re-derive the LAMP1 fixed DN threshold on the IC->RL deconvolved stacks.

Deconv-domain mirror of survey_tmem_dn.py's method, applied to LAMP1 (Step 2 of
the 2026-09-09 plan, CORRECTED 2026-09-09).

Same segment-on-raw / measure-on-deconv / in-cell design as the TMEM survey:
MAP2 cell-body masks come from the raw ND2 (HITL cpsam model, the SAME cache the
TMEM survey builds), LAMP1 640 voxels come from the deconv OME-TIFF, and the Otsu
threshold is read off the *in-cell* voxel distribution -- NOT whole-FOV. Whole-FOV
Otsu folds in non-neuronal cells, debris, and background autofluorescence that the
MAP2 gate excludes; measuring LAMP1 inside neurons is what keeps it comparable to
the (in-cell) TMEM survey and to the per-cell feature pipeline.

Measured in the domain the LAMP1 detector consumes (`lamp1_mask_3d`): subtract the
whole-stack p50 background, then Otsu on the bg-corrected in-cell voxels, so the
chosen DN is directly usable as `lamp1_threshold`.

The one thing that CANNOT mirror TMEM: LAMP1 has lysosomes in every condition, so
there is no biological null to anchor to (KO is not a LAMP1 null). Selection stays
the median Otsu of Control+KI; per-condition medians (incl. KO) are reported for
transparency.

    PYTHONPATH=src python scripts/survey_lamp1_otsu_deconv.py d7 d14 d28

Prereq: the deconv MAP2 mask cache (reports/if_segmentation_pilot/deconv_mask_cache/).
Built by survey_tmem_dn_deconv.py; this script reuses it and will build any missing
masks the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tmem_align.analysis.if_coloc import ColocConfig  # noqa: E402

from deconv_io import condition_from_stem, deconv_fovs, load_deconv_fov_3d  # noqa: E402
from survey_tmem_dn_deconv import cell_masks  # noqa: E402  (shared raw MAP2 mask cache)

OUT = Path(__file__).resolve().parent.parent / "reports" / "if_segmentation_pilot"


def incell_otsu(lamp1_zyx: np.ndarray, cells_yx: np.ndarray, bg_percentile: float) -> tuple[float, float, int]:
    """Otsu on bg-corrected LAMP1 voxels inside MAP2 cell bodies (mirrors lamp1_mask_3d)."""
    bg = float(np.percentile(lamp1_zyx, bg_percentile))  # whole-stack p50, as the detector does
    corrected = np.clip(lamp1_zyx - bg, 0, None)
    mask3d = np.broadcast_to(cells_yx > 0, lamp1_zyx.shape)
    vox = corrected[mask3d]
    n_cells = int(np.unique(cells_yx).size - 1)
    return float(threshold_otsu(vox)), bg, n_cells


def survey(timepoint: str, cfg: ColocConfig) -> pd.DataFrame:
    fovs = deconv_fovs(timepoint)
    print(f"\n### {timepoint}: {len(fovs)} FOVs", flush=True)
    rows = []
    for i, f in enumerate(fovs):
        stem = f.name.replace(".ome.tif", "")
        try:
            cells = cell_masks(f, cfg.map2_channel)  # raw MAP2 masks (cached)
            lamp1 = load_deconv_fov_3d(f)[cfg.lamp1_channel]  # ZYX deconv
            thr, bg, n_cells = incell_otsu(lamp1, cells, cfg.lamp1_bg_percentile)
            if n_cells == 0:
                print(f"  SKIP {stem}: no cells", flush=True)
                continue
            rows.append(
                {
                    "condition": condition_from_stem(stem),
                    "file": stem,
                    "lamp1_p50": bg,
                    "n_cells": n_cells,
                    "otsu_incell": thr,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {stem}: {exc}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(fovs)}]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{timepoint}_lamp1_otsu_survey_deconv.csv", index=False)

    print("=== per-condition in-cell Otsu (bg-corrected) ===")
    print(df.groupby("condition")["otsu_incell"].agg(["median", "mean", "std", "min", "max"]).to_string())
    ctrl_ki = df[df["condition"].isin(["Control", "KI"])]
    chosen = float(ctrl_ki["otsu_incell"].median())
    print(f"--> {timepoint} chosen LAMP1 threshold (median in-cell Otsu Control+KI): {chosen:.1f} DN")
    return df


def main() -> None:
    tps = sys.argv[1:] or ["d7", "d14", "d28"]
    cfg = ColocConfig.from_yaml(str(Path(__file__).resolve().parent.parent / "configs/if_coloc_d7.yaml"))
    summary = {}
    for tp in tps:
        df = survey(tp, cfg)
        ctrl_ki = df[df["condition"].isin(["Control", "KI"])]
        summary[tp] = float(ctrl_ki["otsu_incell"].median())
    print("\n=== deconv LAMP1 thresholds (median in-cell Otsu Control+KI) ===")
    for tp, v in summary.items():
        print(f"  {tp}: {v:.1f} DN   (raw-domain reference: 3765 DN; prior whole-FOV deconv value superseded)")


if __name__ == "__main__":
    main()
