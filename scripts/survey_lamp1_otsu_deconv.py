"""Re-derive the LAMP1 fixed DN threshold on the IC->RL deconvolved stacks.

Deconv-domain mirror of survey_lamp1_otsu.py (Step 2 of the 2026-09-09 plan).
Same rule: per-FOV p50-bg-subtracted Otsu on the LAMP1 MIP, chosen threshold =
median Otsu of Control+KI (KO excluded, it is dimmer). Whole-FOV, no masks.
Writes reports/if_segmentation_pilot/<tp>_lamp1_otsu_survey_deconv.csv per timepoint
(does NOT clobber the raw-domain survey CSVs).

    PYTHONPATH=src python scripts/survey_lamp1_otsu_deconv.py d7 d14 d28
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deconv_io import condition_from_stem, deconv_fovs, load_deconv_fov_3d  # noqa: E402

CH_LAMP1 = "640nm"
BG_PERCENTILE = 50.0
OUT = Path(__file__).resolve().parent.parent / "reports" / "if_segmentation_pilot"


def otsu_p50bg(mip: np.ndarray) -> tuple[float, float]:
    bg = float(np.percentile(mip, BG_PERCENTILE))
    corrected = np.clip(mip - bg, 0, None)
    return float(threshold_otsu(corrected)), bg


def survey(timepoint: str) -> pd.DataFrame:
    fovs = deconv_fovs(timepoint)
    print(f"\n### {timepoint}: {len(fovs)} FOVs", flush=True)
    rows = []
    for i, f in enumerate(fovs):
        stem = f.name.replace(".ome.tif", "")
        try:
            lamp1 = load_deconv_fov_3d(f)[CH_LAMP1]  # ZYX
            mip = lamp1.max(axis=0)
            thr, bg = otsu_p50bg(mip)
            rows.append(
                {
                    "condition": condition_from_stem(stem),
                    "file": stem,
                    "lamp1_p50": bg,
                    "lamp1_mean": float(mip.mean()),
                    "otsu_p50bg": thr,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {stem}: {exc}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(fovs)}]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{timepoint}_lamp1_otsu_survey_deconv.csv", index=False)

    print("=== per-condition Otsu (p50-bg-subtracted) ===")
    print(df.groupby("condition")["otsu_p50bg"].agg(["median", "mean", "std", "min", "max"]).to_string())
    ctrl_ki = df[df["condition"].isin(["Control", "KI"])]
    chosen = float(ctrl_ki["otsu_p50bg"].median())
    print(f"--> {timepoint} chosen LAMP1 threshold (median Otsu Control+KI): {chosen:.1f} DN")
    return df


def main() -> None:
    tps = sys.argv[1:] or ["d7", "d14", "d28"]
    summary = {}
    for tp in tps:
        df = survey(tp)
        ctrl_ki = df[df["condition"].isin(["Control", "KI"])]
        summary[tp] = float(ctrl_ki["otsu_p50bg"].median())
    print("\n=== deconv LAMP1 thresholds (median Otsu Control+KI) ===")
    for tp, v in summary.items():
        print(f"  {tp}: {v:.1f} DN   (raw-domain reference: 3765 DN)")


if __name__ == "__main__":
    main()
