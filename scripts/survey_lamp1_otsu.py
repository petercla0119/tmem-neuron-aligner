"""Survey per-FOV LAMP1 Otsu thresholds across all 72 d7 FOVs.

Loads LAMP1 channel only (no Cellpose), computes bg-subtracted Otsu per FOV,
groups by condition. Outputs summary stats + picks a principled fixed threshold
from Control+KI FOVs to use for the brightness-decoupled re-analysis.

Usage:
    python scripts/survey_lamp1_otsu.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
CH_LAMP1 = "640nm"
BG_PERCENTILE = 50.0
LAMP1_LUT_FLOOR = 120.0  # fixed background floor from DISPLAY_LUT


def otsu_threshold(img: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    img = np.asarray(img, dtype=np.float32)
    corrected = np.clip(img - np.percentile(img, BG_PERCENTILE), 0, None)
    return float(threshold_otsu(corrected)), float(np.percentile(img, BG_PERCENTILE))


def load_lamp1(nd2_path: Path) -> np.ndarray:
    import nd2
    with nd2.ND2File(nd2_path) as f:
        channel_names = [
            str(getattr(getattr(ch, "channel", ch), "name", None) or f"ch{i}")
            for i, ch in enumerate(f.metadata.channels)
        ]
        arr = f.asarray()
    if arr.ndim == 4:
        arr = arr.max(axis=0)
    idx = channel_names.index(CH_LAMP1)
    return arr[idx]


CONDITION_DIRS = {
    "TMEM_KO": "KO",
    "Z59_PLD_Control": "Control",
    "Z60_PLD_TMEMki": "KI",
}

rows = []
for cond_dir, cond_label in CONDITION_DIRS.items():
    nd2_files = sorted((DATA_DIR / cond_dir).glob("*.nd2"))
    print(f"  {cond_label}: {len(nd2_files)} FOVs", flush=True)
    for nd2_path in nd2_files:
        try:
            lamp1 = load_lamp1(nd2_path)
            thr, bg = otsu_threshold(lamp1)
            # also compute Otsu on fixed-floor bg-subtracted
            corrected_fixed = np.clip(lamp1.astype(np.float32) - LAMP1_LUT_FLOOR, 0, None)
            from skimage.filters import threshold_otsu
            thr_fixed_bg = float(threshold_otsu(corrected_fixed))
            rows.append({
                "condition": cond_label,
                "file": nd2_path.name,
                "lamp1_p50": bg,
                "lamp1_mean": float(lamp1.mean()),
                "otsu_p50bg": thr,            # Otsu on p50-subtracted (current pipeline)
                "otsu_fixed_bg": thr_fixed_bg, # Otsu on fixed-floor subtracted
            })
        except Exception as exc:
            print(f"    SKIP {nd2_path.name}: {exc}")

df = pd.DataFrame(rows)
print("\n=== Per-condition Otsu threshold distribution (p50-bg-subtracted) ===")
summary = df.groupby("condition")["otsu_p50bg"].agg(["median", "mean", "std", "min", "max"])
print(summary.to_string())

print("\n=== Per-condition lamp1_mean raw ===")
print(df.groupby("condition")["lamp1_mean"].agg(["median", "mean", "std"]).to_string())

print("\n=== Per-condition Otsu on fixed-bg (LUT floor 120 DN) ===")
print(df.groupby("condition")["otsu_fixed_bg"].agg(["median", "mean", "std", "min", "max"]).to_string())

# Principled fixed threshold: median Otsu from Control + KI (not dragged down by KO dimness)
ctrl_ki = df[df["condition"].isin(["Control", "KI"])]
chosen_thr = float(ctrl_ki["otsu_p50bg"].median())
print("\n=== Chosen fixed threshold ===")
print(f"  Median Otsu (Control+KI, p50-bg): {chosen_thr:.1f} DN")
print("  Rationale: median of non-KO conditions; KO Otsu is expected lower due to dim LAMP1.")

# Also: fixed bg floor version
chosen_fixed_bg_thr = float(ctrl_ki["otsu_fixed_bg"].median())
print(f"  Median Otsu (Control+KI, fixed-bg 120DN): {chosen_fixed_bg_thr:.1f} DN")

# Save survey
out = Path(__file__).resolve().parent.parent / "reports" / "if_segmentation_pilot"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "d7_lamp1_otsu_survey.csv", index=False)
print(f"\nSurvey saved to {out}/d7_lamp1_otsu_survey.csv")
