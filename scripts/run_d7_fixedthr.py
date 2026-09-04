"""Re-run d7 fixed-IF analysis with a fixed LAMP1 threshold.

Fixed threshold: 3765 DN (median Otsu of Control+KI FOVs on p50-bg-subtracted LAMP1).
Derived from scripts/survey_lamp1_otsu.py; see reports/if_segmentation_pilot/d7_lamp1_otsu_survey.csv.

Motivation: per-FOV Otsu adapts down for dimmer KO images, potentially fragmenting
detections and artificially reducing lyso_mean_size_um2 for KO. This run decouples
detection from per-condition brightness to test whether the KO effect is real.

Outputs: reports/if_segmentation_pilot/d7_percell_fixedthr.csv
Log:     reports/if_segmentation_pilot/d7_fixedthr_run.log (stdout)

Mask export (optional):
    python run_d7_fixedthr.py --masks-dir data/masks/d7
    Writes nuclei/cells/lysosomes int32 TIFFs per FOV, napari Labels-ready.
    Layout: {masks_dir}/{timepoint}/{condition}/{stem}__{kind}.tif
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure worktree code is imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmem_align.analysis.if_features import build_table, condition_stats

DATA_DIR = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
OUT_DIR = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Principled fixed threshold: median Otsu from Control+KI FOVs (not KO)
# See d7_lamp1_otsu_survey.csv — Control median 3552, KI median 4223, pooled median 3765
FIXED_THR = 3765.0

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--masks-dir",
    type=Path,
    default=None,
    help="Write napari-ready label TIFFs here (nuclei/cells/lysosomes per FOV). Skipped if not set.",
)
args = parser.parse_args()

nd2_paths = sorted(DATA_DIR.rglob("*.nd2"))
print(f"Found {len(nd2_paths)} ND2 files", flush=True)
print(f"Fixed lyso_threshold = {FIXED_THR} DN (median Control+KI Otsu on p50-bg image)", flush=True)
if args.masks_dir:
    print(f"Mask export → {args.masks_dir}", flush=True)
print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

df = build_table(nd2_paths, lyso_threshold=FIXED_THR, masks_dir=args.masks_dir)

out_csv = OUT_DIR / "d7_percell_fixedthr.csv"
df.to_csv(out_csv, index=False)
print(f"\nDone: {time.strftime('%H:%M:%S')}", flush=True)
print(f"Rows: {len(df):,}, QC-pass: {df['qc_pass'].sum():,}", flush=True)
print(f"Saved: {out_csv}", flush=True)

print("\n=== condition_stats: lyso_mean_size_um2 (fixed thr) ===", flush=True)
r = condition_stats(df, "lyso_mean_size_um2", timepoint="d7")
wm = r["well_means"].groupby("condition")["lyso_mean_size_um2"].agg(["mean", "median", "std"])
print(wm.to_string(), flush=True)
print(f"KW p={r.get('kruskal_p', 'n/a'):.4f}", flush=True)
for cond, pstats in r.get("vs_reference", {}).items():
    print(f"  {cond} vs Control: raw p={pstats['p_raw']:.4f}, BH p={pstats['p_bh']:.4f}", flush=True)

print("\n=== condition_stats: n_lysosomes (fixed thr) ===", flush=True)
r2 = condition_stats(df, "n_lysosomes", timepoint="d7")
wm2 = r2["well_means"].groupby("condition")["n_lysosomes"].agg(["mean", "median", "std"])
print(wm2.to_string(), flush=True)
print(f"KW p={r2.get('kruskal_p', 'n/a'):.4f}", flush=True)
for cond, pstats in r2.get("vs_reference", {}).items():
    print(f"  {cond} vs Control: raw p={pstats['p_raw']:.4f}, BH p={pstats['p_bh']:.4f}", flush=True)

print("\n=== condition_stats: lamp1_mean (fixed thr) ===", flush=True)
r3 = condition_stats(df, "lamp1_mean", timepoint="d7")
wm3 = r3["well_means"].groupby("condition")["lamp1_mean"].agg(["mean", "median", "std"])
print(wm3.to_string(), flush=True)
print(f"KW p={r3.get('kruskal_p', 'n/a'):.4f}", flush=True)
