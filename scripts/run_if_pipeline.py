"""Run the fixed-IF LAMP1 quantification pipeline across all timepoints.

Produces one per-cell CSV per timepoint plus an all-timepoints combined CSV.
Optionally writes napari-ready label TIFFs (nuclei / cells / lysosomes) per FOV.

LAMP1 thresholds
----------------
d7:       3765 DN fixed (calibrated against KO null — median Control+KI Otsu on
          p50-bg-subtracted LAMP1; see reports/if_segmentation_pilot/d7_lamp1_otsu_survey.csv)
d14/d28:  per-FOV Otsu (default) — cross-timepoint fixed thresholds not yet
          surveyed. Survey first with scripts/survey_lamp1_otsu.py, then pass
          --lamp1-thr-d14 / --lamp1-thr-d28 to lock them in.

Usage
-----
# Feature extraction only (no masks):
python scripts/run_if_pipeline.py

# With mask export (recommended — needed for napari QC):
python scripts/run_if_pipeline.py --masks-dir data/masks/cleaved_tmem_pld3_260821

# Single timepoint:
python scripts/run_if_pipeline.py --timepoints d7 --masks-dir data/masks/cleaved_tmem_pld3_260821

Mask layout (napari Labels-ready int32 TIFFs):
    {masks_dir}/{timepoint}/{condition}/{nd2_stem}__{nuclei|cells|lysosomes}.tif

Open in napari:
    File → Open Files → select a set of .tif masks → set layer type to "Labels"
    Or use if_features.view_fov(nd2_path, masks_dir) for a single FOV.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmem_align.analysis.if_features import build_table, condition_stats

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
OUT_DIR = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"

# d7 fixed threshold calibrated against KO null (survey_lamp1_otsu.py, 2026-08-30)
# d14/d28: not yet surveyed — use per-FOV Otsu until survey is run
_LAMP1_THR: dict[str, float | None] = {
    "d7": 3765.0,
    "d14": None,  # per-FOV Otsu
    "d28": None,  # per-FOV Otsu
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--timepoints",
    nargs="+",
    choices=["d7", "d14", "d28"],
    default=["d7", "d14", "d28"],
    metavar="TP",
    help="Timepoints to run (default: all three)",
)
parser.add_argument(
    "--masks-dir",
    type=Path,
    default=None,
    help=(
        "Write napari-ready label TIFFs here (nuclei/cells/lysosomes per FOV). "
        "Layout: {masks_dir}/{timepoint}/{condition}/{stem}__{kind}.tif. "
        "Skipped if not set."
    ),
)
parser.add_argument(
    "--lamp1-thr-d14",
    type=float,
    default=None,
    metavar="DN",
    help="Fixed LAMP1 threshold (DN) for d14. Defaults to per-FOV Otsu.",
)
parser.add_argument(
    "--lamp1-thr-d28",
    type=float,
    default=None,
    metavar="DN",
    help="Fixed LAMP1 threshold (DN) for d28. Defaults to per-FOV Otsu.",
)
args = parser.parse_args()

if args.lamp1_thr_d14 is not None:
    _LAMP1_THR["d14"] = args.lamp1_thr_d14
if args.lamp1_thr_d28 is not None:
    _LAMP1_THR["d28"] = args.lamp1_thr_d28

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
all_frames = []
t0 = time.time()

for tp in args.timepoints:
    data_dir = DATA_ROOT / tp
    if not data_dir.exists():
        print(f"SKIP {tp}: {data_dir} not found", flush=True)
        continue

    nd2_paths = sorted(data_dir.rglob("*.nd2"))
    lyso_thr = _LAMP1_THR[tp]
    thr_label = f"{lyso_thr:.0f} DN (fixed)" if lyso_thr is not None else "per-FOV Otsu"

    print(f"\n{'='*60}", flush=True)
    print(f"{tp}: {len(nd2_paths)} FOVs  |  LAMP1 threshold = {thr_label}", flush=True)
    if args.masks_dir:
        tp_masks = args.masks_dir / tp
        print(f"     Masks → {tp_masks}", flush=True)
    else:
        tp_masks = None
    print(f"     Start: {time.strftime('%H:%M:%S')}", flush=True)

    df = build_table(
        nd2_paths,
        lyso_threshold=lyso_thr,
        masks_dir=tp_masks,
    )

    out_csv = OUT_DIR / f"{tp}_percell_pipeline.csv"
    df.to_csv(out_csv, index=False)
    elapsed = time.time() - t0
    print(f"     Done:  {time.strftime('%H:%M:%S')}  ({elapsed/60:.1f} min total)", flush=True)
    print(f"     Rows:  {len(df):,}  QC-pass: {df['qc_pass'].sum():,}", flush=True)
    print(f"     CSV:   {out_csv}", flush=True)
    all_frames.append(df)

# Combined CSV
if all_frames:
    import pandas as pd
    combined = pd.concat(all_frames, ignore_index=True)
    combined_csv = OUT_DIR / "all_percell_pipeline.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"\nCombined: {len(combined):,} rows → {combined_csv}", flush=True)

# ---------------------------------------------------------------------------
# Quick summary stats (d7 only — fixed threshold, validated finding)
# ---------------------------------------------------------------------------
d7_frames = [f for f in all_frames if "d7" in f["timepoint"].values]
if d7_frames:
    import pandas as pd
    df7 = pd.concat(d7_frames, ignore_index=True)
    print("\n=== d7 condition_stats: lyso_mean_size_um2 ===", flush=True)
    r = condition_stats(df7, "lyso_mean_size_um2", timepoint="d7")
    wm = r["well_means"].groupby("condition")["lyso_mean_size_um2"].agg(["mean", "median", "std"])
    print(wm.to_string(), flush=True)
    print(f"KW p={r.get('kruskal_p', 'n/a'):.4f}", flush=True)
    for cond, ps in r.get("vs_reference", {}).items():
        print(f"  {cond} vs Control: raw p={ps['p_raw']:.4f}  BH p={ps['p_bh']:.4f}", flush=True)

print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min", flush=True)
