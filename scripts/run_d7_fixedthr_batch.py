"""Re-run d7 fixed-IF analysis with a fixed LAMP1 threshold — BATCH version.

Loads all 72 ND2s first, runs nuclei + cell-body Cellpose in batch (list of
images -> model.eval), then applies fixed lyso threshold and computes per-cell
features. Batch mode amortizes model loading and runs all images through the
GPU in one pass, much faster than 72 sequential analyze_fov calls.

Fixed threshold: 3765 DN (median Otsu of Control+KI FOVs on p50-bg-subtracted
LAMP1). See scripts/survey_lamp1_otsu.py.

Outputs: reports/if_segmentation_pilot/d7_percell_fixedthr.csv
         reports/if_segmentation_pilot/d7_fixedthr_run.log (stdout)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from tmem_align.analysis.if_features import (
    PX_UM,
    CONDITION_DIRS,
    detect_lysosomes,
    parse_fov_metadata,
    per_cell_features,
    qc_filter_cells,
    condition_stats,
)
from tmem_align.analysis.if_spatial import (
    CH_DAPI, CH_LAMP1, CH_MAP2, CH_TMEM,
    load_fov,
    apply_display_lut,
)

DATA_DIR = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
OUT_DIR = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_THR = 3765.0  # median Control+KI Otsu on p50-bg; see survey_lamp1_otsu.py
_CELLBODY_MODEL = (
    "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821"
    "/hitl_map2_train/models/map2_cellbody_cpsam"
)

nd2_paths = sorted(DATA_DIR.rglob("*.nd2"))
print(f"Found {len(nd2_paths)} ND2 files", flush=True)
print(f"Fixed lyso_threshold = {FIXED_THR} DN (median Control+KI Otsu on p50-bg)", flush=True)
print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

# --- 1. Load all images ---
print("Loading all ND2 images ...", flush=True)
metas, dapis, maps2, lamp1s, tmems = [], [], [], [], []
for p in nd2_paths:
    try:
        ch = load_fov(p)
        metas.append(parse_fov_metadata(p))
        dapis.append(ch[CH_DAPI])
        maps2.append(ch[CH_MAP2])
        lamp1s.append(ch[CH_LAMP1])
        tmems.append(ch[CH_TMEM])
    except Exception as exc:
        print(f"  SKIP load {Path(p).name}: {exc}", flush=True)
        metas.append(None)
        dapis.append(None)
        maps2.append(None)
        lamp1s.append(None)
        tmems.append(None)

valid_idx = [i for i, m in enumerate(metas) if m is not None]
print(f"Loaded {len(valid_idx)} / {len(nd2_paths)} FOVs. {time.strftime('%H:%M:%S')}", flush=True)

# --- 2. Batch nuclei segmentation (cpsam on DAPI) ---
print("Running batch nuclei segmentation (cpsam) ...", flush=True)
from cellpose import models as cp_models

nuc_model = cp_models.CellposeModel(gpu=True, pretrained_model="cpsam")
dapi_list = [dapis[i].astype(np.float32) for i in valid_idx]
nuclei_masks_list, _, _ = nuc_model.eval(dapi_list, diameter=111, channels=[0, 0])
print(f"Nuclei done. {time.strftime('%H:%M:%S')}", flush=True)

# --- 3. Batch cell-body segmentation (HITL cpsam on MAP2) ---
print("Running batch cell-body segmentation (HITL cpsam) ...", flush=True)
cell_model = cp_models.CellposeModel(gpu=True, pretrained_model=_CELLBODY_MODEL)
# HITL model was trained on fixed-LUT uint8 — apply display lut before inference
map2_u8_list = [
    (apply_display_lut(maps2[i], CH_MAP2) * 255).astype(np.uint8) for i in valid_idx
]
cells_masks_list, _, _ = cell_model.eval(map2_u8_list)
print(f"Cell bodies done. {time.strftime('%H:%M:%S')}", flush=True)

# --- 4. Per-FOV: lyso detect + features + QC ---
print("Computing per-cell features ...", flush=True)
frames = []
for ki, i in enumerate(valid_idx):
    try:
        meta = metas[i]
        lamp1 = lamp1s[i]
        nuclei = nuclei_masks_list[ki].astype(np.int32)
        cells = cells_masks_list[ki].astype(np.int32)
        dapi = dapis[i]

        ch_dict = {
            CH_DAPI: dapi,
            CH_LAMP1: lamp1,
            CH_MAP2: maps2[i],
            CH_TMEM: tmems[i],
        }

        lyso_labels, lamp1_corr = detect_lysosomes(lamp1, threshold=FIXED_THR)
        feats = per_cell_features(cells, nuclei, ch_dict, lyso_labels, lamp1_corr)
        qc = qc_filter_cells(cells, nuclei, dapi)
        df_fov = feats.merge(qc, on="cell_id", how="left")
        for mk, mv in meta.items():
            df_fov[mk] = mv
        frames.append(df_fov)
    except Exception as exc:
        print(f"  SKIP features {nd2_paths[i].name}: {exc}", flush=True)

df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

out_csv = OUT_DIR / "d7_percell_fixedthr.csv"
df_all.to_csv(out_csv, index=False)
print(f"\nDone: {time.strftime('%H:%M:%S')}", flush=True)
print(f"Rows: {len(df_all):,}, QC-pass: {df_all['qc_pass'].sum():,}", flush=True)
print(f"Saved: {out_csv}", flush=True)

print("\n=== condition_stats: lyso_mean_size_um2 (fixed thr 3765 DN) ===", flush=True)
r = condition_stats(df_all, "lyso_mean_size_um2", timepoint="d7")
wm = r["well_means"].groupby("condition")["lyso_mean_size_um2"].agg(["mean", "std"])
print(wm.to_string(), flush=True)
print(f"KW p={r.get('kruskal_p', 'n/a'):.4f}", flush=True)
for cond, ps in r.get("vs_reference", {}).items():
    print(f"  {cond} vs Control: raw={ps['p_raw']:.4f} BH={ps['p_bh']:.4f}", flush=True)

print("\n=== condition_stats: n_lysosomes (fixed thr) ===", flush=True)
r2 = condition_stats(df_all, "n_lysosomes", timepoint="d7")
wm2 = r2["well_means"].groupby("condition")["n_lysosomes"].agg(["mean", "std"])
print(wm2.to_string(), flush=True)
print(f"KW p={r2.get('kruskal_p', 'n/a'):.4f}", flush=True)
for cond, ps in r2.get("vs_reference", {}).items():
    print(f"  {cond} vs Control: raw={ps['p_raw']:.4f} BH={ps['p_bh']:.4f}", flush=True)

print("\n=== condition_stats: lamp1_mean (fixed thr) ===", flush=True)
r3 = condition_stats(df_all, "lamp1_mean", timepoint="d7")
wm3 = r3["well_means"].groupby("condition")["lamp1_mean"].agg(["mean", "std"])
print(wm3.to_string(), flush=True)
print(f"KW p={r3.get('kruskal_p', 'n/a'):.4f}", flush=True)

print("\nAll done.", flush=True)
