#!/usr/bin/env python
"""Within-cell TMEM detection overlay — the exact input to the coloc.

The whole-FOV overlay (plot_tmem_detection_overlay.py) counts every punctum,
including extracellular debris. The coloc only keeps puncta whose (y,x) centroid
falls inside a MAP2 cell-body mask (see analyze_fov_coloc). This renders that
restriction so the KO floor can be judged apples-to-apples:

  cyan outline = MAP2 cell body   red = punctum INSIDE a cell (enters coloc)
  blue = punctum OUTSIDE cells (excluded)

If KO stays populated in RED inside cells, the per-cell coloc floor is suspect.

Run:  PYTHONPATH=src python scripts/plot_tmem_within_cell_overlay.py
Env:  this repo's venv with [nd2] + cellpose>=4 (cpsam model on MPS/CPU).
Also caches masks to reports/if_segmentation_pilot/mask_cache_regen/<stem>.npz
(key 'cells') — rebuilds the fov_cache the coloc config wants.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_coloc import detect_tmem_puncta_3d
from tmem_align.analysis.if_spatial import (
    CH_TMEM,
    load_fov_3d,
    segment_cell_bodies,
)

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
OUT_DIR = Path("reports/if_segmentation_pilot/figures")
MASK_CACHE = Path("reports/if_segmentation_pilot/mask_cache_regen")
COND_DIRS = {"KO": "TMEM_KO", "Control": "Z59_PLD_Control", "KI": "Z60_PLD_TMEMki"}

CALIB_DN = 1763.5
MIN_SIZE_VOX = 2
TMEM_DISPLAY_LUT = (105, 1800)
N_FOV_PER_COND = 3
CH_MAP2 = "488nm"


def _fovs(cond_dir: str, n: int) -> list[Path]:
    return sorted((DATA_ROOT / cond_dir).rglob("*.nd2"))[:n]


def _cached_cells(nd2_path: Path, map2_yx: np.ndarray) -> np.ndarray:
    MASK_CACHE.mkdir(parents=True, exist_ok=True)
    npz = MASK_CACHE / (nd2_path.stem + ".npz")
    if npz.exists():
        return np.load(npz)["cells"]
    cells = segment_cell_bodies(map2_yx)  # gpu=True -> MPS if available
    np.savez_compressed(npz, cells=cells)
    return cells


def _gray_mip(stack_zyx: np.ndarray) -> np.ndarray:
    mip = stack_zyx.max(axis=0)
    lo, hi = TMEM_DISPLAY_LUT
    return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)


def _analyze(nd2_path: Path):
    channels, _ = load_fov_3d(nd2_path)
    tmem = np.asarray(channels[CH_TMEM], dtype=np.float32)  # ZYX
    map2_mip = np.asarray(channels[CH_MAP2], dtype=np.float32).max(axis=0)
    cells = _cached_cells(nd2_path, map2_mip)  # YX labels

    _, centroids = detect_tmem_puncta_3d(tmem, threshold=CALIB_DN, min_size_vox=MIN_SIZE_VOX)
    if centroids.shape[0]:
        iy = np.rint(centroids[:, 1]).astype(int).clip(0, cells.shape[0] - 1)
        ix = np.rint(centroids[:, 2]).astype(int).clip(0, cells.shape[1] - 1)
        inside = cells[iy, ix] > 0
    else:
        iy = ix = np.empty(0, dtype=int)
        inside = np.empty(0, dtype=bool)

    return {
        "gray": _gray_mip(tmem),
        "cells": cells,
        "iy": iy, "ix": ix, "inside": inside,
        "n_total": int(centroids.shape[0]),
        "n_within": int(inside.sum()),
        "n_cells": int(np.unique(cells[cells > 0]).size),
    }


def _render(ax, a, title):
    rgb = np.stack([a["gray"]] * 3, axis=-1)
    bnd = find_boundaries(a["cells"], mode="outer")
    rgb[bnd] = [0.0, 0.8, 0.8]  # cyan cell outlines
    ax.imshow(rgb, interpolation="nearest")
    if a["iy"].size:
        out = ~a["inside"]
        ax.scatter(a["ix"][out], a["iy"][out], s=6, c="#3366ff", marker=".", linewidths=0)
        ax.scatter(a["ix"][a["inside"]], a["iy"][a["inside"]], s=8, c="red", marker=".", linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conds = list(COND_DIRS)
    fig, axes = plt.subplots(len(conds), N_FOV_PER_COND, figsize=(4 * N_FOV_PER_COND, 4 * len(conds)))
    summary = []
    for r, cond in enumerate(conds):
        for c, fov in enumerate(_fovs(COND_DIRS[cond], N_FOV_PER_COND)):
            a = _analyze(fov)
            tag = f"{fov.name.split('_')[-2]}_{fov.name.split('_')[-1][:-4]}"
            per_cell = a["n_within"] / a["n_cells"] if a["n_cells"] else 0.0
            _render(axes[r, c], a,
                    f"{cond} · {tag}  ({a['n_cells']} cells)\n"
                    f"in-cell {a['n_within']}/{a['n_total']} puncta  ·  {per_cell:.2f}/cell")
            summary.append((cond, tag, a["n_cells"], a["n_within"], a["n_total"], per_cell))
            print(f"{cond:8s} {tag:8s} cells={a['n_cells']:3d} "
                  f"in-cell={a['n_within']:4d}/{a['n_total']:4d}  {per_cell:.2f}/cell")
    fig.suptitle(f"Within-cell cleaved-TMEM detection @ {CALIB_DN:g} DN (coloc input)\n"
                 "red = in-cell punctum (counts) · blue = excluded · cyan = MAP2 cell body", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.4, w_pad=0.2, h_pad=0.5)
    out = OUT_DIR / "tmem_within_cell_overlay_by_condition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("\nwrote", out)
    # per-condition mean in-cell puncta/cell — the number the coloc floor rests on
    for cond in conds:
        rows = [s for s in summary if s[0] == cond]
        pc = np.mean([s[5] for s in rows]) if rows else 0.0
        print(f"  {cond:8s} mean in-cell puncta/cell = {pc:.2f}")


if __name__ == "__main__":
    main()
