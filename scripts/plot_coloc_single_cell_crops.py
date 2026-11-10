#!/usr/bin/env python
"""Single-cell coloc crops: LAMP1 mask + TMEM puncta colored by colocalization.

The FOV overlays never showed LAMP1 -- but the coloc enrichment is literally
"fraction of TMEM puncta within 0.5 um of a LAMP1 surface". This crops individual
cells and shows exactly that per cell, for BOTH mask definitions:

  background = LAMP1 640 MIP (grayscale)   magenta = LAMP1 lysosome mask (coloc target)
  white outline = cell mask (soma OR soma+neurite)
  YELLOW punctum = TMEM within 0.5 um of a lysosome (colocalized, the numerator)
  CYAN punctum   = TMEM not colocalized

Classification uses the SAME 3D anisotropic EDT the coloc uses (_sample_distances).
Left column = soma cell; right column = same cell extended into neurites.

Run:  PYTHONPATH=src python scripts/plot_coloc_single_cell_crops.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_coloc import (
    ColocConfig,
    _sample_distances,
    detect_tmem_puncta_3d,
    lamp1_mask_3d,
)
from tmem_align.analysis.if_spatial import (
    CH_LAMP1,
    CH_TMEM,
    expand_to_cell_bodies,
    load_fov_3d,
    segment_cell_bodies,
)

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
CFG_PATH = "configs/if_coloc_d7.yaml"
OUT_DIR = Path("reports/if_segmentation_pilot/figures")
MASK_CACHE = Path("reports/if_segmentation_pilot/mask_cache_regen")
NO_CAP = 100_000
LAMP1_LUT = (120, 12000)  # if_spatial DISPLAY_LUTS[CH_LAMP1]
PAD = 8  # px around bbox

# (condition, FOV dir, filename tag) — chosen high-signal + floor examples
CELLS = [
    ("KI", "Z60_PLD_TMEMki", "D20_F4"),   # top-2 KI cells picked automatically
    ("KI", "Z60_PLD_TMEMki", "D20_F4"),
    ("KO", "TMEM_KO", "C8_F1"),           # KO floor example
    ("Control", "Z59_PLD_Control", "C17_F3"),
]


def _fov(cond_dir: str, tag: str) -> Path:
    hits = [p for p in (DATA_ROOT / cond_dir).rglob("*.nd2") if tag in p.name]
    if not hits:
        raise FileNotFoundError(f"{tag} in {cond_dir}")
    return hits[0]


def _cached(key: str, fn, stem: str):
    MASK_CACHE.mkdir(parents=True, exist_ok=True)
    npz = MASK_CACHE / f"{stem}__{key}.npz"
    if npz.exists():
        return np.load(npz)["m"]
    m = fn()
    np.savez_compressed(npz, m=m)
    return m


def _gray(mip, lut):
    lo, hi = lut
    return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)


def _panel(ax, lamp1_mip, lamp1_mask_mip, cell2d, cent, within, y0, y1, x0, x1, title):
    g = _gray(lamp1_mip[y0:y1, x0:x1], LAMP1_LUT)
    rgb = np.stack([g] * 3, axis=-1)
    lm = lamp1_mask_mip[y0:y1, x0:x1]
    rgb[lm] = rgb[lm] * 0.5 + np.array([0.5, 0.0, 0.5])  # magenta lysosomes
    rgb[find_boundaries(cell2d[y0:y1, x0:x1], mode="outer")] = [1, 1, 1]
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    if cent.shape[0]:
        yy, xx = cent[:, 1] - y0, cent[:, 2] - x0
        ax.scatter(xx[~within], yy[~within], s=22, facecolors="none", edgecolors="cyan", linewidths=1.1)
        ax.scatter(xx[within], yy[within], s=22, facecolors="none", edgecolors="yellow", linewidths=1.3)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _cell_puncta(all_cent, cell2d, dt, thr_um):
    """Puncta whose (y,x) fall in cell2d; classify within thr_um of LAMP1 (3D EDT)."""
    if all_cent.shape[0] == 0:
        return np.empty((0, 3)), np.empty(0, bool)
    iy = np.rint(all_cent[:, 1]).astype(int).clip(0, cell2d.shape[0] - 1)
    ix = np.rint(all_cent[:, 2]).astype(int).clip(0, cell2d.shape[1] - 1)
    inside = cell2d[iy, ix]
    cent = all_cent[inside]
    d = _sample_distances(dt, cent)
    return cent, d <= thr_um


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ColocConfig.from_yaml(CFG_PATH)
    fig, axes = plt.subplots(len(CELLS), 2, figsize=(9, 3.2 * len(CELLS)))

    # cache per-FOV heavy computation
    cache: dict[str, dict] = {}
    ki_labels_used: list[int] = []

    for row, (cond, cond_dir, tag) in enumerate(CELLS):
        fov = _fov(cond_dir, tag)
        key = fov.stem
        if key not in cache:
            channels, sampling = load_fov_3d(fov)
            tmem = np.asarray(channels[CH_TMEM], np.float32)
            lamp1 = np.asarray(channels[CH_LAMP1], np.float32)
            map2_mip = np.asarray(channels["488nm"], np.float32).max(axis=0)
            soma = _cached("cells", lambda m=map2_mip: segment_cell_bodies(m), key)
            whole = _cached("wholecell_soma",
                            lambda s=soma, m=map2_mip: expand_to_cell_bodies(s, m, max_distance=NO_CAP), key)
            _, cent = detect_tmem_puncta_3d(tmem, threshold=cfg.tmem_threshold,
                                            min_size_vox=cfg.min_puncta_size_vox)
            lmask = lamp1_mask_3d(lamp1, bg_percentile=cfg.lamp1_bg_percentile,
                                  threshold=cfg.lamp1_threshold)
            dt = ndi.distance_transform_edt(~lmask, sampling=sampling)
            cache[key] = {"soma": soma, "whole": whole, "cent": cent,
                          "lamp1_mip": lamp1.max(axis=0), "lmask_mip": lmask.max(axis=0), "dt": dt}
        cc = cache[key]

        # pick the cell label: for KI rows pick the top-puncta soma cells (distinct)
        soma = cc["soma"]
        labs = np.unique(soma[soma > 0])
        counts = {int(l): int(_cell_puncta(cc["cent"], soma == l, cc["dt"], cfg.threshold_um)[0].shape[0])
                  for l in labs}
        order = sorted(counts, key=counts.get, reverse=True)
        if cond == "KI":
            lab = next(l for l in order if l not in ki_labels_used)
            ki_labels_used.append(lab)
        else:
            lab = order[0] if order else (int(labs[0]) if labs.size else 0)

        for col, (mtype, masks) in enumerate([("soma", soma), ("soma+neurite", cc["whole"])]):
            cell2d = masks == lab
            if not cell2d.any():
                axes[row, col].axis("off"); continue
            ys, xs = np.where(cell2d)
            y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD, cell2d.shape[0])
            x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD, cell2d.shape[1])
            cent, within = _cell_puncta(cc["cent"], cell2d, cc["dt"], cfg.threshold_um)
            n = cent.shape[0]
            frac = within.mean() if n else 0.0
            _panel(axes[row, col], cc["lamp1_mip"], cc["lmask_mip"], cell2d, cent, within,
                   y0, y1, x0, x1,
                   f"{cond} {tag} cell {lab} · {mtype}\n"
                   f"{int(within.sum())}/{n} puncta coloc'd (≤{cfg.threshold_um}µm) = {frac:.0%}")

    fig.suptitle("Single-cell coloc crops — magenta = LAMP1 lysosomes · yellow = colocalized TMEM · cyan = not",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98], pad=0.4, w_pad=0.2, h_pad=0.5)
    out = OUT_DIR / "coloc_single_cell_crops.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
