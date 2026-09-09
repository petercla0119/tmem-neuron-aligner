#!/usr/bin/env python
"""KI's true enrichment: whole-cell (soma+neurite) masks vs soma-only masks.

The QC (2026-08-31-tmem-detection-mask-qc-report) showed the coloc uses cpsam
SOMA masks, which exclude neurite TMEM signal and undercount KI. This runs the
SAME coloc (analyze_fov_coloc) twice per FOV -- once with soma masks
(segment_cell_bodies) and once with whole-cell masks -- and compares
puncta/cell and enrichment so the undercount is quantified.

Whole-cell mask = expand_to_cell_bodies with the 110px soma cap lifted: the
watershed already floods MAP2 foreground from nucleus seeds; only the distance
cap kept it soma-sized. Lift it -> soma + neurites, partitioned per nucleus.

Run:  PYTHONPATH=src python scripts/coloc_wholecell_vs_soma.py
Env:  this repo's venv ([nd2] + cellpose>=4, cpsam on MPS).
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

from tmem_align.analysis.if_coloc import ColocConfig, analyze_fov_coloc, detect_tmem_puncta_3d
from tmem_align.analysis.if_spatial import (
    CH_TMEM,
    expand_to_cell_bodies,
    load_fov_3d,
    segment_cell_bodies,
)

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
CFG_PATH = "configs/if_coloc_d7.yaml"
OUT_DIR = Path("reports/if_segmentation_pilot/figures")
MASK_CACHE = Path("reports/if_segmentation_pilot/mask_cache_regen")
COND_DIRS = {"KO": "TMEM_KO", "Control": "Z59_PLD_Control", "KI": "Z60_PLD_TMEMki"}
CALIB_DN = 1763.5
TMEM_DISPLAY_LUT = (105, 1800)
N_FOV_PER_COND = 3
NO_CAP = 100_000  # lift the soma-size cap -> whole-cell (soma+neurite)


def _fovs(cond_dir: str, n: int) -> list[Path]:
    return sorted((DATA_ROOT / cond_dir).rglob("*.nd2"))[:n]


def _cached(key: str, fn, stem: str):
    MASK_CACHE.mkdir(parents=True, exist_ok=True)
    npz = MASK_CACHE / f"{stem}__{key}.npz"
    if npz.exists():
        return np.load(npz)["m"]
    m = fn()
    np.savez_compressed(npz, m=m)
    return m


def _masks(nd2_path: Path, map2_mip):
    stem = nd2_path.stem
    soma = _cached("cells", lambda: segment_cell_bodies(map2_mip), stem)
    # Whole-cell = the SAME somata extended into their neurites (soma labels used as
    # watershed seeds, cap lifted). Strict superset of soma -> same cell count, so
    # puncta/cell is a clean neurite-gain metric (nuclei-seeded would change the
    # denominator and confound the comparison).
    whole = _cached("wholecell_soma",
                    lambda: expand_to_cell_bodies(soma, map2_mip, max_distance=NO_CAP), stem)
    return soma, whole


def _agg(rows: list[dict]) -> tuple[float, float, int]:
    """(mean in-cell puncta/cell, mean enrichment over cells w/ puncta, n_cells)."""
    if not rows:
        return 0.0, float("nan"), 0
    ncell = len(rows)
    ppc = float(np.mean([r["n_puncta"] for r in rows]))
    enr = [r["enrichment"] for r in rows if r["n_puncta"] > 0 and np.isfinite(r["enrichment"])]
    return ppc, (float(np.mean(enr)) if enr else float("nan")), ncell


def _gray_mip(stack_zyx):
    lo, hi = TMEM_DISPLAY_LUT
    return np.clip((stack_zyx.max(axis=0) - lo) / (hi - lo + 1e-6), 0, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ColocConfig.from_yaml(CFG_PATH)  # thresholds only; masks passed in directly
    conds = list(COND_DIRS)
    per_cond = {c: {"soma": [], "whole": []} for c in conds}

    fig, axes = plt.subplots(len(conds), N_FOV_PER_COND, figsize=(4 * N_FOV_PER_COND, 4 * len(conds)))
    for r, cond in enumerate(conds):
        for c, fov in enumerate(_fovs(COND_DIRS[cond], N_FOV_PER_COND)):
            channels, sampling = load_fov_3d(fov)
            map2_mip = np.asarray(channels["488nm"], np.float32).max(axis=0)
            soma, whole = _masks(fov, map2_mip)

            rs = analyze_fov_coloc(channels, sampling, soma, cfg, seed=cfg.seed)
            rw = analyze_fov_coloc(channels, sampling, whole, cfg, seed=cfg.seed)
            per_cond[cond]["soma"] += rs
            per_cond[cond]["whole"] += rw

            # overlay: whole-cell outline (green) vs soma outline (cyan) + in-wholecell puncta
            gray = _gray_mip(np.asarray(channels[CH_TMEM], np.float32))
            _, cent = detect_tmem_puncta_3d(np.asarray(channels[CH_TMEM], np.float32),
                                            threshold=CALIB_DN, min_size_vox=cfg.min_puncta_size_vox)
            rgb = np.stack([gray] * 3, axis=-1)
            rgb[find_boundaries(whole, mode="outer")] = [0.1, 0.9, 0.2]  # whole-cell green
            rgb[find_boundaries(soma, mode="outer")] = [0.0, 0.8, 0.8]   # soma cyan
            ax = axes[r, c]
            ax.imshow(rgb, interpolation="nearest")
            if cent.shape[0]:
                iy = np.rint(cent[:, 1]).astype(int).clip(0, whole.shape[0] - 1)
                ix = np.rint(cent[:, 2]).astype(int).clip(0, whole.shape[1] - 1)
                inw = whole[iy, ix] > 0
                ax.scatter(ix[inw], iy[inw], s=8, c="red", marker=".", linewidths=0)
            ppc_s = np.mean([x["n_puncta"] for x in rs]) if rs else 0
            ppc_w = np.mean([x["n_puncta"] for x in rw]) if rw else 0
            ax.set_title(f"{cond} · {fov.name.split('_')[-2]}_{fov.name.split('_')[-1][:-4]}\n"
                         f"puncta/cell soma {ppc_s:.2f} → whole {ppc_w:.2f}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Whole-cell (green, soma+neurite) vs soma-only (cyan) — red = in-whole-cell puncta", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.4, w_pad=0.2, h_pad=0.5)
    out = OUT_DIR / "tmem_wholecell_vs_soma_overlay.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, "\n")

    print(f"{'cond':8s} {'mask':6s} {'cells':>5s} {'puncta/cell':>11s} {'enrichment':>11s}")
    for cond in conds:
        for m in ("soma", "whole"):
            ppc, enr, n = _agg(per_cond[cond][m])
            print(f"{cond:8s} {m:6s} {n:5d} {ppc:11.2f} {enr:11.3f}")
    print("\n(enrichment = mean over cells with >=1 punctum; puncta/cell over all cells)")


if __name__ == "__main__":
    main()
