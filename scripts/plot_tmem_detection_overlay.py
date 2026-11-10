#!/usr/bin/env python
"""Show what the 561 (cleaved-TMEM) detection threshold actually picks up.

The coloc enrichment number is only trustworthy if the TMEM puncta mask is real
signal, not noise. This renders the SAME detection the coloc uses
(`detect_tmem_puncta_3d` on the 3D stack) as a red tint over the 561 MIP, per
condition, plus a KO-vs-KI threshold sweep so the calibrated floor is visible.

Run:  PYTHONPATH=src python scripts/plot_tmem_detection_overlay.py
Env:  this repo's own venv (pip install -e ".[nd2]"); no mask cache needed --
      detection is stack-wide, independent of cell segmentation.

# ponytail: the one knob that matters is CALIB_DN below (the calibrated 1763.5
# DN from survey_tmem_dn.py). SWEEP_DN shows why it was chosen. Re-tune if the
# dataset/laser changes.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from tmem_align.analysis.if_coloc import detect_tmem_puncta_3d
from tmem_align.analysis.if_spatial import CH_TMEM

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
OUT_DIR = Path("reports/if_segmentation_pilot/figures")
COND_DIRS = {"KO": "TMEM_KO", "Control": "Z59_PLD_Control", "KI": "Z60_PLD_TMEMki"}

CALIB_DN = 1763.5     # calibrated absolute DN (median KO per-FOV intra-cell p99.99)
SWEEP_DN = [446.0, 1763.5]  # 446 did NOT floor KO; 1763.5 does
MIN_SIZE_VOX = 2
TMEM_DISPLAY_LUT = (105, 1800)  # matches if_spatial DISPLAY_LUTS[CH_TMEM]
N_FOV_PER_COND = 3


def _fovs(cond_dir: str, n: int) -> list[Path]:
    return sorted((DATA_ROOT / cond_dir).rglob("*.nd2"))[:n]


def _load_tmem_stack(nd2_path: Path) -> np.ndarray:
    from tmem_align.analysis.if_spatial import load_fov_3d

    channels, _ = load_fov_3d(nd2_path)
    return np.asarray(channels[CH_TMEM], dtype=np.float32)  # ZYX


def _gray_mip(stack_zyx: np.ndarray) -> np.ndarray:
    mip = stack_zyx.max(axis=0)
    lo, hi = TMEM_DISPLAY_LUT
    return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)


def _overlay(gray: np.ndarray, mask_mip: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[mask_mip] = rgb[mask_mip] * 0.25 + np.array([0.75, 0.0, 0.0])
    return np.clip(rgb, 0, 1)


def _detect_mask_mip(stack_zyx: np.ndarray, dn: float) -> tuple[np.ndarray, int]:
    labels, centroids = detect_tmem_puncta_3d(
        stack_zyx, threshold=dn, min_size_vox=MIN_SIZE_VOX
    )
    return (labels > 0).max(axis=0), int(centroids.shape[0])


def fig_by_condition() -> None:
    conds = list(COND_DIRS)
    fig, axes = plt.subplots(len(conds), N_FOV_PER_COND, figsize=(4 * N_FOV_PER_COND, 4 * len(conds)))
    for r, cond in enumerate(conds):
        for c, fov in enumerate(_fovs(COND_DIRS[cond], N_FOV_PER_COND)):
            stack = _load_tmem_stack(fov)
            gray = _gray_mip(stack)
            mask_mip, n = _detect_mask_mip(stack, CALIB_DN)
            ax = axes[r, c]
            ax.imshow(_overlay(gray, mask_mip), interpolation="nearest")
            ax.set_title(f"{cond}  ·  {fov.name.split('_')[-2]}_{fov.name.split('_')[-1][:-4]}\n"
                         f"{n} puncta @ {CALIB_DN:g} DN", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Cleaved-TMEM (561) detection @ {CALIB_DN:g} DN — red = puncta going into coloc\n"
                 "KO should floor (few/no puncta); KI should carry real signal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.4, w_pad=0.2, h_pad=0.5)
    out = OUT_DIR / "tmem_detection_overlay_by_condition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def fig_threshold_sweep() -> None:
    # one KO + one KI FOV across the sweep, to show the calibrated floor
    picks = [("KO", _fovs(COND_DIRS["KO"], 1)[0]), ("KI", _fovs(COND_DIRS["KI"], 1)[0])]
    fig, axes = plt.subplots(len(picks), len(SWEEP_DN), figsize=(4 * len(SWEEP_DN), 4 * len(picks)))
    for r, (cond, fov) in enumerate(picks):
        stack = _load_tmem_stack(fov)
        gray = _gray_mip(stack)
        for c, dn in enumerate(SWEEP_DN):
            mask_mip, n = _detect_mask_mip(stack, dn)
            ax = axes[r, c]
            ax.imshow(_overlay(gray, mask_mip), interpolation="nearest")
            flag = "  ← calibrated" if dn == CALIB_DN else ""
            ax.set_title(f"{cond} @ {dn:g} DN{flag}\n{n} puncta", fontsize=9)
            ax.axis("off")
    fig.suptitle("Threshold sweep: 446 DN (does NOT floor KO) vs 1763.5 DN (calibrated floor)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.4, w_pad=0.2, h_pad=0.5)
    out = OUT_DIR / "tmem_detection_threshold_sweep.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_by_condition()
    fig_threshold_sweep()


if __name__ == "__main__":
    main()
