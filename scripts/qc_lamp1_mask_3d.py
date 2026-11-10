#!/usr/bin/env python
"""Visual QC of the 3D LAMP1 lysosome mask (if_coloc.lamp1_mask_3d), raw vs deconv.

lamp1_mask_3d = whole-stack p50-background-subtract then a fixed absolute DN
threshold -> boolean volume the coloc distance transform treats as "lysosome".
The absolute DN is domain-dependent, so this compares the two candidate domains
side by side for one representative FOV per condition (KO / Control / KI, dim->bright):

  RAW    domain, DN = 3765  (median Otsu Control+KI, raw d7, scripts/survey_lamp1_otsu.py)
  DECONV domain, DN = 6515  (median IN-CELL Otsu Control+KI, deconv d7,
                             scripts/survey_lamp1_otsu_deconv.py; raw 3765 superseded)

Per row, 4 panels: raw MIP + green mask boundary + zoom box · deconv MIP + boundary
+ zoom box · raw zoom (green fill) · deconv zoom (green fill). Same zoom window on
both domains (picked on the densest RAW-mask region) so the mask difference is direct.

NB lamp1_mask_3d itself is whole-FOV; the in-cell restriction happens downstream in
analyze_fov_coloc. This QCs the mask the function returns, not the soma-gated subset.

Run (worktree has no venv — use the main tree's):
  PYTHONPATH=src:scripts /Users/pmihack/claire/tmem_2026/tmem-neuron-aligner/.venv/bin/python \
    scripts/qc_lamp1_mask_3d.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

from deconv_io import DECONV_DIR, load_deconv_fov_3d
from tmem_align.io import save_mask_for_napari
from tmem_align.analysis.if_coloc import lamp1_mask_3d
from tmem_align.analysis.if_spatial import CH_LAMP1, load_fov_3d

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
OUT = Path("reports/if_segmentation_pilot/figures/lamp1_mask_3d_qc_raw_vs_deconv.png")
MASKS_DIR = Path("reports/if_segmentation_pilot/lamp1_masks")  # --save-masks target
RAW_LUT = (120, 12000)  # if_spatial.DISPLAY_LUT[CH_LAMP1]
BG_PCT = 50.0
DN_RAW = 3765.0    # survey_lamp1_otsu.py
DN_DECONV = 6515.0  # survey_lamp1_otsu_deconv.py, d7 in-cell
GREEN = np.array([0.13, 0.80, 0.33])  # #22CC55 (never gray on grayscale)
ZOOM = 160  # px

# (condition, ND2 filename tag) — one representative d7 FOV each.
FOVS = [("KO", "C8_F1"), ("Control", "C17_F3"), ("KI", "C20_F2")]


def _raw_fov(tag: str) -> Path:
    hits = [p for p in DATA_ROOT.rglob("*.nd2") if tag in p.name]
    if not hits:
        raise FileNotFoundError(f"raw {tag}")
    return hits[0]


def _deconv_fov(tag: str) -> Path:
    hits = [p for p in DECONV_DIR.glob("*_d7_*.ome.tif") if tag in p.name]
    if not hits:
        raise FileNotFoundError(f"deconv {tag}")
    return hits[0]


def _gray_fixed(mip: np.ndarray, lut: tuple[float, float]) -> np.ndarray:
    lo, hi = lut
    return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)


def _gray_pct(mip: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(mip, [1, 99.5])  # deconv has a different DN domain
    return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)


def _zoom_window(mask_mip: np.ndarray, size: int) -> tuple[int, int]:
    """(y0, x0) of the single densest mask window (box-summed)."""
    d = ndi.uniform_filter(mask_mip.astype(np.float32), size=size)
    cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)
    h, w = mask_mip.shape
    return int(np.clip(cy - size // 2, 0, h - size)), int(np.clip(cx - size // 2, 0, w - size))


def _boundary(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray] * 3, axis=-1)
    rgb[dilation(find_boundaries(mask, mode="inner"), disk(1))] = GREEN
    return np.clip(rgb, 0, 1)


def _fill(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    rgb = np.stack([gray] * 3, axis=-1)
    rgb[mask] = rgb[mask] * (1 - alpha) + GREEN * alpha
    rgb[dilation(find_boundaries(mask, mode="inner"), disk(1))] = GREEN
    return np.clip(rgb, 0, 1)


def _box(ax, y0, x0):
    ax.add_patch(mpatches.Rectangle((x0, y0), ZOOM, ZOOM, fill=False,
                                    edgecolor="yellow", linewidth=1.4))


def _save_masks(cond: str, tag: str, raw, dec, m_raw, m_dec) -> None:
    """Write ZYX LAMP1 stacks + boolean masks as napari-openable TIFFs (io.save_mask_for_napari)."""
    for dom, img, mask in (("raw", raw, m_raw), ("deconv", dec, m_dec)):
        save_mask_for_napari(MASKS_DIR, f"{cond}_{tag}_{dom}", mask, image=img)


def main() -> None:
    save_masks = "--save-masks" in sys.argv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(FOVS), 4, figsize=(4.1 * 4, 4.7 * len(FOVS)))

    for row, (cond, tag) in enumerate(FOVS):
        raw = np.asarray(load_fov_3d(_raw_fov(tag))[0][CH_LAMP1], np.float32)  # RAW, no IC
        dec = np.asarray(load_deconv_fov_3d(_deconv_fov(tag))[CH_LAMP1], np.float32)

        m_raw = lamp1_mask_3d(raw, bg_percentile=BG_PCT, threshold=DN_RAW)
        m_dec = lamp1_mask_3d(dec, bg_percentile=BG_PCT, threshold=DN_DECONV)
        if save_masks:
            _save_masks(cond, tag, raw, dec, m_raw, m_dec)
        raw_mip, dec_mip = raw.max(0), dec.max(0)
        mr_mip, md_mip = m_raw.max(0), m_dec.max(0)
        g_raw, g_dec = _gray_fixed(raw_mip, RAW_LUT), _gray_pct(dec_mip)

        y0, x0 = _zoom_window(mr_mip, ZOOM)  # same window both domains
        sl = (slice(y0, y0 + ZOOM), slice(x0, x0 + ZOOM))

        axes[row, 0].imshow(_boundary(g_raw, mr_mip), interpolation="nearest")
        _box(axes[row, 0], y0, x0)
        axes[row, 0].set_title(f"{cond} {tag} · RAW\nmask vol {m_raw.mean():.2%} · DN={DN_RAW:g}", fontsize=8)

        axes[row, 1].imshow(_boundary(g_dec, md_mip), interpolation="nearest")
        _box(axes[row, 1], y0, x0)
        axes[row, 1].set_title(f"{cond} {tag} · DECONV\nmask vol {m_dec.mean():.2%} · DN={DN_DECONV:g}", fontsize=8)

        axes[row, 2].imshow(_fill(g_raw[sl], mr_mip[sl]), interpolation="nearest")
        axes[row, 2].set_title(f"raw zoom · {mr_mip[sl].mean():.1%} masked", fontsize=8)

        axes[row, 3].imshow(_fill(g_dec[sl], md_mip[sl]), interpolation="nearest")
        axes[row, 3].set_title(f"deconv zoom · {md_mip[sl].mean():.1%} masked", fontsize=8)

        for c in range(4):
            axes[row, c].axis("off")
        print(f"{cond} {tag}: raw vol {m_raw.mean():.3%} / deconv vol {m_dec.mean():.3%}", flush=True)

    fig.suptitle(
        "lamp1_mask_3d QC — green = LAMP1 lysosome mask · RAW (3765 DN) vs DECONV (6515 DN, d7 in-cell)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=0.4, h_pad=2.2, w_pad=0.2)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
