#!/usr/bin/env python
"""Deconv artifact QC gate (Step 4 of the 2026-09-09 threshold re-derivation).

Detect TMEM (and LAMP1) puncta on the DECONV stack at the recommended per-timepoint
deconv thresholds, then outline the SAME detections on both the raw and deconv MIPs.
A real punctum overlaps a compact bright spot present in raw; an RL ring artifact
is a halo offset from a bright soma with no raw counterpart. Zoom row centers on
the brightest detected punctum, where ringing is worst.

    PYTHONPATH=src python scripts/qc_deconv_puncta_gate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tmem_align.analysis.if_coloc import detect_tmem_puncta_3d  # noqa: E402

from deconv_io import (  # noqa: E402
    DECONV_DIR,
    condition_from_stem,
    deconv_fovs,
    load_deconv_fov_3d,
    raw_for_deconv,
)
from tmem_align.analysis.if_spatial import load_fov_3d  # noqa: E402

OUT = Path("reports/if_segmentation_pilot/deconv_qc_gate")
CYAN = np.array([0.20, 0.73, 0.93])

# recommended deconv thresholds (results note, KO-Otsu for TMEM)
TMEM_DN = {"d7": 4975.0, "d14": 6595.0, "d28": 11905.0}
LAMP1_DN = {"d7": 8795.0, "d14": 13021.0, "d28": 12240.0}
MIN_VOX = 2

# curated FOV set: heavy on d14 Control (runs hot), KI=real-puncta ref, KO=floor check
PLAN = {
    "d14": {"Control": 3, "KI": 2, "KO": 2},
    "d7": {"KI": 1, "KO": 1},
    "d28": {"KI": 1, "KO": 1},
}


def _pick(timepoint: str, counts: dict[str, int]) -> list[Path]:
    out = []
    seen = {c: 0 for c in counts}
    for f in deconv_fovs(timepoint):
        c = condition_from_stem(f.name.replace(".ome.tif", ""))
        if c in counts and seen[c] < counts[c]:
            out.append(f)
            seen[c] += 1
    return out


def _norm(mip: np.ndarray, lo_p=1.0, hi_p=99.9) -> np.ndarray:
    lo, hi = np.percentile(mip, [lo_p, hi_p])
    return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)


def _overlay(gray: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray] * 3, axis=-1)
    b = find_boundaries(mask2d, mode="inner")
    from skimage.morphology import dilation, disk

    b = dilation(b, disk(1))
    rgb[b] = CYAN
    return rgb


def _brightest_centroid(centroids: np.ndarray, tmem_zyx: np.ndarray, shape) -> tuple[int, int]:
    if centroids.shape[0] == 0:
        mip = tmem_zyx.max(axis=0)
        y, x = np.unravel_index(int(np.argmax(mip)), mip.shape)
        return int(y), int(x)
    # pick the punctum whose peak DN is highest
    best, by, bx = -1.0, shape[0] // 2, shape[1] // 2
    mip = tmem_zyx.max(axis=0)
    for _, cy, cx in centroids:
        cy, cx = int(cy), int(cx)
        v = float(mip[cy, cx])
        if v > best:
            best, by, bx = v, cy, cx
    return by, bx


def render(tif: Path, timepoint: str) -> dict:
    stem = tif.name.replace(".ome.tif", "")
    cond = condition_from_stem(stem)
    dec = load_deconv_fov_3d(tif)
    raw_ch, _ = load_fov_3d(raw_for_deconv(tif))

    tmem_dec = dec["561nm"]
    tmem_raw = np.asarray(raw_ch["561nm"], dtype=np.float32)
    dn = TMEM_DN[timepoint]
    labels, centroids = detect_tmem_puncta_3d(tmem_dec, threshold=dn, min_size_vox=MIN_VOX)
    pmask = labels.max(axis=0) > 0
    n = int(centroids.shape[0])

    raw_mip = _norm(tmem_raw.max(axis=0))
    dec_mip = _norm(tmem_dec.max(axis=0))
    y, x = _brightest_centroid(centroids, tmem_dec, dec_mip.shape)
    h = 200
    sl = (slice(max(0, y - h), y + h), slice(max(0, x - h), x + h))

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax[0, 0].imshow(raw_mip, cmap="gray"); ax[0, 0].set_title(f"raw 561 MIP")
    ax[0, 1].imshow(dec_mip, cmap="gray"); ax[0, 1].set_title(f"deconv 561 MIP")
    ax[0, 2].imshow(_overlay(dec_mip, pmask)); ax[0, 2].set_title(f"deconv + puncta (DN={dn:.0f}, n={n})")
    ax[0, 3].imshow(_overlay(raw_mip, pmask)); ax[0, 3].set_title("raw + SAME puncta (real if overlaps)")
    ax[1, 0].imshow(raw_mip[sl], cmap="gray"); ax[1, 0].set_title("raw zoom (brightest)")
    ax[1, 1].imshow(dec_mip[sl], cmap="gray"); ax[1, 1].set_title("deconv zoom")
    ax[1, 2].imshow(_overlay(dec_mip, pmask)[sl[0], sl[1]]); ax[1, 2].set_title("deconv zoom + puncta")
    ax[1, 3].imshow(_overlay(raw_mip, pmask)[sl[0], sl[1]]); ax[1, 3].set_title("raw zoom + puncta")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{timepoint} {cond}  |  {stem}  |  TMEM puncta={n} @ {dn:.0f} DN", fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{timepoint}_{cond}_{stem[-12:]}_tmem.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"file": stem, "tp": timepoint, "cond": cond, "n_puncta": n, "png": png.name}


def main() -> None:
    rows = []
    for tp, counts in PLAN.items():
        for tif in _pick(tp, counts):
            r = render(tif, tp)
            rows.append(r)
            print(f"  {r['tp']} {r['cond']:>7}  n={r['n_puncta']:>3}  {r['png']}", flush=True)
    print(f"\n{len(rows)} QC figures -> {OUT}")


if __name__ == "__main__":
    main()
