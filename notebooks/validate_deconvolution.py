"""Richardson-Lucy iteration sweep on one FOV — pick n_iter before the full run.

Deconvolves all 4 channels of one representative d7 KI FOV at n_iter = 5/10/20
and writes before/after max-intensity-projection grids (full-frame + a zoomed
crop) so the sharpening-vs-ringing tradeoff is visible per channel.

Run:  PYTHONPATH=src python notebooks/validate_deconvolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from tmem_align.deconvolve import CANON_ORDER, TARGET, DeconvConfig, compute_psf, deconvolve_stack

FOV = "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7/Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2"
OUT = Path("reports/deconv_sweep")
ITERS = (5, 10, 20)
CROP = (slice(768, 1024), slice(768, 1024))  # 256x256 center-ish crop for detail


def _load_channels(path: str) -> dict[int, np.ndarray]:
    with nd2.ND2File(path) as im:
        axes = list(im.sizes.keys())
        arr = np.asarray(im.asarray())
        c_axis = axes.index("C")
        cmap = {int(c.channel.name[:3]): i for i, c in enumerate(im.metadata.channels)}
    return {nm: np.take(arr, cmap[nm], axis=c_axis) for nm in CANON_ORDER}


def _mip(zyx: np.ndarray) -> np.ndarray:
    return zyx.max(axis=0)


def _show(ax, img, title):
    lo, hi = np.percentile(img, (1, 99.5))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _grid_figure(raw, decon, crop, fname):
    """rows = channels, cols = [raw, iter5, iter10, iter20]."""
    ncols = 1 + len(ITERS)
    fig = plt.figure(figsize=(3 * ncols, 3 * len(CANON_ORDER)))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(CANON_ORDER), ncols), axes_pad=0.15)
    for r, nm in enumerate(CANON_ORDER):
        cells = [(_mip(raw[nm]), "raw")] + [(_mip(decon[nm][it]), f"{it} iter") for it in ITERS]
        for c, (img, label) in enumerate(cells):
            ax = grid[r * ncols + c]
            view = img[crop] if crop else img
            _show(ax, view, f"{nm}nm {TARGET[nm]}\n{label}" if c == 0 else label)
    fig.savefig(OUT / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = _load_channels(FOV)
    decon: dict[int, dict[int, np.ndarray]] = {nm: {} for nm in CANON_ORDER}
    for nm in CANON_ORDER:
        psf = compute_psf(nm, DeconvConfig())
        for it in ITERS:
            decon[nm][it] = deconvolve_stack(raw[nm], psf, n_iter=it)
            print(f"  {nm}nm {TARGET[nm]} @ {it} iter done")

    _grid_figure(raw, decon, None, "sweep_full.png")
    _grid_figure(raw, decon, CROP, "sweep_crop.png")
    print(f"Wrote {OUT}/sweep_full.png and {OUT}/sweep_crop.png")


if __name__ == "__main__":
    main()
