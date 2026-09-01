"""Richardson-Lucy iteration sweep on IC-corrected images — compare to raw deconv.

Applies flat-field correction (dark + flat, sigma=102px, pooled d7+d14+d28)
to the same d7 KI FOV used in the raw sweep, then deconvolves at n_iter=5/10/20.

Outputs to reports/deconv_sweep/:
  sweep_ic_full.png          — full-frame, all channels × [raw, 5, 10, 20 iter]
  sweep_ic_crop.png          — zoomed crop, same layout
  sweep_ic_vs_raw_n10.png    — side-by-side: raw-deconv vs IC-deconv at n_iter=10

Run:  PYTHONPATH=src python notebooks/validate_deconvolution_ic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from tmem_align.deconvolve import CANON_ORDER, TARGET, DeconvConfig, compute_psf, deconvolve_stack

FOV = "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7/Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2"
IC_NPZ = "/Users/pmihack/claire/tmem_2026/data/ic_fields_260821_pooled.npz"
OUT = Path("reports/deconv_sweep")
ITERS = (5, 10, 20)
CROP = (slice(768, 1024), slice(768, 1024))

# Map integer channel key → string npz key
_CH_STR = {488: "488nm", 640: "640nm", 561: "561nm", 405: "405nm"}
IC_FIELD_FLOOR = 0.1


def _load_ic_fields(npz_path: str) -> dict[int, tuple[np.ndarray, float]]:
    """Load pooled IC fields; return {int_nm: (field_yx, darkfield_scalar)}."""
    data = np.load(npz_path)
    result = {}
    for nm in CANON_ORDER:
        key = _CH_STR[nm]
        if key not in data.files:
            continue
        dark_key = f"{key}_darkfield"
        dark = float(data[dark_key]) if dark_key in data.files else 0.0
        result[nm] = (data[key].astype(np.float32), dark)
    return result


def _apply_ic(zyx: np.ndarray, field: np.ndarray, dark: float) -> np.ndarray:
    """Apply dark-subtract + flat-divide to a ZYX uint16 stack. Returns uint16."""
    flat = np.clip(field, IC_FIELD_FLOOR, None)
    corrected = np.clip(zyx.astype(np.float32) - dark, 0.0, None) / flat[np.newaxis]
    return np.clip(np.rint(corrected), 0, 65535).astype(np.uint16)


def _load_channels_raw(path: str) -> dict[int, np.ndarray]:
    with nd2.ND2File(path) as im:
        axes = list(im.sizes.keys())
        arr = np.asarray(im.asarray())
        c_axis = axes.index("C")
        cmap = {int(c.channel.name[:3]): i for i, c in enumerate(im.metadata.channels)}
    return {nm: np.take(arr, cmap[nm], axis=c_axis) for nm in CANON_ORDER}


def _load_channels_ic(path: str, ic_fields: dict[int, tuple[np.ndarray, float]]) -> dict[int, np.ndarray]:
    """Load channels and apply IC before returning."""
    raw = _load_channels_raw(path)
    return {
        nm: _apply_ic(zyx, ic_fields[nm][0], ic_fields[nm][1]) if nm in ic_fields else zyx
        for nm, zyx in raw.items()
    }


def _mip(zyx: np.ndarray) -> np.ndarray:
    return zyx.max(axis=0)


def _show(ax, img, title):
    lo, hi = np.percentile(img, (1, 99.5))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _grid_figure(channels, decon, crop, fname, prefix="IC"):
    """rows = channels, cols = [IC-raw, iter5, iter10, iter20]."""
    ncols = 1 + len(ITERS)
    fig = plt.figure(figsize=(3 * ncols, 3 * len(CANON_ORDER)))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(CANON_ORDER), ncols), axes_pad=0.15)
    for r, nm in enumerate(CANON_ORDER):
        cells = [(_mip(channels[nm]), f"{prefix} raw")] + [
            (_mip(decon[nm][it]), f"{it} iter") for it in ITERS
        ]
        for c, (img, label) in enumerate(cells):
            ax = grid[r * ncols + c]
            view = img[crop] if crop else img
            _show(ax, view, f"{nm}nm {TARGET[nm]}\n{label}" if c == 0 else label)
    fig.savefig(OUT / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / fname}", flush=True)


def _comparison_figure(raw_channels, ic_channels, raw_decon, ic_decon, n_iter, crop, fname):
    """4 columns: raw-input, raw-decon, IC-input, IC-decon. Rows = channels."""
    ncols = 4
    fig = plt.figure(figsize=(3 * ncols, 3 * len(CANON_ORDER)))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(CANON_ORDER), ncols), axes_pad=0.15)
    col_labels = [f"Raw input", f"Raw →RL {n_iter}iter", f"IC-corrected input", f"IC →RL {n_iter}iter"]
    for r, nm in enumerate(CANON_ORDER):
        cells = [
            _mip(raw_channels[nm])[crop],
            _mip(raw_decon[nm][n_iter])[crop],
            _mip(ic_channels[nm])[crop],
            _mip(ic_decon[nm][n_iter])[crop],
        ]
        for c, img in enumerate(cells):
            ax = grid[r * ncols + c]
            lo, hi = np.percentile(cells[0], (1, 99.5))  # same LUT as raw input for fair compare
            ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            title = f"{nm}nm {TARGET[nm]}\n{col_labels[c]}" if r == 0 else col_labels[c]
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Raw vs IC-corrected deconvolution (n_iter={n_iter})", fontsize=11)
    fig.savefig(OUT / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / fname}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading IC fields...", flush=True)
    ic_fields = _load_ic_fields(IC_NPZ)
    for nm, (field, dark) in ic_fields.items():
        print(f"  {nm}nm: field shape={field.shape} dark={dark:.1f} ADU", flush=True)

    print(f"\nLoading raw FOV: {Path(FOV).name}", flush=True)
    raw_channels = _load_channels_raw(FOV)
    ic_channels = _load_channels_ic(FOV, ic_fields)

    for nm in CANON_ORDER:
        raw_mean = raw_channels[nm].mean()
        ic_mean = ic_channels[nm].mean()
        print(f"  {nm}nm: raw mean={raw_mean:.0f}  IC mean={ic_mean:.0f}  Δ={(ic_mean-raw_mean)/raw_mean*100:+.1f}%", flush=True)

    print("\nDeconvolving IC-corrected channels...", flush=True)
    ic_decon: dict[int, dict[int, np.ndarray]] = {nm: {} for nm in CANON_ORDER}
    for nm in CANON_ORDER:
        psf = compute_psf(nm, DeconvConfig())
        for it in ITERS:
            ic_decon[nm][it] = deconvolve_stack(ic_channels[nm], psf, n_iter=it)
            print(f"  {nm}nm @ {it} iter done", flush=True)

    # Also load the pre-existing raw deconv results if available, else re-run
    raw_sweep_exists = (OUT / "sweep_full.png").exists()
    if not raw_sweep_exists:
        print("\nRaw deconv results not found — re-running raw sweep for comparison...", flush=True)
        raw_decon: dict[int, dict[int, np.ndarray]] = {nm: {} for nm in CANON_ORDER}
        for nm in CANON_ORDER:
            psf = compute_psf(nm, DeconvConfig())
            for it in ITERS:
                raw_decon[nm][it] = deconvolve_stack(raw_channels[nm], psf, n_iter=it)
                print(f"  raw {nm}nm @ {it} iter done", flush=True)
    else:
        print("\nRaw deconv sweep figures already exist — re-running raw deconv for comparison figure only...", flush=True)
        raw_decon = {nm: {} for nm in CANON_ORDER}
        for nm in CANON_ORDER:
            psf = compute_psf(nm, DeconvConfig())
            raw_decon[nm][10] = deconvolve_stack(raw_channels[nm], psf, n_iter=10)
            print(f"  raw {nm}nm @ 10 iter done", flush=True)

    print("\nGenerating figures...", flush=True)
    _grid_figure(ic_channels, ic_decon, None, "sweep_ic_full.png")
    _grid_figure(ic_channels, ic_decon, CROP, "sweep_ic_crop.png")
    _comparison_figure(raw_channels, ic_channels, raw_decon, ic_decon, 10, CROP, "sweep_ic_vs_raw_n10.png")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
