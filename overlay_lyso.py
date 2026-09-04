"""Sanity-check the KO "smaller lysosome" result: per-condition detection overlays.

Picks the d7 FOV whose cells' median lyso size is closest to that condition's median
(a representative field), overlays detected lysosome contours on the fixed-LUT LAMP1,
and prints per-condition Otsu thresholds + puncta-size distributions -- so we can tell
a real size difference from an Otsu-threshold artifact.

  PYTHONPATH=src python overlay_lyso.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

from tmem_align.analysis.if_features import PX_UM, detect_lysosomes  # noqa: E402
from tmem_align.analysis.if_spatial import CH_LAMP1, apply_display_lut, load_fov  # noqa: E402

CSV = "reports/if_segmentation_pilot/d7_percell_full.csv"
OUT = "reports/if_segmentation_pilot/lyso_detection_overlay_by_condition.png"
CONDS = ["Control", "KI", "KO"]
ZOOM = 256


def representative_fov(df: pd.DataFrame, cond: str) -> str:
    """file_path of the FOV whose median lyso size is closest to the condition median."""
    d = df[(df["condition"] == cond) & (df["qc_pass"] == True)]  # noqa: E712
    per_fov = d.groupby("file_path")["lyso_mean_size_um2"].median()
    target = d["lyso_mean_size_um2"].median()
    return (per_fov - target).abs().idxmin()


def zoom_box(corrected: np.ndarray) -> tuple[slice, slice]:
    """Crop window centered on the densest LAMP1 region."""
    from scipy import ndimage as ndi

    dens = ndi.gaussian_filter(corrected, 20)
    cy, cx = np.unravel_index(int(dens.argmax()), dens.shape)
    h, w = corrected.shape
    y0 = min(max(cy - ZOOM // 2, 0), h - ZOOM)
    x0 = min(max(cx - ZOOM // 2, 0), w - ZOOM)
    return slice(y0, y0 + ZOOM), slice(x0, x0 + ZOOM)


def main() -> None:
    df = pd.read_csv(CSV)
    # plain subplots, not ImageGrid: full-FOV and zoom are different pixel sizes and
    # ImageGrid forces one shared extent -> the small crop renders as a thumbnail.
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), gridspec_kw={"width_ratios": [3, 2]})

    print(f"{'cond':8} {'otsu_thr':>9} {'n_puncta':>9} {'median_um2':>11} "
          f"{'p90_um2':>9}  (representative FOV, whole frame)")
    for i, cond in enumerate(CONDS):
        fp = representative_fov(df, cond)
        lamp1 = load_fov(fp)[CH_LAMP1]
        labels, corrected = detect_lysosomes(lamp1)
        thr = float(threshold_otsu(corrected))
        areas = np.bincount(labels.ravel())[1:]
        areas = areas[areas > 0] * PX_UM**2
        med = float(np.median(areas)) if areas.size else 0.0
        p90 = float(np.percentile(areas, 90)) if areas.size else 0.0
        print(f"{cond:8} {thr:9.1f} {areas.size:9d} {med:11.3f} {p90:9.3f}  {fp.split('/')[-1]}")

        disp = apply_display_lut(lamp1, CH_LAMP1)
        zy, zx = zoom_box(corrected)
        for ax, (sy, sx) in ((axes[i, 0], (slice(None), slice(None))), (axes[i, 1], (zy, zx))):
            ax.imshow(disp[sy, sx], cmap="gray", vmin=0, vmax=1, aspect="equal")
            offy, offx = (sy.start or 0), (sx.start or 0)
            for c in find_contours((labels[sy, sx] > 0).astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], "-", color="#ff3b30", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(cond, fontsize=13, fontweight="bold")
        axes[i, 0].set_title(f"{cond} — full FOV (median {med:.2f} µm², p90 {p90:.2f})", fontsize=9)
        axes[i, 1].set_title(f"{cond} — zoom (red = detected)", fontsize=9)

    fig.suptitle("d7 LAMP1 lysosome detection (red = detected puncta)", y=0.99)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT}")


# --- Per-cell QC overlays across the size distribution (KO/Control/KI) -----------
# Reuses the cached per-FOV masks (cells, lyso_labels) from the quantification run so
# we don't re-segment; only raw LAMP1 is loaded (fixed-LUT display). The point: eyeball
# whether "KO lysosomes smaller" is real, or a cell-mask/segmentation artifact — so we
# sample cells spanning small→median→LARGE per genotype (the effect lives in the big-blob
# tail).
FOV_CACHE = (
    "/Users/pmihack/claire/tmem_2026/tmem-if-features/reports/if_segmentation_pilot/fov_cache"
)
N_PER_COND = 24  # ~4x6 grid per genotype
PAD_PX = 40


def _cache_path(nd2_file_path: str) -> str:
    import os

    return os.path.join(FOV_CACHE, os.path.basename(nd2_file_path).replace(".nd2", ".npz"))


def _span_sample(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick n cells evenly across the lyso-size distribution (small→large tail)."""
    d = sub[sub["n_lysosomes"] >= 1].sort_values("lyso_mean_size_um2").reset_index(drop=True)
    if len(d) <= n:
        return d
    idx = np.linspace(0, len(d) - 1, n).round().astype(int)
    return d.iloc[np.unique(idx)].reset_index(drop=True)


def per_cell_qc_overlays() -> None:
    import os

    from skimage.measure import find_contours

    df = pd.read_csv(CSV)
    df = df[df["qc_pass"] == True]  # noqa: E712
    for cond in CONDS:
        sel = _span_sample(df[df["condition"] == cond], N_PER_COND)
        ncols, nrows = 4, int(np.ceil(len(sel) / 4))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        # group by FOV so each ND2 + cache loads once
        loaded: dict[str, tuple] = {}
        for j, (_, r) in enumerate(sel.iterrows()):
            fp = r["file_path"]
            if fp not in loaded:
                cache = np.load(_cache_path(fp))
                lamp1 = load_fov(fp)[CH_LAMP1]
                loaded[fp] = (cache["cells"], cache["lyso_labels"], apply_display_lut(lamp1, CH_LAMP1))
            cells, lyso, disp = loaded[fp]
            cid = int(r["cell_id"])
            ys, xs = np.where(cells == cid)
            if ys.size == 0:
                continue
            y0, y1 = max(ys.min() - PAD_PX, 0), min(ys.max() + PAD_PX, disp.shape[0])
            x0, x1 = max(xs.min() - PAD_PX, 0), min(xs.max() + PAD_PX, disp.shape[1])
            ax = axes[j]
            ax.imshow(disp[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, aspect="equal")
            # MAP2 cell-body mask outline (cyan)
            for c in find_contours((cells[y0:y1, x0:x1] == cid).astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], "-", color="#00e5ff", lw=1.0)
            # LAMP1 lysosome contours within this cell (red)
            cell_lyso = np.where(cells[y0:y1, x0:x1] == cid, lyso[y0:y1, x0:x1], 0)
            for c in find_contours((cell_lyso > 0).astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], "-", color="#ff3b30", lw=0.7)
            ax.set_title(
                f"{r['well']}_F{int(r['fov'])} c{cid}\n"
                f"{r['lyso_mean_size_um2']:.2f} µm²  (n={int(r['n_lysosomes'])})",
                fontsize=8,
            )
        med = df[df["condition"] == cond]["lyso_mean_size_um2"].median()
        fig.suptitle(
            f"d7 {cond} — per-cell QC (cyan=MAP2 body, red=LAMP1 lyso); "
            f"cells span size distribution (cond median {med:.2f} µm²)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = f"reports/if_segmentation_pilot/d7_qc_overlays_{cond.lower()}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  ({len(sel)} cells, {len(loaded)} FOVs)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "percell":
        per_cell_qc_overlays()
    else:
        main()
