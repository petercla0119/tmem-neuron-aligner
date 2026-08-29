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


if __name__ == "__main__":
    main()
