"""Compare IC-corrected (no deconv) vs IC+RL deconvolved z-slices.

One figure per channel: rows=[IC | IC+RL n=<n_iter>], cols=evenly-spaced z-slices.
Shared per-channel LUT (both stacks) so sharpening is visible.

Output: reports/deconv_comparison/zslice_<nm>nm_<name>_n<iter>.png  (4 files)

Usage:
    # n=10 from precomputed batch output:
    PYTHONPATH=src python scripts/compare_ic_vs_deconv_zslices.py
    # n=5 deconvolved on the fly (needs preprocess worktree src):
    PYTHONPATH=.claude/worktrees/preprocess/src python scripts/compare_ic_vs_deconv_zslices.py --n-iter 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
import tifffile
from mpl_toolkits.axes_grid1 import ImageGrid

IC_FIELD_FLOOR = 0.1
IC_NPZ = "/Users/pmihack/claire/tmem_2026/data/ic_fields_260821_pooled.npz"
ND2_PATH = (
    "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7"
    "/Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2"
)
DECONV_PATH = (
    "/Users/pmihack/claire/tmem_2026/data/deconvolved_260821"
    "/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.ome.tif"
)

CANON_ORDER = [488, 640, 561, 405]
CH_STR = {488: "488nm", 640: "640nm", 561: "561nm", 405: "405nm"}
CH_NAME = {488: "MAP2", 640: "LAMP1", 561: "cl-TMEM", 405: "DAPI"}
N_SLICES = 5
OUT = Path("reports/deconv_comparison")


def load_ic_fields(npz_path: str) -> dict[int, tuple[np.ndarray, float]]:
    data = np.load(npz_path)
    return {
        nm: (data[CH_STR[nm]].astype(np.float32), float(data.get(f"{CH_STR[nm]}_darkfield", 0.0)))
        for nm in CANON_ORDER
        if CH_STR[nm] in data.files
    }


def apply_ic(zyx: np.ndarray, field: np.ndarray, dark: float) -> np.ndarray:
    flat = np.clip(field, IC_FIELD_FLOOR, None)
    corrected = np.clip(zyx.astype(np.float32) - dark, 0.0, None) / flat[np.newaxis]
    return np.clip(np.rint(corrected), 0, 65535).astype(np.uint16)


def load_nd2_by_channel(path: str) -> dict[int, np.ndarray]:
    """Load ND2 → {nm: ZYX uint16}."""
    with nd2.ND2File(path) as f:
        axes = list(f.sizes.keys())
        arr = np.asarray(f.asarray())
        c_axis = axes.index("C")
        cmap = {int(c.channel.name[:3]): i for i, c in enumerate(f.metadata.channels)}
    return {nm: np.take(arr, cmap[nm], axis=c_axis) for nm in CANON_ORDER if nm in cmap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()
    n_iter = args.n_iter

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading IC fields...", flush=True)
    ic_fields = load_ic_fields(IC_NPZ)

    print("Loading ND2 + applying IC...", flush=True)
    raw_ch = load_nd2_by_channel(ND2_PATH)
    ic_ch = {nm: apply_ic(raw_ch[nm], *ic_fields[nm]) for nm in CANON_ORDER}

    if n_iter == 10:
        print("Loading precomputed deconvolved OME-TIFF (n=10)...", flush=True)
        deconv = tifffile.imread(DECONV_PATH)  # CZYX, CANON_ORDER
        deconv_ch = {nm: deconv[i] for i, nm in enumerate(CANON_ORDER)}
    else:
        from tmem_align.deconvolve import DeconvConfig, compute_psf, deconvolve_stack
        print(f"Deconvolving IC-corrected channels at n_iter={n_iter}...", flush=True)
        cfg = DeconvConfig()
        deconv_ch = {}
        for nm in CANON_ORDER:
            psf = compute_psf(nm, cfg)
            deconv_ch[nm] = deconvolve_stack(ic_ch[nm], psf, n_iter=n_iter)
            print(f"  {nm}nm done", flush=True)

    n_z = min(ic_ch[488].shape[0], deconv_ch[488].shape[0])
    z_indices = np.linspace(0, n_z - 1, N_SLICES, dtype=int)
    print(f"Z-slices: {z_indices.tolist()} (of {n_z})", flush=True)

    for nm in CANON_ORDER:
        name = CH_NAME[nm]
        ic = ic_ch[nm]
        dc = deconv_ch[nm]

        # Shared LUT from both stacks so neither is clipped relative to the other
        both = np.concatenate([ic[z_indices].ravel(), dc[z_indices].ravel()])
        lo, hi = np.percentile(both, (1, 99.5))
        hi = max(hi, lo + 1)

        fig = plt.figure(figsize=(2.5 * N_SLICES, 5))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, N_SLICES), axes_pad=0.08)

        for col, z in enumerate(z_indices):
            for row, (stack, label) in enumerate([(ic, "IC (no deconv)"), (dc, f"IC + RL n={n_iter}")]):
                ax = grid[row * N_SLICES + col]
                ax.imshow(stack[z], cmap="gray", vmin=lo, vmax=hi)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(label, fontsize=8)
                if row == 0:
                    ax.set_title(f"z={z}", fontsize=8)

        fig.suptitle(f"{nm} nm  {name}  — IC vs IC+RL n={n_iter}  (d7 C20_F1)", fontsize=10)
        fname = OUT / f"zslice_{nm}nm_{name.lower().replace('-', '')}_n{n_iter}.png"
        fig.savefig(fname, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {fname}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
