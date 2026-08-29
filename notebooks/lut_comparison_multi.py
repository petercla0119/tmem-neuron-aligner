"""LUT comparison figure across multiple FOVs.

Layout: 2 rows per FOV (per-image percentile top, fixed LUT bottom), 4 channel columns.
Uses ImageGrid so imshow aspect ratio doesn't leave horizontal dead space.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from tmem_align.analysis.if_spatial import (
    CH_DAPI, CH_LAMP1, CH_MAP2, CH_TMEM,
    DISPLAY_LUT, apply_display_lut, load_fov,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"

FOVS = [
    ("TMEM_KO / C8_F1",      DATA / "TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C8_F1.nd2"),
    ("TMEM_KO / E8_F3",      DATA / "TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_E8_F3.nd2"),
    ("PLD_Control / C17_F1", DATA / "Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C17_F1.nd2"),
    ("PLD_Control / G17_F3", DATA / "Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_G17_F3.nd2"),
    ("PLD_TMEMki / C20_F1",  DATA / "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2"),
    ("PLD_TMEMki / E20_F3",  DATA / "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_E20_F3.nd2"),
]

PANELS = [(CH_MAP2, "MAP2/488"), (CH_LAMP1, "LAMP1/640"), (CH_TMEM, "TMEM/561"), (CH_DAPI, "DAPI/405")]
N = len(FOVS)
NCOLS, NROWS = 4, 2 * N

fig = plt.figure(figsize=(16, 4 * N))
fig.suptitle(
    "Display LUT: per-image percentile (top of each pair) vs fixed per-channel LUT (bottom)",
    fontsize=12, y=0.995,
)

grid = ImageGrid(
    fig, 111,
    nrows_ncols=(NROWS, NCOLS),
    axes_pad=0.04,  # uniform padding between all cells (inches)
    label_mode="L",
)
fig.subplots_adjust(top=0.97)

images = []
for fi, (label, path) in enumerate(FOVS):
    print(f"Loading {label} …", flush=True)
    fov = load_fov(path)
    for key, _ in PANELS:
        img = fov[key].astype(np.float32)
        lo_p, hi_p = np.percentile(img, [1, 99.5])
        images.append(("per", key, np.clip((img - lo_p) / (hi_p - lo_p + 1e-6), 0, 1)))
    for key, _ in PANELS:
        img = fov[key].astype(np.float32)
        images.append(("fixed", key, apply_display_lut(img, key)))

# ImageGrid fills row-major: row 0 = per-image for FOV0, row 1 = fixed for FOV0, etc.
for idx, (ax, (lut_type, key, img)) in enumerate(zip(grid, images)):
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Column headers: right-side ylabel on each top-row cell
for col, (key, ch_label) in enumerate(PANELS):
    ax = grid[col]
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"{ch_label}", fontsize=8, rotation=90, labelpad=4)

# FOV row labels on left column
for fi, (label, _) in enumerate(FOVS):
    grid[fi * 2 * NCOLS].text(-0.02, 0.5, f"{label}\nper-image", fontsize=7,
                               transform=grid[fi * 2 * NCOLS].transAxes,
                               ha="right", va="center", rotation=90)
    lut_lines = "\n".join(f"{ch_label}: {DISPLAY_LUT[key]}" for key, ch_label in PANELS)
    grid[fi * 2 * NCOLS + NCOLS].text(-0.02, 0.5, f"fixed LUT\n{lut_lines}", fontsize=6,
                                       transform=grid[fi * 2 * NCOLS + NCOLS].transAxes,
                                       ha="right", va="center", rotation=90)

# Fixed LUT values: right-side ylabel on each bottom-row cell (replaces overlapping set_title)
for fi in range(N):
    for col, (key, _) in enumerate(PANELS):
        ax = grid[(fi * 2 + 1) * NCOLS + col]
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"{DISPLAY_LUT[key]}", fontsize=6, rotation=90,
                      labelpad=4, color="0.4")

out = REPORT / "lut_comparison_multi.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
