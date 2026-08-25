"""Montage of Cellpose-SAM cell-body segmentation on MAP2 across several FOVs.

Sanity check for how badly direct-MAP2 segmentation does across more examples
(the pilot notebook only showed one FOV per condition). Cyan = mask outlines.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_spatial import CH_MAP2, load_fov, segment_nuclei

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
REPORT.mkdir(parents=True, exist_ok=True)

# 3 FOVs per condition, spread across different wells
fovs = {
    "TMEM_KO": [
        "TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C8_F2.nd2",
        "TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_E8_F1.nd2",
        "TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_F8_F3.nd2",
    ],
    "PLD_Control": [
        "Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C17_F2.nd2",
        "Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_E17_F1.nd2",
        "Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_F17_F3.nd2",
    ],
    "PLD_TMEMki": [
        "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F2.nd2",
        "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_E20_F1.nd2",
        "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_F20_F3.nd2",
    ],
}

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for row, (cond, paths) in enumerate(fovs.items()):
    for col, rel in enumerate(paths):
        ax = axes[row, col]
        chs = load_fov(DATA / rel)
        map2 = chs[CH_MAP2].astype(np.float32)
        masks = segment_nuclei(map2, diameter=None)  # same cpsam, cytoplasmic input
        lo, hi = np.percentile(map2, [1, 99.5])
        gray = np.clip((map2 - lo) / (hi - lo + 1e-6), 0, 1)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[find_boundaries(masks, mode="outer")] = [0, 1, 1]
        ax.imshow(rgb)
        well = Path(rel).stem.split("_")[-2] + "_" + Path(rel).stem.split("_")[-1]
        ax.set_title(f"{cond}  {well} — {int(masks.max())} bodies", fontsize=11)
        ax.axis("off")
        print(f"{cond} {well}: {int(masks.max())} bodies")

plt.suptitle("Cellpose-SAM on MAP2 (cytoplasmic) — 3 FOVs per condition, d7", fontsize=14)
plt.tight_layout()
out = REPORT / "map2_cellbody_examples.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
