"""max_distance sweep for seeded cell-body expansion (DAPI nuclei → MAP2).

Rows = conditions, cols = max_distance in px. Cyan = cell body, red = nucleus.
Pick the cap that keeps bodies soma-sized without flooding down neurites.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_spatial import (
    CH_DAPI,
    CH_MAP2,
    expand_to_cell_bodies,
    load_fov,
    segment_nuclei,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
REPORT.mkdir(parents=True, exist_ok=True)

DISTANCES = [90, 110, 140]  # px at 0.108 µm/px → ~10, 12, 15 µm
fovs = {
    "TMEM_KO": "TMEM_KO/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C8_F1.nd2",
    "PLD_Control": "Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C17_F1.nd2",
    "PLD_TMEMki": "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2",
}

fig, axes = plt.subplots(len(fovs), len(DISTANCES), figsize=(6 * len(DISTANCES), 6 * len(fovs)))
for row, (cond, rel) in enumerate(fovs.items()):
    chs = load_fov(DATA / rel)
    nuclei = segment_nuclei(chs[CH_DAPI])
    map2 = chs[CH_MAP2].astype(np.float32)
    lo, hi = np.percentile(map2, [1, 99.5])
    gray = np.clip((map2 - lo) / (hi - lo + 1e-6), 0, 1)
    for col, dist in enumerate(DISTANCES):
        bodies = expand_to_cell_bodies(nuclei, map2, max_distance=dist)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[find_boundaries(bodies, mode="outer")] = [0, 1, 1]
        rgb[find_boundaries(nuclei, mode="outer")] = [1, 0, 0]
        ax = axes[row, col]
        ax.imshow(rgb)
        ax.set_title(f"{cond} — max_distance={dist}px (~{dist * 0.108:.0f}µm)", fontsize=10)
        ax.axis("off")
        print(f"{cond} dist={dist}: {int(bodies.max())} bodies")

plt.suptitle("Seeded cell-body expansion — max_distance sweep, d7", fontsize=14)
plt.tight_layout()
out = REPORT / "map2_expansion_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
