"""Derive fixed per-channel LUT display ranges for the d7 fixed-IF dataset.

Per-image percentile stretch makes conditions incomparable. This pools intensity
percentiles across a spread of FOVs per channel and recommends a fixed [lo, hi]
display range (in raw DN) to use for every crop going forward.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tmem_align.analysis.if_spatial import (
    CH_DAPI,
    CH_LAMP1,
    CH_MAP2,
    CH_TMEM,
    DISPLAY_LUT,
    apply_display_lut,
    load_fov,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
REPORT.mkdir(parents=True, exist_ok=True)

# Spread: 3 wells x 2 FOVs per condition = 18 FOVs, across the plate
fovs = []
for cond, sub, pre, col in [
    ("TMEM_KO", "TMEM_KO", "TMEMKO", 8),
    ("PLD_Control", "Z59_PLD_Control", "PLD3Control", 17),
    ("PLD_TMEMki", "Z60_PLD_TMEMki", "PLD3TMEM106B", 20),
]:
    for row in ("C", "E", "G"):
        for f in ("F1", "F3"):
            fovs.append(
                DATA / sub / f"{pre}_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_{row}{col}_{f}.nd2"
            )

CHANNELS = {"MAP2/488": CH_MAP2, "LAMP1/640": CH_LAMP1, "TMEM/561": CH_TMEM, "DAPI/405": CH_DAPI}
pooled = {name: [] for name in CHANNELS}

for path in fovs:
    if not path.exists():
        print(f"SKIP missing {path.name}")
        continue
    chs = load_fov(path)
    for name, key in CHANNELS.items():
        pooled[name].append(chs[key].ravel())

print(f"\nPooled over {len(fovs)} FOVs. Per-channel percentiles (raw DN, uint16):\n")
qs = [0.5, 1, 50, 99, 99.5, 99.9, 99.99, 100]
header = "channel      " + "  ".join(f"p{q:>5}" for q in qs)
print(header)
print("-" * len(header))
rec = {}
for name in CHANNELS:
    allpx = np.concatenate(pooled[name])
    pv = np.percentile(allpx, qs)
    print(f"{name:12s} " + "  ".join(f"{int(v):6d}" for v in pv))
    # Recommended LUT: lo = p1 (background floor), hi = p99.9 (show puncta,
    # tolerate a hair of clipping on the very brightest somata).
    rec[name] = (int(np.percentile(allpx, 1)), int(np.percentile(allpx, 99.9)))

print("\nRecommended fixed LUT ranges (lo=p1, hi=p99.9):")
for name, (lo, hi) in rec.items():
    print(f"  {name:12s} [{lo:6d}, {hi:6d}]")

# --- Before/after comparison figure: per-image percentile vs fixed LUT ---
demo = load_fov(
    DATA / "Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2"
)
panels = [(CH_MAP2, "MAP2/488"), (CH_LAMP1, "LAMP1/640"), (CH_TMEM, "TMEM/561"), (CH_DAPI, "DAPI/405")]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for col, (key, label) in enumerate(panels):
    img = demo[key].astype(np.float32)
    lo, hi = np.percentile(img, [1, 99.5])
    axes[0, col].imshow(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
    axes[0, col].set_title(f"{label}\nper-image p1–p99.5", fontsize=10)
    axes[1, col].imshow(apply_display_lut(img, key), cmap="gray")
    axes[1, col].set_title(f"fixed LUT {DISPLAY_LUT[key]}", fontsize=10)
    for r in (0, 1):
        axes[r, col].axis("off")
axes[0, 0].set_ylabel("per-image percentile", fontsize=11)
plt.suptitle("Display LUT: per-image percentile (top) vs fixed per-channel LUT (bottom) — PLD_TMEMki C20_F1", fontsize=13)
plt.tight_layout()
plt.savefig(REPORT / "lut_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved {REPORT / 'lut_comparison.png'}")

# --- Write the LUT reference report ---
lines = [
    "# Fixed display LUT ranges — d7 fixed-IF dataset",
    "",
    f"Pooled over {len([p for p in fovs if p.exists()])} FOVs across the plate "
    "(3 conditions x 3 rows x 2 fields).",
    "Raw uint16 DN. Use these instead of per-image percentile stretch so brightness",
    "is comparable across conditions and does not drift image-to-image.",
    "",
    "| Channel | LUT lo | LUT hi | Basis |",
    "|---|---|---|---|",
]
basis = {
    "MAP2/488": "p1 floor → p99.9; bright somata clip slightly (intended)",
    "LAMP1/640": "p1 floor → p99.9; punctate lysosomes",
    "TMEM/561": "p1 floor → ~p99.9; very dim/sparse — hi≈1800 is the biology",
    "DAPI/405": "p1 floor → p99.9; bright nuclei",
}
keymap = {"MAP2/488": CH_MAP2, "LAMP1/640": CH_LAMP1, "TMEM/561": CH_TMEM, "DAPI/405": CH_DAPI}
for name in CHANNELS:
    lo, hi = DISPLAY_LUT[keymap[name]]
    lines.append(f"| {name} | {lo} | {hi} | {basis[name]} |")
lines += [
    "",
    "Camera/background floor ≈ 105 DN (all channels bottom out there).",
    "For figures emphasizing dim puncta, drop hi to ~p99.5 (LAMP1 ~5000, TMEM ~400).",
    "Source: `notebooks/lut_range_analysis.py`. Applied via "
    "`tmem_align.analysis.if_spatial.apply_display_lut(img, channel)`.",
    "",
]
(REPORT / "lut_ranges.md").write_text("\n".join(lines))
print(f"Saved {REPORT / 'lut_ranges.md'}")
