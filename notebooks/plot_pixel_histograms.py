"""Plot per-channel pixel-value histograms (MIP) across all d7 ND2 FOVs.

Saves reports/if_segmentation_pilot/pixel_histograms.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tmem_align.analysis.if_spatial import load_fov, DISPLAY_LUT

ND2_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
OUT = Path("/Users/pmihack/claire/tmem_2026/tmem-neuron-aligner/reports/if_segmentation_pilot/pixel_histograms.png")

nd2_files = sorted(ND2_ROOT.rglob("*.nd2"))
print(f"Found {len(nd2_files)} ND2 files")

# Accumulate per-channel pixel samples (subsample to keep memory sane)
SUBSAMPLE = 4  # take every 4th pixel
pixels: dict[str, list] = {}

for i, path in enumerate(nd2_files):
    print(f"  [{i+1}/{len(nd2_files)}] {path.name}", flush=True)
    fov = load_fov(path)
    for ch, arr in fov.items():
        flat = arr.ravel()[::SUBSAMPLE]
        pixels.setdefault(ch, []).append(flat)

# Merge and plot
channels = sorted(pixels.keys())
fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 4), sharey=False)
if len(channels) == 1:
    axes = [axes]

for ax, ch in zip(axes, channels):
    vals = np.concatenate(pixels[ch])
    lo, hi = DISPLAY_LUT.get(ch, (vals.min(), np.percentile(vals, 99.9)))
    ax.hist(vals, bins=200, range=(0, min(hi * 2, vals.max())), color="steelblue", log=True)
    ax.axvline(lo, color="orange", lw=1, label=f"LUT lo={lo}")
    ax.axvline(hi, color="red", lw=1, label=f"LUT hi={hi}")
    ax.set_title(ch)
    ax.set_xlabel("Pixel value (DN)")
    ax.set_ylabel("Count (log)")
    ax.legend(fontsize=7)

fig.suptitle(f"MIP pixel histograms — {len(nd2_files)} FOVs, d7 fixed-IF", y=1.01)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
