#!/usr/bin/env python3
"""QC check: LAMP1 640 dim at d14.

Hypothesis: the −77% Δmean in d14 KI LAMP1 640 is either
  (a) all d14 FOVs are dim in 640 (plate/staining batch effect), or
  (b) F2 specifically is dim (focus issue, dim FOV).

Generates:
  lamp1_d14_qc/
    raw_distribution_by_timepoint.png  — 640 distributions: d7/d14/d28 × KO/Control/KI (F2)
    raw_mean_per_fov.png               — per-FOV mean 640 intensity, all F1-F4 across all wells/timepoints
    z_profile_comparison.png           — Z-plane mean 640 for d14 KI F1-F4 vs d7 KI F2 + d28 KI F2
    midplane_montage_d14.png           — middle Z-plane 640 image for every d14 FOV (visual check)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
OUT_DIR = REPO_ROOT / "reports" / "lamp1_d14_qc"

TIMEPOINTS = ["d7", "d14", "d28"]
WELL_DIRS = ["TMEM_KO", "Z59_PLD_Control", "Z60_PLD_TMEMki"]
WELL_LABELS = {"TMEM_KO": "KO", "Z59_PLD_Control": "Control", "Z60_PLD_TMEMki": "KI"}
CH_640 = "640nm"


def load_channel(path: Path, ch_name: str) -> tuple[np.ndarray, list[str]]:
    """Load one ND2, return (ZYX uint16 array for ch_name, all_ch_names)."""
    import nd2
    with nd2.ND2File(path) as f:
        ch_names = [c.channel.name for c in f.metadata.channels]
        arr = f.asarray()
    ci = ch_names.index(ch_name) if ch_name in ch_names else None
    if ci is None:
        raise ValueError(f"{ch_name} not in {ch_names} for {path.name}")
    return arr[:, ci].astype(np.float32), ch_names


def fov_paths(data_root: Path, tp: str, well: str) -> list[Path]:
    d = data_root / tp / well
    return sorted(d.glob("*.nd2")) if d.exists() else []


# ---------------------------------------------------------------------------
# Plot 1 — raw 640 distribution per timepoint × well type (F2 only)
# ---------------------------------------------------------------------------

def plot_distributions_by_timepoint(out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    fig.suptitle("LAMP1 640 raw intensity distribution — F2 FOV, all Z-planes\n(checking if d14 is dim plate-wide)", fontsize=11)

    for ri, tp in enumerate(TIMEPOINTS):
        for ci, well in enumerate(WELL_DIRS):
            ax = axes[ri][ci]
            paths = fov_paths(DATA_ROOT, tp, well)
            if not paths:
                ax.axis("off"); continue
            path = paths[1] if len(paths) > 1 else paths[0]  # F2
            try:
                zyx, _ = load_channel(path, CH_640)
            except Exception as e:
                ax.set_title(f"{tp} {WELL_LABELS[well]}\nERROR: {e}", fontsize=7)
                continue

            flat = zyx.ravel()
            lo, hi = float(np.percentile(flat, 0.5)), float(np.percentile(flat, 99.5))
            bins = np.linspace(lo, hi, 100)
            ax.hist(flat, bins=bins, density=True, color="crimson", alpha=0.7, histtype="stepfilled")
            ax.set_title(f"{tp} {WELL_LABELS[well]}\nmean={flat.mean():.0f}  min={flat.min():.0f}", fontsize=8)
            ax.set_xlabel("Intensity (ADU)", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Plot 2 — per-FOV mean 640 across all wells and timepoints
# ---------------------------------------------------------------------------

def plot_mean_per_fov(out_path: Path) -> None:
    """Dot plot: mean 640 intensity per FOV, grouped by timepoint × well."""
    data = {}  # (tp, well_label) -> [mean_per_fov]
    for tp in TIMEPOINTS:
        for well in WELL_DIRS:
            label = f"{tp}\n{WELL_LABELS[well]}"
            means = []
            for path in fov_paths(DATA_ROOT, tp, well):
                try:
                    zyx, _ = load_channel(path, CH_640)
                    means.append(float(zyx.mean()))
                except Exception:
                    pass
            if means:
                data[label] = means

    labels = list(data.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    for xi, label in enumerate(labels):
        vals = data[label]
        ax.scatter([xi] * len(vals), vals, alpha=0.7, s=30, color="crimson")
        ax.plot([xi - 0.3, xi + 0.3], [np.mean(vals), np.mean(vals)], "k-", linewidth=2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean pixel intensity (ADU, all Z-planes)")
    ax.set_title("LAMP1 640 mean intensity per FOV — all timepoints × well types\n(each dot = one F1/F2/F3/F4 file; bar = group mean)")
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="~camera floor (100 ADU)")
    ax.legend(fontsize=8)

    # Shade d14 columns
    d14_indices = [i for i, l in enumerate(labels) if l.startswith("d14")]
    if d14_indices:
        ax.axvspan(min(d14_indices) - 0.5, max(d14_indices) + 0.5, alpha=0.08, color="orange", label="d14")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Plot 3 — Z-profile comparison: d14 KI F1-F4 vs d7/d28 KI F2
# ---------------------------------------------------------------------------

def plot_z_profiles(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("LAMP1 640 — mean intensity per Z-plane\nd14 KI (all FOVs) vs d7 KI F2 vs d28 KI F2", fontsize=10)

    # d14 KI all FOVs
    for i, path in enumerate(fov_paths(DATA_ROOT, "d14", "Z60_PLD_TMEMki")):
        try:
            zyx, _ = load_channel(path, CH_640)
            z_means = zyx.mean(axis=(1, 2))
            ax.plot(z_means, color="darkorange", alpha=0.7, linewidth=1.2,
                    label=f"d14 KI F{i+1}" if i < 4 else "_")
        except Exception:
            pass

    # d7 KI F2 reference
    paths_d7 = fov_paths(DATA_ROOT, "d7", "Z60_PLD_TMEMki")
    if len(paths_d7) > 1:
        zyx, _ = load_channel(paths_d7[1], CH_640)
        ax.plot(zyx.mean(axis=(1, 2)), "b-", linewidth=2, label="d7 KI F2")

    # d28 KI F2 reference
    paths_d28 = fov_paths(DATA_ROOT, "d28", "Z60_PLD_TMEMki")
    if len(paths_d28) > 1:
        zyx, _ = load_channel(paths_d28[1], CH_640)
        ax.plot(zyx.mean(axis=(1, 2)), "g-", linewidth=2, label="d28 KI F2")

    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="~camera floor")
    ax.set_xlabel("Z-plane index"); ax.set_ylabel("Mean intensity (ADU)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Plot 4 — middle-plane montage of 640 for all d14 FOVs
# ---------------------------------------------------------------------------

def plot_midplane_montage(out_path: Path) -> None:
    """Show middle Z-plane of LAMP1 640 for every d14 FOV (KO / Control / KI)."""
    all_imgs: list[tuple[str, np.ndarray]] = []
    for well in WELL_DIRS:
        for path in fov_paths(DATA_ROOT, "d14", well):
            try:
                zyx, _ = load_channel(path, CH_640)
                mid = zyx.shape[0] // 2
                all_imgs.append((f"d14 {WELL_LABELS[well]}\n{path.name[-10:-4]}", zyx[mid]))
            except Exception:
                pass

    n = len(all_imgs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).ravel()
    fig.suptitle("LAMP1 640 — middle Z-plane, all d14 FOVs\n(visual check: focus issues, staining dropout, dim wells)", fontsize=10)

    # Compute a consistent display range across all images (p1–p99.5 of all)
    all_vals = np.concatenate([img.ravel() for _, img in all_imgs])
    vlo, vhi = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99.5))

    for i, (label, img) in enumerate(all_imgs):
        ax = axes[i]
        ax.imshow(img, cmap="Reds", vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_title(label, fontsize=7)
        ax.axis("off")
        # Annotate mean
        ax.text(0.02, 0.02, f"mean={img.mean():.0f}", transform=ax.transAxes,
                color="white", fontsize=6, va="bottom")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}", flush=True)

    print("[1/4] Distribution by timepoint (F2 FOVs)...", flush=True)
    plot_distributions_by_timepoint(OUT_DIR / "raw_distribution_by_timepoint.png")

    print("[2/4] Per-FOV mean (all F1-F4, all timepoints)...", flush=True)
    plot_mean_per_fov(OUT_DIR / "raw_mean_per_fov.png")

    print("[3/4] Z-profile comparison (d14 KI vs d7/d28 KI)...", flush=True)
    plot_z_profiles(OUT_DIR / "z_profile_comparison.png")

    print("[4/4] Middle-plane montage (all d14 FOVs)...", flush=True)
    plot_midplane_montage(OUT_DIR / "midplane_montage_d14.png")

    print(f"\nDone → {OUT_DIR}", flush=True)
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}", flush=True)


if __name__ == "__main__":
    main()
