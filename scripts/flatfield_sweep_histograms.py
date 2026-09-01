#!/usr/bin/env python3
"""Add histogram panels to the flat-field sweep report.

Generates two new figures:
  fov_histograms.png       — per-channel histograms raw vs corrected at σ=102 for all 3 well types
  fov_sigma_hist_ki.png    — KI FOV histograms at each sigma (MAP2 + cl-TMEM)

Loads the representative FOVs fresh from ND2 (same files as the sweep),
recomputes the IC field at σ=102 using the pooled sample, and plots.

Usage:
    python scripts/flatfield_sweep_histograms.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tmem_align.preprocess import IC_FIELD_FLOOR, calculate_ic_fields_by_channel  # noqa: E402

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
OUT_DIR = REPO_ROOT / "reports" / "ab_flatfield_sweep"

CHANNELS = ["488nm", "561nm", "640nm", "405nm"]
CH_LABELS = {"488nm": "MAP2 488", "561nm": "cl-TMEM 561", "640nm": "LAMP1 640", "405nm": "DAPI 405"}
WELL_DIRS = ["TMEM_KO", "Z59_PLD_Control", "Z60_PLD_TMEMki"]
WELL_LABELS = {"TMEM_KO": "KO", "Z59_PLD_Control": "Control", "Z60_PLD_TMEMki": "KI"}
SIGMA_CHOSEN = 102
SIGMA_GRID = [8, 25, 51, 102, 256]
SAMPLE_FRACTION = 0.25
SEED = 0

CH_COLORS = {"488nm": "green", "561nm": "darkorange", "640nm": "crimson", "405nm": "royalblue"}


def load_nd2_channel_means(path: Path) -> dict[str, np.ndarray]:
    import nd2
    with nd2.ND2File(path) as f:
        ch_names = [c.channel.name for c in f.metadata.channels]
        arr = f.asarray().astype(np.float32)
    if arr.ndim == 4:
        return {name: arr[:, i].mean(axis=0) for i, name in enumerate(ch_names)}
    elif arr.ndim == 3:
        return {name: arr[i] for i, name in enumerate(ch_names)}
    raise ValueError(f"Unexpected shape {arr.shape}")


def load_nd2_zcyx(path: Path) -> tuple[np.ndarray, list[str]]:
    import nd2
    with nd2.ND2File(path) as f:
        ch_names = [c.channel.name for c in f.metadata.channels]
        arr = f.asarray()
    return arr, ch_names


def build_pooled_ibc(data_root: Path) -> dict[str, list[np.ndarray]]:
    import random
    rng = random.Random(SEED)
    pooled: dict[str, list] = defaultdict(list)
    for tp in ["d7", "d14", "d28"]:
        for well in WELL_DIRS:
            d = data_root / tp / well
            if not d.exists():
                continue
            paths = sorted(d.glob("*.nd2"))
            sampled = rng.sample(paths, max(1, int(len(paths) * SAMPLE_FRACTION)))
            for path in sampled:
                try:
                    for ch, yx in load_nd2_channel_means(path).items():
                        if ch in CHANNELS:
                            pooled[ch].append(yx)
                except Exception as e:
                    print(f"  WARNING: {path.name}: {e}", flush=True)
    return dict(pooled)


def apply_field_zyx(raw_zyx: np.ndarray, field: np.ndarray, dark: float = 0.0) -> np.ndarray:
    """Apply dark-subtract + flat-divide to a ZYX stack. Returns corrected ZYX (float32)."""
    f = np.clip(field, IC_FIELD_FLOOR, None)
    return np.clip(raw_zyx.astype(np.float32) - dark, 0, None) / f[np.newaxis]


def plot_fov_histograms(
    fovs: list[tuple[str, np.ndarray, list[str]]],
    fields: dict[str, np.ndarray],
    darkfields: dict[str, float],
    channels_to_show: list[str],
    out_path: Path,
) -> None:
    """Per-channel overlaid histograms (raw vs corrected) for all well types."""
    n_ch = len(channels_to_show)
    n_wells = len(fovs)
    fig, axes = plt.subplots(n_wells, n_ch, figsize=(n_ch * 4, n_wells * 2.8), squeeze=False)
    fig.suptitle(f"Intensity histograms — raw vs corrected (σ={SIGMA_CHOSEN}, darkfield ON)\nAll Z-planes concatenated (not MIP)", fontsize=11)

    for row, (well_label, zcyx, ch_names) in enumerate(fovs):
        for col, ch in enumerate(channels_to_show):
            ax = axes[row][col]
            if ch not in ch_names:
                ax.axis("off")
                continue
            ci = ch_names.index(ch)
            raw_zyx = zcyx[:, ci].astype(np.float32)   # (Z, Y, X) — all planes
            field = fields.get(ch, np.ones(raw_zyx.shape[1:]))
            dark = darkfields.get(ch, 0.0)
            corrected_zyx = apply_field_zyx(raw_zyx, field, dark)

            raw_flat = raw_zyx.ravel()
            corr_flat = corrected_zyx.ravel()

            lo = float(np.percentile(raw_flat, 0.5))
            hi = float(np.percentile(raw_flat, 99.5))
            bins = np.linspace(lo, hi, 120)

            ax.hist(raw_flat, bins=bins, alpha=0.5, color="gray",
                    density=True, label="Raw", histtype="stepfilled", linewidth=0)
            ax.hist(corr_flat, bins=bins, alpha=0.7,
                    color=CH_COLORS.get(ch, "steelblue"),
                    density=True, label="Corrected", histtype="step", linewidth=1.5)

            if row == 0:
                ax.set_title(CH_LABELS[ch], fontsize=10)
            if col == 0:
                ax.set_ylabel(well_label, fontsize=10)
            ax.set_xlabel("Intensity (AU)", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

            raw_mean = float(np.mean(raw_flat))
            corr_mean = float(np.mean(corr_flat))
            pct = (corr_mean - raw_mean) / max(raw_mean, 1) * 100
            ax.axvline(raw_mean, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(corr_mean, color=CH_COLORS.get(ch, "steelblue"), linestyle="--", linewidth=0.8, alpha=0.9)
            ax.text(0.97, 0.95, f"Δmean: {pct:+.1f}%", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    color="black" if abs(pct) < 2 else "red")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def plot_sigma_histograms(
    well_label: str,
    zcyx: np.ndarray,
    ch_names: list[str],
    fields_by_sigma: list[dict[str, np.ndarray]],
    channels_to_show: list[str],
    out_path: Path,
) -> None:
    """For one well (KI), overlay raw + each sigma's corrected histogram per channel."""
    n_ch = len(channels_to_show)
    fig, axes = plt.subplots(1, n_ch, figsize=(n_ch * 4, 4), squeeze=False)
    fig.suptitle(f"{well_label} — raw vs corrected at each σ (MAP2 + cl-TMEM, all Z-planes)", fontsize=11)

    for col, ch in enumerate(channels_to_show):
        ax = axes[0][col]
        if ch not in ch_names:
            ax.axis("off")
            continue
        ci = ch_names.index(ch)
        raw_zyx = zcyx[:, ci].astype(np.float32)
        raw_flat = raw_zyx.ravel()
        lo = float(np.percentile(raw_flat, 0.5))
        hi = float(np.percentile(raw_flat, 99.5))
        bins = np.linspace(lo, hi, 120)

        ax.hist(raw_flat, bins=bins, alpha=0.4, color="gray",
                density=True, label="Raw", histtype="stepfilled", linewidth=0)

        palette = plt.cm.plasma(np.linspace(0.15, 0.9, len(SIGMA_GRID)))
        for si, (sigma, fields) in enumerate(zip(SIGMA_GRID, fields_by_sigma)):
            field = fields.get(ch, np.ones(raw_zyx.shape[1:]))
            corr_flat = apply_field_zyx(raw_zyx, field).ravel()
            ax.hist(corr_flat, bins=bins, alpha=0.8, density=True,
                    label=f"σ={sigma}", histtype="step", linewidth=1.5, color=palette[si])

        ax.set_title(CH_LABELS[ch], fontsize=10)
        ax.set_xlabel("Intensity (AU)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


def main():
    print("Building pooled IC fields...", flush=True)
    pooled_ibc = build_pooled_ibc(DATA_ROOT)

    print(f"Estimating field at chosen sigma (σ={SIGMA_CHOSEN}) with darkfield...", flush=True)
    result_with_dark = calculate_ic_fields_by_channel(
        pooled_ibc, smooth=SIGMA_CHOSEN, estimate_darkfield=True
    )
    fields_chosen: dict[str, np.ndarray] = {}
    darkfields: dict[str, float] = {}
    for ch, val in result_with_dark.items():
        if isinstance(val, tuple):
            fields_chosen[ch], darkfields[ch] = val
        else:
            fields_chosen[ch] = val
            darkfields[ch] = 0.0
    print(f"  Darkfields: { {ch: f'{v:.1f}' for ch, v in darkfields.items()} }", flush=True)

    print("Estimating fields for all sigmas (for sigma histogram plot)...", flush=True)
    fields_by_sigma: list[dict[str, np.ndarray]] = []
    for sigma in SIGMA_GRID:
        print(f"  sigma={sigma}...", flush=True)
        fields_by_sigma.append(
            calculate_ic_fields_by_channel(pooled_ibc, smooth=sigma)
        )

    print("Loading representative FOVs (all 3 timepoints × 3 well types)...", flush=True)
    import nd2
    # fov_examples: list of (row_label, zcyx, ch_names)
    # one F2 FOV per timepoint × well type, 9 rows total
    fov_examples: list[tuple[str, np.ndarray, list[str]]] = []
    ki_d7_zcyx, ki_d7_ch_names = None, None
    for tp in ["d7", "d14", "d28"]:
        for well in WELL_DIRS:
            well_path = DATA_ROOT / tp / well
            nd2_files = sorted(well_path.glob("*.nd2")) if well_path.exists() else []
            if not nd2_files:
                print(f"  WARNING: no files in {well_path}", flush=True)
                continue
            path = nd2_files[1] if len(nd2_files) > 1 else nd2_files[0]  # F2
            label = f"{tp} {WELL_LABELS[well]}"
            print(f"  {label}: {path.name}", flush=True)
            zcyx, ch_names = load_nd2_zcyx(path)
            fov_examples.append((label, zcyx, ch_names))
            if tp == "d7" and well == "Z60_PLD_TMEMki":
                ki_d7_zcyx, ki_d7_ch_names = zcyx, ch_names

    print("Generating histogram figures...", flush=True)
    plot_fov_histograms(
        fov_examples, fields_chosen, darkfields,
        channels_to_show=["488nm", "561nm", "640nm", "405nm"],
        out_path=OUT_DIR / "fov_histograms.png",
    )

    if ki_d7_zcyx is not None:
        plot_sigma_histograms(
            "KI d7", ki_d7_zcyx, ki_d7_ch_names,
            fields_by_sigma=fields_by_sigma,
            channels_to_show=["488nm", "561nm"],
            out_path=OUT_DIR / "fov_sigma_histograms_ki.png",
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
