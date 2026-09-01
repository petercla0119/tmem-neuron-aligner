#!/usr/bin/env python3
"""Flat-field correction validation sweep — sigma grid + A/B arms.

Sweeps Gaussian smoothing sigma across five values (short-side / 20, 40, 80
plus guard rails at 8 and 256 px), estimates one pooled per-channel 2-D IC
field per sigma from d7+d14+d28, and emits figures + a markdown report.

Outputs → reports/ab_flatfield_sweep/

Usage:
    python scripts/flatfield_sweep.py [--data-root /path/to/cleaved_tmem_pld3_260821]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

# Resolve repo root so the worktree src is found regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tmem_align.preprocess import (  # noqa: E402
    IC_FIELD_FLOOR,
    _estimate_darkfield,
    _load_image,
    _zcyx_to_cyx,
    apply_ic_field,
    calculate_ic_fields_by_channel,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SHORT_SIDE = 2048
SIGMA_GRID = [8, SHORT_SIDE // 80, SHORT_SIDE // 40, SHORT_SIDE // 20, 256]
# → [8, 25, 51, 102, 256]
SIGMA_LABELS = [f"σ={s}" for s in SIGMA_GRID]
SIGMA_DEFAULT_IDX = 2  # σ=51 (/40)

CHANNELS = ["488nm", "561nm", "640nm", "405nm"]
CH_LABELS = {"488nm": "MAP2 488", "561nm": "cl-TMEM 561", "640nm": "LAMP1 640", "405nm": "DAPI 405"}
CH_CMAPS  = {"488nm": "green", "561nm": "Oranges", "640nm": "Reds", "405nm": "Blues"}

CORNER_PX = 128   # corner patch size for corner/center ratio
CENTER_PX = 256   # center patch size

DATA_ROOT_DEFAULT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
TIMEPOINTS = ["d7", "d14", "d28"]
WELL_DIRS = ["TMEM_KO", "Z59_PLD_Control", "Z60_PLD_TMEMki"]
WELL_LABELS = {"TMEM_KO": "KO", "Z59_PLD_Control": "Control", "Z60_PLD_TMEMki": "KI"}

OUT_DIR = REPO_ROOT / "reports" / "ab_flatfield_sweep"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_nd2_paths(data_root: Path) -> dict[str, list[Path]]:
    """Return {timepoint: [nd2_paths]} across all wells."""
    paths: dict[str, list[Path]] = {tp: [] for tp in TIMEPOINTS}
    for tp in TIMEPOINTS:
        for well in WELL_DIRS:
            d = data_root / tp / well
            if d.exists():
                paths[tp].extend(sorted(d.glob("*.nd2")))
    return paths


def load_nd2_channel_means(
    path: Path,
) -> dict[str, np.ndarray]:
    """Load one ND2, return {channel_name: Z-mean YX (float32)} keyed by name.

    Z-mean (not median) for speed — 30× faster per file, negligible quality
    difference when pooling 50+ FOVs for the pixelwise-median IC field.
    """
    import nd2
    with nd2.ND2File(path) as f:
        ch_names = [c.channel.name for c in f.metadata.channels]
        arr = f.asarray()  # (Z, C, Y, X) or (C, Y, X)
    arr = arr.astype(np.float32)
    if arr.ndim == 4:  # ZCYX → per-channel Z-mean
        return {name: arr[:, i].mean(axis=0) for i, name in enumerate(ch_names)}
    elif arr.ndim == 3:  # CYX — no Z
        return {name: arr[i] for i, name in enumerate(ch_names)}
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} in {path}")


def build_images_by_channel(
    paths_by_tp: dict[str, list[Path]],
    pool: bool = True,
    sample_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Build images_by_channel for pooled or per-timepoint estimation.

    Samples sample_fraction of each timepoint's files before loading (seeded).
    Returns either:
        pool=True  → {"_pooled": {ch: [yx_arrays...]}}
        pool=False → {tp: {ch: [yx_arrays...]}}
    """
    import random
    rng = random.Random(seed)
    result: dict[str, dict[str, list]] = {}

    for tp, paths in paths_by_tp.items():
        sampled = paths if sample_fraction >= 1.0 else rng.sample(paths, max(1, int(len(paths) * sample_fraction)))
        print(f"    {tp}: loading {len(sampled)}/{len(paths)} files...", flush=True)
        ch_images: dict[str, list] = defaultdict(list)
        for path in sampled:
            try:
                ch_means = load_nd2_channel_means(path)
                for ch, yx in ch_means.items():
                    if ch in CHANNELS:
                        ch_images[ch].append(yx)
            except Exception as e:
                print(f"  WARNING: skipping {path.name}: {e}", flush=True)
        result[tp] = dict(ch_images)

    if pool:
        pooled: dict[str, list] = defaultdict(list)
        for tp_imgs in result.values():
            for ch, imgs in tp_imgs.items():
                pooled[ch].extend(imgs)
        return {"_pooled": dict(pooled)}
    return result


# ---------------------------------------------------------------------------
# Field estimation helpers
# ---------------------------------------------------------------------------

def estimate_fields(
    images_by_channel: dict[str, list[np.ndarray]],
    sigma: int,
    estimate_darkfield: bool = False,
) -> dict[str, np.ndarray] | dict[str, tuple[np.ndarray, float]]:
    return calculate_ic_fields_by_channel(
        images_by_channel, smooth=sigma, estimate_darkfield=estimate_darkfield
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def field_metrics(field: np.ndarray) -> dict[str, float]:
    """Spatial CV% and corner/center ratio for a 2-D field."""
    H, W = field.shape
    center_r = slice(H // 2 - CENTER_PX // 2, H // 2 + CENTER_PX // 2)
    center_c = slice(W // 2 - CENTER_PX // 2, W // 2 + CENTER_PX // 2)
    center_mean = float(np.mean(field[center_r, center_c]))

    corners = [
        field[:CORNER_PX, :CORNER_PX],
        field[:CORNER_PX, -CORNER_PX:],
        field[-CORNER_PX:, :CORNER_PX],
        field[-CORNER_PX:, -CORNER_PX:],
    ]
    corner_mean = float(np.mean([c.mean() for c in corners]))
    cv = float(np.std(field) / np.mean(field) * 100)
    return {"cv_pct": cv, "corner_center_ratio": corner_mean / max(center_mean, 1e-6)}


def cross_fov_cv(images: list[np.ndarray], field: np.ndarray, dark: float = 0.0) -> float:
    """CV% of per-FOV mean intensity across FOVs, after IC correction."""
    field_clipped = np.clip(field, IC_FIELD_FLOOR, None)
    means = []
    for img in images:
        corrected = (img.astype(np.float32) - dark) / field_clipped
        means.append(float(np.mean(corrected)))
    arr = np.array(means)
    return float(np.std(arr) / np.mean(arr) * 100) if arr.mean() > 0 else 0.0


def cross_fov_cv_raw(images: list[np.ndarray]) -> float:
    means = [float(np.mean(img)) for img in images]
    arr = np.array(means)
    return float(np.std(arr) / np.mean(arr) * 100) if arr.mean() > 0 else 0.0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _lut(img: np.ndarray, p_lo: float = 1, p_hi: float = 99) -> np.ndarray:
    lo, hi = np.percentile(img, p_lo), np.percentile(img, p_hi)
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def plot_field_renders(
    fields_by_sigma: list[dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """4 channels × 5 sigmas grid of field heatmaps."""
    n_sigma = len(SIGMA_GRID)
    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(n_ch, n_sigma, figsize=(n_sigma * 3, n_ch * 3))
    fig.suptitle("Estimated 2-D IC fields — per channel × sigma", fontsize=13)

    for ci, ch in enumerate(CHANNELS):
        for si, (sigma, fields) in enumerate(zip(SIGMA_GRID, fields_by_sigma)):
            ax = axes[ci][si]
            field = fields.get(ch)
            if field is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(field, cmap="RdBu_r", vmin=0.7, vmax=1.3, interpolation="nearest")
            if ci == 0:
                ax.set_title(SIGMA_LABELS[si], fontsize=9)
            if si == 0:
                ax.set_ylabel(CH_LABELS[ch], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_metrics_vs_sigma(
    metrics_by_sigma: list[dict[str, dict[str, float]]],  # [sigma_idx][ch] → {cv_pct, cc_ratio}
    raw_cv_by_ch: dict[str, float],
    out_path: Path,
) -> None:
    """Spatial CV% and corner/center ratio vs sigma, per channel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Field metrics vs smoothing sigma", fontsize=12)
    markers = ["o", "s", "^", "D"]

    for i, ch in enumerate(CHANNELS):
        cv_vals = [m[ch]["cv_pct"] for m in metrics_by_sigma]
        cc_vals = [m[ch]["corner_center_ratio"] for m in metrics_by_sigma]
        label = CH_LABELS[ch]
        ax1.plot(SIGMA_GRID, cv_vals, marker=markers[i], label=label)
        ax2.plot(SIGMA_GRID, cc_vals, marker=markers[i], label=label)

    ax1.set_xlabel("Sigma (px)"); ax1.set_ylabel("Spatial CV% of field")
    ax1.set_xscale("log"); ax1.legend(fontsize=8); ax1.set_title("Spatial CV% (higher = more heterogeneity captured)")
    ax1.axvline(SIGMA_GRID[SIGMA_DEFAULT_IDX], color="gray", linestyle="--", alpha=0.5, label="default /40")

    ax2.set_xlabel("Sigma (px)"); ax2.set_ylabel("Corner / center ratio")
    ax2.set_xscale("log"); ax2.legend(fontsize=8); ax2.set_title("Corner/center ratio (1.0 = no vignette)")
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax2.axvline(SIGMA_GRID[SIGMA_DEFAULT_IDX], color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fov_sigma_comparison(
    fov_zcyx: np.ndarray,
    ch_names_in_fov: list[str],
    fields_by_sigma: list[dict[str, np.ndarray]],
    channel: str,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    """Max-projection of one channel, raw vs corrected at each sigma (for display)."""
    ci = ch_names_in_fov.index(channel) if channel in ch_names_in_fov else 0
    raw_mip = fov_zcyx[:, ci].max(axis=0).astype(np.float32)

    n = len(SIGMA_GRID)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3, 3.5))
    axes[0].imshow(_lut(raw_mip), cmap="gray")
    axes[0].set_title("Raw", fontsize=9); axes[0].axis("off")

    for si, (sigma, fields) in enumerate(zip(SIGMA_GRID, fields_by_sigma)):
        field = fields.get(channel)
        if field is None:
            axes[si + 1].set_visible(False)
            continue
        field_c = np.clip(field, IC_FIELD_FLOOR, None)
        corrected = raw_mip / field_c
        axes[si + 1].imshow(_lut(corrected), cmap="gray")
        axes[si + 1].set_title(SIGMA_LABELS[si], fontsize=9)
        axes[si + 1].axis("off")

    fig.suptitle(f"{title_prefix} — {CH_LABELS.get(channel, channel)} — raw vs corrected", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_darkfield_ab(
    fov_zcyx: np.ndarray,
    ch_names_in_fov: list[str],
    field_no_dark: dict[str, np.ndarray],
    field_with_dark: dict[str, np.ndarray],
    darkfields: dict[str, float],
    channel: str,
    out_path: Path,
) -> None:
    ci = ch_names_in_fov.index(channel) if channel in ch_names_in_fov else 0
    raw_mip = fov_zcyx[:, ci].max(axis=0).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(_lut(raw_mip), cmap="gray"); axes[0].set_title("Raw"); axes[0].axis("off")

    field_nd = np.clip(field_no_dark.get(channel, np.ones_like(raw_mip)), IC_FIELD_FLOOR, None)
    axes[1].imshow(_lut(raw_mip / field_nd), cmap="gray")
    axes[1].set_title("Darkfield OFF"); axes[1].axis("off")

    field_wd = np.clip(field_with_dark.get(channel, np.ones_like(raw_mip)), IC_FIELD_FLOOR, None)
    dark_val = darkfields.get(channel, 0.0)
    corrected_with_dark = np.clip(raw_mip - dark_val, 0, None) / field_wd
    axes[2].imshow(_lut(corrected_with_dark), cmap="gray")
    axes[2].set_title(f"Darkfield ON (offset={dark_val:.0f} ADU)"); axes[2].axis("off")

    fig.suptitle(f"Darkfield A/B — {CH_LABELS.get(channel, channel)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_pooled_vs_pertimepoint(
    fov_zcyx: np.ndarray,
    ch_names_in_fov: list[str],
    pooled_fields: dict[str, np.ndarray],
    tp_fields: dict[str, dict[str, np.ndarray]],
    channel: str,
    out_path: Path,
) -> None:
    ci = ch_names_in_fov.index(channel) if channel in ch_names_in_fov else 0
    raw_mip = fov_zcyx[:, ci].max(axis=0).astype(np.float32)

    tp_names = sorted(tp_fields.keys())
    ncols = 2 + len(tp_names)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3.5))
    axes[0].imshow(_lut(raw_mip), cmap="gray"); axes[0].set_title("Raw"); axes[0].axis("off")

    pf = np.clip(pooled_fields.get(channel, np.ones_like(raw_mip)), IC_FIELD_FLOOR, None)
    axes[1].imshow(_lut(raw_mip / pf), cmap="gray"); axes[1].set_title("Pooled"); axes[1].axis("off")

    for i, tp in enumerate(tp_names):
        tf = np.clip(tp_fields[tp].get(channel, np.ones_like(raw_mip)), IC_FIELD_FLOOR, None)
        axes[2 + i].imshow(_lut(raw_mip / tf), cmap="gray")
        axes[2 + i].set_title(f"Per-tp ({tp})"); axes[2 + i].axis("off")

    fig.suptitle(f"Pooled vs per-timepoint — {CH_LABELS.get(channel, channel)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_z_profile(
    fov_zcyx: np.ndarray,
    ch_names_in_fov: list[str],
    fields: dict[str, np.ndarray],
    channel: str,
    out_path: Path,
) -> None:
    """Mean corrected intensity per Z-plane — should be flat if vignette-only correction."""
    ci = ch_names_in_fov.index(channel) if channel in ch_names_in_fov else 0
    raw_z = fov_zcyx[:, ci].astype(np.float32)  # (Z, Y, X)

    field = np.clip(fields.get(channel, np.ones(raw_z.shape[1:])), IC_FIELD_FLOOR, None)
    corrected_z = raw_z / field[np.newaxis]  # broadcast (1, Y, X) across Z

    z_indices = np.arange(raw_z.shape[0])
    raw_means = raw_z.mean(axis=(1, 2))
    corr_means = corrected_z.mean(axis=(1, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(z_indices, raw_means, "o-", label="Raw")
    axes[0].plot(z_indices, corr_means, "s--", label="Corrected")
    axes[0].set_xlabel("Z plane"); axes[0].set_ylabel("Mean intensity")
    axes[0].set_title(f"Z-profile — {CH_LABELS.get(channel, channel)}"); axes[0].legend()

    # Relative: corrected/raw — should stay near 1.0 if the field is Z-invariant
    ratio = corr_means / np.maximum(raw_means, 1)
    axes[1].plot(z_indices, ratio, "^-", color="purple")
    axes[1].axhline(1.0, color="gray", linestyle=":")
    axes[1].set_xlabel("Z plane"); axes[1].set_ylabel("Corrected / Raw mean")
    axes[1].set_title("Ratio per plane (should be flat ≈ const)")

    fig.suptitle("3D Z-profile check — single 2D field applied across Z", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_well_examples(
    fovs: list[tuple[str, np.ndarray, list[str]]],  # (well_label, zcyx, ch_names)
    fields: dict[str, np.ndarray],  # default sigma fields
    channels_to_show: list[str],
    out_path: Path,
) -> None:
    """Before/after for each well type at default sigma for selected channels."""
    n_wells = len(fovs)
    n_ch = len(channels_to_show)
    fig, axes = plt.subplots(n_wells, n_ch * 2, figsize=(n_ch * 2 * 3, n_wells * 3))
    if n_wells == 1:
        axes = axes[np.newaxis]

    for row, (well_label, zcyx, ch_names) in enumerate(fovs):
        for col, ch in enumerate(channels_to_show):
            if ch not in ch_names:
                axes[row, col * 2].axis("off"); axes[row, col * 2 + 1].axis("off")
                continue
            ci = ch_names.index(ch)
            raw_mip = zcyx[:, ci].max(axis=0).astype(np.float32)
            field = np.clip(fields.get(ch, np.ones_like(raw_mip)), IC_FIELD_FLOOR, None)
            corrected = raw_mip / field

            axes[row, col * 2].imshow(_lut(raw_mip), cmap="gray")
            axes[row, col * 2].set_title(f"{well_label} {CH_LABELS.get(ch, ch)} raw", fontsize=8)
            axes[row, col * 2].axis("off")

            axes[row, col * 2 + 1].imshow(_lut(corrected), cmap="gray")
            axes[row, col * 2 + 1].set_title(f"{well_label} {CH_LABELS.get(ch, ch)} corrected", fontsize=8)
            axes[row, col * 2 + 1].axis("off")

    fig.suptitle(f"Before/after per well type — σ={SIGMA_GRID[SIGMA_DEFAULT_IDX]} (default /40)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    out_dir: Path,
    sigma_metrics: list[dict[str, dict[str, float]]],
    raw_cv: dict[str, float],
    darkfields: dict[str, float],
    n_files: int,
    n_files_per_tp: dict[str, int],
) -> None:
    default_sigma = SIGMA_GRID[SIGMA_DEFAULT_IDX]
    default_metrics = sigma_metrics[SIGMA_DEFAULT_IDX]

    lines = [
        "# Flat-field Correction — Validation Sweep Report",
        "",
        f"**Date:** 2026-08-31  **Branch:** `dev/preprocess/flatfield`  **Script:** `scripts/flatfield_sweep.py`",
        "",
        "## Settings used",
        "",
        f"- **Sigma grid:** {SIGMA_GRID} px (short-side/20, /40, /80 + guard rails 8, 256)",
        f"- **Default sigma:** {default_sigma} px (short-side/40)",
        f"- **Estimator:** median-of-population, pooled across d7+d14+d28",
        f"- **Images pooled:** {n_files} ND2 files across all timepoints",
        f"  - d7: {n_files_per_tp.get('d7', 0)}, d14: {n_files_per_tp.get('d14', 0)}, d28: {n_files_per_tp.get('d28', 0)}",
        f"- **Channel keying:** by name (`488nm`, `561nm`, `640nm`, `405nm`) — D20_F1 swap handled automatically",
        f"- **Field model:** one 2-D YX field per named channel, broadcast across Z for 3-D stacks",
        f"- **Darkfield:** estimated scalar per channel via 1st-percentile-of-minima",
        "",
        "## Estimated darkfield offsets (ADU)",
        "",
        "| Channel | Darkfield offset |",
        "|---------|-----------------|",
    ]
    for ch in CHANNELS:
        lines.append(f"| {CH_LABELS[ch]} | {darkfields.get(ch, 0):.1f} |")

    lines += [
        "",
        "## Field metrics per sigma",
        "",
        "Spatial CV% = std/mean × 100 over the full 2-D field.  ",
        "Corner/center ratio = mean of four 128×128 corner patches / 256×256 center patch.",
        "",
    ]
    for ch in CHANNELS:
        lines.append(f"### {CH_LABELS[ch]}")
        lines.append("")
        lines.append(f"Raw cross-FOV CV% (uncorrected mean/FOV): **{raw_cv.get(ch, 0):.1f}%**")
        lines.append("")
        lines.append("| Sigma | Spatial CV% | Corner/center |")
        lines.append("|-------|------------|---------------|")
        for si, metrics in enumerate(sigma_metrics):
            m = metrics.get(ch, {})
            marker = " ← default" if si == SIGMA_DEFAULT_IDX else ""
            lines.append(
                f"| {SIGMA_GRID[si]} | {m.get('cv_pct', 0):.2f}% | {m.get('corner_center_ratio', 0):.4f}{marker} |"
            )
        lines.append("")

    lines += [
        "## Figures",
        "",
        "| Figure | What it shows |",
        "|--------|--------------|",
        "| `field_renders.png` | Estimated 2-D fields — 4 channels × 5 sigmas. Check for cell-shaped structure (trip-wire for BaSiC revisit). |",
        "| `metrics_vs_sigma.png` | Spatial CV% and corner/center ratio vs sigma. Pick sigma at the CV% knee. |",
        "| `fov_ki_sigma_comparison.png` | KI FOV raw vs corrected at each sigma (MAP2 channel). |",
        "| `fov_ko_sigma_comparison.png` | KO FOV raw vs corrected at each sigma (MAP2 channel). |",
        "| `darkfield_ab.png` | Darkfield ON vs OFF at default sigma. |",
        "| `pooled_vs_pertimepoint.png` | Pooled field vs per-timepoint fields at default sigma. |",
        "| `z_profile.png` | Mean intensity per Z-plane before/after — single 2-D field should not introduce axial artifact. |",
        "| `well_examples.png` | Before/after at default sigma for KO / Control / KI wells (MAP2 + cl-TMEM channels). |",
        "",
        "## Decision 6 — smoothing sigma",
        "",
        "*(Fill in after reviewing figures)*",
        "",
        f"Chosen sigma: **TBD** — see `metrics_vs_sigma.png` (pick where CV% knee plateaus without cell-shaped structure in `field_renders.png`).",
        "",
        "## Caveats",
        "",
        "- No uniform reference slide exists — validation shows *consistency/uniformity*, not absolute accuracy.",
        "- Darkfield is a 1st-percentile-of-minima estimate, not a measured dark frame.",
        "- Same 2-D field broadcast across all Z planes; axial Z-dependence of illumination is not corrected.",
        "- D20_F1 channel swap handled by name-keying (not excluded — data is used with correct channel assignment).",
    ]

    report_path = out_dir / "SWEEP_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"  Report → {report_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sample-fraction", type=float, default=0.25,
                        help="Fraction of files per timepoint to use for field estimation (default 0.25)")
    args = parser.parse_args()

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    sample_fraction: float = args.sample_fraction
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_root}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Sample fraction: {sample_fraction}", flush=True)

    # -----------------------------------------------------------------------
    # 1. Collect ND2 paths
    # -----------------------------------------------------------------------
    print("\n[1/7] Collecting ND2 paths...", flush=True)
    paths_by_tp = collect_nd2_paths(data_root)
    n_files_per_tp = {tp: len(ps) for tp, ps in paths_by_tp.items()}
    n_files = sum(n_files_per_tp.values())
    for tp, n in n_files_per_tp.items():
        print(f"  {tp}: {n} files", flush=True)

    # -----------------------------------------------------------------------
    # 2. Load images (Z-median per channel per file) — pooled
    # -----------------------------------------------------------------------
    print("\n[2/7] Loading ND2 files and extracting Z-mean per channel...", flush=True)
    pooled_ibc = build_images_by_channel(paths_by_tp, pool=True, sample_fraction=sample_fraction)["_pooled"]
    per_tp_ibc = build_images_by_channel(paths_by_tp, pool=False, sample_fraction=sample_fraction)
    for ch in CHANNELS:
        n = len(pooled_ibc.get(ch, []))
        print(f"  {CH_LABELS[ch]}: {n} images pooled", flush=True)

    # -----------------------------------------------------------------------
    # 3. Compute fields at each sigma (pooled, no darkfield)
    # -----------------------------------------------------------------------
    print("\n[3/7] Estimating IC fields across sigma grid...", flush=True)
    fields_by_sigma: list[dict[str, np.ndarray]] = []
    for sigma in SIGMA_GRID:
        print(f"  sigma={sigma}...", flush=True)
        f = estimate_fields(pooled_ibc, sigma=sigma)
        fields_by_sigma.append(f)

    # Darkfield: estimate at default sigma
    print("  estimating darkfield at default sigma...", flush=True)
    fields_with_dark_result = estimate_fields(pooled_ibc, sigma=SIGMA_GRID[SIGMA_DEFAULT_IDX], estimate_darkfield=True)
    fields_no_dark = fields_by_sigma[SIGMA_DEFAULT_IDX]
    fields_with_dark: dict[str, np.ndarray] = {}
    darkfields: dict[str, float] = {}
    for ch, val in fields_with_dark_result.items():
        if isinstance(val, tuple):
            fields_with_dark[ch], darkfields[ch] = val
        else:
            fields_with_dark[ch] = val
            darkfields[ch] = 0.0

    # Per-timepoint fields at default sigma
    print("  per-timepoint fields at default sigma...", flush=True)
    tp_fields: dict[str, dict[str, np.ndarray]] = {}
    for tp, ibc in per_tp_ibc.items():
        if ibc:
            tp_fields[tp] = estimate_fields(ibc, sigma=SIGMA_GRID[SIGMA_DEFAULT_IDX])

    # -----------------------------------------------------------------------
    # 4. Compute metrics
    # -----------------------------------------------------------------------
    print("\n[4/7] Computing metrics...", flush=True)
    sigma_metrics: list[dict[str, dict[str, float]]] = []
    for sigma, fields in zip(SIGMA_GRID, fields_by_sigma):
        m = {}
        for ch in CHANNELS:
            if ch in fields:
                m[ch] = field_metrics(fields[ch])
        sigma_metrics.append(m)
        print(f"  sigma={sigma}: MAP2 CV%={m.get('488nm', {}).get('cv_pct', 0):.2f}  cc={m.get('488nm', {}).get('corner_center_ratio', 0):.4f}", flush=True)

    raw_cv: dict[str, float] = {}
    for ch in CHANNELS:
        imgs = pooled_ibc.get(ch, [])
        if imgs:
            raw_cv[ch] = cross_fov_cv_raw(imgs)

    # -----------------------------------------------------------------------
    # 5. Load representative FOVs for visualization
    # -----------------------------------------------------------------------
    print("\n[5/7] Loading representative FOVs...", flush=True)
    import nd2

    fov_examples: list[tuple[str, np.ndarray, list[str]]] = []
    for well in WELL_DIRS:
        well_path = data_root / "d7" / well
        nd2_files = sorted(well_path.glob("*.nd2")) if well_path.exists() else []
        if not nd2_files:
            continue
        path = nd2_files[1]  # F2 — avoid F1 which might have the channel swap
        print(f"  loading {path.name} ({WELL_LABELS[well]})...", flush=True)
        with nd2.ND2File(path) as f:
            ch_names = [c.channel.name for c in f.metadata.channels]
            zcyx = f.asarray()
        fov_examples.append((WELL_LABELS[well], zcyx, ch_names))

    # -----------------------------------------------------------------------
    # 6. Figures
    # -----------------------------------------------------------------------
    print("\n[6/7] Generating figures...", flush=True)

    print("  field renders...", flush=True)
    plot_field_renders(fields_by_sigma, out_dir / "field_renders.png")

    print("  metrics vs sigma...", flush=True)
    plot_metrics_vs_sigma(sigma_metrics, raw_cv, out_dir / "metrics_vs_sigma.png")

    if fov_examples:
        ki_label, ki_zcyx, ki_ch_names = fov_examples[-1]
        ko_label, ko_zcyx, ko_ch_names = fov_examples[0]
        print("  KI sigma comparison...", flush=True)
        plot_fov_sigma_comparison(ki_zcyx, ki_ch_names, fields_by_sigma, "488nm",
                                  out_dir / "fov_ki_sigma_comparison.png", ki_label)
        print("  KO sigma comparison...", flush=True)
        plot_fov_sigma_comparison(ko_zcyx, ko_ch_names, fields_by_sigma, "488nm",
                                  out_dir / "fov_ko_sigma_comparison.png", ko_label)

        print("  darkfield A/B...", flush=True)
        plot_darkfield_ab(ki_zcyx, ki_ch_names, fields_no_dark, fields_with_dark, darkfields,
                          "488nm", out_dir / "darkfield_ab.png")

        print("  pooled vs per-timepoint...", flush=True)
        plot_pooled_vs_pertimepoint(ki_zcyx, ki_ch_names, fields_no_dark, tp_fields,
                                    "488nm", out_dir / "pooled_vs_pertimepoint.png")

        print("  Z-profile...", flush=True)
        plot_z_profile(ki_zcyx, ki_ch_names, fields_by_sigma[SIGMA_DEFAULT_IDX],
                       "488nm", out_dir / "z_profile.png")

        print("  well examples (before/after)...", flush=True)
        plot_well_examples(fov_examples, fields_by_sigma[SIGMA_DEFAULT_IDX],
                           ["488nm", "561nm"], out_dir / "well_examples.png")

    # -----------------------------------------------------------------------
    # 7. Report
    # -----------------------------------------------------------------------
    print("\n[7/7] Writing report...", flush=True)
    write_report(out_dir, sigma_metrics, raw_cv, darkfields, n_files, n_files_per_tp)

    print(f"\nDone → {out_dir}", flush=True)
    print("Figures generated:", flush=True)
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}", flush=True)


if __name__ == "__main__":
    main()
