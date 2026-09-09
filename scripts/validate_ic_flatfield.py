#!/usr/bin/env python
"""Validate the retrospective IC/flat-field (σ=102) against the existing FOV cache.

Two questions, no calibration slide required (imaging is done — see the IC vault
note, 2026-09-02):

  A. PUNCTA PRESERVATION — does dividing by the smooth field distort the puncta
     being quantified? Metric: per-pixel LOCAL CONTRAST = img / gaussian_blur(img,
     local_sigma). A punctum's contrast against its immediate background. Under a
     field that is smooth at the punctum scale, local contrast is invariant to the
     correction (numerator and denominator scale by ~the same factor). We compare
     local contrast RAW vs CORRECTED on the puncta channel's brightest pixels and
     report the median fractional change. Near 0 ⇒ σ=102 does NOT distort puncta.

  B. CENTER/EDGE SHADING — does the field remove a real vignette without
     inverting/adding structure? Metric: robust center-region level / edge-region
     level of a densely-filling channel, RAW vs CORRECTED. Raw > ~1.05–1.10 means
     correction is warranted; corrected should move toward 1.0, never overshoot < 1
     or invert. (Somewhat self-confirming since the field encodes this gradient, so
     it's a sanity/direction check — test A is the load-bearing one for the worry.)

Runs on the existing FOV cache = the ND2s under the dataset root, read via
tmem_align.analysis.if_spatial.load_fov (raw = ic_fields=None; corrected = the
pre-computed .npz). No worktree / engine import needed — load_fov and
load_ic_fields are the application side, present on this branch.

Usage (worktrees carry no venv — use the main tree's interpreter):
    PYTHONPATH=src .venv/bin/python scripts/validate_ic_flatfield.py
    PYTHONPATH=src .venv/bin/python scripts/validate_ic_flatfield.py --self-check
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

DATA_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
IC_NPZ = Path("/Users/pmihack/claire/tmem_2026/data/ic_fields_260821_pooled.npz")
OUT_DIR = Path("reports/ic_validation")

TIMEPOINTS = ["d7", "d14", "d28"]
# Filenames encode channel biology: TMEM561 LAMP1640 MAP2488 DAPI405.
# Both punctate channels are quantified in the coloc, so test both.
PUNCTA_CHANNELS = ["561nm", "640nm"]  # cl-TMEM, LAMP1
EPS = 1e-6


# --- metrics ---------------------------------------------------------------


def local_contrast(img: np.ndarray, local_sigma: float) -> np.ndarray:
    """img / blur(img). Invariant to any gain field smooth relative to local_sigma."""
    f = img.astype(np.float32)
    return f / (gaussian_filter(f, sigma=local_sigma) + EPS)


def puncta_preservation(
    raw: np.ndarray, corr: np.ndarray, local_sigma: float, pct: float
) -> dict[str, float]:
    """Compare local contrast raw vs corrected on the brightest (punctum) pixels.

    Pixel set is fixed from the RAW image so both are evaluated at identical
    locations. Returns median |Δ| fractional change and Pearson r of the contrasts.
    """
    lc_raw = local_contrast(raw, local_sigma)
    lc_corr = local_contrast(corr, local_sigma)
    mask = raw >= np.percentile(raw, pct)
    a, b = lc_raw[mask], lc_corr[mask]
    frac = np.abs(b - a) / (np.abs(a) + EPS)
    r = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    return {
        "n_px": int(mask.sum()),
        "median_frac_change": float(np.median(frac)),
        "p95_frac_change": float(np.percentile(frac, 95)),
        "pearson_r": r,
    }


def strip_dark(ic_fields):
    """flat+dark IC dict → flat-only (darkfield zeroed)."""
    return {k: (v[0] if isinstance(v, tuple) else v) for k, v in ic_fields.items()}


def center_edge_ratio(img: np.ndarray, center_frac: float, edge_band: float) -> float:
    """Robust (median) center level / edge level. Vignette ⇒ > 1 (bright center)."""
    h, w = img.shape
    ch, cw = int(h * center_frac), int(w * center_frac)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    center = img[r0 : r0 + ch, c0 : c0 + cw]

    bh, bw = int(h * edge_band), int(w * edge_band)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:bh], edge_mask[-bh:], edge_mask[:, :bw], edge_mask[:, -bw:] = (True,) * 4

    edge_level = float(np.median(img[edge_mask]))
    return float(np.median(center)) / (edge_level + EPS)


# --- driver ----------------------------------------------------------------


def sample_fovs(root: Path, timepoints: list[str], n: int, seed: int) -> dict[str, list[Path]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[Path]] = {}
    for tp in timepoints:
        files = sorted((root / tp).rglob("*.nd2"))  # ND2s live in per-well subdirs
        if not files:
            print(f"  ! no .nd2 under {root / tp}", file=sys.stderr)
            continue
        take = files if len(files) <= n else [files[i] for i in rng.choice(len(files), n, replace=False)]
        out[tp] = sorted(take)
    return out


def run(args: argparse.Namespace) -> None:
    from tmem_align.analysis.if_spatial import load_fov, load_ic_fields

    ic_fields = load_ic_fields(args.ic_npz)
    if args.flat_only:
        # Strip darkfield to isolate the σ-smoothing effect from dark subtraction.
        ic_fields = strip_dark(ic_fields)
        print("  (flat-only: darkfield zeroed)")
    fovs = sample_fovs(args.data_root, args.timepoints, args.n_fovs, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for tp, paths in fovs.items():
        for p in paths:
            raw = load_fov(p, ic_fields=None)
            corr = load_fov(p, ic_fields=ic_fields)

            for pch in args.puncta_channels:
                if pch in raw:
                    pp = puncta_preservation(
                        raw[pch], corr[pch], args.local_sigma, args.puncta_pct,
                    )
                    rows.append({"timepoint": tp, "fov": p.name, "test": "puncta",
                                 "channel": pch, **pp})

            for ch in raw:
                rows.append({
                    "timepoint": tp, "fov": p.name, "test": "center_edge", "channel": ch,
                    "ratio_raw": center_edge_ratio(raw[ch], args.center_frac, args.edge_band),
                    "ratio_corr": center_edge_ratio(corr[ch], args.center_frac, args.edge_band),
                })
            print(f"  {tp}/{p.name} done")

    csv_path = args.out / "ic_validation.csv"
    fields = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    _summary(rows)
    print(f"\nWrote {csv_path}  ({len(rows)} rows)")


def _summary(rows: list[dict]) -> None:
    pun = [r for r in rows if r["test"] == "puncta"]
    if pun:
        print("\n[A] Puncta local-contrast median fractional change (per channel):")
        print("    (<0.02 ⇒ σ field does not distort puncta contrast)")
        for ch in sorted({r["channel"] for r in pun}):
            med = float(np.median([r["median_frac_change"] for r in pun if r["channel"] == ch]))
            verdict = "PRESERVED" if med < 0.02 else "CHECK — >2% shift"
            print(f"    {ch:>6}: {med:.4f}  → {verdict}")

    ce = [r for r in rows if r["test"] == "center_edge"]
    if ce:
        print("\n[B] Center/edge ratio (median across FOVs), per channel — want raw→corr toward 1.0:")
        for ch in sorted({r["channel"] for r in ce}):
            raw = np.median([r["ratio_raw"] for r in ce if r["channel"] == ch])
            corr = np.median([r["ratio_corr"] for r in ce if r["channel"] == ch])
            flag = " ⚠ INVERTED/overshoot" if corr < 0.95 else ""
            print(f"    {ch:>6}: raw {raw:.3f} → corr {corr:.3f}{flag}")


def _find_ki_fov(root: Path, timepoints: list[str]) -> Path:
    """First PLD3TMEM106B (KI) FOV — has real cl-TMEM puncta for the visual check."""
    for tp in timepoints:
        hits = sorted((root / tp).rglob("PLD3TMEM106B*.nd2"))
        if hits:
            return hits[0]
    any_nd2 = next((root / timepoints[0]).rglob("*.nd2"), None)
    if any_nd2 is None:
        raise FileNotFoundError(f"no .nd2 under {root}/{timepoints[0]}")
    return any_nd2


def make_figures(args: argparse.Namespace) -> None:
    """Visual QC: raw vs flat-only vs flat+dark, full FOV + zoomed center/edge crops.

    Same fixed per-channel DISPLAY_LUT across all three conditions (honest — a
    correction that changed the picture would show; renormalizing each panel would
    hide it). Full FOV shows de-vignetting; crops show puncta contrast is preserved.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import ImageGrid

    from tmem_align.analysis import if_spatial as ifs
    from tmem_align.analysis.if_spatial import apply_display_lut, load_fov, load_ic_fields

    full = load_ic_fields(args.ic_npz)
    flat = strip_dark(full)
    fov = Path(args.fig_fov) if args.fig_fov else _find_ki_fov(args.data_root, args.timepoints)
    print(f"  figures from {fov.name}")

    raw_d = load_fov(fov, ic_fields=None)
    flat_d = load_fov(fov, ic_fields=flat)
    full_d = load_fov(fov, ic_fields=full)
    conds = [("raw", raw_d), ("flat-only", flat_d), ("flat+dark (full)", full_d)]
    args.out.mkdir(parents=True, exist_ok=True)

    # crop windows: center + worst-vignette corner (5% inset)
    h, w = raw_d[ifs.CH_MAP2].shape
    c = args.crop_px // 2
    cen = (slice(h // 2 - c, h // 2 + c), slice(w // 2 - c, w // 2 + c))
    m = int(0.05 * min(h, w))
    edge = (slice(m, m + args.crop_px), slice(m, m + args.crop_px))

    def _panel(ax, img, ch):
        ax.imshow(apply_display_lut(img, ch), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Figure 1: full FOV, rows = channels, cols = conditions ---
    fig_channels = [(ifs.CH_MAP2, "MAP2 488 (vignette)"), (ifs.CH_TMEM, "cl-TMEM 561"),
                    (ifs.CH_LAMP1, "LAMP1 640")]
    fig = plt.figure(figsize=(9, 9))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(fig_channels), 3), axes_pad=0.06)
    for ri, (ch, label) in enumerate(fig_channels):
        for ci, (cname, d) in enumerate(conds):
            ax = grid[ri * 3 + ci]
            _panel(ax, d[ch], ch)
            ce = center_edge_ratio(d[ch], args.center_frac, args.edge_band)
            ax.text(0.02, 0.98, f"C/E {ce:.3f}", color="yellow", fontsize=8,
                    va="top", ha="left", transform=ax.transAxes)
            if ri == 0:
                ax.set_title(cname, fontsize=10)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
                for (ys, xs), col in ((cen, "cyan"), (edge, "magenta")):
                    ax.add_patch(Rectangle((xs.start, ys.start), args.crop_px, args.crop_px,
                                           ec=col, fc="none", lw=1.2))
    fig.suptitle(f"IC full-FOV QC — {fov.name}\ncyan=center crop  magenta=edge crop  "
                 "(same fixed LUT across columns)", fontsize=9)
    p1 = args.out / "ic_qc_full_fov.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: zoomed crops, rows = {puncta ch}×{center,edge}, cols = conditions ---
    rows = [(ifs.CH_TMEM, "cl-TMEM 561", cen, "center"), (ifs.CH_TMEM, "cl-TMEM 561", edge, "edge"),
            (ifs.CH_LAMP1, "LAMP1 640", cen, "center"), (ifs.CH_LAMP1, "LAMP1 640", edge, "edge")]
    fig = plt.figure(figsize=(9, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(rows), 3), axes_pad=0.06)
    for ri, (ch, label, win, where) in enumerate(rows):
        for ci, (cname, d) in enumerate(conds):
            ax = grid[ri * 3 + ci]
            _panel(ax, d[ch][win], ch)
            if ri == 0:
                ax.set_title(cname, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{label}\n{where}", fontsize=9)
    fig.suptitle(f"IC puncta-crop QC ({args.crop_px}px) — {fov.name}\n"
                 "flat-only should preserve puncta; edge row shows de-vignette lift", fontsize=9)
    p2 = args.out / "ic_qc_puncta_crops.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p1}\n  wrote {p2}")


def self_check() -> None:
    """Synthetic: puncta on a known vignette. Correcting must preserve local
    contrast (test A) and flatten center/edge (test B). No ND2 files needed."""
    rng = np.random.default_rng(0)
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]
    # smooth vignette: bright center, ~30% dimmer corners (σ-102-like, no fine structure)
    r2 = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (h / 2) ** 2
    field = 1.3 - 0.5 * r2
    clean = np.full((h, w), 100.0)
    for cy, cx in rng.integers(20, h - 20, size=(60, 2)):  # bright puncta
        clean[cy - 1 : cy + 2, cx - 1 : cx + 2] += 4000.0
    raw = (clean * field).astype(np.float32)
    corr = raw / field  # perfect correction

    pp = puncta_preservation(raw, corr, local_sigma=5.0, pct=99.0)
    assert pp["median_frac_change"] < 0.02, pp  # smooth field ⇒ contrast preserved
    r_before = center_edge_ratio(raw, 0.4, 0.15)
    r_after = center_edge_ratio(corr, 0.4, 0.15)
    assert r_before > 1.1, r_before  # vignette present
    assert abs(r_after - 1.0) < 0.05, r_after  # flattened
    print(f"self-check OK: puncta Δ={pp['median_frac_change']:.4f}, "
          f"center/edge {r_before:.3f}→{r_after:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true", help="run synthetic assert check, no ND2 needed")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--ic-npz", type=Path, default=IC_NPZ)
    ap.add_argument("--timepoints", nargs="+", default=TIMEPOINTS)
    ap.add_argument("--puncta-channels", nargs="+", default=PUNCTA_CHANNELS)
    ap.add_argument("--n-fovs", type=int, default=8, help="FOVs sampled per timepoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-sigma", type=float, default=5.0, help="px; punctum-scale blur for local contrast")
    ap.add_argument("--puncta-pct", type=float, default=99.0, help="brightest-pixel percentile = punctum set")
    ap.add_argument("--center-frac", type=float, default=0.4)
    ap.add_argument("--edge-band", type=float, default=0.15)
    ap.add_argument("--flat-only", action="store_true",
                    help="zero the darkfield to isolate the σ-smoothing effect on puncta")
    ap.add_argument("--figures", action="store_true",
                    help="emit raw/flat/full QC PNGs (full FOV + center/edge puncta crops)")
    ap.add_argument("--fig-fov", type=Path, default=None, help="specific ND2 for --figures")
    ap.add_argument("--crop-px", type=int, default=256, help="zoom crop size for --figures")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return
    if args.figures:
        make_figures(args)
        return
    run(args)


if __name__ == "__main__":
    main()
