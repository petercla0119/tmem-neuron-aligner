#!/usr/bin/env python
"""Color-coded temporal overlay of a well across days — see how cells move vs day 0.

Each timepoint's stable (morphology) channel is tinted a distinct color and additively
overlaid: where cells sit in the same place every day the colors sum to white; where they
drift you see colored fringes. Shown RAW (physical drift) and AFTER registration (B_pilot
masked path — cells should lock together). A third panel plots each day's estimated shift
relative to day 0 as a colored trajectory (the literal "shift between days").

Reuses the pilot ND2 loaders. Needs the [nd2] extra and the real data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from tmem_align.register import apply_shift, register_translation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_260213_longitudinal_pilot as pilot  # noqa: E402

OUT = Path("reports/alignment_comparison/real_data/day_overlays")


def norm(frame, scale):
    """Percentile-normalize a 2D frame to ~[0,1] with a scale so a few days saturate to white."""
    lo, hi = np.percentile(frame, [1, 99])
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    return np.clip(scale * (frame.astype(np.float32) - lo) / (hi - lo), 0, 1)


def composite(frames, colors):
    """Additive RGB overlay: sum(day_image * day_color), clipped."""
    rgb = np.zeros((*frames[0].shape, 3), dtype=np.float32)
    for img, c in zip(frames, colors):
        rgb += img[..., None] * np.asarray(c[:3], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--wells", nargs="+", default=["E05", "F05"])
    ap.add_argument("--max-timepoints", type=int, default=6)
    ap.add_argument("--max-sites", type=int, default=1)
    ap.add_argument("--max-read-bytes", type=int, default=2 * 1024**3)
    ap.add_argument("--stable-channel", default="488")
    ap.add_argument("--scale", type=float, default=1.4, help="brightness scale for the overlay")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    selected = pilot.select_pilot_files(args.data_root, args.wells, max_timepoints=args.max_timepoints)

    for well, files in selected.items():
        if not files:
            print(f"WARN {well}: no ND2 files — skipping")
            continue
        days = [f["day"] for f in files]
        stable = []
        for f in files:
            loaded = pilot.load_nd2_cyx(f["path"], args.max_sites, args.max_read_bytes)
            idx = pilot.choose_channel_index(loaded["channel_names"], args.stable_channel)
            stable.append(loaded["array"][idx])
        # crop to common shape if needed
        min_y = min(s.shape[0] for s in stable)
        min_x = min(s.shape[1] for s in stable)
        stable = [s[:min_y, :min_x] for s in stable]

        colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(days)))

        # register each day to day 0 on the stable channel (B_pilot masked path)
        ref = stable[0]
        shifts = [(0.0, 0.0)]
        registered = [stable[0]]
        for s in stable[1:]:
            _, (dy, dx), _ = register_translation(ref, s, robust_preprocess=False, mask_percentile=20.0)
            shifts.append((dy, dx))
            registered.append(apply_shift(s, dy, dx))

        raw_n = [norm(s, args.scale) for s in stable]
        reg_n = [norm(s, args.scale) for s in registered]

        fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
        ax[0].imshow(composite(raw_n, colors))
        ax[0].set_title(f"{well} — RAW overlay (physical drift)")
        ax[1].imshow(composite(reg_n, colors))
        ax[1].set_title(f"{well} — registered overlay (cells locked = good)")
        for a in ax[:2]:
            a.set_axis_off()

        # shift trajectory: estimated offset of each day relative to day 0 (image px)
        # recovered shift ~= -(applied drift); negate so the arrow points the way cells moved.
        xs = [-dx for _, dx in shifts]
        ys = [-dy for dy, _ in shifts]
        ax[2].plot(xs, ys, "-", color="0.6", lw=1, zorder=1)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax[2].scatter(x, y, color=colors[i], s=90, zorder=2, edgecolor="k", linewidth=0.4)
            ax[2].annotate(f"d{days[i]}", (x, y), fontsize=8, xytext=(4, 4),
                           textcoords="offset points")
        ax[2].scatter(0, 0, marker="*", s=200, color=colors[0], edgecolor="k", zorder=3)
        ax[2].set_title(f"{well} — cell shift vs day {days[0]} (px)")
        ax[2].set_xlabel("x shift (px)")
        ax[2].set_ylabel("y shift (px)")
        ax[2].invert_yaxis()  # match image coordinates (y down)
        ax[2].axhline(0, color="0.85", lw=0.5)
        ax[2].axvline(0, color="0.85", lw=0.5)
        ax[2].set_aspect("equal", adjustable="datalim")
        ax[2].grid(True, alpha=0.2)

        sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(days[0], days[-1]))
        fig.colorbar(sm, ax=ax[1], fraction=0.046, pad=0.02, label="day")

        out = OUT / f"{well}_day_overlay.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        drift = np.hypot(xs[-1], ys[-1])
        print(f"{well}: {len(days)} days {days} | net drift day{days[0]}->day{days[-1]} = "
              f"{drift:.1f} px -> {out}")


if __name__ == "__main__":
    main()
