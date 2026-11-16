"""Reproducible full-field Cellpose-SAM counts and quantitative mCherry trends."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tmem-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
from cellpose import models
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from tmem_align.analysis.longitudinal_counts import (
    add_baseline_changes,
    prepare_channels,
    quantify_frame,
    segment_nuclei,
)

PARAMETERS = {
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "normalize": True,
    "min_size": 15,
    "batch_size": 32,
}
PATTERN = re.compile(r"_d(\d+)_.*_Well([A-Z]\d{2})_")


def inventory(root, wells, days):
    rows = []
    for p in sorted(root.rglob("*.nd2")):
        match = PATTERN.search(p.name)
        if not match or match[2] not in wells or int(match[1]) not in days:
            continue
        with nd2.ND2File(p) as f:
            channels = [c.channel.name for c in f.metadata.channels]
            if channels != ["488nm", "561nm"] or list(f.sizes) != ["C", "Y", "X"]:
                raise ValueError(f"Unexpected channels/axes: {p}: {channels}, {f.sizes}")
            ev = f.events()[0]
            julian = f.frame_metadata(0).channels[0].time.absoluteJulianDayNumber
            stamp = (datetime(1858, 11, 17) + timedelta(days=julian - 2400000.5)).isoformat(
                timespec="seconds"
            )
            rows.append(
                {
                    "well": match[2],
                    "day": int(match[1]),
                    "path": str(p),
                    "timestamp": stamp,
                    "height": f.sizes["Y"],
                    "width": f.sizes["X"],
                    "pixel_um": f.voxel_size().x,
                    "dtype": str(f.dtype),
                    "channels": ";".join(channels),
                    "significant_bits": f.attributes.bitsPerComponentSignificant,
                    "exposure_488_ms": ev.get("Exposure Time(488nm) [ms]"),
                    "exposure_561_ms": ev.get("Exposure Time(561nm) [ms]"),
                    "pfs_status": ev.get("PFS Status"),
                    "size_bytes": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No matching ND2 files")
    if df.duplicated(["well", "day"]).any():
        raise ValueError("Duplicate well/day acquisitions")
    missing = {(w, d) for w in wells for d in days} - set(zip(df.well, df.day))
    if missing:
        raise ValueError(f"Missing well/day acquisitions: {sorted(missing)}")
    if df[["exposure_488_ms", "exposure_561_ms"]].isna().any().any():
        raise ValueError("Missing exposure metadata")
    if df[["exposure_488_ms", "exposure_561_ms"]].nunique().max() != 1:
        raise ValueError("Exposure settings changed; review before intensity comparison")
    return df.sort_values(["day", "well"]).reset_index(drop=True)


def display(a, limits=None):
    lo, hi = limits if limits else np.percentile(a, [1, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1), 0, 1)


def overlay(green, red, labels, row, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(display(green), cmap="gray")
    axs[0].set_title("Complete imaged field · mNG")
    y, x = np.array(green.shape) // 2
    r = min(220, y, x)
    sl = np.s_[y - r : y + r, x - r : x + r]
    g = display(green[sl])
    rgb = np.repeat(g[..., None], 3, axis=-1)
    rgb[find_boundaries(labels[sl])] = [0, 1, 1]
    axs[1].imshow(rgb)
    axs[1].set_title("Center detail · nuclear outlines")
    axs[2].imshow(red[sl], cmap="magma", vmin=90, vmax=600)
    axs[2].set_title("mCherry · fixed display 90–600 ADU")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(
        f"{row['well']} · day {row['day']} · {row['nuclei_count']:,} nuclei · "
        f"{row['area_mm2']:.2f} mm² sampled",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=135)
    plt.close(fig)


def plots(df, out):
    colors = dict(zip(sorted(df.well.unique()), plt.get_cmap("tab10").colors))
    panels = [
        ("nuclei_count", "Detected nuclei", "nuclei"),
        ("nuclei_count_fold_baseline", "Nuclei / day 5", "fold"),
        ("mcherry_mean_raw", "mCherry mean · raw", "ADU"),
        ("mcherry_mean_bg_corrected", "mCherry mean · background corrected", "ADU"),
        (
            "mcherry_integrated_per_nucleus",
            "Total corrected mCherry / nucleus",
            "ADU·pixels / nucleus",
        ),
        ("mcherry_mean_bg_corrected_fold_baseline", "Corrected mCherry / day 5", "fold"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, title, unit) in zip(axs.flat, panels):
        for well, group in df.groupby("well"):
            ax.plot(group.day, group[key], "-o", ms=3, lw=1.7, label=well, color=colors[well])
        ax.set(title=title, xlabel="Experimental day", ylabel=unit)
        ax.grid(alpha=0.2)
        ax.spines[["right", "top"]].set_visible(False)
        if "fold" in key:
            ax.axhline(1, color="gray", ls=":", lw=1)
    axs[0, 0].legend(ncol=2, frameon=False)
    fig.suptitle(
        f"{df.well.nunique()}-well longitudinal analysis · Cellpose-SAM\nComplete acquired fields; counts are not tracked survival",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "longitudinal_summary.png", dpi=180)
    fig.savefig(out / "longitudinal_summary.pdf")
    plt.close(fig)
    for groupname, keys in [("nuclei_counts", panels[:2]), ("mcherry_signal", panels[2:])]:
        fig, axs = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.6), squeeze=False)
        for ax, (key, title, unit) in zip(axs.flat, keys):
            for well, g in df.groupby("well"):
                ax.plot(g.day, g[key], "-o", label=well, color=colors[well], ms=4)
            ax.set(title=title, xlabel="Experimental day", ylabel=unit)
            ax.grid(alpha=0.2)
            ax.spines[["right", "top"]].set_visible(False)
        axs[0, 0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{groupname}.png", dpi=160)
        fig.savefig(out / f"{groupname}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wells", nargs="+", default=["E04", "E06", "F05", "F08", "F10"])
    ap.add_argument("--days", type=int, nargs="+", default=list(range(5, 18)))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for d in ("frames", "qc", "preprocessed"):
        (out / d).mkdir(exist_ok=True)
    weights = Path.home() / ".cellpose/models/cpsam"
    config = {
        "input": str(args.input.resolve()),
        "wells": args.wells,
        "days": args.days,
        "scale": args.scale,
        "model": "cpsam",
        "weights_sha256": hashlib.file_digest(weights.open("rb"), "sha256").hexdigest(),
        "parameters": PARAMETERS,
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["cellpose", "torch", "nd2", "numpy", "scikit-image"]
        },
        "measurement": "full acquired field; valid nonzero blocks; original intensity units",
        "background": "10th percentile of block-averaged valid red pixels",
        "raw_source_policy": "read-only",
        "scope": "first-pass population counts, not tracking",
    }
    cfg = out / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("Configuration differs from cache; choose a new output directory")
    cfg.write_text(json.dumps(config, indent=2))
    inv = inventory(args.input, args.wells, args.days)
    inv.to_csv(out / "image_inventory.csv", index=False)
    print(f"Inventory: {len(inv)} frames; scale {args.scale}", flush=True)
    model = models.CellposeModel(gpu=not args.cpu, pretrained_model=str(weights))
    if not args.cpu and model.device.type == "cpu":
        raise RuntimeError("GPU requested but unavailable")
    (out / "device.txt").write_text(str(model.device) + "\n")
    selected = inv.iloc[: args.limit] if args.limit else inv
    for index, row in selected.iterrows():
        key = f"{row.well}_d{row.day}"
        result = out / "frames" / f"{key}.json"
        artifacts = [
            out / "preprocessed" / f"{key}.npz",
            out / "frames" / f"{key}_labels.npz",
            out / "frames" / f"{key}_nuclei.csv",
            out / "qc" / f"{key}.png",
        ]
        if result.exists() and all(p.exists() for p in artifacts):
            cached = json.loads(result.read_text())
            if (
                cached["source_size_bytes"] != row.size_bytes
                or cached["source_mtime_ns"] != row.mtime_ns
            ):
                raise ValueError(f"Source file changed: {row.path}")
            print(f"{key}: cached", flush=True)
            continue
        start = time.monotonic()
        with nd2.ND2File(row.path) as f:
            raw = f.asarray()
        green, red, valid = prepare_channels(raw, args.scale)
        h, w = green.shape
        trim = raw[:, : h * args.scale, : w * args.scale]
        saturation = float(np.mean(trim[1] >= 2**row.significant_bits - 1))
        del raw, trim
        masks = segment_nuclei(model, green, PARAMETERS)
        masks[~valid] = 0
        # Retain all model components; boundary counts are reported separately.
        bg = float(np.percentile(red[valid], 10))
        metrics, cells = quantify_frame(red, masks, valid, row.pixel_um, args.scale, bg)
        metrics.update(
            well=row.well,
            day=int(row.day),
            timestamp=row.timestamp,
            source_file=row.path,
            source_size_bytes=int(row.size_bytes),
            source_mtime_ns=int(row.mtime_ns),
            scale=args.scale,
            pixel_um=float(row.pixel_um),
            trimmed_height=h * args.scale,
            trimmed_width=w * args.scale,
            exposure_488_ms=float(row.exposure_488_ms),
            exposure_561_ms=float(row.exposure_561_ms),
            pfs_status=int(row.pfs_status),
            mcherry_saturated_fraction=saturation,
            green_focus_laplacian_variance=float(ndi.laplace(green).var()),
            valid_fraction=float(valid.mean()),
            qc_status="automated_not_manually_adjudicated",
        )
        # Float32 block means are lossless for uint16 block sums divided by a power of two.
        np.savez_compressed(out / "preprocessed" / f"{key}.npz", green=green, red=red, valid=valid)
        np.savez_compressed(out / "frames" / f"{key}_labels.npz", labels=masks)
        pd.DataFrame(
            cells,
            columns=[
                "label",
                "centroid_y_px",
                "centroid_x_px",
                "area_um2",
                "nuclear_mcherry_mean_raw",
                "nuclear_mcherry_mean_bg_corrected",
                "touches_field_boundary",
            ],
        ).to_csv(out / "frames" / f"{key}_nuclei.csv", index=False)
        overlay(green, red, masks, metrics, out / "qc" / f"{key}.png")
        metrics["seconds"] = time.monotonic() - start
        result.write_text(json.dumps(metrics, indent=2, allow_nan=True))
        print(
            f"{index + 1}/{len(inv)} {key}: {metrics['nuclei_count']} nuclei, "
            f"red {metrics['mcherry_mean_raw']:.2f}, {metrics['seconds']:.1f}s",
            flush=True,
        )
        # Rebuild only from this run's expected frames, never from arbitrary folder contents.
        rows = [
            json.loads((out / "frames" / f"{r.well}_d{r.day}.json").read_text())
            for r in inv.itertuples()
            if (out / "frames" / f"{r.well}_d{r.day}.json").exists()
        ]
        df = add_baseline_changes(pd.DataFrame(rows))
        df.to_csv(out / "well_day_metrics.csv", index=False)
        plots(df, out)
    rows = [
        json.loads((out / "frames" / f"{r.well}_d{r.day}.json").read_text())
        for r in inv.itertuples()
        if (out / "frames" / f"{r.well}_d{r.day}.json").exists()
    ]
    df = add_baseline_changes(pd.DataFrame(rows))
    df.to_csv(out / "well_day_metrics.csv", index=False)
    plots(df, out)
    print(f"FINISHED: {len(df)}/{len(inv)} frames in {out}", flush=True)


if __name__ == "__main__":
    main()
