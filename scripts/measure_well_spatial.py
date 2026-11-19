"""Measure nine spatial tiles per saved frame and render fixed-scale image examples."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

from tmem_align.analysis.spatial_comparisons import spatial_tiles

DAYS = (5, 11, 17)
GREEN = (80, 2000)
RED = (90, 600)


def scaled(a, limits):
    return np.clip((a.astype(float) - limits[0]) / (limits[1] - limits[0]), 0, 1)


def bar(ax, pixel_um, scale, size=100):
    width = size / (pixel_um * scale)
    ax.plot([20, 20 + width], [355, 355], color="white", lw=3)
    ax.text(20, 340, f"{size} µm", color="white", fontsize=7)


def examples(green, red, valid, labels, row, tiles, dest):
    h, w = green.shape
    py = row["pixel_um"]
    scale = row["scale"]
    well = row["well"]
    day = row["day"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout="constrained")
    for ax, a, limits, title in zip(
        axes[:2], [green, red], [GREEN, RED], ["mNG full field", "mCherry full field"], strict=True
    ):
        thumb = np.ma.masked_where(~valid[::8, ::8], a[::8, ::8])
        ax.imshow(
            thumb,
            cmap="gray",
            vmin=limits[0],
            vmax=limits[1],
            extent=(0, w, h, 0),
            interpolation="nearest",
        )
        for edge in (h / 3, 2 * h / 3):
            ax.axhline(edge, color="cyan", lw=0.5)
        for edge in (w / 3, 2 * w / 3):
            ax.axvline(edge, color="cyan", lw=0.5)
        ax.add_patch(
            Rectangle(
                (w // 2 - 192, h // 2 - 192), 384, 384, fill=False, edgecolor="yellow", lw=0.8
            )
        )
        ax.plot([150, 150 + 500 / (py * scale)], [h - 180, h - 180], color="white", lw=3)
        ax.text(150, h - 280, "500 µm", color="white", fontsize=7)
        ax.set_title(f"{title}\n{limits[0]}–{limits[1]} ADU", fontsize=10)
        ax.axis("off")
    y, x = h // 2 - 192, w // 2 - 192
    crop = (slice(y, y + 384), slice(x, x + 384))
    ok = valid[crop]
    for ax, a, limits, title in zip(
        axes[2:], [green, red], [GREEN, RED], ["Center mNG + masks", "Center mCherry"], strict=True
    ):
        im = scaled(a[crop], limits)
        rgb = np.repeat(im[:, :, None], 3, axis=2)
        rgb[~ok] = 0.12
        if a is green:
            rgb[find_boundaries(labels[crop], mode="inner")] = [0, 1, 1]
        ax.imshow(rgb, interpolation="nearest")
        bar(ax, py, scale)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{well} · {row['condition']} · day {day} | {int(row['nuclei_count']):,} candidates | field mCherry {row['mcherry_mean_raw']:.1f} | nuclear mNG {row['mng_mean_per_nucleus_raw']:.1f} ADU",
        fontsize=11,
    )
    fig.savefig(dest / f"{well}_d{day}_overview.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.8), layout="constrained")
    for ax, t in zip(axes.ravel(), tiles.itertuples(), strict=True):
        y = (t.y0 + t.y1) // 2 - 192
        x = (t.x0 + t.x1) // 2 - 192
        crop = (slice(y, y + 384), slice(x, x + 384))
        rgb = np.zeros((384, 384, 3))
        rgb[:, :, 0] = scaled(red[crop], RED)
        rgb[:, :, 1] = scaled(green[crop], GREEN)
        rgb[~valid[crop]] = 0.12
        rgb[find_boundaries(labels[crop], mode="inner")] = [0, 1, 1]
        ax.imshow(rgb, interpolation="nearest")
        bar(ax, py, scale)
        ax.axis("off")
        ax.set_title(
            f"R{t.tile_row + 1}C{t.tile_column + 1} · {t.nuclei_count} candidates\nTile red {t.mcherry_mean_raw:.1f} · nuclear green {t.mng_mean_per_nucleus_raw:.1f}",
            fontsize=9,
        )
    fig.suptitle(
        f"{well} · day {day} · nine fixed-location crops\nGreen 80–2000 ADU · red 90–600 ADU · cyan nuclear outlines\nNumbers summarize entire tiles, not just displayed crops",
        fontsize=11,
    )
    fig.savefig(dest / f"{well}_d{day}_regions.png", dpi=125)
    plt.close(fig)
    return {
        "green_display_above_max_fraction": float(np.mean(green[valid] > GREEN[1])),
        "red_display_above_max_fraction": float(np.mean(red[valid] > RED[1])),
        "green_display_below_min_fraction": float(np.mean(green[valid] < GREEN[0])),
        "red_display_below_min_fraction": float(np.mean(red[valid] < RED[0])),
    }


def process(task):
    root, out, row, codehash = task
    root = Path(root)
    out = Path(out)
    well, day = row["well"], int(row["day"])
    key = f"{well}_d{day}"
    wd = root / "wells" / well
    paths = [
        wd / "preprocessed" / f"{key}.npz",
        wd / "frames" / f"{key}_labels.npz",
        wd / "frames" / f"{key}_nuclei.csv",
    ]
    signature = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    cache = out / "frames" / f"{key}.json"
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored["input_signature"] == signature and stored["code_sha256"] == codehash:
            if day not in DAYS or all(
                (out / "images" / f"{key}_{s}.png").exists() for s in ["overview", "regions"]
            ):
                return stored
    with np.load(paths[0]) as z:
        green, red, valid = z["green"], z["red"], z["valid"]
    with np.load(paths[1]) as z:
        labels = z["labels"]
    cells = pd.read_csv(paths[2])
    tiles = spatial_tiles(green, red, valid, labels, cells, row["pixel_um"], row["scale"])
    assert tiles.nuclei_count.sum() == row["nuclei_count"], key
    assert np.isclose(tiles.valid_area_mm2.sum(), row["area_mm2"], rtol=1e-10), key
    assert np.isclose(
        np.average(tiles.mcherry_mean_raw, weights=tiles.valid_pixels),
        row["mcherry_mean_raw"],
        rtol=1e-7,
    ), key
    occupied = tiles[tiles.nuclei_count > 0]
    assert np.isclose(
        np.average(occupied.mng_mean_per_nucleus_raw, weights=occupied.nuclei_count),
        row["mng_mean_per_nucleus_raw"],
        rtol=1e-10,
    ), key
    display = examples(green, red, valid, labels, row, tiles, out / "images") if day in DAYS else {}
    tiles["well"] = well
    tiles["day"] = day
    tiles["condition"] = row["condition"]
    tiles["plate_block"] = row["plate_block"]
    result = {
        "input_signature": signature,
        "code_sha256": codehash,
        "well": well,
        "day": day,
        "conservation": "pass",
        "tiles": tiles.to_dict("records"),
        "display": display,
    }
    temp = cache.with_suffix(".tmp")
    temp.write_text(json.dumps(result))
    temp.replace(cache)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = a.output / "comparisons"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    source = a.output / "longitudinal_summary_metrics.csv"
    sourcehash = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source)
    assert len(df) == 1248 and not df.duplicated(["well", "day"]).any()
    codehash = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__)
        .parents[1]
        .joinpath("src/tmem_align/analysis/spatial_comparisons.py")
        .read_bytes()
    ).hexdigest()
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [
            pool.submit(process, (str(a.output), str(out), r, codehash))
            for r in df.to_dict("records")
        ]
        for i, f in enumerate(as_completed(futures), 1):
            records.append(f.result())
            if i % 48 == 0:
                print(f"Spatial frames verified: {i}/1248", flush=True)
    tiles = pd.DataFrame([t for r in records for t in r["tiles"]]).sort_values(
        ["well", "day", "tile"]
    )
    assert len(tiles) == 11232 and not tiles.duplicated(["well", "day", "tile"]).any()
    tiles.to_csv(out / "spatial_tile_metrics.csv", index=False)
    displays = pd.DataFrame(
        [{"well": r["well"], "day": r["day"], **r["display"]} for r in records if r["display"]]
    ).sort_values(["well", "day"])
    assert len(displays) == 288
    displays.to_csv(out / "image_display_clipping.csv", index=False)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == sourcehash
    (out / "measurement_provenance.json").write_text(
        json.dumps(
            {
                "frames": len(records),
                "tiles": len(tiles),
                "image_example_frames": len(displays),
                "input_csv_sha256": sourcehash,
                "code_sha256": codehash,
                "conservation_checks": "1248 passed",
                "grid": "3x3 equal coordinate tiles; nuclei assigned once by centroid; whole-mask green mean",
                "green_display_adu": GREEN,
                "red_display_adu": RED,
                "crop_size_reduced_pixels": 384,
                "spatial_alignment": "none; same coordinates do not imply same cells",
            },
            indent=2,
        )
    )
    print("Spatial measurements and all-well image examples complete.", flush=True)


if __name__ == "__main__":
    main()
