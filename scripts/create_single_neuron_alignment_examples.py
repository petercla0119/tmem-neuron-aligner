#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tif
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage import filters, measure, morphology
from skimage.registration import phase_cross_correlation

from tmem_align.analysis.mcherry_metrics import quantify_mcherry_timeseries
from tmem_align.register import apply_shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create single-neuron-focused time-series examples from registered pilot stacks."
    )
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--wells", nargs="+", default=["E05", "F05"])
    parser.add_argument("--days", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--alignment-channel", type=int, default=2)
    parser.add_argument("--mcherry-channel", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--duration-ms", type=int, default=950)
    parser.add_argument("--max-local-shift", type=float, default=35.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.report_root / "single_neuron_examples"
    figures = out_dir / "figures"
    stacks_dir = out_dir / "registered_roi_stacks"
    figures.mkdir(parents=True, exist_ok=True)
    stacks_dir.mkdir(parents=True, exist_ok=True)

    roi_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    metric_tables: list[pd.DataFrame] = []
    roi_stacks: dict[str, np.ndarray] = {}

    for well in args.wells:
        stack_path = args.report_root / "registered_stacks" / f"{well}_registered_common_overlap_tcyx.ome.tif"
        if not stack_path.exists():
            print(f"Skipping {well}; missing {stack_path}")
            continue
        stack = np.asarray(tif.imread(stack_path))
        roi = select_stable_single_neuron_roi(
            stack,
            alignment_channel=args.alignment_channel,
            mcherry_channel=args.mcherry_channel,
            crop_size=args.crop_size,
        )
        local_stack, local_shifts = local_register_roi_stack(
            stack,
            roi,
            alignment_channel=args.alignment_channel,
            max_local_shift=args.max_local_shift,
        )
        roi_stacks[well] = local_stack
        tif.imwrite(
            stacks_dir / f"{well}_single_neuron_registered_tcyx.ome.tif",
            local_stack,
            photometric="minisblack",
            metadata={"axes": "TCYX"},
            ome=True,
        )

        roi_rows.append({"well": well, **roi})
        for time_index, shift in enumerate(local_shifts):
            shift_rows.append(
                {
                    "well": well,
                    "time_index": time_index,
                    "day": args.days[time_index] if time_index < len(args.days) else time_index,
                    "local_y_shift": shift[0],
                    "local_x_shift": shift[1],
                    "local_shift_pixels": float(np.hypot(*shift)),
                    "local_qc_pass": bool(np.hypot(*shift) <= args.max_local_shift),
                }
            )

        metadata_rows = [
            {
                "well": well,
                "timepoint_day": args.days[index] if index < len(args.days) else index,
                "cell_roi_id": f"{well}_single_neuron_candidate",
                "site_fov": "site0_single_neuron_crop",
            }
            for index in range(local_stack.shape[0])
        ]
        metrics = quantify_mcherry_timeseries(
            local_stack[:, args.mcherry_channel],
            mask_stack=local_stack[:, args.alignment_channel],
            metadata_rows=metadata_rows,
        )
        metric_tables.append(metrics)

        write_single_well_montage(
            local_stack,
            well=well,
            days=args.days,
            mcherry_channel=args.mcherry_channel,
            alignment_channel=args.alignment_channel,
            path=figures / f"{well}_single_neuron_alignment_montage.png",
        )
        write_single_well_gifs(
            local_stack,
            well=well,
            days=args.days,
            mcherry_channel=args.mcherry_channel,
            alignment_channel=args.alignment_channel,
            figures=figures,
            duration_ms=args.duration_ms,
        )

    pd.DataFrame(roi_rows).to_csv(out_dir / "single_neuron_roi_selection.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(out_dir / "single_neuron_local_registration_qc.csv", index=False)
    if metric_tables:
        metrics = pd.concat(metric_tables, ignore_index=True)
        metrics.to_csv(out_dir / "single_neuron_mcherry_metrics.csv", index=False)
        write_metric_plot(metrics, figures / "single_neuron_mcherry_metric_over_time.png")
    if len(roi_stacks) >= 2:
        write_side_by_side_gif(
            roi_stacks,
            days=args.days,
            mcherry_channel=args.mcherry_channel,
            path=figures / "E05_vs_F05_single_neuron_mcherry.gif",
            duration_ms=args.duration_ms,
        )
        write_side_by_side_montage(
            roi_stacks,
            days=args.days,
            mcherry_channel=args.mcherry_channel,
            path=figures / "E05_vs_F05_single_neuron_mcherry_montage.png",
        )

    write_readme(out_dir, args, roi_rows)
    print(f"Wrote single-neuron examples under {out_dir}")


def select_stable_single_neuron_roi(
    stack: np.ndarray,
    *,
    alignment_channel: int,
    mcherry_channel: int,
    crop_size: int,
) -> dict[str, Any]:
    components_by_time = [
        detect_compact_components(stack[time_index, alignment_channel])
        for time_index in range(stack.shape[0])
    ]
    if not components_by_time or any(not components for components in components_by_time):
        return select_single_timepoint_roi(
            stack[0, alignment_channel],
            stack[0, mcherry_channel],
            crop_size=crop_size,
        )

    scored = []
    for first in components_by_time[0]:
        matched = [first]
        distances = []
        for components in components_by_time[1:]:
            nearest = min(
                components,
                key=lambda component: np.hypot(
                    component["centroid_y"] - first["centroid_y"],
                    component["centroid_x"] - first["centroid_x"],
                ),
            )
            distance = float(
                np.hypot(
                    nearest["centroid_y"] - first["centroid_y"],
                    nearest["centroid_x"] - first["centroid_x"],
                )
            )
            matched.append(nearest)
            distances.append(distance)

        max_distance = max(distances) if distances else 0.0
        mean_x = float(np.mean([component["centroid_x"] for component in matched]))
        mean_y = float(np.mean([component["centroid_y"] for component in matched]))
        height, width = stack.shape[-2:]
        margin = crop_size // 2 + 40
        if mean_y < margin or mean_x < margin or mean_y > height - margin or mean_x > width - margin:
            continue
        mcherry_values = [
            float(stack[index, mcherry_channel, int(component["centroid_y"]), int(component["centroid_x"])])
            for index, component in enumerate(matched)
        ]
        score = -2.0 * max_distance + 0.05 * float(np.mean([c["area"] for c in matched])) + float(np.mean(mcherry_values))
        scored.append((score, max_distance, matched, mean_y, mean_x))

    if not scored:
        return select_single_timepoint_roi(
            stack[0, alignment_channel],
            stack[0, mcherry_channel],
            crop_size=crop_size,
        )

    _, max_distance, matched, cy, cx = max(scored, key=lambda item: item[0])
    height, width = stack.shape[-2:]
    x_start = int(round(cx - crop_size / 2))
    y_start = int(round(cy - crop_size / 2))
    x_start = max(0, min(x_start, width - crop_size))
    y_start = max(0, min(y_start, height - crop_size))
    return {
        "roi_id": "single_neuron_candidate_stable_auto",
        "x_start": x_start,
        "y_start": y_start,
        "width": crop_size,
        "height": crop_size,
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "component_area_pixels": int(np.mean([component["area"] for component in matched])),
        "max_centroid_distance_pixels": float(max_distance),
        "selection_channel": "488",
        "selection_note": (
            "automatic compact foreground component selected for nearest centroid consistency "
            "across aligned days; manual same-neuron review recommended"
        ),
    }


def detect_compact_components(alignment_frame: np.ndarray) -> list[dict[str, Any]]:
    align = alignment_frame.astype(np.float32)
    lo, hi = np.percentile(align, [2, 99.5])
    normalized = np.clip((align - lo) / max(hi - lo, 1), 0, 1)
    smooth = gaussian_filter(normalized, sigma=2)
    threshold = max(float(filters.threshold_otsu(smooth)), float(np.percentile(smooth, 72)))
    mask = smooth > threshold
    mask = morphology.remove_small_objects(mask, max_size=127)
    mask = binary_fill_holes(mask)
    labels = measure.label(mask)
    props = measure.regionprops(labels, intensity_image=align)
    height, width = align.shape
    margin = 80

    components = []
    for prop in props:
        y0, x0, y1, x1 = prop.bbox
        if prop.area < 300 or prop.area > 3000:
            continue
        cy, cx = prop.centroid
        if cy < margin or cx < margin or cy > height - margin or cx > width - margin:
            continue
        bbox_h = y1 - y0
        bbox_w = x1 - x0
        aspect = max(bbox_h / max(bbox_w, 1), bbox_w / max(bbox_h, 1))
        if aspect > 3.5:
            continue
        intensity_mean = float(prop.intensity_mean if hasattr(prop, "intensity_mean") else prop.mean_intensity)
        components.append(
            {
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "area": int(prop.area),
                "bbox": prop.bbox,
                "intensity_mean": intensity_mean,
            }
        )
    return components


def select_single_timepoint_roi(
    alignment_frame: np.ndarray,
    mcherry_frame: np.ndarray,
    *,
    crop_size: int,
) -> dict[str, Any]:
    components = detect_compact_components(alignment_frame)
    height, width = alignment_frame.shape
    scored = []
    for component in components:
        cy = component["centroid_y"]
        cx = component["centroid_x"]
        label_y = int(round(cy))
        label_x = int(round(cx))
        local_mcherry = float(mcherry_frame[label_y, label_x])
        center_distance = np.hypot(cy - height / 2, cx - width / 2) / np.hypot(height, width)
        score = local_mcherry + 0.2 * component["intensity_mean"] - 100 * center_distance + 0.02 * component["area"]
        scored.append((score, component))

    if not scored:
        raise ValueError("No single-neuron candidate ROI found; try a different crop size or threshold.")

    _, best = max(scored, key=lambda item: item[0])
    cy = best["centroid_y"]
    cx = best["centroid_x"]
    x_start = int(round(cx - crop_size / 2))
    y_start = int(round(cy - crop_size / 2))
    x_start = max(0, min(x_start, width - crop_size))
    y_start = max(0, min(y_start, height - crop_size))
    return {
        "roi_id": "single_neuron_candidate_auto",
        "x_start": x_start,
        "y_start": y_start,
        "width": crop_size,
        "height": crop_size,
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "component_area_pixels": int(best["area"]),
        "max_centroid_distance_pixels": "",
        "selection_channel": "488",
        "selection_note": "automatic compact foreground component from Day 8 stable channel; manual review recommended",
    }


def local_register_roi_stack(
    stack: np.ndarray,
    roi: dict[str, Any],
    *,
    alignment_channel: int,
    max_local_shift: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    y0 = int(roi["y_start"])
    x0 = int(roi["x_start"])
    height = int(roi["height"])
    width = int(roi["width"])
    crops = stack[:, :, y0 : y0 + height, x0 : x0 + width]
    reference = robust_registration_image(crops[0, alignment_channel])
    registered = [crops[0]]
    shifts = [(0.0, 0.0)]
    for time_index in range(1, crops.shape[0]):
        moving = robust_registration_image(crops[time_index, alignment_channel])
        shift, _, _ = phase_cross_correlation(reference, moving, upsample_factor=20)
        dy, dx = float(shift[0]), float(shift[1])
        if np.hypot(dy, dx) > max_local_shift:
            dy, dx = 0.0, 0.0
        registered.append(apply_shift(crops[time_index], dy, dx))
        shifts.append((dy, dx))
    return np.stack(registered, axis=0), shifts


def robust_registration_image(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(np.float32)
    lo, hi = np.percentile(image, [5, 99])
    return gaussian_filter(np.clip((image - lo) / max(hi - lo, 1), 0, 1), sigma=1)


def write_single_well_montage(
    stack: np.ndarray,
    *,
    well: str,
    days: list[int],
    mcherry_channel: int,
    alignment_channel: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, stack.shape[0], figsize=(3.2 * stack.shape[0], 6.2), constrained_layout=True)
    for index in range(stack.shape[0]):
        overlay = rgb_overlay(stack[index, mcherry_channel], stack[index, alignment_channel])
        axes[0, index].imshow(overlay)
        axes[0, index].set_title(f"{well} Day {days[index]} overlay")
        axes[1, index].imshow(stack[index, mcherry_channel], cmap="magma", vmax=np.percentile(stack[:, mcherry_channel], 99.5))
        axes[1, index].set_title("mCherry")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_single_well_gifs(
    stack: np.ndarray,
    *,
    well: str,
    days: list[int],
    mcherry_channel: int,
    alignment_channel: int,
    figures: Path,
    duration_ms: int,
) -> None:
    mcherry_frames = []
    overlay_frames = []
    mcherry_limits = robust_limits(stack[:, mcherry_channel])
    for index in range(stack.shape[0]):
        mcherry = normalize_to_uint8(stack[index, mcherry_channel], mcherry_limits)
        overlay = (rgb_overlay(stack[index, mcherry_channel], stack[index, alignment_channel]) * 255).astype(np.uint8)
        mcherry_frames.append(annotate(Image.fromarray(mcherry, mode="L").convert("RGB"), well, days[index], "single-neuron mCherry"))
        overlay_frames.append(annotate(Image.fromarray(overlay, mode="RGB"), well, days[index], "red=mCherry green=488"))
    imageio.mimsave(figures / f"{well}_single_neuron_mcherry.gif", mcherry_frames, duration=duration_ms, loop=0)
    imageio.mimsave(figures / f"{well}_single_neuron_overlay.gif", overlay_frames, duration=duration_ms, loop=0)


def write_side_by_side_gif(
    stacks: dict[str, np.ndarray],
    *,
    days: list[int],
    mcherry_channel: int,
    path: Path,
    duration_ms: int,
) -> None:
    wells = list(stacks)
    limits = robust_limits(np.concatenate([stacks[well][:, mcherry_channel].ravel() for well in wells]))
    frames = []
    for index in range(min(stacks[well].shape[0] for well in wells)):
        panels = []
        for well in wells:
            frame = normalize_to_uint8(stacks[well][index, mcherry_channel], limits)
            panels.append(annotate(Image.fromarray(frame, mode="L").convert("RGB"), well, days[index], "single-neuron mCherry"))
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        frames.append(canvas)
    imageio.mimsave(path, frames, duration=duration_ms, loop=0)


def write_side_by_side_montage(
    stacks: dict[str, np.ndarray],
    *,
    days: list[int],
    mcherry_channel: int,
    path: Path,
) -> None:
    wells = list(stacks)
    fig, axes = plt.subplots(len(wells), len(days), figsize=(3.0 * len(days), 3.0 * len(wells)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    vmax = np.percentile(np.concatenate([stacks[well][:, mcherry_channel].ravel() for well in wells]), 99.5)
    for row, well in enumerate(wells):
        for col, day in enumerate(days):
            axes[row, col].imshow(stacks[well][col, mcherry_channel], cmap="magma", vmax=vmax)
            axes[row, col].set_title(f"{well} Day {day}")
            axes[row, col].set_axis_off()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_metric_plot(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4), constrained_layout=True)
    for well, df in metrics.groupby("well", sort=True):
        ax.plot(df["timepoint_day"], df["diffuse_to_punctate_ratio"], marker="o", linewidth=2, label=well)
    ax.set_xlabel("Day")
    ax.set_ylabel("Diffuse / punctate ratio")
    ax.set_title("Single-neuron crop mCherry metric")
    ax.legend(title="Well")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(out_dir: Path, args: argparse.Namespace, roi_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Single-Neuron Time-Series Alignment Examples",
        "",
        "These examples crop one automatically selected, compact 488-positive foreground component",
        "from each well's globally registered stack, then perform local crop-level registration on",
        "the 488 channel and apply the same transform to mCherry.",
        "",
        "Use these as presentation examples of the alignment concept, not as final same-neuron",
        "biological calls until the ROI identity is manually reviewed.",
        "",
        f"Report root: `{args.report_root}`",
        f"Crop size: `{args.crop_size} x {args.crop_size}` pixels",
        "",
        "Outputs:",
        "- `figures/*_single_neuron_alignment_montage.png`",
        "- `figures/*_single_neuron_mcherry.gif`",
        "- `figures/*_single_neuron_overlay.gif`",
        "- `figures/E05_vs_F05_single_neuron_mcherry.gif`",
        "- `single_neuron_roi_selection.csv`",
        "- `single_neuron_local_registration_qc.csv`",
        "- `single_neuron_mcherry_metrics.csv`",
        "",
        "Selected ROIs:",
    ]
    for row in roi_rows:
        lines.append(
            f"- {row['well']}: x={row['x_start']}, y={row['y_start']}, "
            f"area={row['component_area_pixels']} px"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rgb_overlay(mcherry: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    red = normalize_to_float(mcherry, robust_limits(mcherry))
    green = normalize_to_float(alignment, robust_limits(alignment))
    rgb = np.zeros((*red.shape, 3), dtype=np.float32)
    rgb[..., 0] = red
    rgb[..., 1] = green
    return rgb


def robust_limits(arr: np.ndarray) -> tuple[float, float]:
    values = arr.astype(np.float32)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def normalize_to_float(frame: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    lo, hi = limits
    return np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)


def normalize_to_uint8(frame: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return (normalize_to_float(frame, limits) * 255).astype(np.uint8)


def annotate(image: Image.Image, well: str, day: int, subtitle: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text = f"{well} Day {day} | {subtitle}"
    draw.rectangle((0, 0, canvas.width, 18), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255), font=font)
    return canvas


if __name__ == "__main__":
    main()
