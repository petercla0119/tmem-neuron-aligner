#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tmem_align.analysis.mcherry_metrics import quantify_mcherry_timeseries

from run_260213_longitudinal_pilot import (
    choose_channel_index,
    infer_channel_string,
    infer_day,
    infer_sequence,
    load_nd2_cyx,
    register_stack,
)
from tmem_align.registration_qc import crop_tcyx


ROW_CONDITIONS = {
    "C": "PLD3_only_no_mCherry",
    "D": "PLD3_TMEM106B_no_mCherry",
    "E": "PLD3_mCherry_reporter_control",
    "F": "PLD3_TMEM106B_mCherry_primary",
    "G": "PLD3_only_no_mCherry",
    "H": "PLD3_TMEM106B_no_mCherry",
    "I": "PLD3_mCherry_reporter_control",
    "J": "PLD3_TMEM106B_mCherry_primary",
    "K": "PLD3_only_no_mCherry",
    "L": "PLD3_TMEM106B_no_mCherry",
    "M": "PLD3_mCherry_reporter_control",
    "N": "PLD3_TMEM106B_mCherry_primary",
}
MCHERRY_ROWS = {"E", "F", "I", "J", "M", "N"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all-well longitudinal registration QC and mCherry-valid measurement for the "
            "260213 dataset. Raw ND2 files are read-only."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--channels", nargs="+", default=["488", "561"])
    parser.add_argument("--max-sites", type=int, default=1)
    parser.add_argument("--max-read-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--limit-wells", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1 = sequential).")
    parser.add_argument(
        "--ref-mode",
        choices=["to_first", "anchored"],
        default="to_first",
        help="Temporal registration mode: 'to_first' (register every day to day 0, default) or "
        "'anchored' (re-anchor to the last good frame when correlation drops).",
    )
    parser.add_argument(
        "--anchor-corr-thresh",
        type=float,
        default=0.10,
        help="Anchored mode: re-anchor when post-corr to the current anchor drops below this. "
        "Ignored for --ref-mode to_first.",
    )
    parser.add_argument(
        "--min-post-correlation",
        type=float,
        default=0.07,
        help="QC gate: a timepoint fails when its post-registration correlation is below this.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _process_one_well(
    well: str,
    rows: list[dict[str, Any]],
    days: list[int],
    channels: list[str],
    max_sites: int,
    max_read_bytes: int,
    ref_mode: str = "to_first",
    anchor_corr_thresh: float = 0.10,
    min_post_correlation: float = 0.07,
) -> dict[str, Any]:
    try:
        if len(rows) != len(days):
            raise ValueError(f"Expected days {days}, found {[row['day'] for row in rows]}")
        loaded = [load_nd2_cyx(row["path"], max_sites, max_read_bytes) for row in rows]
        channel_names = loaded[0]["channel_names"]
        alignment_index = choose_channel_index(channel_names, channels[0])
        mcherry_index = choose_channel_index(channel_names, channels[1])
        raw_stack = np.stack([item["array"] for item in loaded], axis=0)
        registered, well_qc, common_crop = register_stack(
            raw_stack,
            well=well,
            rows=rows,
            alignment_channel_index=alignment_index,
            alignment_channel_label=channel_names[alignment_index],
            ref_mode=ref_mode,
            anchor_corr_thresh=anchor_corr_thresh,
            min_post_correlation=min_post_correlation,
        )
        condition = condition_for_well(well)
        for row in well_qc:
            row["condition"] = condition
            row["row"] = well[0]
            row["column"] = well[1:]
            row["common_crop"] = str(common_crop)

        metrics = None
        if has_mcherry_reporter(well):
            common = crop_tcyx(registered, common_crop)
            metadata_rows = [
                {
                    "well": well,
                    "row": well[0],
                    "column": well[1:],
                    "condition": condition,
                    "site_fov": "site0",
                    "timepoint_day": row["day"],
                    "file_name": row["path"].name,
                    "mcherry_channel": channel_names[mcherry_index],
                    "registration_channel": channel_names[alignment_index],
                }
                for row in rows
            ]
            metrics = quantify_mcherry_timeseries(
                common[:, mcherry_index],
                mask_stack=common[:, alignment_index],
                metadata_rows=metadata_rows,
            )
        return {"qc": well_qc, "metrics": metrics, "error": None}
    except Exception as exc:
        return {
            "qc": [],
            "metrics": None,
            "error": {
                "well": well,
                "row": well[0],
                "column": well[1:],
                "condition": condition_for_well(well),
                "error": repr(exc),
                "days_requested": "|".join(map(str, days)),
            },
        }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    selected = select_all_well_files(args.data_root.expanduser().resolve(), args.days)
    if args.limit_wells:
        selected = dict(list(selected.items())[: args.limit_wells])
    selected_files = selected_files_table(selected)
    selected_files.to_csv(output / "all_wells_selected_files.csv", index=False)

    if args.dry_run:
        write_readme(output, args, selected_files, None, None, dry_run=True)
        print(f"Dry run wrote all-well file selection: {output / 'all_wells_selected_files.csv'}")
        print(f"Wells selected: {len(selected)}")
        return

    qc_rows: list[dict[str, Any]] = []
    metric_tables: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []

    total = len(selected)
    well_items = list(selected.items())

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one_well, well, rows, args.days, args.channels,
                    args.max_sites, args.max_read_bytes,
                    args.ref_mode, args.anchor_corr_thresh, args.min_post_correlation,
                ): (idx, well)
                for idx, (well, rows) in enumerate(well_items, start=1)
            }
            for future in as_completed(futures):
                idx, well = futures[future]
                result = future.result()
                print(f"[{idx}/{total}] {well} done", flush=True)
                if result["error"]:
                    failure_rows.append(result["error"])
                    print(f"  failed {well}: {result['error']['error']}", flush=True)
                else:
                    qc_rows.extend(result["qc"])
                    if result["metrics"] is not None:
                        metric_tables.append(result["metrics"])
    else:
        for index, (well, rows) in enumerate(well_items, start=1):
            print(f"[{index}/{total}] {well} loading {len(rows)} days", flush=True)
            result = _process_one_well(
                well, rows, args.days, args.channels,
                args.max_sites, args.max_read_bytes,
                args.ref_mode, args.anchor_corr_thresh, args.min_post_correlation,
            )
            if result["error"]:
                failure_rows.append(result["error"])
                print(f"  failed {well}: {result['error']['error']}", flush=True)
            else:
                qc_rows.extend(result["qc"])
                if result["metrics"] is not None:
                    metric_tables.append(result["metrics"])

    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["well", "row", "column", "condition", "error", "days_requested"],
    )
    measurements = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
    summary = build_all_well_summary(qc, measurements, failures)

    qc.to_csv(output / "all_wells_registration_qc.csv", index=False)
    failures.to_csv(output / "all_wells_failures.csv", index=False)
    measurements.to_csv(output / "all_wells_mcherry_measurements.csv", index=False)
    summary.to_csv(output / "all_wells_summary_stats.csv", index=False)
    write_plate_heatmaps(qc, measurements, failures, figures)
    write_condition_summary_figure(measurements, figures / "all_wells_mcherry_condition_summary.png")
    write_readme(output, args, selected_files, summary, failures, dry_run=False)

    print(f"Wrote all-well batch outputs under {output}")
    print(summary.to_string(index=False))


def select_all_well_files(data_root: Path, days: list[int]) -> dict[str, list[dict[str, Any]]]:
    by_well: dict[str, dict[int, Path]] = {}
    for path in sorted(data_root.rglob("*.nd2")):
        name_lower = path.name.lower()
        if "brightfield" in name_lower:
            continue
        day = infer_day(path.name)
        if day not in days:
            continue
        well = infer_well(path.name)
        if not well:
            continue
        by_well.setdefault(well, {})[int(day)] = path
    selected: dict[str, list[dict[str, Any]]] = {}
    for well in sorted(by_well):
        if all(day in by_well[well] for day in days):
            selected[well] = [{"well": well, "day": day, "path": by_well[well][day]} for day in days]
    return selected


def selected_files_table(selected: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for well, items in selected.items():
        for item in items:
            path = item["path"]
            rows.append(
                {
                    "well": well,
                    "row": well[0],
                    "column": well[1:],
                    "condition": condition_for_well(well),
                    "mcherry_analysis_valid": has_mcherry_reporter(well),
                    "day": item["day"],
                    "path": str(path),
                    "file_name": path.name,
                    "file_size_bytes": path.stat().st_size,
                    "inferred_channel": infer_channel_string(path.name),
                    "site_fov": infer_sequence(path.name),
                }
            )
    return pd.DataFrame(rows)


def build_all_well_summary(
    qc: pd.DataFrame,
    measurements: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not qc.empty:
        for (condition, day), df in qc.groupby(["condition", "timepoint_day"], sort=True):
            rows.append(
                {
                    "summary_type": "registration_qc",
                    "condition": condition,
                    "day": int(day),
                    "wells": int(df["well"].nunique()),
                    "observations": int(len(df)),
                    "qc_pass": int(df["qc_pass"].sum()),
                    "qc_total": int(len(df)),
                    "mean_overlap_fraction": float(df["overlap_fraction"].mean()),
                    "mean_diffuse_to_punctate_ratio": np.nan,
                    "median_diffuse_to_punctate_ratio": np.nan,
                    "mean_puncta_count": np.nan,
                }
            )
    if not measurements.empty:
        for (condition, day), df in measurements.groupby(["condition", "timepoint_day"], sort=True):
            rows.append(
                {
                    "summary_type": "mcherry_valid_measurement",
                    "condition": condition,
                    "day": int(day),
                    "wells": int(df["well"].nunique()),
                    "observations": int(len(df)),
                    "qc_pass": np.nan,
                    "qc_total": np.nan,
                    "mean_overlap_fraction": np.nan,
                    "mean_diffuse_to_punctate_ratio": float(df["diffuse_to_punctate_ratio"].mean()),
                    "median_diffuse_to_punctate_ratio": float(df["diffuse_to_punctate_ratio"].median()),
                    "mean_puncta_count": float(df["puncta_count"].mean()),
                }
            )
    if not failures.empty:
        rows.append(
            {
                "summary_type": "failures",
                "condition": "all",
                "day": "",
                "wells": int(failures["well"].nunique()),
                "observations": int(len(failures)),
                "qc_pass": np.nan,
                "qc_total": np.nan,
                "mean_overlap_fraction": np.nan,
                "mean_diffuse_to_punctate_ratio": np.nan,
                "median_diffuse_to_punctate_ratio": np.nan,
                "mean_puncta_count": np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_plate_heatmaps(
    qc: pd.DataFrame,
    measurements: pd.DataFrame,
    failures: pd.DataFrame,
    figures: Path,
) -> None:
    rows = list("CDEFGHIJKLMN")
    columns = [f"{value:02d}" for value in range(5, 21)]

    if not qc.empty:
        qc_summary = qc.groupby("well")["qc_pass"].mean().to_dict()
        save_plate_heatmap(
            qc_summary,
            rows,
            columns,
            figures / "all_wells_registration_qc_pass_fraction.png",
            "Registration QC pass fraction, selected days",
            vmin=0,
            vmax=1,
        )
    if not measurements.empty:
        slope_by_well = {}
        for well, df in measurements.groupby("well"):
            ordered = df.sort_values("timepoint_day")
            if len(ordered) >= 2:
                slope_by_well[well] = float(
                    np.polyfit(
                        ordered["timepoint_day"].to_numpy(dtype=float),
                        ordered["diffuse_to_punctate_ratio"].to_numpy(dtype=float),
                        1,
                    )[0]
                )
        save_plate_heatmap(
            slope_by_well,
            rows,
            columns,
            figures / "all_wells_mcherry_ratio_slope_heatmap.png",
            "mCherry-valid wells: diffuse/punctate slope per day",
        )
    if not failures.empty:
        failure_by_well = {well: 1.0 for well in failures["well"].unique()}
        save_plate_heatmap(
            failure_by_well,
            rows,
            columns,
            figures / "all_wells_failures_heatmap.png",
            "Processing failures",
            vmin=0,
            vmax=1,
        )


def save_plate_heatmap(
    values: dict[str, float],
    rows: list[str],
    columns: list[str],
    path: Path,
    title: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    arr = np.full((len(rows), len(columns)), np.nan)
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            arr[r, c] = values.get(f"{row}{column}", np.nan)
    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    for r in range(len(rows)):
        for c in range(len(columns)):
            if np.isfinite(arr[r, c]):
                ax.text(c, r, f"{arr[r, c]:.2g}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_condition_summary_figure(measurements: pd.DataFrame, path: Path) -> None:
    if measurements.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for condition, df in measurements.groupby("condition", sort=True):
        summary = (
            df.groupby("timepoint_day")["diffuse_to_punctate_ratio"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax.errorbar(
            summary["timepoint_day"],
            summary["mean"],
            yerr=summary["sem"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=condition,
        )
    ax.set_xlabel("Day")
    ax.set_ylabel("Diffuse / punctate ratio")
    ax.set_title("All mCherry-valid wells by condition")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    output: Path,
    args: argparse.Namespace,
    selected_files: pd.DataFrame,
    summary: pd.DataFrame | None,
    failures: pd.DataFrame | None,
    *,
    dry_run: bool,
) -> None:
    wells_total = selected_files["well"].nunique()
    mcherry_wells = selected_files[selected_files["mcherry_analysis_valid"]]["well"].nunique()
    lines = [
        "# 260213 All-Well Batch",
        "",
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Data root: `{args.data_root}`",
        f"Days requested: `{', '.join(map(str, args.days))}`",
        f"Wells selected: `{wells_total}`",
        f"mCherry-valid wells measured: `{mcherry_wells}`",
        "",
        "Why this is broader than the E05/F05 pilot: the first run was a tiny proof-of-pipeline",
        "using the first reporter-control/experimental pair. This run includes every well with",
        "the requested days present.",
        "",
        "Important interpretation rule: rows without mCherry reporter are included for registration",
        "QC and plate coverage, but they are not treated as zero-mCherry puncta samples.",
        "",
        "Outputs:",
        "- `all_wells_selected_files.csv`",
        "- `all_wells_registration_qc.csv`",
        "- `all_wells_mcherry_measurements.csv` for E/F/I/J/M/N rows only",
        "- `all_wells_summary_stats.csv`",
        "- `all_wells_failures.csv`",
        "- `figures/all_wells_registration_qc_pass_fraction.png`",
        "- `figures/all_wells_mcherry_ratio_slope_heatmap.png`",
        "- `figures/all_wells_mcherry_condition_summary.png`",
    ]
    if dry_run:
        lines.append("")
        lines.append("Dry run only: no pixels were loaded.")
    elif summary is not None:
        lines.extend(["", "## Summary", "", "```text", summary.to_string(index=False), "```"])
    if failures is not None and not failures.empty:
        lines.extend(["", "## Failures", "", "```text", failures.to_string(index=False), "```"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_well(name: str) -> str | None:
    import re

    match = re.search(r"Well([C-N]\d{2})", name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def condition_for_well(well: str) -> str:
    return ROW_CONDITIONS.get(well[0].upper(), "unknown")


def has_mcherry_reporter(well: str) -> bool:
    return well[0].upper() in MCHERRY_ROWS


if __name__ == "__main__":
    main()
