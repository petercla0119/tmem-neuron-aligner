#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROCESSED_ROOT = Path("/Users/makennarodriguez/Documents/TMEM106B_processed")

REPORTER_ROWS = {"E", "I", "M"}
PRIMARY_ROWS = {"F", "J", "N"}
PAIR_FOR_ROW = {"E": "E_F", "F": "E_F", "I": "I_J", "J": "I_J", "M": "M_N", "N": "M_N"}

METRICS = [
    ("puncta_count", "Puncta count"),
    ("punctate_mean", "Punctate mean intensity"),
    ("diffuse_mean", "Diffuse mean intensity"),
    ("rupture_like_score", "Diffuse / punctate mean"),
    ("mean_puncta_area_pixels", "Mean puncta area"),
    ("max_puncta_intensity", "Max puncta intensity"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create graphical summaries from local mCherry longitudinal pilot metrics."
    )
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot_root = args.processed_root / "pilot"
    output_dir = pilot_root / "mcherry_graphical_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = collect_metrics(pilot_root)
    if combined.empty:
        raise FileNotFoundError(f"No mCherry metrics found under {pilot_root}")

    group_summary = summarize_by_condition(combined)
    paired_delta = summarize_paired_delta(combined)

    combined_path = output_dir / "combined_mcherry_metrics.csv"
    group_path = output_dir / "condition_day_summary.csv"
    delta_path = output_dir / "paired_primary_minus_control_delta.csv"
    combined.to_csv(combined_path, index=False)
    group_summary.to_csv(group_path, index=False)
    paired_delta.to_csv(delta_path, index=False)

    write_metric_grid(combined, output_dir / "mcherry_metric_trajectories.png")
    write_group_summary(group_summary, output_dir / "mcherry_condition_mean_sem.png")
    write_delta_figure(paired_delta, output_dir / "mcherry_primary_minus_control_delta.png")
    write_puncta_diffuse_scatter(combined, output_dir / "mcherry_puncta_diffuse_scatter.png")

    print(f"Wrote combined metrics: {combined_path}")
    print(f"Wrote condition/day summary: {group_path}")
    print(f"Wrote paired deltas: {delta_path}")
    print(f"Wrote figures under: {output_dir}")
    print(group_summary.to_string(index=False))


def collect_metrics(pilot_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(pilot_root.glob("*_longitudinal/*_mcherry_metrics.csv")):
        well = infer_well(path)
        if well is None or not is_mcherry_valid(well):
            continue
        df = pd.read_csv(path)
        row = well[0]
        column = well[1:]
        df.insert(0, "well", well)
        df.insert(1, "row", row)
        df.insert(2, "column", column)
        df.insert(3, "condition", condition_for_row(row))
        df.insert(4, "condition_label", condition_label_for_row(row))
        df.insert(5, "replicate_pair", f"{PAIR_FOR_ROW[row]}_{column}")
        df.insert(6, "source_metrics_csv", str(path))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["column", "replicate_pair", "well", "day"])


def infer_well(path: Path) -> str | None:
    match = re.search(r"([A-P]\d{2})_days_", path.name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def is_mcherry_valid(well: str) -> bool:
    return well[0] in REPORTER_ROWS | PRIMARY_ROWS


def condition_for_row(row: str) -> str:
    if row in REPORTER_ROWS:
        return "PLD3_mCherry_reporter_control"
    if row in PRIMARY_ROWS:
        return "PLD3_TMEM106B_mCherry_primary"
    raise ValueError(f"Row {row} is not mCherry-valid")


def condition_label_for_row(row: str) -> str:
    if row in REPORTER_ROWS:
        return "PLD3 + mCherry"
    if row in PRIMARY_ROWS:
        return "PLD3 + TMEM106B + mCherry"
    raise ValueError(f"Row {row} is not mCherry-valid")


def summarize_by_condition(combined: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined.groupby(["condition", "condition_label", "day"], sort=True)
        .agg(
            n_wells=("well", "nunique"),
            puncta_count_mean=("puncta_count", "mean"),
            puncta_count_sem=("puncta_count", sem),
            punctate_mean_mean=("punctate_mean", "mean"),
            punctate_mean_sem=("punctate_mean", sem),
            diffuse_mean_mean=("diffuse_mean", "mean"),
            diffuse_mean_sem=("diffuse_mean", sem),
            rupture_like_score_mean=("rupture_like_score", "mean"),
            rupture_like_score_sem=("rupture_like_score", sem),
            mean_puncta_area_pixels_mean=("mean_puncta_area_pixels", "mean"),
            mean_puncta_area_pixels_sem=("mean_puncta_area_pixels", sem),
            max_puncta_intensity_mean=("max_puncta_intensity", "mean"),
            max_puncta_intensity_sem=("max_puncta_intensity", sem),
        )
        .reset_index()
    )
    return summary


def summarize_paired_delta(combined: pd.DataFrame) -> pd.DataFrame:
    pivot = combined.pivot_table(
        index=["column", "replicate_pair", "day"],
        columns="condition",
        values=[metric for metric, _ in METRICS],
        aggfunc="mean",
    )
    rows = []
    for index, values in pivot.iterrows():
        column, replicate_pair, day = index
        row = {"column": column, "replicate_pair": replicate_pair, "day": day}
        for metric, _ in METRICS:
            try:
                primary = values[(metric, "PLD3_TMEM106B_mCherry_primary")]
                control = values[(metric, "PLD3_mCherry_reporter_control")]
            except KeyError:
                primary = np.nan
                control = np.nan
            row[f"{metric}_primary_minus_control"] = primary - control
        rows.append(row)
    return pd.DataFrame(rows).dropna(how="all", subset=[f"{metric}_primary_minus_control" for metric, _ in METRICS])


def sem(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    return float(series.sem())


def write_metric_grid(combined: pd.DataFrame, figure_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {
        "PLD3 + mCherry": "#2f6f9f",
        "PLD3 + TMEM106B + mCherry": "#b55d15",
    }
    markers = {"05": "o", "06": "s", "07": "^", "08": "D"}
    for ax, (metric, title) in zip(axes.ravel(), METRICS, strict=True):
        for well, df in combined.groupby("well", sort=True):
            label = well
            condition_label = df["condition_label"].iloc[0]
            column = df["column"].iloc[0]
            ax.plot(
                df["day"],
                df[metric],
                marker=markers.get(column, "o"),
                linewidth=1.8,
                alpha=0.85,
                label=label,
                color=colors[condition_label],
            )
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 8))
    fig.suptitle("mCherry puncta and diffusion metrics by well")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def write_group_summary(group_summary: pd.DataFrame, figure_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metrics = [
        ("puncta_count", "Puncta count"),
        ("punctate_mean", "Punctate mean intensity"),
        ("diffuse_mean", "Diffuse mean intensity"),
        ("rupture_like_score", "Diffuse / punctate mean"),
    ]
    colors = {
        "PLD3 + mCherry": "#2f6f9f",
        "PLD3 + TMEM106B + mCherry": "#b55d15",
    }
    for ax, (metric, title) in zip(axes.ravel(), metrics, strict=True):
        for condition_label, df in group_summary.groupby("condition_label", sort=True):
            ax.errorbar(
                df["day"],
                df[f"{metric}_mean"],
                yerr=df[f"{metric}_sem"],
                marker="o",
                linewidth=2.4,
                capsize=4,
                label=condition_label,
                color=colors[condition_label],
            )
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Condition mean +/- SEM for processed mCherry-valid wells")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def write_delta_figure(paired_delta: pd.DataFrame, figure_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    metrics = [
        ("rupture_like_score_primary_minus_control", "Diffuse / punctate delta"),
        ("puncta_count_primary_minus_control", "Puncta count delta"),
    ]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for replicate_pair, df in paired_delta.groupby("replicate_pair", sort=True):
            ax.axhline(0, color="#555555", linewidth=1, alpha=0.7)
            ax.plot(df["day"], df[metric], marker="o", linewidth=2, label=replicate_pair)
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.set_ylabel("Primary minus matched control")
        ax.grid(True, alpha=0.25)
    axes[-1].legend(title="Pair")
    fig.suptitle("Matched primary-control differences")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def write_puncta_diffuse_scatter(combined: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = {
        "PLD3 + mCherry": "#2f6f9f",
        "PLD3 + TMEM106B + mCherry": "#b55d15",
    }
    for condition_label, df in combined.groupby("condition_label", sort=True):
        scatter = ax.scatter(
            df["punctate_mean"],
            df["diffuse_mean"],
            s=np.clip(df["puncta_count"] / 4, 25, 260),
            c=df["day"],
            cmap="viridis",
            edgecolor=colors[condition_label],
            linewidth=1.5,
            alpha=0.8,
            label=condition_label,
        )
    ax.set_xlabel("Punctate mean intensity")
    ax.set_ylabel("Diffuse mean intensity")
    ax.set_title("Diffuse versus punctate mCherry signal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Day")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
