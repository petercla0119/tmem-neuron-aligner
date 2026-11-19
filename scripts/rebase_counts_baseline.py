"""Recompute the counts-stage baseline columns, plots and endpoint tables for a finished output tree.

Reuses the saved per-frame measurements, so Cellpose does not rerun. Raw measurement columns are
untouched; only ``baseline_day`` and the ``*_change_baseline`` / ``*_fold_baseline`` columns move.
Run the summary and comparisons stages afterwards so every report agrees.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from run_longitudinal_counts import plots

from tmem_align.analysis.longitudinal_counts import add_baseline_changes

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ["baseline_day"] + [
    f"{k}_{kind}_baseline"
    for k in (
        "nuclei_count",
        "mcherry_mean_raw",
        "mcherry_mean_bg_corrected",
        "mcherry_integrated_per_nucleus",
    )
    for kind in ("change", "fold")
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline-day", type=int, default=6)
    ap.add_argument(
        "--report-backups",
        type=Path,
        required=True,
        help="Summary-stage backup tree; its per-well report.html copies are refreshed",
    )
    ap.add_argument(
        "--keep",
        type=Path,
        required=True,
        help="Folder that receives the pre-rebase CSVs (never overwritten)",
    )
    a = ap.parse_args()
    root = a.output
    wells = sorted(p.parent for p in (root / "wells").glob("*/well_day_metrics.csv"))
    before = sha(root / "well_day_metrics.csv")
    for folder in wells + [root]:
        rel = folder.relative_to(root)
        for name in ("well_day_metrics.csv", "endpoint_summary.csv"):
            dst = a.keep / rel / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(folder / name, dst)
    for folder in wells:
        df = pd.read_csv(folder / "well_day_metrics.csv")
        columns = list(df.columns)
        raw = df.drop(columns=DERIVED)
        df = add_baseline_changes(raw, a.baseline_day)[columns]
        assert (df.baseline_day == a.baseline_day).all(), folder
        df.to_csv(folder / "well_day_metrics.csv", index=False)
        plots(df, folder, a.baseline_day)
        # report_longitudinal_counts rewrites report.html in the original (pre-summary) layout;
        # park that copy in the summary-stage backups so compact_well picks up the new table,
        # and restore the current compacted report until the summary stage reruns.
        current = (folder / "report.html").read_text()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/report_longitudinal_counts.py"),
                "--output",
                str(folder),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        backup = a.report_backups / "wells" / folder.name / "report.html"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "report.html", backup)
        (folder / "report.html").write_text(current)
        print(folder.name, flush=True)
    frames = [pd.read_csv(f / "well_day_metrics.csv") for f in wells]
    pd.concat(frames, ignore_index=True).sort_values(["well", "day"]).to_csv(
        root / "well_day_metrics.csv", index=False
    )
    pd.concat([pd.read_csv(f / "endpoint_summary.csv") for f in wells], ignore_index=True).to_csv(
        root / "endpoint_summary.csv", index=False
    )
    after = sha(root / "well_day_metrics.csv")
    print(f"wells: {len(wells)}; baseline day {a.baseline_day}")
    print(f"well_day_metrics.csv sha256 before {before}")
    print(f"well_day_metrics.csv sha256 after  {after}")


if __name__ == "__main__":
    main()
