"""Run the validated longitudinal analysis one well at a time on the complete plate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WELLS = [f"{row}{col:02d}" for row in "CDEFGHIJKLMN" for col in range(3, 11)]
DAYS = list(range(5, 18))
MODEL_SHA = "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2"


def check_inputs(root: Path, manifest: list[dict]):
    """Fail before GPU work if any expected acquisition is absent or incomplete."""
    expected = {(w, d) for w in WELLS for d in DAYS}
    found = {(r["well"], r["day"]) for r in manifest}
    if found != expected or len(manifest) != len(expected):
        raise ValueError("Manifest must contain exactly 96 wells × 13 days without duplicates")
    problems = []
    for row in manifest:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe relative path in source manifest")
        p = root / relative
        if not p.is_file():
            problems.append(f"Missing: {relative}")
        elif p.stat().st_size != row["size_bytes"]:
            problems.append(f"Incomplete or changed size: {relative}")
    if problems:
        raise ValueError(
            f"{len(problems)} input problems; finish the data copy first.\n"
            + "\n".join(problems[:12])
        )
    return len(manifest)


def aggregate(out: Path):
    """Aggregate only wells that passed their independent saved-artifact checks."""
    import pandas as pd

    completed = []
    metrics, endpoints = [], []
    for well in WELLS:
        folder = out / "wells" / well
        completion = folder / "completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("complete"):
            frame = pd.read_csv(folder / "well_day_metrics.csv")
            if set(zip(frame.well, frame.day)) != {(well, d) for d in DAYS} or len(frame) != 13:
                raise ValueError(f"Unexpected result rows for {well}")
            completed.append(well)
            metrics.append(frame)
            endpoints.append(pd.read_csv(folder / "endpoint_summary.csv"))
    if metrics:
        pd.concat(metrics, ignore_index=True).sort_values(["well", "day"]).to_csv(
            out / "well_day_metrics.csv", index=False
        )
        pd.concat(endpoints, ignore_index=True).to_csv(out / "endpoint_summary.csv", index=False)
    cells = "".join(
        f"<tr><td>{w}</td><td>{'verified' if w in completed else 'pending'}</td><td>"
        + (f'<a href="wells/{w}/report.html">Report and plots</a>' if w in completed else "")
        + "</td></tr>"
        for w in WELLS
    )
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Full-plate longitudinal analysis</title>'
        "<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto}td,th{padding:6px 24px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>"
        f"<h1>Full-plate nuclei and mCherry</h1><p>{len(completed)}/96 wells verified; {13 * len(completed)}/1248 images. Days5–17.</p>"
        "<p>Same first-pass measurement method as the five-well pilot. No retrospective illumination correction, individual-cell tracking, or survival inference. Masks require biological review.</p>"
        '<p><a href="well_day_metrics.csv">Combined measurements CSV</a> · <a href="endpoint_summary.csv">Endpoint summary CSV</a></p>'
        "<table><tr><th>Well</th><th>Status</th><th>Results</th></tr>" + cells + "</table>"
    )
    return completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process only E04 day5 into a separate smoke-test directory",
    )
    args = ap.parse_args()
    manifest = json.loads((ROOT / "handoff/source_manifest.json").read_text())
    n = check_inputs(args.input, manifest)
    weights = Path.home() / ".cellpose/models/cpsam"
    if (
        not weights.exists()
        or hashlib.file_digest(weights.open("rb"), "sha256").hexdigest() != MODEL_SHA
    ):
        raise ValueError(
            "Missing or different cpsam weights; copy the pilot checkpoint before running"
        )
    import torch

    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        raise RuntimeError("GPU unavailable; do not launch the plate on CPU")
    print(
        f"Preflight passed: {n} files match expected paths and sizes; pinned cpsam hash and GPU available.",
        flush=True,
    )
    print(
        "Size checks are not checksums of image contents. Finish the copying program before starting.",
        flush=True,
    )
    if args.preflight_only:
        return
    if args.output is None:
        ap.error("--output is required for processing")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # One process per GPU. File lock is released automatically on exit/crash.
    import fcntl

    lock = (out / ".run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another plate process is using this output directory") from None
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    status = out / "job_status.json"

    def update(state, **kw):
        status.write_text(
            json.dumps(
                dict(state=state, updated=datetime.now().isoformat(), pid=os.getpid(), **kw),
                indent=2,
            )
        )

    config = {
        "input": str(args.input.resolve()),
        "wells": WELLS,
        "days": DAYS,
        "scale": 2,
        "model_sha256": MODEL_SHA,
        "source_manifest_sha256": hashlib.sha256(
            (ROOT / "handoff/source_manifest.json").read_bytes()
        ).hexdigest(),
    }
    previous = out / "plate_config.json"
    if previous.exists() and json.loads(previous.read_text()) != config:
        raise ValueError("Plate configuration changed; use a new output directory")
    previous.write_text(json.dumps(config, indent=2))
    selected = ["E04"] if args.smoke_test else WELLS
    update("running", expected_images=1 if args.smoke_test else 1248)
    try:
        for well in selected:
            folder = out / "smoke_E04_d5" if args.smoke_test else out / "wells" / well
            update("running", active_well=well, expected_images=1 if args.smoke_test else 1248)
            command = [
                sys.executable,
                str(ROOT / "scripts/run_longitudinal_counts.py"),
                "--input",
                str(args.input.resolve()),
                "--output",
                str(folder),
                "--wells",
                well,
                "--scale",
                "2",
            ]
            if args.smoke_test:
                command += ["--days", "5"]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            if (ROOT / "handoff/resolution_qc").exists():
                shutil.copytree(
                    ROOT / "handoff/resolution_qc", folder / "resolution_qc", dirs_exist_ok=True
                )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/report_longitudinal_counts.py"),
                    "--output",
                    str(folder),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            done = json.loads((folder / "completion.json").read_text())
            if not done["complete"]:
                raise RuntimeError(f"Incomplete verified output for {well}")
            if not args.smoke_test:
                completed = aggregate(out)
                update(
                    "running", completed_wells=len(completed), completed_images=len(completed) * 13
                )
        if args.smoke_test:
            update("smoke_complete", completed_images=1)
        else:
            completed = aggregate(out)
            if len(completed) != 96:
                raise RuntimeError("Full plate is incomplete")
            update("complete", completed_wells=96, completed_images=1248)
    except Exception as exc:
        update("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
