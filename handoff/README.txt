Longitudinal nuclei and mCherry — portable release 20261115-v2

Contains the uncommitted analysis implementation from isolated branch
codex/longitudinal-nuclei-mcherry, based on 6a8a12adb4587aadee65f81ab5b32b8891ad8029.
Code hashes are in handoff/code_manifest.json. This is a code-only snapshot,
not a clone of the worktree or a copy of its virtual environment.

On pmihack:
  cd /Users/pmihack/claire/tmem_2026/analysis/longitudinal-counts-20261115-v2
  bash handoff/bootstrap.sh
After the data copy finishes:
  .venv/bin/python scripts/run_plate_counts.py --input /Users/pmihack/claire/tmem_2026/data/mCherry-mNG-longitudinal --preflight-only
  nohup bash handoff/run-all.sh > run.log 2>&1 &
  tail -f run.log

Use the same launch command to resume after interruption. Do not change input
files during a run. Completed frames are reused after configuration and input
metadata checks. One process runs at a time in a given output folder. Use only
one full-plate process on this GPU. The laptop need not stay connected.

Default output:
  /Users/pmihack/claire/tmem_2026/outputs/mcherry-mng-counts-scale2-v1
Root report.html links per-well plots and reports; well_day_metrics.csv and
endpoint_summary.csv combine verified wells; job_status.json records progress.
For a one-frame test, add --smoke-test and --output /path/to/separate/smoke-output
to the Python command instead of --preflight-only. Full input preflight still applies.

All 96 wells C03–N10, days 5–17 = 1,248 experimental images. Preflight verifies
expected relative paths and file sizes, not image-content checksums. It also
requires the exact pilot cpsam checkpoint and an available GPU. Preserve source
subdirectories while copying; the two nonexperimental test files are unused.

Measurements match the five-well pilot: 2x2 block averaging, zero-padding
exclusion, green-only Cellpose-SAM, raw and per-frame p10-background-subtracted
red signal, and field-integrated signal divided by nuclei count. No retrospective
illumination correction, tracking, whole-cell assignment, or death inference.
Masks and intensity backgrounds require biological QC. Quantitative pixels
are retained separately from the normalized segmentation input.

The resolution_qc material is the pilot sensitivity check, not validation of
new wells. The report green-unit sanity threshold comes from this dataset.
Direct dependencies are pinned; installed-environment.txt records resolved
transitive versions. The existing pmihack repository and environment are unused.

Version 2 changes only lossless NPZ compression to DEFLATE level 1.
Existing scientific configurations, masks, measurements and output paths are
compatible with version 1. Completed frames remain reusable. The original v1
release is retained on pmihack; v2 has separate source and environment.
