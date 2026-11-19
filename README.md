# TMEM longitudinal analysis

Use this repository for future longitudinal analysis on pmihack:
`/Users/pmihack/claire/tmem_2026/analysis/tmem-longitudinal`

This is a standalone repository assembled from the deployed analysis snapshots. It does not contain the complete upstream tmem-neuron-aligner Git history. Pipeline and reporting code were copied unchanged.

## Preserved releases

- `longitudinal-counts-20261115-v1`: original deployed source; began the plate run (the run record attributes the first 70 frames to v1).
- `longitudinal-counts-20261115-v2`: faster lossless NPZ compression; resumed and completed the plate. Segmentation and measurement settings stayed the same.
- `longitudinal-summary-20261116`: snapshot of deployed nuclear mNG measurement and longitudinal plotting code at consolidation time.
- `provenance/deployment-snapshots.json`: original paths, source checksums, imported commits, and verified archive checksums.

Original deployment directories, logs, and complete Python environments are retained under `../archive/consolidation-20261116/releases/`. Their original paths remain compatibility symlinks. Do not use those snapshots for new development. Source-and-record archives are beside the archived directories; these tar files exclude virtual environments and caches, which remain in the archived directories.

The original `../longitudinal-summary-20261116/` workspace is retained for ongoing work in the main conversation. This repository contains a snapshot, not an automatic synchronization of subsequent edits. Reconcile any later reporting edits here before using them.

## Environment and checks

The independent `.venv` uses Python 3.12.13 and the exact 57 package versions recorded by v2. Recreate it if needed:

```bash
cd /Users/pmihack/claire/tmem_2026/analysis/tmem-longitudinal
bash setup.sh
PYTHONPATH=src .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

## Input and results

- Input: `/Users/pmihack/claire/tmem_2026/data/mCherry-mNG-longitudinal/`
- Completed full-plate results: `/Users/pmihack/claire/tmem_2026/outputs/mcherry-mng-counts-scale2-v1/`
- Main report: the results directory's `report.html`.
- Historical `scale2-v1` output naming is retained; it does not mean every frame was generated with code release v1.

Consolidation does not rerun segmentation, measurements, or reports. It does not alter raw images or existing results.

## Commands for future runs

Read-only input preflight (requires the existing cpsam checkpoint and MPS GPU):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_plate_counts.py --input /Users/pmihack/claire/tmem_2026/data/mCherry-mNG-longitudinal --preflight-only
```

For a future run, explicitly choose the destination; resuming an existing destination reuses compatible completed frames. Run one plate process at a time:

```bash
PYTHONPATH=src bash handoff/run-all.sh /path/to/nd2-data /path/to/output
```

Additional measurements and report generation modify the specified results directory; they are not needed merely to consolidate code:

```bash
PYTHONPATH=src .venv/bin/python scripts/measure_nuclear_mng.py --output /path/to/output --workers 8
PYTHONPATH=src .venv/bin/python scripts/build_longitudinal_plate_summary.py --output /path/to/output --plate-map longitudinal_plate_map.csv --backups /path/to/report-backups
```

Within-well and within-condition comparisons (the spatial tile measurement is the slow step; the report builder reads the baseline day back from `longitudinal_summary_provenance.json`, so run it again after any summary rerun):

```bash
PYTHONPATH=src .venv/bin/python scripts/measure_well_spatial.py --output /path/to/output --workers 6
PYTHONPATH=src .venv/bin/python scripts/build_well_comparisons.py --output /path/to/output
```

`--baseline-day` (default 6, chosen 2026-11-16 at the biologist check-in) sets the day each well's fold readouts are normalized to; the fold columns are named `<readout>_day<N>_fold`. On a rerun the plate report keeps sections appended by later stages and only the fold definition is swapped. The counts stage (`run_longitudinal_counts.py --baseline-day`, default 6) writes the same baseline into `baseline_day` and the `*_change_baseline` / `*_fold_baseline` columns; a well that has not reached the baseline day yet falls back to its first observed day. The comparisons builder used to hardcode day 5; it is now in this repository and follows the provenance value.

To move an already finished output tree to a new baseline without rerunning Cellpose:

```bash
PYTHONPATH=src .venv/bin/python scripts/rebase_counts_baseline.py --output /path/to/output --report-backups /path/to/report-backups --keep /path/to/output/counts_day5_20261116
```

then rerun the summary and comparisons builders above. Raw measurement columns are unchanged; the file hash of `well_day_metrics.csv` moves, so provenance JSONs written afterwards record the new one.

`handoff/README.txt` and `handoff/code_manifest.json` describe the historical v2 deployment. Use this README and `setup.sh` for the consolidated repository. Git tags preserve the original handoff files exactly.

## Recovery

Use `git show <tag>:<path>` to inspect a historical source file without changing the working tree. Original logs, environments, and files remain in the archive. To undo a compatibility relocation, remove only the relevant original-path symlink and move its corresponding archived directory back; never overwrite an existing directory. The complete canonical Git history is also backed up as `../archive/consolidation-20261116/tmem-longitudinal.bundle`.
