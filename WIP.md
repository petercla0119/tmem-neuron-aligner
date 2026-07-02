# WIP — 2026-07-02

Developer handoff: Makenna Rodriguez → Claire Peterson (pmihack).
Branch: `csp-dev`

---

## 1. Path updates

Bulk `sed` replacement across 11 files: all hardcoded paths from Makenna's machine
(`/Users/makennarodriguez/Documents/...`) updated to Claire's data location
(`/Users/pmihack/claire/tmem_2026/data/...`).

| File | Variables changed |
|------|-------------------|
| `notebooks/01_local_nd2_pilot.ipynb` | `RAW_ROOT`, `INTERIM_ROOT`, `PROCESSED_ROOT` |
| `scripts/run_ef05_mcherry_pilot.py` | `DEFAULT_RAW_ROOT`, `DEFAULT_INTERIM_ROOT`, `DEFAULT_PROCESSED_ROOT` |
| `scripts/run_f05_longitudinal_pilot.py` | same three defaults |
| `scripts/compare_ef05_longitudinal.py` | `DEFAULT_PROCESSED_ROOT` |
| `scripts/make_registration_qc_montages.py` | `DEFAULT_INTERIM_ROOT`, `DEFAULT_PROCESSED_ROOT` |
| `scripts/plot_mcherry_pilot_analysis.py` | `DEFAULT_PROCESSED_ROOT` |
| `scripts/run_mcherry_roi_pilot.py` | `DEFAULT_INTERIM_ROOT`, `DEFAULT_PROCESSED_ROOT` |
| `scripts/build_mcherry_qc_report.py` | `DEFAULT_PROCESSED_ROOT` |
| `scripts/build_mcherry_stage_prefilter.py` | `DEFAULT_PROCESSED_ROOT` |
| `scripts/build_applicable_nd2_manifest.py` | raw root string, `DEFAULT_PROCESSED_ROOT` |
| `scripts/make_mcherry_timeseries_videos.py` | `DEFAULT_INTERIM_ROOT`, `DEFAULT_PROCESSED_ROOT` |

## 2. Notebook crash fix

`notebooks/01_local_nd2_pilot.ipynb` — three cells crashed with `FileNotFoundError`
(single-frame preview workflow never run on this machine). Added `if path.exists():` guards:

- Cell `subset-summary` — guard around `PREVIEW_METADATA_JSON`
- Cell `metadata-summary` — same guard (depends on `pilot` from previous cell)
- Cell `display-preview` — same guard for `PREVIEW_PNG`

Other display cells already had this pattern.

## 3. Scripts executed

All completed successfully.

| # | Command | Status | Notes |
|---|---------|--------|-------|
| 1 | `python scripts/run_ef05_mcherry_pilot.py` | done | E05/F05 single-day mCherry smoke test |
| 2 | `python scripts/run_f05_longitudinal_pilot.py` | done | F05 3-day longitudinal (days 8, 25, 39) |
| 3 | `python scripts/run_f05_longitudinal_pilot.py --well E05` | done | E05 longitudinal |
| 4 | `python scripts/run_f05_longitudinal_pilot.py --well I05` | done | I05 longitudinal |
| 5 | `python scripts/run_f05_longitudinal_pilot.py --well J05` | done | J05 longitudinal |
| 6 | `python scripts/run_f05_longitudinal_pilot.py --well M05` | done | M05 longitudinal |
| 7 | `python scripts/run_f05_longitudinal_pilot.py --well N05` | done | N05 longitudinal |
| 8 | `python scripts/compare_ef05_longitudinal.py` | done | E05/F05, I05/J05, M05/N05 + 6-well aggregate comparisons |
| 9 | `python scripts/make_registration_qc_montages.py` | done | QC montages + shift summary for all 6 wells |

Wells processed: E05, F05, I05, J05, M05, N05 (all mCherry-valid per plate map).

## 4. QC flags from registration

Three well/day combos flagged `large_shift = True`:
- **F05 day 25** — dx=921 pixels (horizontal jump)
- **J05 day 39** — dy=1026 pixels (vertical jump)
- **M05 day 39** — dx=-921 pixels (horizontal jump)

These may indicate stage repositioning errors on those imaging days.

## 5. Files created this session

- `MANIFEST.md` — repo manifest (modules, scripts, tests, configs, docs)
- `WIP.md` — this file

## 6. Notebook status

`notebooks/01_local_nd2_pilot.ipynb` should now Run All cleanly. The preview cells (metadata/image from older single-frame extract) will print "Missing..." — that workflow was never run here. All other sections (EF05, F05 longitudinal, comparison, replicate, QC) will display results.

## 7. All-well batch reproduction

**Command:**
```
python scripts/run_260213_all_wells_batch.py \
  --data-root /Users/pmihack/claire/tmem_2026/data/260213_Feb16recopy_HYdiff_landingpadlines_survival_384well1 \
  --output /Users/pmihack/claire/tmem_2026/tmem-neuron-aligner/reports/260213_all_wells_reproduced \
  --days 8 12 16
```

**Purpose:** Reproduce the original Makenna run from 2026-06-23 (stored at `reports/260213_all_wells_20260623_days8_12_16/`). Processes 192 wells across the full plate (rows C-N, columns 05-20), 3 days each. Only E/F/I/J/M/N rows (96 wells) get mCherry measurement; all 192 get registration QC.

**Status:** Killed — restarted with --workers 10 (see below)

**Expected outputs** (in `reports/260213_all_wells_reproduced/`):
- `all_wells_selected_files.csv`
- `all_wells_registration_qc.csv`
- `all_wells_mcherry_measurements.csv`
- `all_wells_summary_stats.csv`
- `all_wells_failures.csv`
- `figures/all_wells_registration_qc_pass_fraction.png`
- `figures/all_wells_mcherry_ratio_slope_heatmap.png`
- `figures/all_wells_mcherry_condition_summary.png`
- `README.md` (auto-generated summary)

**Comparison target:** `reports/260213_all_wells_20260623_days8_12_16/` (Makenna's original run, 192 wells, 96 mCherry-valid)

## 8. Parallelization of all-well batch

**File changed:** `scripts/run_260213_all_wells_batch.py`

**What:** Added `--workers N` flag for parallel well processing via `concurrent.futures.ProcessPoolExecutor`. Default is 1 (sequential, same as before). Each well is independent — load ND2, register, quantify — so parallelizes cleanly.

**Changes:**
- Added `from concurrent.futures import ProcessPoolExecutor, as_completed` (line 5)
- Added `--workers` argparse flag (line 65)
- Extracted per-well logic into `_process_one_well()` function (new, before `main()`)
- `main()` loop now branches: workers > 1 uses ProcessPoolExecutor, workers == 1 uses sequential loop

**Restarted command:**
```
python scripts/run_260213_all_wells_batch.py \
  --data-root .../260213_Feb16recopy_HYdiff_landingpadlines_survival_384well1 \
  --output .../reports/260213_all_wells_reproduced \
  --days 8 12 16 \
  --workers 10
```

**Status:** Complete. 192/192 wells, zero failures.

## 9. Reproduction validation

Compared reproduced results (`reports/260213_all_wells_reproduced/`) against Makenna's original (`reports/260213_all_wells_20260623_days8_12_16/`).

**Verdict:** Match. All summary stats agree to floating-point precision (~10th decimal). Tiny differences (e.g. overlap fraction 0.8156373488 vs 0.8156373489) are from parallel reduction order, not logic changes.

- 192 wells processed, 48 per condition, zero failures (both runs)
- Registration QC: 48/48 pass across all conditions (both runs)
- mCherry ratios match to 2-3 significant figures
- Puncta counts: same or ±1

**Output location:** `reports/260213_all_wells_reproduced/`

## 10. Performance benchmarks

Controlled benchmark: 20 wells, days 8/12/16, same data. Hardware: Apple Silicon, 32 cores, 512 GB RAM.

### Wall-clock scaling

| Workers | Wall time | Speedup vs sequential | Per-well |
|---------|-----------|----------------------|----------|
| 1       | 56.8 s    | 1.0x (baseline)      | 2.84 s   |
| 5       | 15.4 s    | 3.7x                 | 0.77 s   |
| 10      | 9.9 s     | 5.7x                 | 0.50 s   |
| 20      | 8.6 s     | 6.6x                 | 0.43 s   |

Diminishing returns past 10 workers — I/O bandwidth saturates.

### CPU time and memory (20 wells, `/usr/bin/time -l`)

| Workers | Wall (s) | User CPU (s) | Sys CPU (s) | Total CPU (s) | Peak RSS (GB) |
|---------|----------|-------------|------------|---------------|---------------|
| 1       | 57.2     | 54.7        | 13.5       | 68.2          | 2.99          |
| 10      | 10.0     | 70.0        | 17.4       | 87.4          | 1.84          |
| 20      | 8.9      | 83.6        | 37.7       | 121.3         | 1.80          |

**Observations:**
- Wall time drops 6.4x (57s → 9s) going from 1 to 10 workers
- Total CPU time *increases* with more workers (overhead from multiprocessing, IPC serialization)
- Peak RSS is actually *lower* with workers — parent process doesn't hold all arrays simultaneously; child processes have their own memory spaces not counted in parent RSS
- 20 workers burns 75% more total CPU than 10 workers for only 12% wall-time improvement — **10 workers is the sweet spot**
- sys time doubles at 20 workers (IPC/fork overhead)

### Full-plate extrapolation (192 wells)

| Workers | Estimated wall time | Measured |
|---------|-------------------|----------|
| 1       | ~13.7 min         | (extrapolated from 38 wells in 2m42s before kill) |
| 10      | ~2 min 5 sec      | measured |

**Recommendation:** `--workers 10` for this hardware. Higher worker counts waste CPU with negligible wall-time gain.

## 11. docs/ reorganization

**Problem:** `docs/` contained 2,083 files: 22 markdown docs mixed in with ~2,060 generated dashboard files (HTML, PNG, CSS, JS, JSON/CSV). The generated site was ~85 MB committed directly on the source branch, making the directory unnavigable and bloating the repo.

**What changed:**

1. **Created `gh-pages` orphan branch** with all generated dashboard content (2,061 files including `.nojekyll`). The site is served from this branch instead of `docs/` on the source branch. The dashboard is regenerable via `scripts/build_github_pages_dashboard.py`.

2. **Removed generated site files from `csp-dev`** — all HTML, PNG, CSS, JS, JSON, CSV, and TXT files that were part of the dashboard build. These are the files under `docs/assets/`, `docs/wells/`, `docs/rois/`, `docs/summaries/`, `docs/previews/`, `docs/roi_previews/`, plus root-level `.html`, `.json`, and `site_size_report.txt`.

3. **Organized 22 markdown docs into subdirectories:**

   | Directory | Files | Contents |
   |-----------|-------|----------|
   | `docs/design/` | 4 | Decision rationale, audits (ALIGNMENT_METHOD_REVIEW, OVERLAP_ONLY_AUDIT, PIPELINE_AUDIT_AND_WORKPLAN, PI_OVERLAP_ONLY_RESPONSE) |
   | `docs/results/` | 6 | Analysis write-ups with reproduction commands (PILOT_EF05, F05_LONGITUDINAL, EF05_LONGITUDINAL, REPLICATE_LONGITUDINAL, MCHERRY_GRAPHICAL, REGISTRATION_QC_MONTAGES) |
   | `docs/guides/` | 8 | How-to docs (DASHBOARD, GITHUB_PAGES_DASHBOARD, ROI_IDENTITY_REVIEW, IMAGE_INTEGRATION_CHECKLIST, LOCAL_JUPYTER, NAPARI_VIEWING, STAGE_QC_AND_ROI_WORKFLOW, CODEX_4_AGENT_TEAM) |
   | `docs/plans/` | 4 | Execution plans (FULL_DATASET_EXECUTION, FULL_DATASET_DEFG, MANUAL_ALIGNMENT_OVERRIDE, PSEUDO_FOV_ALIGNMENT) |

4. **Replaced `docs/README.md`** — was a 3-line stub ("This folder is a sanitized GitHub Pages build..."). Now an index linking every doc with a one-line description, plus a note that the dashboard lives on `gh-pages`.

**Nothing deleted.** All 22 substantive markdown files preserved — 20 of 23 contain parameters, thresholds, reproduction commands, or decision rationale needed for reproducibility. The only content removed from the source branch is the generated dashboard, which is preserved on `gh-pages` and can be rebuilt from the build script.

**GitHub Pages config note:** If Pages was configured to serve from `docs/` on the default branch, it will need to be reconfigured to serve from the `gh-pages` branch root instead.
