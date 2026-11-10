# GRN longitudinal single-cell tracking — design

Date: 2026-11-10 · Branch: `feat/grn-longitudinal` (off `csp-dev` @ `6a8a12a`) · Status: draft for Claire's approval

## Intent

Follow every neuron on the GRN longitudinal plate from d5 to d17, measure its perinuclear mCherry every day, and record the day it dies. The product is a per-cell table in survival format so a Finkbeiner-style risk-of-death analysis (Cox proportional hazards, mCherry as covariate, genotype × line as factors) can be run on it.

This is a **new dataset and a new module**. It does not extend the 260213 pilot code, the fixed-IF LAMP1 code, or Makenna's upstream atlas. It reuses only ND2 reading and the phase-correlation registration primitive already in `tmem_align`.

## The data

- **Location** — `/Users/clairepeterson/Desktop/TMEM mCherry mNG longitudinal/` on the laptop (`pmi-local-tunnel`). 414 GB, laptop only. Not copied anywhere for the pilot.
- **Sessions** — 13 daily folders `2026MMDD_HHMMSS_mmm/`, d5 → d17 (2026-10-10 → 10-22). Folder `20261013_162922_585` is a 2-file brightfield test: skip it.
- **Files** — one ND2 per well per day, 96 wells per session (rows C–N × columns 03–10). Name stem `2610DD_dN_Z61_Z62_GRN_mCh_mNG_Well<R><CC>_Channel488nm,561nm_Seq<NNNN>.nd2`. The `_mNG` token is absent on d14–d16; parse the day from `_d(\d+)_` and the well from `_Well([A-N]\d{2})_`, never from the stem as a whole.
- **Per file** — `CYX` `2 × 9421 × 9421` uint16, single plane, single position, 20× / 0.75 NA, **0.3238 µm/px** (3.05 mm field = whole well, pre-stitched by NIS). 355 MB.
- **Channels** — index 0 `488nm` = **mNeonGreen nucleus**, identity only. Index 1 `561nm` = **mCherry**, readout only. Embedded ex/em metadata is wrong on both channels; trust the names. mCherry is roughly 5× dimmer than mNG (d5 F05: mNG p99 1615, mCherry p99 306).
- **Plate map** — vault `03_contexts/2026_tmem/reference/mng-longitudinal-plate-map.md`. Four conditions, WT vs GRN-mutant (KO) × lines 61 / 62; each column holds one condition in rows C–H and another in rows I–N; 24 imaged wells per condition, 12 per row block. Ship it as `configs/grn_longitudinal_plate_map.csv` (long format: `column,block,rows,genotype,line`).

## Goals

1. **Tracks** — every mNG nucleus detected on its first day is followed day to day until it is censored at d17, dies, or is lost; each track end carries a reason.
2. **Readout** — per cell per day: mean and integrated mCherry in a perinuclear ring, background-subtracted.
3. **Survival table** — one row per track, ready for `lifelines`: `start_day, stop_day, event, baseline_mcherry, genotype, line, block, well, track_id`, plus a long form with the time-varying mCherry.
4. **QC that a person can look at** — per-well nucleus count over days, per-well track montage, per-track end-state reason.
5. **Runs on the laptop for 4 wells** within an afternoon; scales to 96 wells on pmihack without redesign.

## Non-goals (v1)

- Fitting the Cox model. v1 emits the table; the model is a notebook on top of it.
- mCherry puncta-versus-diffuse analysis, or any spatial pattern metric.
- Stitching, Z, multipoint — none exist in this acquisition.
- Any use of the 561 channel for detection, linking, or registration.
- Merging or importing Makenna's upstream `audit_v1` / atlas code.
- A GUI. napari is for looking, not for annotation, in v1.

## Anti-goals (would make it a failure even if it "works")

- A dead cell reassigned to a neighbor so the track continues. Conservative linking beats long tracks.
- "Lost" and "died" conflated in the survival table. They are different end states with different downstream treatment.
- Phenotype leaking into geometry: a measurement region whose shape depends on the mCherry signal.
- Numbers that cannot be re-derived: every track must be reproducible from raw ND2 + config + code, no hand-edited intermediates.

## Decisions already taken (Claire's unless marked *mine*)

| Decision | Value | Why |
|---|---|---|
| Identity channel | 488 mNG only | stable nuclear label; same rule as 260213 |
| Readout channel | 561 mCherry only | the biology being measured |
| Readout region | *mine* ring from nuclear boundary to **5 µm** outside (≈ 15 px), Voronoi-clipped against neighbors, nucleus excluded | mCherry is perinuclear (Claire); nucleus excluded so mNG bleed cannot inflate it; width is a config parameter |
| Background | *mine* per well per day: mode of 561 pixels outside all rings and nuclei | removes day-to-day illumination drift without touching cells |
| Death rule | *mine* nucleus absent on **2 consecutive days** within the search radius → died, event day = first absent day; a single missing day is a gap, not an event | guards against one-day segmentation misses; replace once Claire has seen real deaths |
| Lost rule | *mine* track ends because the predicted position is within a margin of the image edge, inside a registration-failed region, or the nucleus merges irrecoverably with a neighbor | these are not biology |
| Censoring | alive on d17 → censored at 17 | last imaging day |
| Statistical factors | genotype × line, row block as stratum | plate is balanced 24/24/24/24 with blocks as replicates |
| Imaged set | columns 03–10 only | the acquisition; map columns 11–18 are not on disk |
| Compute | pilot on the laptop (128 GB, 18 cores, no GPU); full plate later on pmihack | data is on the laptop; no copy until the pilot proves the method |
| Package placement | new subpackage `src/tmem_align/longitudinal/` + CLI group `tmem-align longitudinal …` | separate dataset, same repo (2026-08-29 architecture decision: don't split repos) |
| Python | 3.12 via uv on the laptop; the repo's `zarr<3` pin and ruff line length 100 apply | laptop env built 2026-11-10 |

## Pipeline per well

```
13 ND2 (d5..d17)
 → 1. discover + validate      files, days, plate-map lookup, skip test session
 → 2. register (mNG)           coarse at 1/8 scale → refine full-res; shift per day vs anchor; QC residual
 → 3. segment nuclei (mNG)     per day; label image + table (y, x, area, intensity)
 → 4. link                     day→day, reciprocal NN + global assignment, gap ≤ 1, no reassignment of vanished nuclei
 → 5. end states               censored / died / lost, with reason
 → 6. measure (mCherry)        ring per nucleus per day, Voronoi-clipped, background-subtracted
 → 7. write                    cells.parquet, tracks.parquet, survival.csv, counts.csv, qc/
 → 8. montage                  per-well figure: tracks over days, deaths marked
```

**Registration** — reuse `register_translation` on mNG at 1/8 scale (1178²) for the coarse shift, then refine at full resolution on a central crop or tiled, float32, to keep memory in check. Anchor = d5 (or the first available day). Record per-day shift and correlation; flag a day when the residual exceeds a threshold and mark that day's measurements `registration_failed`. Translation only; no rotation in v1 (plate is not removed between consecutive days — confirm from shifts in the pilot).

**Segmentation** — left to the builder, with this guidance: start classical (background subtraction, Gaussian, local threshold, watershed on distance transform) because nuclei are round, bright, and the laptop has no GPU. Target is **hundreds of nuclei per well**, not dozens; global Otsu on this assay found 23–47 and is known to be wrong. Cellpose `nuclei` on CPU is the fallback if classical cannot separate touching nuclei. Whatever is chosen, cache label images per well per day so linking and measuring can be re-run without re-segmenting.

**Linking** — the four failure modes in `docs/tracking_hardening_plan.md` (ID collision, cumulative drift, drift-swap, no QC signal) are the requirements list. Link **consecutive days** (not day 0 → day N), require reciprocal nearest neighbors, resolve the remainder with a global assignment (Hungarian on distance) under a max displacement in µm, allow a one-day gap, and never let a nucleus that has vanished be matched to anything on a later day. Every link records distance and whether it was reciprocal; every track records how it ended.

**Measurement** — ring = `dilate(nucleus, r_px) − nucleus`, intersected with that nucleus's Voronoi cell computed on all nuclei present that day, intersected with the registered field. Report `ring_mean`, `ring_integrated`, `ring_area_px`, `nucleus_area_px`, `nucleus_mng_mean`, `background`. If the ring is clipped below a minimum area, flag it.

## Outputs

`outputs/longitudinal/<plate>/<well>/`

- `cells.parquet` — one row per nucleus per day: `well, day, track_id, label, y, x, area_px, mng_mean, ring_mean, ring_integrated, ring_area_px, background, linked_from, link_dist_px, reciprocal, flags`
- `tracks.parquet` — one row per track: `well, track_id, first_day, last_day, end_state ∈ {censored, died, lost}, end_reason, n_days, n_gaps, baseline_ring_mean, baseline_day`
- `survival.csv` — per track in counting-process form for `lifelines`: `track_id, well, genotype, line, block, start, stop, event, baseline_mcherry, mcherry_tv`
- `counts.csv` — per day: nuclei detected, tracks alive, deaths, losses
- `registration.csv` — per day: dy, dx, residual, status
- `qc/montage.png`, `qc/tracks_<n>.png` — visual checks
- `run.json` — config, code version, timings

## Pilot (laptop)

- **Wells** — one per condition, all from row block C–H so block is held constant: `C03` WT 61, `C04` KO 61, `C05` WT 62, `C06` KO 62. Second pass adds the I–N counterparts if time allows.
- **Days** — all 13.
- **Entry point** — `tmem-align longitudinal track-well --config configs/grn_longitudinal.yaml --well C03`, plus a `scripts/run_grn_pilot.py` that loops the four wells.
- **Pass criteria**
  - registration residual ≤ 2 px at full resolution on ≥ 12 of 13 days per well
  - ≥ 100 nuclei detected per well on d5, and the builder shows a segmentation overlay that Claire agrees looks right
  - of d5 nuclei, ≥ 70 % end as censored-at-d17 or died; lost ≤ 10 %
  - the montage lets Claire confirm or replace the death rule by eye on at least five real deaths
  - mCherry trajectories for a handful of cells are smooth across days (no sawtooth from mis-links)
  - one well runs end to end in under 15 minutes on the laptop
- **What the pilot decides** — segmentation method, ring width, death rule, max displacement, whether rotation is needed. Each is a config value, not code.

## Testing

- **Synthetic well** — generated field with N round nuclei, known drift per day, known deaths (nuclei removed from day k on), known mCherry per cell; assert tracks, end states, event days, and ring means recover the truth. This is the main regression test; it runs in seconds at reduced size.
- **Unit** — reciprocal matching, gap handling, Voronoi clipping, end-state assignment, filename parser (including the d14–d16 naming slip), plate-map lookup.
- **No test reads a real ND2**; real-data checks are the pilot pass criteria above.

## Performance budget (laptop)

- One well raw in RAM: 13 × 355 MB ≈ 4.6 GB uint16. Fine. Do not hold more than one well at a time.
- Full-res FFT on 9421² float32 ≈ 355 MB per array; coarse-then-refine keeps full-res correlation to a crop. Segmentation per day on 9421² classical ≈ tens of seconds; 13 days × 4 wells is minutes.
- Parallelism across wells only (one process per well, `OMP_NUM_THREADS` capped), never within a well in v1.

## Left to the builder

- Segmentation algorithm and its parameters, within the guidance above.
- Exact downsample factor and refine strategy for registration.
- Label-cache format (npz or zarr; zarr must stay `<3`).
- Hungarian vs. greedy-after-reciprocal for the residual assignment.
- Whether `survival.csv` time-varying form is wide or long; `lifelines` accepts both.
- Montage layout.

## What this does not decide yet (needs Claire, after the pilot)

- The real death signature in mNG, and whether the 2-day rule stands.
- Whether baseline mCherry is the d5 value or the mean of the first three days.
- When and how the 414 GB moves to pmihack for the full plate.

## References in this repo and the vault

- `src/tmem_align/register.py` — `register_translation`, reused as is.
- `src/tmem_align/nd2_tools.py` — ND2 reading, reused.
- `scripts/make_tracking_gif.py` and `docs/tracking_hardening_plan.md` — prior greedy tracker and its failure analysis; requirements source, not code source.
- Vault: `03_contexts/2026_tmem/reference/mng-longitudinal-plate-map.md`, `TMEM Project Overview 2026` (dataset entry), `02_knowledge/microscopy/cell_segmentation_methods_tmem_neuron_aligner.md` (segmentation target).
