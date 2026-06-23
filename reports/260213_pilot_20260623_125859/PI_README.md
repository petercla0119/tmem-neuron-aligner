# 260213 E05/F05 Longitudinal Pilot

Run started: 2026-06-23T12:59:23

## What Was Analyzed

This is a tiny real-data pilot from `/Users/makennarodriguez/Documents/260213_Feb16recopy_HYdiff_landingpadlines_survival_384well1`. The requested folder prefix was
`260213_Feb15recopy`; the local folder found and used was `260213_Feb16recopy_HYdiff_landingpadlines_survival_384well1`.
This naming mismatch should be verified with the acquisition/copy notes before presentation.

Wells analyzed:
- `E05`: PLD3 + mCherry reporter control.
- `F05`: PLD3 + TMEM106B + mCherry primary experimental well.

Timepoints: 8, 12, 16

Channels: registration used 488, measurement used 561/mCherry. The mCherry channel was not used
as the primary registration reference because mCherry redistribution is the phenotype being
screened.

## What The Aligner Did

For each well, the script loaded the first 3 fluorescence ND2 timepoints,
estimated X/Y drift with phase cross-correlation on the stable 488 channel, applied the same
transform to the 561/mCherry channel, cropped to common overlap, and measured mCherry punctate
versus diffuse signal inside a foreground mask.

## Outputs

- `dataset_inventory.csv`: dataset-wide inventory of 1,920 ND2 files.
- `selected_pilot_files.csv`: the six files loaded for this E05/F05 pilot when a full
  `dataset_inventory.csv` was already present.
- `registration_qc.csv`: shifts, correlations, overlap, and pass/fail flags.
- `mcherry_measurements.csv`: per-well/timepoint mCherry measurements.
- `summary_stats.csv`: pilot slopes and QC counts.
- `figures/registration_before_after.png`: 488-channel registration QC.
- `figures/aligned_timeseries_mcherry.png`: aligned mCherry time series.
- `figures/*_aligned_mcherry_timeseries.gif`: animated aligned mCherry examples.
- `figures/mcherry_metric_over_time.png`: mCherry metrics over time.
- `registered_stacks/`: small registered OME-TIFF stacks for review.
- `single_neuron_examples/`: cropped local ROI examples focused on automatically selected
  single-neuron candidates, including mCherry GIFs, overlay GIFs, local registration QC, and
  crop-level mCherry metrics.
- `single_neuron_examples/ome_zarr/`: chunked OME-Zarr exports of the registered local ROI stacks
  for scalable image viewing.

## Single-Neuron Candidate Examples

The `single_neuron_examples/` folder provides presentation-friendly local crops from the aligned
E05/F05 stacks. These were selected automatically from compact 488-positive components with
nearest-centroid consistency across Days 8, 12, and 16, then locally re-aligned on 488. They are
useful for showing how the aligner can focus on neuron-scale time series, but same-neuron identity
should be manually reviewed before making biological claims.

## Pilot Result

E05 (PLD3_mCherry_reporter_control) increased from 2.546 to 3.043 (slope 0.06211 per day). F05 (PLD3_TMEM106B_mCherry_primary) increased from 3.189 to 4.385 (slope 0.1495 per day).

This is real local microscopy data, but it is still a small screening pilot. The metric is a
longitudinal punctate-to-diffuse reporter redistribution score. It is not proof of lysosomal
rupture by itself.

## How This Helps The TMEM106B Paper

This workflow converts raw longitudinal imaging into same-well quantitative trajectories. It can
test whether PLD3+TMEM106B+mCherry shows progressive punctate-to-diffuse reporter behavior
relative to mCherry reporter controls, giving a light-microscopy bridge to the paper's model of
lysosomal TMEM106B fibril accumulation and rupture-like phenotypes. It also helps prioritize
wells, timepoints, and neurons for cryo-CLEM, immunostaining, and lysosome assays.

## Limitations And Next Steps

- Add orthogonal rupture markers: Galectin-3/Galectin-8 recruitment, LAMP1/LAMP2 morphology,
  LysoTracker loss, p62/LC3, or LLOMe positive control.
- Expand to more wells, sites, cells, and replicate pairs before inferential statistics.
- Validate segmentation/tracking manually for same-neuron claims.
- Tighten registration QC thresholds after reviewing failed or large-shift alignments.
- The current pilot has 6/6 registration QC rows passing.
- Not enough independent replicates for mixed-effects or inferential statistics.
