# mCherry graphical pilot analysis

This note summarizes the current local graphical analysis of mCherry puncta and diffusion metrics. It uses fluorescence files only, and only wells with mCherry:

- Reporter controls: `E`, `I`, `M` rows (`PLD3 + mCherry`)
- Primary condition: `F`, `J`, `N` rows (`PLD3 + TMEM106B + mCherry`)

Rows without mCherry are excluded from puncta/diffusion interpretation. Missing mCherry in `C/D/G/H/K/L` rows is not treated as zero puncta.

## Current processed subset

The current expanded pilot includes two columns of matched replicate wells:

- Column 05: `E05/F05`, `I05/J05`, `M05/N05`
- Column 06: `E06/F06`, `I06/J06`, `M06/N06`
- Days: `8`, `25`, `39`
- Registration channel: `488nm Binned`, channel index `2`
- mCherry phenotype channel: `561nm Binned`, channel index `1`

This is still a pilot. It is not a full-plate analysis and not proof of lysosomal rupture.

## Graphical outputs

Generated figures and tables are local outputs outside Git:

```text
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/mcherry_graphical_analysis/
  combined_mcherry_metrics.csv
  condition_day_summary.csv
  paired_primary_minus_control_delta.csv
  mcherry_metric_trajectories.png
  mcherry_condition_mean_sem.png
  mcherry_primary_minus_control_delta.png
  mcherry_puncta_diffuse_scatter.png
```

The most useful starting figure is:

```text
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/mcherry_graphical_analysis/mcherry_condition_mean_sem.png
```

## Current condition means

Mean values across the 6 processed reporter-control wells and 6 processed primary wells:

| Condition | Day | n wells | Puncta count | Punctate mean | Diffuse mean | Diffuse / punctate |
|---|---:|---:|---:|---:|---:|---:|
| PLD3 + mCherry | 8 | 6 | 849.3 | 195.6 | 8.78 | 0.045 |
| PLD3 + mCherry | 25 | 6 | 689.7 | 131.1 | 7.25 | 0.055 |
| PLD3 + mCherry | 39 | 6 | 827.0 | 214.3 | 8.31 | 0.039 |
| PLD3 + TMEM106B + mCherry | 8 | 6 | 867.8 | 137.2 | 8.78 | 0.064 |
| PLD3 + TMEM106B + mCherry | 25 | 6 | 733.0 | 44.1 | 7.32 | 0.166 |
| PLD3 + TMEM106B + mCherry | 39 | 6 | 888.7 | 69.4 | 6.84 | 0.098 |

The clearest pilot signal is the Day 25 increase in diffuse/punctate mCherry score in the primary condition. The primary wells also show much lower punctate mean intensity than reporter controls at Day 25 and Day 39.

## Applicable dataset manifest

The filename/size-only manifest is:

```text
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/dataset_manifest/
  nd2_filename_size_manifest.csv
  mcherry_applicable_nd2_manifest.csv
  mcherry_applicable_summary.csv
```

The valid fluorescence mCherry subset contains 864 ND2 files:

- `432` reporter-control fluorescence ND2 files
- `432` primary fluorescence ND2 files

This is still too much to run as a casual next step. Scale by selected columns and days.

## QC cautions

Registration QC for column 05 is under:

```text
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/registration_qc/
```

Registration QC for column 06 is under:

```text
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/registration_qc_column06/
```

Large-shift flags occurred in both columns, so the next optimization should be registration quality control before larger batch quantification. Large shifts may represent stage-position differences, FOV mismatch, or registration ambiguity, and can bias whole-frame quantification even when common-overlap cropping is used.

## Further optimization and uses

Recommended optimizations:

1. Build an automated QC gate that excludes or flags wells with large registration shifts before group statistics.
2. Use stage coordinates to pre-check whether Day 8, Day 25, and Day 39 are likely the same field of view before pixel registration.
3. Add neuron or cell-body ROIs so quantification is not dominated by empty well area or field-level drift.
4. Test multiple segmentation thresholds on a small hand-reviewed subset before locking the puncta metric.
5. Keep using the 488nm channel for registration and 561nm for phenotype; do not use mCherry punctation itself as the registration target.
6. For larger viewing outputs, prefer chunked OME-Zarr rather than broad TIFF conversion.
7. Add a lab review notebook that displays the registered stack, QC overlay, metrics, and pass/fail status per well.

Practical uses now:

- Identify wells with strongest diffuse mCherry signal.
- Compare reporter controls against matched TMEM106B+mCherry wells.
- Prioritize wells/days for manual visual review.
- Generate figures for lab discussion.
- Decide which subset is worth converting to a more viewer-friendly time-series format.
