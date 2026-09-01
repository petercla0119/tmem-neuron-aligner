# Flat-field Correction — Validation Sweep Report

**Date:** 2026-08-31  **Branch:** `dev/preprocess/flatfield`  **Script:** `scripts/flatfield_sweep.py`

## Settings used

- **Sigma grid:** [8, 25, 51, 102, 256] px (short-side/20, /40, /80 + guard rails 8, 256)
- **Default sigma:** 51 px (short-side/40)
- **Estimator:** median-of-population, pooled across d7+d14+d28
- **Images pooled:** 209 ND2 files across all timepoints
  - d7: 72, d14: 72, d28: 65
- **Channel keying:** by name (`488nm`, `561nm`, `640nm`, `405nm`) — D20_F1 swap handled automatically
- **Field model:** one 2-D YX field per named channel, broadcast across Z for 3-D stacks
- **Darkfield:** estimated scalar per channel via 1st-percentile-of-minima

## Estimated darkfield offsets (ADU)

| Channel | Darkfield offset |
|---------|-----------------|
| MAP2 488 | 95.8 |
| cl-TMEM 561 | 91.3 |
| LAMP1 640 | 99.7 |
| DAPI 405 | 97.8 |

## Field metrics per sigma

Spatial CV% = std/mean × 100 over the full 2-D field.  
Corner/center ratio = mean of four 128×128 corner patches / 256×256 center patch.

### MAP2 488

Raw cross-FOV CV% (uncorrected mean/FOV): **23.1%**

| Sigma | Spatial CV% | Corner/center |
|-------|------------|---------------|
| 8 | 13.58% | 0.8736 |
| 25 | 10.48% | 0.8761 |
| 51 | 8.58% | 0.8821 ← default |
| 102 | 6.62% | 0.9030 |
| 256 | 4.00% | 0.9325 |

### cl-TMEM 561

Raw cross-FOV CV% (uncorrected mean/FOV): **11.2%**

| Sigma | Spatial CV% | Corner/center |
|-------|------------|---------------|
| 8 | 0.64% | 1.0081 |
| 25 | 0.55% | 1.0080 |
| 51 | 0.49% | 1.0076 ← default |
| 102 | 0.42% | 1.0054 |
| 256 | 0.29% | 1.0017 |

### LAMP1 640

Raw cross-FOV CV% (uncorrected mean/FOV): **18.8%**

| Sigma | Spatial CV% | Corner/center |
|-------|------------|---------------|
| 8 | 3.37% | 1.0224 |
| 25 | 3.32% | 1.0225 |
| 51 | 3.25% | 1.0230 ← default |
| 102 | 3.08% | 1.0236 |
| 256 | 2.41% | 1.0170 |

### DAPI 405

Raw cross-FOV CV% (uncorrected mean/FOV): **40.3%**

| Sigma | Spatial CV% | Corner/center |
|-------|------------|---------------|
| 8 | 4.11% | 0.9141 |
| 25 | 3.68% | 0.9150 |
| 51 | 3.36% | 0.9157 ← default |
| 102 | 3.02% | 0.9198 |
| 256 | 2.30% | 0.9389 |

## Figures

| Figure | What it shows |
|--------|--------------|
| `field_renders.png` | Estimated 2-D fields — 4 channels × 5 sigmas. Check for cell-shaped structure (trip-wire for BaSiC revisit). |
| `metrics_vs_sigma.png` | Spatial CV% and corner/center ratio vs sigma. Pick sigma at the CV% knee. |
| `fov_ki_sigma_comparison.png` | KI FOV raw vs corrected at each sigma (MAP2 channel). |
| `fov_ko_sigma_comparison.png` | KO FOV raw vs corrected at each sigma (MAP2 channel). |
| `darkfield_ab.png` | Darkfield ON vs OFF at default sigma. |
| `pooled_vs_pertimepoint.png` | Pooled field vs per-timepoint fields at default sigma. |
| `z_profile.png` | Mean intensity per Z-plane before/after — single 2-D field should not introduce axial artifact. |
| `well_examples.png` | Before/after at default sigma for KO / Control / KI wells (MAP2 + cl-TMEM channels). |

## Decision 6 — smoothing sigma

**Chosen sigma: 102 px (short-side/20)**

Rationale (from visual inspection of `field_renders.png` and `metrics_vs_sigma.png`):

- σ=8 and σ=25: **rejected** — MAP2 field shows individual cell somata and neurite networks as dark blobs (trip-wire: cell-shaped structure → BaSiC revisit would be needed, but at σ=102 the trip-wire is not triggered).
- σ=51 (prior default): **marginal** — MAP2 field still has broad cloud-like structure consistent with biological contamination in the median.
- **σ=102: first clean sigma** — MAP2 field render shows a smooth gradient with no distinguishable cell shapes. LAMP1 640 retains the real corner-vignette (corner/center = 1.024). Spatial CV% = 6.6% (meaningful illumination heterogeneity still captured, not oversmoothed).
- σ=256: **over-smoothed** — MAP2 CV% drops to 4.0% and corner/center rises to 0.93, suggesting the real vignette signal is being lost.

**Action:** update `smooth_sigma` default from `short_side/40` (51 px) to `short_side/20` (102 px) in the preprocess pipeline. Write back to Decision 6 of `flatfield_correction_plan.md`.

## Caveats

- No uniform reference slide exists — validation shows *consistency/uniformity*, not absolute accuracy.
- Darkfield is a 1st-percentile-of-minima estimate, not a measured dark frame.
- Same 2-D field broadcast across all Z planes; axial Z-dependence of illumination is not corrected.
- D20_F1 channel swap handled by name-keying (not excluded — data is used with correct channel assignment).
