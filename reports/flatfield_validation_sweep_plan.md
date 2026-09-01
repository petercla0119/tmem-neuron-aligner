# Flat-field Correction — Validation Sweep Plan

**Branch:** `dev/preprocess/flatfield` · **Companion to:** `reports/flatfield_correction_plan.md`
**Vault copy:** `03_contexts/2026_tmem/_artifacts/2026-08-31-flatfield-validation-sweep-plan.md`

Purpose: the flat-field correction decisions are locked (see the correction plan), but two things must be chosen/confirmed from data before a production run — the **smoothing sigma (Decision 6, deferred here)** and that the other locked decisions behave as expected on real images. This plan is empirical: run the sweep, read the figures, pick the sigma, confirm the decisions. Reuses the A/B report pattern (`reports/ab_ic_test/`), output to `reports/ab_flatfield_sweep/`.

## Primary axis — smoothing sigma

Sweep the Gaussian smoothing sigma applied to the per-channel median field (`preprocess.py:497`, auto = short-side/40 ≈ 51 px). Candidate grid: **short-side/20 (~102 px), /40 (~51 px, current default), /80 (~26 px)**, plus two guard rails — a near-unsmoothed field (~8 px) and a heavily smoothed field (~256 px). Estimate one pooled per-channel field per sigma (Decision 1).

**Goal:** pick the *smallest* sigma that flattens illumination without the field taking on cell-shaped structure. Too small → the median's residual biology leaks into the "field"; too large → real vignette under-corrected.

## Secondary axes — confirm the locked decisions hold

Run these as A/B arms, not a full grid (keep it cheap):
- **Darkfield on vs off** (Decision 3) — confirm on ≥ off for corner uniformity, no new artifact.
- **Pooled vs per-timepoint field** (Decision 1) — confirm pooled ≈ per-timepoint on uniformity and does not shift the biology more than per-timepoint does.
- **Corrected-3D vs max-proj-2D readout** (Decision 2) — confirm the 3D path preserves the KI ≫ KO > Control ordering.

## Metrics / figures (per channel, before vs after, per sigma)

1. **Field render** — the estimated 2D field must be smooth and centered on 1, with **no cell-shaped structure** (this is the Decision-5 "centered-cell bias" trip-wire: if the field looks like a cell at the chosen sigma, that triggers the BaSiC (Background and Shading Correction) revisit).
2. **Cross-FOV uniformity** — per-channel **CV%** and **corner/center ratio** across the four non-adjacent fields (F1–F4), before vs after. Correction should reduce inter-FOV spread; plot CV% vs sigma to find the knee.
3. **Registration invariance** — shifts unchanged to ±0.1 px (correction must not perturb alignment; matches `AB_COMPARISON.md`).
4. **Biological-effect stability** — E05 vs F05 slope + F05/E05 ratio across sigma; pick the sigma that flattens illumination while moving the readout no more than the baseline A/B already does.
5. **3D-specific** — plane-to-plane corrected-intensity **Z-profile**, confirming the single 2D field removed lateral vignette without introducing an axial artifact (the Decision-2 / R5 check).

## Data

Pooled per-channel fields from the full d7+d14+d28 population (seeded sample, exclude D17, name-keyed channels). Apply to the E05/F05 pilot wells plus one KO / Control / KI FOV per timepoint. Cap threads if run on Cheaha (HPC thread rule).

## Success criteria

- A chosen sigma where per-channel CV% is near its minimum **and** the field render is smooth (no cell structure).
- Darkfield-on ≥ darkfield-off on uniformity, no artifact.
- Pooled field within tolerance of per-timepoint on uniformity and biology shift.
- 3D corrected path preserves KI ≫ KO > Control.
- **Ground-truth caveat:** no uniform reference slide exists, so this shows *consistency/uniformity*, not absolute accuracy — state it in the report.

## Deliverable

An A/B-style report at `reports/ab_flatfield_sweep/` (markdown + figures), and the chosen sigma written back into Decision 6 of `reports/flatfield_correction_plan.md`.
