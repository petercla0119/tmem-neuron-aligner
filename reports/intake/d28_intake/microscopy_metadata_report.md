# Microscopy Metadata Report — d28

**Dataset:** `/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d28`
**Scan timestamp:** 2026-08-29
**Files:** 65 ND2 | **Channels:** 260 (4 per image) | **Samples/wells:** 15 (+ 3 F-row wells inferred from filename)
**Total size:** 26.9 GB | **Parse errors:** 0

## Dataset snapshot

| Property | Value |
|---|---|
| Formats | ND2 (all) |
| Conditions | TMEM_KO (wells C/D/E/F/G/H 8), Z59_PLD_Control (C/E/F/G/H 17 — **no D17, correct per design**), Z60_PLD_TMEMki (C/D/E/F/G/H 20) |
| Fields per well | 4 (**non-adjacent, non-overlapping FOVs — not tiles**) |
| Image dimensions | ZCYX — 2048×2048 px, 4 channels, variable Z |
| Pixel size | 0.108 µm x 0.108 µm (observed, Plan Apo IR 60x WI DIC N2, NA 1.27) |
| Z-spacing | 0.3 µm (observed); **depth set adaptively per field** |
| Z-slice range | 9–23 per field (median ~13) — adaptive, expected |
| Acquisition date | 2026-08-28 (all files same day, mtime 16:58–20:07 UTC) |
| Stitching | N/A — FOVs are independent fields, not stitched |
| Channels (observed) | 488 nm, 561 nm, 640 nm, 405 nm (excitation confirmed from ND2 ExW field) |
| Exposure (observed) | 488 → 300 ms · 561/640 → 900 ms · 405 → 500 ms |
| Laser power (observed) | 488 → 100 · 640 → 100 · 561 → 65 · 405 → 83 — **identical to d7 and d14** |
| Conversion gain (observed) | HDR, all channels — **identical to d7 and d14** |
| Raw vs processed | Assumed raw |

---

## Metadata dashboard

| Category | Found | Missing / Conflicting | Why it matters | Severity | Action |
|---|---|---|---|---|---|
| 1. Dataset organization | 3 condition folders, 16 wells (Z59_PLD_Control has C/E/F/G/H 17 — **no D17, correct per design**) | — | D17 is not part of the design; its absence here is correct | INFO | None — d14's D17 is the anomaly, not this |
| 2. Image dimensions & axis order | ZCYX, 2048×2048 confirmed | — | — | INFO | — |
| 3. Spatial calibration | Pixel size 0.108 µm, z-spacing 0.3 µm — observed from ND2 | — | — | INFO | — |
| 4. Channels & biological targets | Same 4-channel panel as d14 (488/561/640/405). **Excitation recovered from ND2 ExW field and written to manifest.** Filename encodes: TMEM=561, LAMP1=640, MAP2=488, DAPI=405 | Biological targets not yet in manifest | Channel identity required before quantification | HIGH | Fill channels table below |
| 5. Objective & optical configuration | Plan Apo IR 60x WI DIC N2, NA 1.27; spinning-disk confocal, 50 µm pinhole — same as d14 | — | — | INFO | — |
| 6. Detector & intensity encoding | uint16, Prime BSI, HDR gain, laser power per channel (488/640=100, 561=65, 405=83) — recovered from ND2 description | — | Intensity comparison to d7/d14 | INFO | **Verified identical across d7/d14/d28** — safe to compare intensities |
| 7. Z-stack structure | 0.3 µm spacing, adaptive depth (9–23 planes) | `D8_F4.nd2` has 23 slices and `D8_F1.nd2` 19 (both TMEM_KO D8) — deepest fields | Adaptive Z is by design; deep D8 fields likely thick neuron clusters | INFO | Use per-field z-count; optionally eyeball D8_F4/F1 |
| 8. Time-series structure | Single time point per file | — | — | INFO | — |
| 9. Tiling & stitching | 4 independent FOVs per well (F1–F4), **non-adjacent, non-overlapping** | — | Fields are separate neurons/regions, not tiles | INFO | Do NOT stitch. Ignore the `tile_overlap` estimate in dataset_summary.json |
| 10. Biological sample identity | Well IDs observed from ND2 for non-F-row wells. F-row wells (F8, F17, F20 — 12 files) parsed from filename. | Condition not mapped in manifest | | HIGH | Fill-in below |
| 11. Experimental design & replication | 3 conditions; Z59_PLD_Control = C/E/F/G/H 17 (5 wells, correct — no D17) | — | Balanced per design | INFO | Confirm biological replicate unit |
| 12. Preprocessing & provenance | Assumed raw | | | INFO | — |
| 13. File format & storage | ND2, lossless uint16 | — | — | INFO | — |
| 14. Registration | Not yet performed | — | — | INFO | — |
| 15. Analysis-specific | Same TMEM/LAMP1/MAP2/DAPI IF panel; per-field independent FOVs | Channel-to-target mapping not in manifest | | HIGH | Fill channels table below |

---

## Fill in: Channel biological targets

*Excitation now recovered and written to `channels.csv`. Only biological target needs confirming.*

| Channel (laser) | Biological target (confirm) | Excitation nm (observed) | Laser power | Exposure |
|---|---|---|---|---|
| 488 nm | MAP2 ? | 488 | 100 | 300 ms |
| 561 nm | TMEM106B ? | 561 | 65 | 900 ms |
| 640 nm | LAMP1 ? | 640 | 100 | 900 ms |
| 405 nm | DAPI ? | 405 | 83 | 500 ms |

---

## Fill in: Sample/condition mapping

| Well | Condition (from folder) | Biological replicate unit | Notes |
|---|---|---|---|
| C8, D8, E8, F8, G8, H8 | TMEM_KO | | |
| C17, E17, F17, G17, H17 | Z59_PLD_Control | | No D17 — correct per design |
| C20, D20, E20, F20, G20, H20 | Z60_PLD_TMEMki | | |

---

## Consolidated questions

### Critical before any analysis

1. **F-row well metadata** — F8, F17, F20 have no well_id in ND2 metadata; inferred from filename. No data missing — just confirming the parse.

### Needed for quantitative comparisons

2. **Experimental unit** — Is one well = one biological replicate? The 4 non-adjacent FOVs per well are independent fields, likely technical replicates within a well.

### Resolved

- ✅ **D17** — correctly absent per experimental design. (d14's D17 is the anomaly and should be excluded there.)
- ✅ **Z-stack depth** — adaptive per field, by design. D8_F4 (23 slices) / D8_F1 (19) are the deepest fields, likely thick neuron clusters.
- ✅ **Stitching** — FOVs are non-adjacent; no stitching, no overlap. `tile_overlap` in the summary JSON is meaningless here.
- ✅ **Laser power & gain** — verified identical across d7/d14/d28. Intensity comparison across timepoints is safe.
- ✅ **Excitation wavelengths** — recovered from ND2 and written to `channels.csv`.

---

*Output files:* `images.csv`, `channels.csv`, `samples.csv`, `dataset_summary.json` — `reports/intake/d28_intake/`
*Re-run with `--update` after editing fill-in tables to merge values into the manifest.*
