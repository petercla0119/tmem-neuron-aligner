# Microscopy Metadata Report — d14

**Dataset:** `/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d14`
**Scan timestamp:** 2026-08-29
**Files:** 72 ND2 | **Channels:** 288 (4 per image) | **Samples/wells:** 16 (+ 3 F-row wells inferred from filename)
**Total size:** 27.7 GB | **Parse errors:** 0

## Dataset snapshot

| Property | Value |
|---|---|
| Formats | ND2 (all) |
| Conditions | TMEM_KO (wells C/D/E/F/G/H 8), Z59_PLD_Control (C/D/E/F/G/H 17), Z60_PLD_TMEMki (C/D/E/F/G/H 20) |
| Fields per well | 4 (**non-adjacent, non-overlapping FOVs — not tiles**) |
| Image dimensions | ZCYX — 2048×2048 px, 4 channels, variable Z |
| Pixel size | 0.108 µm x 0.108 µm (observed, Plan Apo IR 60x WI DIC N2, NA 1.27) |
| Z-spacing | 0.3 µm (observed); **depth set adaptively per field** |
| Z-slice range | 8–18 per field (median ~12) — adaptive, expected |
| Acquisition date | 2026-08-25 → 2026-08-28 (file mtime; no embedded timestamp extracted) |
| Stitching | N/A — FOVs are independent fields, not stitched |
| Channels (observed) | 488 nm, 561 nm, 640 nm, 405 nm (excitation confirmed from ND2 ExW field) |
| Exposure (observed) | 488 → 300 ms · 561/640 → 900 ms · 405 → 500 ms |
| Laser power (observed) | 488 → 100 · 640 → 100 · 561 → 65 · 405 → 83 — **identical to d7 and d28** |
| Conversion gain (observed) | HDR, all channels — **identical to d7 and d28** |
| Raw vs processed | Assumed raw |

---

## Metadata dashboard

| Category | Found | Missing / Conflicting | Why it matters | Severity | Action |
|---|---|---|---|---|---|
| 1. Dataset organization | 3 condition folders, 18 wells, 4 fields/well | — | — | INFO | — |
| 2. Image dimensions & axis order | ZCYX, 2048×2048 confirmed | — | — | INFO | — |
| 3. Spatial calibration | Pixel size 0.108 µm, z-spacing 0.3 µm — observed from ND2 | — | Required for morphometry, synapse sizing, co-localization | INFO | — |
| 4. Channels & biological targets | Channel names = laser lines (488/561/640/405); **excitation recovered from ND2 ExW field and written to manifest** (488/561/640/405). Emission 515 nm for 488/561/640 is a NIS-Elements artifact (last laser line written to all). Filename encodes: TMEM=561, LAMP1=640, MAP2=488, DAPI=405 | Biological targets not yet confirmed in manifest | Channel identity required before quantification | HIGH | Confirm targets from filename (fill-in below) |
| 5. Objective & optical configuration | Plan Apo IR 60x WI DIC N2, NA 1.27; spinning-disk confocal (CSU-W1), 50 µm pinhole, PFS on — observed | — | — | INFO | — |
| 6. Detector & intensity encoding | uint16, Prime BSI camera, HDR conversion gain, laser power per channel (488/640=100, 561=65, 405=83) — all recovered from ND2 description | — | Intensity comparison across sessions | INFO | **Verified identical across d7/d14/d28** — safe to compare intensities |
| 7. Z-stack structure | 0.3 µm spacing, adaptive depth (8–18 planes) | — | Adaptive per-field Z is by design | INFO | Use per-field z-count; do not assume fixed depth |
| 8. Time-series structure | Single time point per file | — | — | INFO | — |
| 9. Tiling & stitching | 4 independent FOVs per well (F1–F4), **non-adjacent, non-overlapping** | — | Fields are separate neurons/regions, not tiles | INFO | Do NOT stitch. Ignore the `tile_overlap` estimate in dataset_summary.json — it is meaningless for non-adjacent FOVs |
| 10. Biological sample identity | Well IDs observed (ND2 metadata): C/D/E/G/H per condition. F-row wells (F8, F17, F20 — 12 files) missing from ND2 metadata; parseable from filename | Condition not mapped in manifest | Sample identity needed for grouping | HIGH | See fill-in below; apply condition mapping |
| 11. Experimental design & replication | 3 conditions × wells × 4 fields. **D17 present here but should not exist per experimental design** | Fields ≠ biological replicates (same well). Unexpected D17 well. Experimental unit unclear. | D17 must be excluded; pseudoreplication risk | HIGH | Exclude D17 (see finding 1); confirm biological replicate unit |
| 12. Preprocessing & provenance | Assumed raw | Not stated in metadata | — | INFO | Correct if flat-field or other preprocessing applied before export |
| 13. File format & storage | ND2, lossless uint16 | — | — | INFO | — |
| 14. Registration | Not yet performed | — | — | INFO | — |
| 15. Analysis-specific | TMEM/LAMP1/MAP2/DAPI IF panel on neurons; per-field independent FOVs | Channel-to-target mapping not in manifest | Blocks automated quantification routing | HIGH | Fill channels table below |

---

## Fill in: Channel biological targets

*Excitation is now recovered and written to `channels.csv`. Only biological target needs confirming — edit and re-run with `--update`.*

| Channel (laser) | Biological target (confirm) | Excitation nm (observed) | Laser power | Exposure |
|---|---|---|---|---|
| 488 nm | MAP2 ? | 488 | 100 | 300 ms |
| 561 nm | TMEM106B ? | 561 | 65 | 900 ms |
| 640 nm | LAMP1 ? | 640 | 100 | 900 ms |
| 405 nm | DAPI ? | 405 | 83 | 500 ms |

---

## Fill in: Sample/condition mapping

*F-row wells were not captured in ND2 well metadata; all other wells confirmed from ND2. Condition column is blank for all wells — fill in below. **Exclude D17 — not part of the design.***

| Well | Condition (from folder) | Biological replicate unit | Notes |
|---|---|---|---|
| C8, D8, E8, F8, G8, H8 | TMEM_KO | | |
| C17, ~~D17~~, E17, F17, G17, H17 | Z59_PLD_Control | | D17 present in data but should not exist — exclude |
| C20, D20, E20, F20, G20, H20 | Z60_PLD_TMEMki | | |

---

## Consolidated questions

### Critical before any analysis

1. **Unexpected D17 well** — d14 contains D17 (Z59_PLD_Control, 4 files) but per the experimental design D17 should not exist (d28 correctly has none). These 4 files should be excluded from analysis. Confirm exclusion, or clarify if D17 was a deliberate extra.

2. **F-row well metadata** — Wells F8, F17, F20 (12 files total) have no well_id in the ND2 file metadata; the well is inferred from the filename. No data is missing — just confirming the parse is correct.

### Needed for quantitative comparisons

3. **Experimental unit** — Which unit constitutes one biological replicate: a single well, a single neuron, or something else? Each of the 4 FOVs per well is an independent, non-adjacent field — likely technical replicates within a well.

### Resolved

- ✅ **Z-stack depth** — adaptive per field, by design. Variable 8–18 slices is expected.
- ✅ **Stitching** — FOVs are non-adjacent; no stitching, no overlap. `tile_overlap` in the summary JSON is meaningless here.
- ✅ **Laser power & gain** — verified identical across d7/d14/d28 (see snapshot). Intensity comparison across timepoints is safe.
- ✅ **Excitation wavelengths** — recovered from ND2 and written to `channels.csv`.

---

*Output files:* `images.csv`, `channels.csv`, `samples.csv`, `dataset_summary.json` — `reports/intake/d14_intake/`
*Re-run with `--update` after editing fill-in tables to merge values into the manifest.*
