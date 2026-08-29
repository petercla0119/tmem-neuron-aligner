# Microscopy Intake Report — `cleaved_tmem_pld3_260821`

Scanned 2026-08-21. Extractor: `intake.py` (nd2 reader 0.11.3). No pixels loaded
during extraction; a small QC sample was loaded separately (see QC section).

## Dataset snapshot

| Property | Value |
|---|---|
| Files | 72 ND2, all parsed OK (0 failures) |
| Total size | 27.8 GB |
| Format | Nikon ND2 (raw, quantitatively safe — no PNG/JPEG exports) |
| Dimensions | ZCYX per file: 4 channels, **Z = 6–16 slices (varies)**, 2048×2048, uint16 |
| Time | none (single timepoint, `d7`) |
| Pixel size | 0.1083 µm/px (X=Y), **consistent across all 72** |
| Z-spacing | 0.3 µm |
| Objective | 60× water, NA 1.27 (Plan Apo IR 60x WI DIC N2), consistent |
| Design | **6 rows (C–H) × 3 conditions × 4 fields = 72** (**D17 present but should not exist — exclude, 4 files**) |
| File mtime range | 2026-08-12 → 2026-08-21 (embedded acquisition timestamp: none) |
| Laser power (observed) | 488 → 100 · 640 → 100 · 561 → 65 · 405 → 83 — **identical to d14 and d28** |
| Conversion gain (observed) | HDR, all channels — **identical to d14 and d28** |

**Experimental layout (recovered from folder + filename):**

| Condition | Folder | Filename prefix | Plate column | Wells |
|---|---|---|---|---|
| TMEM KO | `TMEM_KO` | `TMEMKO_` | 8 | C8–H8 |
| PLD3 control | `Z59_PLD_Control` | `PLD3Control_` | 17 | C17–H17 |
| PLD3 + TMEM106B ki | `Z60_PLD_TMEMki` | `PLD3TMEM106B_` | 20 | C20–H20 |

Each well has 4 fields (F1–F4). This is **not the mCherry survival assay** in
the main pipeline — it's a fixed 4-channel immunofluorescence panel.

## Findings

| Category | Found | Missing / Conflicting | Why it matters | Severity | Action |
|---|---|---|---|---|---|
| **Channel identity** | `TMEM561 LAMP1640 MAP2488 DAPI405` → C0=488/MAP2, C1=640/LAMP1, C2=561/**cleaved TMEM106B**, C3=405/DAPI. **C2=561=cleaved TMEM106B confirmed by user.** | MAP2/LAMP1/DAPI targets still `inferred` from filename | Every downstream measurement keys off which channel is cleaved TMEM106B | MEDIUM | Confirm remaining targets in fill-in table |
| **Channel order** | 71/72 files: order `[488, 640, 561, 405]` (561/TMEM at index **C2**) | **1 file swapped:** `Z60_PLD_TMEMki/..._D20_F1.nd2` is `[488, 640, 405, 561]` — 561/TMEM at **C3**. **Verified in both embedded metadata and pixels** (C2 there is bright nuclear = DAPI; C3 is dim = TMEM) | A fixed channel-index → target map (C2=TMEM) would read **DAPI as cleaved TMEM106B** for this one file | HIGH | Index channels by name/wavelength, not fixed position; or drop D20_F1 |
| **Well identity** | Corrected to true wells C8–H20 (see below) | Extractor's regex matched **"D3" from "PLD3"** → 48/72 mislabeled `D3`. **Fixed in this manifest** (re-parsed `_<row><col>_F<n>` from filename) | Well = sample grouping; the raw parse collapsed 48 files onto one fake well | HIGH (resolved) | Confirm wells map to the intended conditions |
| **Condition** | Recovered from folder/prefix into `samples.csv` (`condition` column) | Was absent from raw `samples.csv` | Needed to group replicates | MEDIUM (resolved) | Confirm labels |
| **Z-depth** | Z varies 6–16 slices across files | **Confirmed expected** (autofocus range) — not a defect | Projections & any z-aware step must handle variable depth; per-slice normalization differs | LOW | Decide projection strategy (max/sum/single); no acquisition follow-up needed |
| **Excitation** | **Recovered from ND2 description ExW field and written to `channels.csv`** (488/561/640/405, keyed by channel name so the D20_F1 swap is handled). Emission values (515/450) remain NIS placeholders, not real. | Real per-channel *emission* still unknown | Reproducibility / spectral unmixing docs | LOW | Emission optional; fill from acquisition log if needed |
| **Laser power & gain** | Recovered from ND2 description: power 488/640=100, 561=65, 405=83; conversion gain HDR all channels | — | Cross-session intensity comparability | INFO | **Verified identical across d7/d14/d28** — intensity comparison across timepoints is safe |
| **Exposure** | Embedded: C0=300ms, C1/C2=900ms, C3(405)=500ms — **identical across d7/d14/d28** | — | Intensity comparability across channels | INFO | Confirmed |
| **D17 well** | Present in d7 (`PLD3Control_..._D17_F1–F4`, 4 files) | **Should not exist per experimental design** (d28 correctly has no D17) | D17 must be excluded from analysis | HIGH | Exclude the 4 D17 files |
| **Fields (F1–F4)** | **Confirmed: 4 separate FOVs per well** (not tiles) | — | Do **not** stitch. Fields are **technical replicates within a well** — averaging/pooling them is pseudoreplication if the well is the experimental unit | MEDIUM | Aggregate field measurements to the well before cross-condition stats |
| **Tile overlap** | Extractor estimated ~29% X / 20% Y | **Meaningless — fields are separate FOVs, not a tiled scan.** Ignore `tile_overlap` in the summary | Would only matter for stitching, which does not apply | INFO (resolved) | None |
| **Saturation / blanks** | Sampled 2 files (TMEM_KO E8, PLD_TMEMki E20): 0% saturation, all 4 channels populated | Only 2/72 sampled | Clipped or blank channels invalidate quantification | INFO | — (expand QC if needed) |

## Fill in: Channel metadata
*Excitation now recovered and written to `channels.csv`. Only biological target
(and optionally real emission) needs confirming — re-run with `--update`.*

| Channel index | Filename tag | Biological target (confirm) | Excitation nm (observed) | Laser power | Exposure ms | Notes |
|---|---|---|---|---|---|---|
| C0 | 488 | MAP2 | 488 | 100 | 300 | neuronal marker? |
| C1 | 640 | LAMP1 | 640 | 100 | 900 | lysosome |
| C2 | 561 | cleaved TMEM106B (confirmed) | 561 | 65 | 900 | primary readout; at C3 in D20_F1 |
| C3 | 405 | DAPI | 405 | 83 | 500 | nuclei |

## Fill in: Condition / well confirmation
*Confirm each column maps to the intended biological condition. **Exclude D17 — not part of the design.***

| Plate column | Wells | Condition (recovered) | Correct? | Notes |
|---|---|---|---|---|
| 8 | C8–H8 | TMEM_KO | | |
| 17 | C17, ~~D17~~, E17–H17 | PLD_Control | | D17 present in data but should not exist — exclude |
| 20 | C20–H20 | PLD_TMEMki | | |

## Projection guidance (variable Z, punctate readout)

Z-depth is **balanced across conditions** (PLD_Control 11.3, PLD_TMEMki 11.7,
TMEM_KO 11.4 slices mean; median 12 all three; ranges 6–16 overlap). So the
depth-scaling of sum/max projections will **not bias the between-condition
comparison** — the projection choice is about signal fidelity, not confounding.

For punctate cleaved TMEM106B (C2/561):
- **Max** — recommended 2D choice; preserves puncta peaks regardless of z-plane.
  Noise-sensitive → apply a puncta threshold/denoise. Mild per-image depth
  dependence, harmless here since depth is condition-balanced.
- **Sum** — scales with each image's slice count; harder to interpret. Avoid
  unless normalized by slice count (then ≈ mean).
- **Mean** — depth-robust but dilutes sparse puncta into background.
- **Median** — removes puncta (z-outliers); use only for diffuse/background.

Best fidelity: **3D puncta segmentation on the stack** (no projection at all).

## QC (sampled — do not generalize)

2 files, mid-z plane. No saturation (0% ≥ 65500 DN). Signal present in all
channels; TMEM (C2) is dim/punctate as expected, DAPI (C3) and MAP2 (C0) bright.

## Manifest corrections applied this run

- `images.csv`: `well_id`, `field_id`, `sample_id` re-parsed from filename
  (fixed the "PLD3"→"D3" collision on 48 files).
- `samples.csv`: rebuilt with 18 true wells + `condition` from folder/prefix.
- `channels.csv`: `excitation_nm` recovered from ND2 description ExW field (keyed
  by channel name, so the D20_F1 channel-order swap is handled correctly).
- Pixel-size/objective left as extracted (`observed`).

## Open questions (please answer)

**Critical before analysis**
1. **Channel target map** — confirm the fill-in table. Is C2/561 the *cleaved*
   TMEM106B antibody or total? This is the whole point of the assay.
2. **The one swapped file** (`Z60_PLD_TMEMki/...D20_F1.nd2`) has 405/561 in
   reversed order (561/TMEM at C3 not C2) — verified in metadata + pixels. Was it
   re-acquired with a different config, or should it be dropped? Either way,
   downstream must map channels by wavelength/name, not fixed index.

3. **D17 well** — d7 contains D17 (PLD_Control, 4 files) but D17 should not exist
   per the experimental design (d28 correctly has none). Confirm exclusion.

**Needed for specific analyses**
4. **Variable Z** — confirmed expected (adaptive autofocus range). Only open
   question: which projection do you want downstream (max, sum, single plane)?
5. **Experimental unit** — is one *well* the biological replicate, or one *field*,
   or is the well the technical unit and the line/condition the replicate? (The 4
   FOVs per well are non-adjacent independent fields, likely technical replicates.)

**Resolved**
- ✅ **Fields F1–F4** — 4 independent, non-adjacent FOVs per well. No stitching;
  `tile_overlap` in the summary JSON is meaningless.
- ✅ **Laser power, gain, exposure** — verified identical across d7/d14/d28.
  Cross-timepoint intensity comparison is safe.
- ✅ **Excitation wavelengths** — recovered from ND2 and written to `channels.csv`.
  (Only real per-channel *emission* remains unknown — optional.)
