# Flat-field / Illumination Correction Plan (2D + 3D)

**Branch:** `dev/preprocess/flatfield` (from `preprocess`) · **Target module:** `src/tmem_align/preprocess.py`
**Status:** plan only — no implementation code here.

This plan extends the IC capability already merged on `preprocess` so that flat-field
correction is correct for **both single-plane 2D (Z=1)** and **3D z-stacks (Z>1)**, keyed by
channel *name*, and validated with the existing A/B pattern.

---

## Decisions (locked 2026-08-31, by user)

All six previously-open questions are decided. Brief justifications below; these are authoritative for implementation.

1. **Field grouping — POOLED: one 2D flat-field per named channel across d7+d14+d28.** Laser power, HDR gain, and exposures are identical across the three timepoints (given), so the illumination field is physically the same; pooling maximizes the sample feeding the pixelwise median → more robust field, one artifact to reason about. Keep the per-timepoint path (`calculate_ic_fields_by_timepoint`, `preprocess.py:249`) available for the A/B sanity check only.

2. **Quantification runs on the corrected 3D ZCYX stack.** Correction is applied to the full z-stack and quantify/register consume the corrected 3D data — no pre-IC max-projection. Remove/bypass the `standardize_to_cyx` Z max-projection (`run_260213_longitudinal_pilot.py:301`) for this path; any max-projection for display/QC happens *after* correction.

3. **Darkfield ON by default, scalar per channel.** Dark-subtract before flat-divide (`corrected = (raw − dark) / flat`); skipping it lets the ~100 ADU camera bias get amplified in dim corners. Scalar (not spatial) because the back-illuminated sCMOS (Prime BSI) offset is ~uniform and no dark frames exist to justify a per-pixel map. Estimate via the existing 1st-percentile-of-minima path (`preprocess.py:451`) — an estimate, not a measurement.

4. **Flat-field + dark-subtract run BEFORE deconvolution.** Deconvolution assumes a shading-free, offset-free input; correcting after would fold the illumination profile into the reconstruction. Deconvolution lives on a separate worktree — the handoff artifact is the corrected (dark+flat) ZCYX stack. Order: dark → flat → deconv → quantify.

5. **Keep the median-of-population estimator for now.** Robust pixelwise median → Gaussian smooth → mean-normalize (`preprocess.py:88-99`, `_rescale_field:514`); no new dependency, already tested, robust to bright biological outliers. Revisit BaSiC (Background and Shading Correction, Peng et al. 2017) only if validation shows the median inherited a centered-cell bias on the dense structural channels (488 MAP2 / 405 DAPI). Full comparison: `02_knowledge/methods/Retrospective Illumination Correction — Median-of-Population vs BaSiC.md`.

6. **Smoothing sigma: 102 px (short-side/20). Decided from sweep data 2026-08-31.** σ=8 and σ=25 show cell-shaped structure in the MAP2 field (trip-wire). σ=51 (prior default) is marginal — broad cloud-like contamination. σ=102 is the first clean sigma: smooth gradient, no cell structure, spatial CV%=6.6% (real vignette preserved). σ=256 over-smooths (CV% 4%, corner/center near 1, vignette lost). Update `smooth_sigma` default to short-side/20 in the pipeline. Sweep report: `reports/ab_flatfield_sweep/SWEEP_REPORT.md`.

---

## Existing-code gap analysis

What exists today (all on `preprocess` / this branch), and what is missing for 2D+3D.

**Shipped and working:**
- `calculate_ic_field()` (`preprocess.py:30`) — robust pixelwise median across sampled images,
  Gaussian smooth, mean-normalize (centered on 1). Handles 2D (YX) and 3D **CYX** input; optional
  scalar darkfield return.
- `apply_ic_field()` / `_apply_ic_field_float()` (`preprocess.py:106`, `:374`) — `corrected =
  (image - darkfield) / flatfield`, float throughout, single quantize to uint16
  (`_quantize:352`), zero-field guard (`:390`), field floor at 0.1 (`_rescale_field:526`).
- `_broadcast_like()` (`preprocess.py:357`) — broadcasts a 2D or 3D field against 2D/3D/4D images.
- Per-timepoint orchestration `calculate_ic_fields_by_timepoint()` (`:249`) +
  `preprocess_with_lookup()` (`:298`), CLI `compute-ic-fields` (`cli.py:82`), standalone
  `scripts/compute_ic_fields.py`, and `--illumination-correct` in the pilot
  (`run_260213_longitudinal_pilot.py:64`).
- 90 tests (`tests/test_preprocess.py`, `tests/test_preprocess_integration.py`) covering 2D/CYX/TCYX
  application, darkfield subtraction, seeded sampling, edge cases.

**Gaps that block the 2D+3D requirement:**

- **G1 — `calculate_ic_field` crashes on true z-stacks (ZCYX, 4D).** For a 4D array,
  `multichannel = first.ndim == 3 and channel is None` is False (`preprocess.py:86`), so it falls to
  the else branch and calls `_extract_channel(img, None)` → `normalize_to_2d()`
  (`preprocess.py:485`), which raises `ValueError("Cannot normalize …")` for ndim>3
  (`io.py:47`). Real ND2s are ZCYX with adaptive Z (6–23 planes); `read_image`→`nd2.imread`
  (`io.py:22`) returns the full ZCYX array. Field estimation on raw z-stacks is therefore broken
  today; it only "works" because the pilot max-projects Z first (`run_260213_longitudinal_pilot.py:301`).

- **G2 — Axis meaning is guessed from `ndim`, not from labels.** `_broadcast_like` treats 3D as CYX
  and 4D as TCYX (`preprocess.py:363-371`); `calculate_ic_field` treats `shape[0]` as channels
  (`:88`). A ZCYX z-stack (4D) would be silently misread as TCYX, and a squeezed single-channel
  z-stack ZYX (3D) as CYX. There is no Z-axis awareness anywhere in `preprocess.py`.

- **G3 — Channels keyed by index, not name.** `calculate_ic_field` pools channel `c` positionally
  across files (`preprocess.py:88-92`). Given the `D20_F1.nd2` swap gotcha (channels 2/3 swapped),
  pooling by index mixes 561/cleaved-TMEM into the wrong field. The pilot already resolves channels
  by name for quantification (`choose_channel_index`, `run_260213_longitudinal_pilot.py:668`) but the
  IC field builder does not.

- **G4 — Darkfield not wired.** Estimator exists (`preprocess.py:451`) but neither the CLI
  (`cli.py:82`) nor the pilot passes it; best-practice dark-before-flat is not exercised
  (`AB_COMPARISON.md:47`).

- **G5 — No 2D-vs-3D detection contract.** Nothing decides "is this Z=1 or Z>1" and applies the
  field accordingly; it relies on the caller having already collapsed Z.

---

## Method

**Estimator (keep the shipped one; flag BaSiC as the alternative in Open Q5).**
Per channel, pixelwise **median across the sampled image population** → Gaussian smooth (low-freq
vignette) → mean-normalize to center on 1. This is already implemented and tested
(`preprocess.py:88-99`, `_rescale_field:514`); it needs no new dependency (numpy/scipy/skimage
only). Median is robust to bright biological outliers — validated by
`test_robust_to_single_bright_outlier` (`test_preprocess.py:333`).

**One 2D flat-field per named channel — the core model change.** Illumination vignetting is a
lateral property of the optics + excitation path for a given channel; it is (to first order)
Z-invariant across a single FOV's short stack. So the estimated object is a **dict keyed by channel
name → 2D YX field** (four keys: MAP2/488, LAMP1/640, cleaved-TMEM/561, DAPI/405). This directly
fixes G2/G3 and makes 2D and 3D application fall out of the same 2D field (see next section).

**Per-channel handling / gotchas (given facts, verified against code):**
- Key by channel NAME, resolved from `image.metadata.channels[*].channel.name` as the pilot does
  (`run_260213_longitudinal_pilot.py:278`), NOT by index — fixes the `D20_F1` swap (G3).
- Ignore `emission_nm` (NIS artifact, untrustworthy) — never use it to key channels.
- Exclude well **D17** from the image population before estimation.

**Population for the median — pooled vs per-timepoint (Open Q1).** Proposed default: **pooled across
d7+d14+d28**, per channel, because acquisition settings are identical (given) so one field is
physically justified and the larger sample tightens the median. Keep the per-timepoint path
(`calculate_ic_fields_by_timepoint`) available for the A/B check. Tradeoff: pooling assumes no
lamp/alignment drift between sessions; per-timepoint absorbs drift but has a smaller, noisier sample
and (per `AB_COMPARISON.md`) can shift the biological contrast. **Flagged, not decided.**

**Dark/offset (Q3, proposed default ON + scalar).** Estimate scalar darkfield per channel via the
existing `estimate_darkfield` path (`preprocess.py:451`) and apply dark-before-flat (already the order
in `_apply_ic_field_float:382`). Scalar (not spatial) until a vignetted dark structure is actually
observed in the flats. See "Darkfield explained" above for the on/off and scalar/spatial rationale.

**Non-adjacent FOVs:** no mosaic/overlap illumination estimation is possible (given) — retrospective
population median is the only source. This matches the shipped approach; no change.

### Darkfield explained (Q3 walkthrough)

The correction model is `corrected = (raw − darkfield) / flatfield`. Two independent choices:

- **On vs off** = whether you subtract the camera's zero-light offset before dividing. A camera
  reports a nonzero value even in total darkness: a fixed electronic bias (~100 ADU on the Prime
  BSI). If you divide *without* subtracting it, that constant offset gets scaled by the flatfield —
  amplified most in the dim/vignetted corners — so it distorts intensity ratios and the puncta/diffuse
  measurement. Best practice = **darkfield ON** (subtract first). Cost here: we have no dark frames,
  so it's an *estimate* (1st-percentile-of-minima, `preprocess.py:451`), not a measurement.
- **Scalar vs spatial** = what shape the darkfield is.
  - *Scalar:* one number subtracted everywhere — assumes the offset is uniform across the sensor.
    Simple, robust, hard to overfit. This is what the code does now.
  - *Spatial:* a full 2D per-pixel offset image — captures sensor non-uniformity (amplifier glow,
    thermal gradient, hot pixels). More accurate *only if* such structure actually exists; needs more
    data to estimate and can inject noise if it doesn't.
  - For a modern back-illuminated sCMOS with a near-uniform bias, **scalar is normally enough**. Go
    spatial only if the estimated flats reveal dark structure. **Proposed default: ON + scalar,
    per channel.**

---

## 2D vs 3D handling (concrete logic)

**Detection — from axis labels, not `ndim` (fixes G2/G5).**
- For ND2: read `image.sizes` (an ordered dict of axis→size, as the pilot already does at
  `run_260213_longitudinal_pilot.py:259`). `Z` present with size >1 → **3D z-stack**; `Z` absent or
  size 1 → **2D single-plane**.
- For TIFF: read the OME `axes` string from tifffile metadata (same source `standardize_to_cyx`
  uses). Fall back to: squeeze singleton dims, then interpret remaining axes by label.
- The correction functions should accept an explicit `axes=` argument (e.g. `"ZCYX"`, `"CYX"`,
  `"YX"`) rather than guessing from `ndim`. This is the one real API change to `preprocess.py`.

**Field estimation input.** Before pooling into the median, reduce every image to **CYX** per file:
- 2D file (CYX or YX): use as-is (add channel axis if YX).
- 3D file (ZCYX): reduce Z so each file contributes one YX plane per channel to the population.
  Proposed reduction = **median over Z** (robust; a max-projection would bias the flat-field toward
  the brightest plane). This is the fix for G1 — `calculate_ic_field` must Z-reduce ZCYX inputs to
  CYX instead of calling `normalize_to_2d` and crashing.

**Field application.**
- **2D (Z=1), image CYX:** per-channel 2D division — already works
  (`apply_ic_field`, tested `test_2d_field_3d_image` / `test_3d_field_3d_image`,
  `test_preprocess.py:221,229`). Look up each channel's 2D field by name and divide.
- **3D (Z>1), image ZCYX:** **broadcast the per-channel 2D field across every Z plane** — the same
  lateral field corrects each plane of that channel. Concretely, arrange the field as `(1, C, Y, X)`
  and divide the `(Z, C, Y, X)` stack; `_broadcast_like` already produces a `[newaxis]` leading axis
  for a 3D field against a 4D image (`preprocess.py:369`), so the broadcast mechanics exist — what's
  missing is (a) selecting the right per-channel plane by name and (b) treating the leading 4D axis
  as **Z, not T**. Same-field-every-plane is the correct default because vignetting is lateral;
  per-plane fields would require a per-Z reference we do not have.
- Darkfield (scalar or 2D per channel) subtracts first, broadcasting identically.

**Contract summary:** input axes ∈ {YX, CYX, ZCYX} → detect via labels → estimate per-channel 2D
field → apply as 2D (CYX) or Z-broadcast (ZCYX). TCYX (time) stays supported for the existing pilot
path.

---

## Pipeline integration

Pipeline flow (from `CLAUDE.md`): ND2 → stitch → **preprocess (IC)** → register → ROI → quantify.
FOVs are non-adjacent so "stitch" is a no-op grouping here; IC still slots **after read, before
registration and quantification** — matching AD-4 (`preprocess_surveillance_log.md:42`) and the
current pilot, which corrects each frame *before* `register_stack` (`run_260213_longitudinal_pilot.py:84`).

- **Correction order:** dark-subtract → flat-field divide → (optional rolling-ball background) →
  quantify. Order already enforced and tested (`_apply_ic_field_float:382`,
  `test_ic_applied_before_bg`, `test_preprocess_integration.py:342`). Deconvolution (separate
  worktree, Q4) slots strictly **after** flat-field/dark: dark → flat → (deconv) → quantify.
- **Do NOT** feed IC into the registration *decision* — the A/B report shows IC gives no registration
  benefit (`AB_COMPARISON.md:22`) and registration must not use the mCherry/561 channel (CLAUDE.md).
  Corrected frames can still be the ones registered; the alignment channel stays 488/MAP2.
- **2D vs 3D at the integration point (Q2, DECIDED = 3D):** the fixed-IF quantify path carries the
  **corrected ZCYX stack** through — no pre-IC max-projection; `standardize_to_cyx`'s Z max-projection
  (`run_260213_longitudinal_pilot.py:301`) is removed/bypassed for this path. Any max-projection for
  display/QC happens *after* correction, on the corrected stack. Correction still supports 2D (Z=1)
  inputs unchanged.
- **Deconvolution handoff (Q4):** a deconv step is being built on a separate worktree. IC output
  (dark-subtracted + flat-fielded ZCYX) is the required *input* to deconvolution — flat-field/dark
  must precede deconv. Keep the corrected-but-not-yet-deconvolved stack available as the handoff
  artifact; do not quantify puncta on pre-deconv data if the deconv path supersedes it.

---

## Testing strategy

> **Note (Q6, TBD/later):** the synthetic-gradient recovery test below is kept as a cheap unit-level
> sanity check, but whether it stays the *primary* validation is deferred — we may replace it with a
> real-data or physically-motivated check later. Do not treat it as the authoritative validator.

**Unit tests (synthetic, no real data) — extend `tests/test_preprocess.py`:**
- *Recover a known 2D gradient:* build N images = flat_base × known_illumination_gradient(YX);
  assert estimated field ≈ gradient (up to mean-normalization) and that applying it drops CV%. The
  pattern already exists (`test_ic_reduces_illumination_gradient`,
  `test_preprocess_integration.py:48`) — extend it to assert the recovered field *shape matches the
  gradient*, not just CV drop. (Provisional per Q6.)
- *3D z-stack estimation (new, covers G1):* feed ZCYX arrays (e.g. Z=8, C=2) with a per-channel
  lateral gradient constant across Z; assert `calculate_ic_field` returns a per-channel 2D field
  (no crash) and recovers each channel's gradient.
- *3D application / Z-broadcast (new, covers G5):* apply a per-channel 2D field to a ZCYX stack;
  assert every Z plane is corrected identically and output shape == input shape.
- *2D single-plane still works:* CYX and YX inputs (existing coverage, keep).
- *Channel keyed by name (new, covers G3):* build a population where one "file" has channels swapped;
  assert the name-keyed field for 561 is not contaminated by the swapped file.
- *Dark-before-flat (existing):* `test_scalar_darkfield_subtracts_before_dividing`
  (`test_preprocess.py:270`) — keep; add a 3D variant.
- *Uniform image → field ≈ 1* (`test_uniform_gives_flat_ic`, `test_preprocess_integration.py:412`) —
  extend to ZCYX.
- Ponytail: one runnable assert-based check per new branch (Z-reduce, name-keying, Z-broadcast); no
  new framework.

**Real-data validation — reuse the A/B pattern (`reports/ab_ic_test/`):**
- Rerun the two-arm pilot (baseline vs `--illumination-correct`) exactly as
  `AB_COMPARISON.md:59-66` documents, now with (a) darkfield ON and (b) a true-3D arm if Open Q2
  says quantify-3D.
- **Metrics/figures that demonstrate success:**
  - *Cross-FOV uniformity:* per-channel CV% and corner-vs-center ratio across the F1–F4 non-adjacent
    fields, before vs after — IC should reduce inter-FOV intensity spread.
  - *Field sanity:* render each per-channel 2D field (range, e.g. shipped [1.00, 1.49] from the
    surveillance smoke test, `preprocess_surveillance_log.md:99`); it must be smooth and centered
    on 1.
  - *Registration invariance:* shifts unchanged to ±0.1 px (confirms IC didn't perturb alignment,
    as `AB_COMPARISON.md:22`).
  - *Biological effect:* E05 vs F05 slope + F05/E05 ratio table as in `AB_COMPARISON.md:31-36`, so
    any change to the readout is visible and reviewed, not silent.
  - *3D-specific:* plane-to-plane corrected-intensity profile down Z, showing the flat-field removed
    lateral vignette without introducing an axial artifact.
- **Ground-truth caveat (unchanged):** no flat-field slide / uniform reference exists, so validation
  shows *consistency and uniformity*, not accuracy (`AB_COMPARISON.md:52-55`). State this in the
  report.

---

## Application / CLI

- **Estimate fields:** `tmem-align compute-ic-fields <plate_dir>` (`cli.py:82`) already writes an
  `.npz`. Changes needed:
  - Add `--pool / --per-timepoint` (default per Open Q1) so one command can produce a pooled
    per-channel field set or one per timepoint.
  - Add `--estimate-darkfield` to also store the per-channel scalar offset (Open Q3).
  - Keys in the `.npz` become **channel names** (or `timepoint/channel`), and each value is a **2D**
    field — the on-disk format changes from the current per-timepoint whole-array (`cli.py:95`).
    Exclude D17 in the file-gathering step.
- **Apply during a run:** `run_260213_longitudinal_pilot.py --illumination-correct` stays the entry
  point; it must load the name-keyed field set and call the axis-aware apply (2D or Z-broadcast).
  Add `--darkfield` to opt darkfield in.
- **Config:** optionally surface an `illumination:` block in `configs/template_experiment.yaml`
  (there is none today — only `diffuse_percentile_background`, `template_experiment.yaml:84`) with
  `enabled`, `pool`, `estimate_darkfield`, `smooth_sigma`. Ponytail: only add config keys the run
  actually reads; skip speculative knobs.
- **Run over d7/d14/d28:** `compute-ic-fields` once per plate (pooled) → cached `.npz` → each pilot/
  batch run applies it. Standalone `scripts/compute_ic_fields.py` mirrors the CLI for HPC.

---

## Risks

- **R1 — Silent axis misread (G2).** A ZCYX stack read as TCYX corrupts every field/apply. Mitigate
  by making axes explicit end-to-end and asserting `Z`-labelled data is never treated as `T`.
- **R2 — Channel contamination via D20_F1 swap (G3).** Index-keyed pooling mixes channels; name-key
  everything and add the swap test above.
- **R3 — Memory on real frames.** Pooling all d7+d14+d28 (209 files × ZCYX × 2048²) into RAM for a
  median is large; the existing code already sampled (`sample_fraction` default 0.25) and warns of
  this ceiling (`preprocess.py:80`). Keep seeded sampling; Z-reduce per file *before* stacking so
  only one YX plane per channel per file enters the median. Cap threads per CLAUDE.md HPC rule if run
  on Cheaha.
- **R4 — IC changes the biology (validated real risk).** IC moved the F05/E05 effect size −19%
  (`AB_COMPARISON.md:35`). Keep IC **off by default**, opt-in via flag, and always run the A/B before
  trusting a corrected readout.
- **R5 — Same-field-across-Z assumption.** If the optics have Z-dependent illumination, one 2D field
  per channel under-corrects deep planes. No axial reference exists to model this; flag as a known
  limitation and inspect the Z-profile figure in validation.
- **R6 — Darkfield is an estimate, not a measurement.** No dark frames exist; the scalar offset is a
  percentile heuristic (`preprocess.py:451`). Acceptable for a constant camera bias; revisit only if
  flats show dark structure.
- **R7 — Darkfield subtraction dramatically reduces 561/640 signal for KO/Control (RESOLVED — expected).** Histograms showed darkfield removing 77–81% of mean 561 for KO/Control and 51–62% for 640. Concern was whether NIS-Elements had already subtracted the ADC bias. **Resolved from raw ND2 metadata:** (a) raw pixel minima across all channels are 83–109 ADU — if offset had been subtracted the floor would be ≈ 0; (b) `UseIntenzityCorrection = False` in unstructured metadata — NIS-Elements correction was explicitly off. The ~91–100 ADU darkfield IS the camera ADC bias; it should be subtracted. The large 561/640 shifts for KO/Control are correct biology — those cells have near-zero cleaved-TMEM signal (~29 ADU true signal above the camera floor). Decision 3 (darkfield ON) is confirmed.
