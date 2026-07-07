# Preprocess (Illumination Correction) Surveillance Log

## Session Metadata

| Field | Value |
|-------|-------|
| Date started | 2026-07-06 |
| Branch | `preprocess` |
| Base branch | `csp-dev` |
| Base commit | `0e545f7` (Consolidate registration QC functions into library module (#1)) |
| Reference impl | BrieFlow v1.5.0 `workflow/lib/shared/illumination_correction.py` |
| Target file | `src/tmem_align/preprocess.py` |

## Decision Log

| # | Decision | Choice | Rationale | Status |
|---|----------|--------|-----------|--------|
| 1 | Source reference | BrieFlow v1.5.0 `illumination_correction.py` | Proven CellProfiler-style IC, already validated in production | Accepted |
| 2 | Architecture | Single file `src/tmem_align/preprocess.py` | Ponytail: fewest files. All IC logic in one module | Accepted |
| 3 | IC approach | Median-filter-of-averaged-images, division-based correction | CellProfiler standard. Supports per-well (default) and per-plate grouping | Accepted |
| 4 | Pipeline order | ND2 -> stitch -> preprocess(IC) -> register -> ROI -> quantify | IC must run on stitched images before registration aligns them | Accepted |
| 5 | Dependencies | scikit-image, numpy, scipy (existing only) | No new packages. Ponytail: already-installed deps | Accepted |
| 6 | Testing strategy | Unit tests (synthetic data) first, then integration with edge cases | Fast feedback loop, no dependency on real data for CI | Accepted |
| 7 | Demo | Jupyter notebook, before/after with actual ND2 data | Visual validation from `/Users/pmihack/claire/tmem_2026/data/` | Accepted |
| 8 | Multi-channel IC strategy | Per-channel recursion in `calculate_ic_field()` | CYX images get independent 2D IC field per channel (C,Y,X output). Each fluorescence channel has a different illumination profile | Accepted |
| 9 | ND2 support | Lazy `import nd2` in `io.py` `read_image()` | Optional dep already in pyproject.toml. IC can work directly on raw ND2 files without pre-conversion | Accepted |

## Architecture Decisions

### AD-1: Single-file module over multi-file package
IC correction is a self-contained transform: compute illumination function from a group of images, then divide each image by it. No need for a package directory. One file, importable functions.

### AD-2: Per-well default grouping
Per-well grouping uses images from the same well across all tiles to compute the illumination function. This matches the typical experimental setup where illumination varies by well position. Per-plate option available for cases with uniform illumination across the plate.

### AD-3: Division-based correction over subtraction
Division preserves relative intensity relationships between features. Subtraction can introduce negative values and distort signal ratios. Division is the CellProfiler standard.

### AD-4: Preprocessing before registration
Registration alignment (translation/rotation) should operate on corrected images so that intensity-based matching is not biased by illumination gradients. IC is a per-image operation with no spatial alignment dependency.

### AD-5: Per-channel IC recursion
For multi-channel (CYX) images, `calculate_ic_field()` recurses per channel, computing an independent 2D IC field for each. Returns a (C,Y,X) IC field. Biological rationale: each fluorescence channel (e.g. DAPI, GFP, mCherry) has a distinct illumination profile due to different excitation/emission optics.

### AD-6: ND2 lazy import
ND2 reading added to `io.py` via lazy `import nd2` inside `read_image()`. The `nd2` package is already listed in pyproject.toml as an optional dependency. This avoids import-time failures when nd2 is not installed (not needed for non-ND2 workflows).

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Status |
|---|------|-----------|--------|------------|--------|
| R1 | Median filter too slow on large stitched images | Medium | Medium | Use scipy.ndimage.median_filter with reasonable kernel; profile if needed | Open |
| R2 | Division by zero in illumination function | Low | High | Clip illumination function minimum, add epsilon | Open |
| R3 | Per-well grouping has too few images for reliable IC | Low | Medium | Warn if < 3 images; fall back to per-plate | Open |
| R4 | IC changes break downstream registration | Low | High | Run full pipeline test with known-good data | Open |
| R5 | Memory pressure from loading all well images at once | Medium | Medium | Process images incrementally (running average) | Open |

## Output Tracking

### Files Created
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/tmem_align/preprocess.py` | IC module (calculate_ic_field, apply_ic_field, subtract_background, preprocess_image, well/plate wrappers) | 291 | Complete |
| `notebooks/02_preprocessing_before_after.ipynb` | 7-section demo (data load, IC calc, IC viz, before/after, intensity profiles, BG sub, stats) | - | Complete |
| `tests/test_preprocess.py` | Unit tests | - | In progress |

### Files Modified
| File | Change | Status |
|------|--------|--------|
| `src/tmem_align/io.py` | `read_image()` handles ND2 via lazy import; `find_images()` includes `.nd2` suffix | Complete |

### Pipeline Integration Points
| Integration | File | Status |
|-------------|------|--------|
| Snakemake rule for IC step | TBD | Not started |
| Config schema update | TBD | Not started |

### Verification Results
| Test | Data | Result |
|------|------|--------|
| Synthetic 2D IC | Generated gradient + noise | Pass |
| Multi-channel CYX IC | Synthetic 3-channel | Pass |
| Real ND2 data loading | 192 wells, 3-channel 2868x2868 | Pass |

## Progress Timeline

| Timestamp | Event | Notes |
|-----------|-------|-------|
| 2026-07-06 | Session started | Initial plan established, 7 decisions accepted |
| 2026-07-06 | Task 1 complete | Worktree created on `preprocess` branch from `csp-dev` |
| 2026-07-06 | Task 2 complete | `preprocess.py` implemented (291 lines): calculate_ic_field, apply_ic_field, subtract_background, preprocess_image, well/plate wrappers |
| 2026-07-06 | io.py updated | ND2 support added (lazy import nd2, .nd2 in find_images) |
| 2026-07-06 | Task 3 complete | Demo notebook `02_preprocessing_before_after.ipynb` with 7 sections |
| 2026-07-06 | Decisions 8-9 added | Multi-channel IC strategy (per-channel recursion), ND2 lazy import |
| 2026-07-06 | Task 4 in progress | Unit test agent spawned |
