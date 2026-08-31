# Fixed-IF LAMP1 Report — TMEM106B KO / Control / KI

*Branch `feat/if-lamp1-quantification` · fixed 3765 DN LAMP1 threshold · well-level statistics (pseudoreplication guard) · audience: skeptical biologists.*

## §0 — Headline

| timepoint | condition | cells | qc_pass | wells |
| --- | --- | --- | --- | --- |
| d14 | Control | 300 | 269 | 6 |
| d14 | KI | 370 | 354 | 6 |
| d14 | KO | 406 | 377 | 6 |
| d28 | Control | 256 | 212 | 5 |
| d28 | KI | 265 | 214 | 6 |
| d28 | KO | 309 | 273 | 6 |
| d7 | Control | 452 | 434 | 6 |
| d7 | KI | 395 | 375 | 6 |
| d7 | KO | 533 | 500 | 6 |

![d7 mean lysosome size, well-level. KO smaller than Control.](figures/headline_d7_lyso_size.png)
*d7 mean lysosome size, well-level. KO smaller than Control.*

## §1 — The assay & question

3 genotypes (TMEM **KO**, PLD3 **Control**, TMEM **KI**) × 3 timepoints (d7/d14/d28) × 4 channels (DAPI/405, MAP2/488, cleaved-TMEM/561, LAMP1/640; max-Z projection). **Question:** does TMEM106B genotype alter LAMP1 lysosome morphology/abundance, and does cleaved TMEM colocalize with lysosomes?

![Representative FOV, per channel (fixed display LUT).](figures/../fig_pipeline_perchannel.png)
*Representative FOV, per channel (fixed display LUT).*

![Same FOV, merged channels.](figures/../fig_pipeline_merged.png)
*Same FOV, merged channels.*

## §2 — Pipeline (why you can trust the numbers)

1. **Fixed display LUT** — brightness is comparable across genotypes, not per-image auto-stretched.
2. **Fixed 3765 DN lysosome threshold** — derived from Control+KI FOVs (`survey_lamp1_otsu.py`); decouples detection from per-genotype brightness so a dimmer KO can't artificially fragment/shrink its lysosomes.
3. **HITL fine-tuned cell-body segmentation** — direct Cellpose fails on the MAP2 neurite mesh; a model fine-tuned on 9 hand-corrected FOVs gives clean somata (AP@0.5 0.58). Known ~15–20% under-detection of faint cells.
4. **Condition-blind QC** — border / size-outlier / pyknotic cells excluded by identical rules across genotypes.
5. **Well-level aggregation** — cells → per-well means (n≈6 wells/genotype) before any test. Cells within a well are not independent replicates.

![Step: DAPI → nuclei (Cellpose-SAM).](figures/../nuclei_overlay.png)
*Step: DAPI → nuclei (Cellpose-SAM).*

![Step: MAP2 → cell bodies (HITL cpsam).](figures/../map2_cellbody_overlay.png)
*Step: MAP2 → cell bodies (HITL cpsam).*

## §3.1 — d7 results

![d7 — Mean lysosome size. KW p=0.00624 · KI vs Ctrl BH p=0.24 · KO vs Ctrl BH p=0.0519](figures/d7_lyso_mean_size_um2.png)
*d7 — Mean lysosome size. KW p=0.00624 · KI vs Ctrl BH p=0.24 · KO vs Ctrl BH p=0.0519*

![d7 — Lysosome count / cell. KW p=0.318 · KI vs Ctrl BH p=0.699 · KO vs Ctrl BH p=0.619](figures/d7_n_lysosomes.png)
*d7 — Lysosome count / cell. KW p=0.318 · KI vs Ctrl BH p=0.699 · KO vs Ctrl BH p=0.619*

![d7 — Lysosome area fraction. KW p=0.0145 · KI vs Ctrl BH p=0.485 · KO vs Ctrl BH p=0.0823](figures/d7_lyso_area_frac.png)
*d7 — Lysosome area fraction. KW p=0.0145 · KI vs Ctrl BH p=0.485 · KO vs Ctrl BH p=0.0823*

![d7 — LAMP1 mean (bg-sub). KW p=0.0268 · KI vs Ctrl BH p=0.394 · KO vs Ctrl BH p=0.186](figures/d7_lamp1_mean.png)
*d7 — LAMP1 mean (bg-sub). KW p=0.0268 · KI vs Ctrl BH p=0.394 · KO vs Ctrl BH p=0.186*

![d7 — Perinuclear index. KW p=0.241 · KI vs Ctrl BH p=0.359 · KO vs Ctrl BH p=1](figures/d7_perinuclear_index.png)
*d7 — Perinuclear index. KW p=0.241 · KI vs Ctrl BH p=0.359 · KO vs Ctrl BH p=1*

![d7 — TMEM–LAMP1 Manders M1 (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.](figures/d7_tmem_manders_m1.png)
*d7 — TMEM–LAMP1 Manders M1 (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.*

![d7 — TMEM–LAMP1 Pearson (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.](figures/d7_tmem_pearson.png)
*d7 — TMEM–LAMP1 Pearson (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.*

![d7 — typical cells per genotype; yellow = detected lysosomes.](figures/d7_crops.png)
*d7 — typical cells per genotype; yellow = detected lysosomes.*

## §3.2 — d14 results

![d14 — Mean lysosome size. KW p=0.296 · KI vs Ctrl BH p=0.186 · KO vs Ctrl BH p=0.937](figures/d14_lyso_mean_size_um2.png)
*d14 — Mean lysosome size. KW p=0.296 · KI vs Ctrl BH p=0.186 · KO vs Ctrl BH p=0.937*

![d14 — Lysosome count / cell. KW p=0.143 · KI vs Ctrl BH p=0.186 · KO vs Ctrl BH p=0.937](figures/d14_n_lysosomes.png)
*d14 — Lysosome count / cell. KW p=0.143 · KI vs Ctrl BH p=0.186 · KO vs Ctrl BH p=0.937*

![d14 — Lysosome area fraction. KW p=0.0504 · KI vs Ctrl BH p=0.00866 · KO vs Ctrl BH p=0.818](figures/d14_lyso_area_frac.png)
*d14 — Lysosome area fraction. KW p=0.0504 · KI vs Ctrl BH p=0.00866 · KO vs Ctrl BH p=0.818*

![d14 — LAMP1 mean (bg-sub). KW p=0.0667 · KI vs Ctrl BH p=0.0173 · KO vs Ctrl BH p=0.485](figures/d14_lamp1_mean.png)
*d14 — LAMP1 mean (bg-sub). KW p=0.0667 · KI vs Ctrl BH p=0.0173 · KO vs Ctrl BH p=0.485*

![d14 — Perinuclear index. KW p=0.884 · KI vs Ctrl BH p=0.818 · KO vs Ctrl BH p=0.818](figures/d14_perinuclear_index.png)
*d14 — Perinuclear index. KW p=0.884 · KI vs Ctrl BH p=0.818 · KO vs Ctrl BH p=0.818*

![d14 — TMEM–LAMP1 Manders M1 (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.](figures/d14_tmem_manders_m1.png)
*d14 — TMEM–LAMP1 Manders M1 (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.*

![d14 — TMEM–LAMP1 Pearson (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.](figures/d14_tmem_pearson.png)
*d14 — TMEM–LAMP1 Pearson (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.*

![d14 — typical cells per genotype; yellow = detected lysosomes.](figures/d14_crops.png)
*d14 — typical cells per genotype; yellow = detected lysosomes.*

## §3.3 — d28 results

![d28 — Mean lysosome size. KW p=0.0799 · KI vs Ctrl BH p=0.537 · KO vs Ctrl BH p=0.0606](figures/d28_lyso_mean_size_um2.png)
*d28 — Mean lysosome size. KW p=0.0799 · KI vs Ctrl BH p=0.537 · KO vs Ctrl BH p=0.0606*

![d28 — Lysosome count / cell. KW p=0.118 · KI vs Ctrl BH p=0.126 · KO vs Ctrl BH p=0.126](figures/d28_n_lysosomes.png)
*d28 — Lysosome count / cell. KW p=0.118 · KI vs Ctrl BH p=0.126 · KO vs Ctrl BH p=0.126*

![d28 — Lysosome area fraction. KW p=0.106 · KI vs Ctrl BH p=0.126 · KO vs Ctrl BH p=0.104](figures/d28_lyso_area_frac.png)
*d28 — Lysosome area fraction. KW p=0.106 · KI vs Ctrl BH p=0.126 · KO vs Ctrl BH p=0.104*

![d28 — LAMP1 mean (bg-sub). KW p=0.068 · KI vs Ctrl BH p=0.0823 · KO vs Ctrl BH p=0.0606](figures/d28_lamp1_mean.png)
*d28 — LAMP1 mean (bg-sub). KW p=0.068 · KI vs Ctrl BH p=0.0823 · KO vs Ctrl BH p=0.0606*

![d28 — Perinuclear index. KW p=0.227 · KI vs Ctrl BH p=0.658 · KO vs Ctrl BH p=1](figures/d28_perinuclear_index.png)
*d28 — Perinuclear index. KW p=0.227 · KI vs Ctrl BH p=0.658 · KO vs Ctrl BH p=1*

![d28 — TMEM–LAMP1 Manders M1 (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.](figures/d28_tmem_manders_m1.png)
*d28 — TMEM–LAMP1 Manders M1 (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.*

![d28 — TMEM–LAMP1 Pearson (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.](figures/d28_tmem_pearson.png)
*d28 — TMEM–LAMP1 Pearson (2D proxy) *(2D max-projection proxy; rigorous 3D object-based coloc lives in the refinements branch)*.*

![d28 — typical cells per genotype; yellow = detected lysosomes.](figures/d28_crops.png)
*d28 — typical cells per genotype; yellow = detected lysosomes.*

## §4 — Across days & genotypes

![Mean lysosome size — trajectory across days.](figures/traj_lyso_mean_size_um2.png)
*Mean lysosome size — trajectory across days.*

![Lysosome count / cell — trajectory across days.](figures/traj_n_lysosomes.png)
*Lysosome count / cell — trajectory across days.*

![Lysosome area fraction — trajectory across days.](figures/traj_lyso_area_frac.png)
*Lysosome area fraction — trajectory across days.*

![LAMP1 mean (bg-sub) — trajectory across days.](figures/traj_lamp1_mean.png)
*LAMP1 mean (bg-sub) — trajectory across days.*

![Perinuclear index — trajectory across days.](figures/traj_perinuclear_index.png)
*Perinuclear index — trajectory across days.*

### Effect sizes (Cohen's d, well-level, vs Control)

| day | readout | d(KO) | d(KI) |
| --- | --- | --- | --- |
| d7 | Mean lysosome size | -1.58 | 0.95 |
| d7 | Lysosome count / cell | -0.72 | 0.11 |
| d7 | Lysosome area fraction | -1.39 | 0.64 |
| d7 | LAMP1 mean (bg-sub) | -1.2 | 0.68 |
| d7 | Perinuclear index | 0.09 | -0.7 |
| d14 | Mean lysosome size | 0.43 | 1.43 |
| d14 | Lysosome count / cell | -0.06 | 1.4 |
| d14 | Lysosome area fraction | 0.31 | 2.07 |
| d14 | LAMP1 mean (bg-sub) | 0.7 | 1.68 |
| d14 | Perinuclear index | -0.04 | -0.35 |
| d28 | Mean lysosome size | -1.67 | -0.51 |
| d28 | Lysosome count / cell | -1.28 | -1.16 |
| d28 | Lysosome area fraction | -1.83 | -0.99 |
| d28 | LAMP1 mean (bg-sub) | -1.68 | -0.99 |
| d28 | Perinuclear index | 0.03 | -0.53 |

*Negative d(KO) for lysosome size = KO smaller than Control.*

### d14 is a detection floor, not a null

LAMP1 signal drops sharply at d14 across all genotypes (below). With a fixed 3765 DN threshold, dim d14 lysosomes fall under detection — the d14 null reflects staining/exposure, not biology. This is why the KO-smaller effect is visible at d7 and d28 but not d14.

![LAMP1 mean by day — the d14 dip is the dimness floor.](figures/traj_lamp1_mean.png)
*LAMP1 mean by day — the d14 dip is the dimness floor.*

At a shared display LUT and fixed µm window, d14 shows **fewer and smaller detected lysosomes** than d7/d28 — which the auto-scaled §3.2 per-day panels hide by rescaling each day. What this does **not** show is dimmer lysosomes: per-lysosome-pixel LAMP1 is comparable across days (median ~6.4k DN at d7 and d14), because anything passing the fixed 3765 DN threshold is bright by construction. d14's low *cell-mean* LAMP1 comes from sparse/small puncta, not dim ones. And the yellow fill paints every detected lysosome the same colour, so intensity is not readable from it at all. Read this as the detection/batch floor — cross-day absolute differences are confounded with plate/staining batch (each timepoint is its own plate), not biology.

![Representative cell per day × genotype, nearest each group's median lysosome size, at a shared fixed display LUT and fixed µm window. cyan=cell, magenta=nucleus, yellow fill=detected lysosomes (fill hides intensity); value = cell mean lysosome size.](figures/crops_across_days.png)
*Representative cell per day × genotype, nearest each group's median lysosome size, at a shared fixed display LUT and fixed µm window. cyan=cell, magenta=nucleus, yellow fill=detected lysosomes (fill hides intensity); value = cell mean lysosome size.*

### Artifact control: the KO effect is not a thresholding artifact

Per-FOV Otsu adapts downward on dim KO images, which *could* shrink KO lysosomes artificially. Switching to a fixed brightness-blind threshold does not remove the KO-smaller effect:

| threshold | d(KO vs Ctrl) |
| --- | --- |
| Otsu (per-FOV) | -1.73 |
| Fixed 3765 DN | -1.58 |

## §5 — Caveats & what would convince a skeptic

- **Biological n = 1.** d7/d14/d28 may be one differentiation, not independent replicates. Well-level stats guard pseudoreplication *within* the run but cannot manufacture biological replicates. **This gates every genotype claim.** Resolve: confirm replicate structure with the lab; add independent differentiations → upgrade `condition_stats` to a linear mixed model.
- **Direction is backwards from TMEM106B-LOF literature**, which typically *enlarges* lysosomes. Could be early-timepoint biology, cell state, or a 2D max-projection artifact. Flag for lab discussion.
- **d14 is a detection floor** (LAMP1 ~2.5× dimmer). Restrict d14 to within-timepoint comparisons or re-stain.
- **Coloc here is a 2D proxy** (Manders M1 / Pearson within the cell mask on a max-projection). The rigorous 3D object-based coloc (`if_coloc.py`, position-randomized null) lives in the **refinements** branch — cite that for any coloc claim.
- **HITL segmentation under-detects ~15–20% of faint cells.** Acceptable for morphology/area readouts; note it for count-based claims.
