# Fixed display LUT ranges — d7 fixed-IF dataset

Pooled over 18 FOVs across the plate (3 conditions x 3 rows x 2 fields).
Raw uint16 DN. Use these instead of per-image percentile stretch so brightness
is comparable across conditions and does not drift image-to-image.

| Channel | LUT lo | LUT hi | Basis |
|---|---|---|---|
| MAP2/488 | 110 | 21000 | p1 floor → p99.9; bright somata clip slightly (intended) |
| LAMP1/640 | 120 | 12000 | p1 floor → p99.9; punctate lysosomes |
| TMEM/561 | 105 | 1800 | p1 floor → ~p99.9; very dim/sparse — hi≈1800 is the biology |
| DAPI/405 | 130 | 21000 | p1 floor → p99.9; bright nuclei |

Camera/background floor ≈ 105 DN (all channels bottom out there).
For figures emphasizing dim puncta, drop hi to ~p99.5 (LAMP1 ~5000, TMEM ~400).
Source: `notebooks/lut_range_analysis.py`. Applied via `tmem_align.analysis.if_spatial.apply_display_lut(img, channel)`.
