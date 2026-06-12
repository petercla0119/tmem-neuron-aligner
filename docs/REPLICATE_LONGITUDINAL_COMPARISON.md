# Replicate Longitudinal Comparison

This pilot extends the same-well longitudinal workflow to two matched reporter-control/primary pairs:

- Pair 1: `E05` reporter-control and `F05` primary
- Pair 2: `I05` reporter-control and `J05` primary

Each well is registered to its own Day 8 reference. Wells are compared only after same-well registration; different wells are not registered to each other.

## Local Outputs

Generated outputs are intentionally outside Git:

```text
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/e05_f05_i05_j05_longitudinal_comparison/
  E05_F05_I05_J05_days_day8_day25_day39_comparison_metrics.csv
  E05_F05_I05_J05_days_day8_day25_day39_comparison_summary.png
```

Each well also has its own local stack outputs under:

```text
/Users/makennarodriguez/Documents/TMEM106B_interim/pilot/{well}_longitudinal/
/Users/makennarodriguez/Documents/TMEM106B_processed/pilot/{well}_longitudinal/
```

## Preliminary Pattern

Reporter-control wells remain lower on the diffuse-to-punctate score:

| well | condition | day 8 | day 25 | day 39 |
|---|---|---:|---:|---:|
| E05 | PLD3 + mCherry | 0.046 | 0.047 | 0.034 |
| I05 | PLD3 + mCherry | 0.046 | 0.059 | 0.040 |

Primary TMEM106B wells are higher, especially at Day 25:

| well | condition | day 8 | day 25 | day 39 |
|---|---|---:|---:|---:|
| F05 | PLD3 + TMEM106B + mCherry | 0.063 | 0.170 | 0.090 |
| J05 | PLD3 + TMEM106B + mCherry | 0.063 | 0.153 | 0.106 |

This supports continuing the workflow, but it is not yet proof of lysosomal rupture. The current metric is a screening measure for mCherry redistribution.

## Registration QC Flags

Large shifts require visual review before scaling:

| well | day | dy | dx | note |
|---|---:|---:|---:|---|
| F05 | 25 | -1.0 | 921.0 | large horizontal shift |
| J05 | 39 | 1026.0 | -2.0 | large vertical shift |

The scripts crop to the common overlap before quantification, but these large shifts may indicate a different field of view, strong stage offset, or registration ambiguity.

## Reproduce

From the repo root:

```bash
source .venv/bin/activate
python scripts/run_f05_longitudinal_pilot.py --well E05
python scripts/run_f05_longitudinal_pilot.py --well F05
python scripts/run_f05_longitudinal_pilot.py --well I05
python scripts/run_f05_longitudinal_pilot.py --well J05
python scripts/compare_ef05_longitudinal.py
```

Next, visually review the registered stacks for F05 Day 25 and J05 Day 39, then add adjacent days or additional matched pairs only after the registration behavior is understood.
