# Local-only Jupyter workflow for the TMEM106B ND2 dataset

This workflow runs entirely on your computer. GitHub contains code and small metadata files only; the 99 GB ND2 dataset stays on your local disk or external SSD.

## Recommended storage

```text
TMEM106B_data/
  raw_nd2/       # original files; never modify
  pilot/         # optional small copied subset
  interim/       # extracted/stiched/registered files
  processed/     # OME-Zarr, ROI stacks, CSV measurements
```

Keep the GitHub repository elsewhere, for example `~/Documents/tmem-neuron-aligner`.

## Install once with Miniforge

1. Install Miniforge for your computer.
2. Open Terminal.
3. Change into the repository folder.
4. Create the environment:

```bash
conda env create -f environment.yml
conda activate tmem-align
python -m ipykernel install --user --name tmem-align --display-name "Python (tmem-align)"
```

## Launch JupyterLab

```bash
conda activate tmem-align
cd ~/Documents/tmem-neuron-aligner
jupyter lab
```

Open `notebooks/01_local_nd2_pilot.ipynb` and select the `Python (tmem-align)` kernel.

## Safe order for the 99 GB dataset

1. Set `RAW_ROOT` to the local ND2 folder.
2. Inspect one representative ND2 file.
3. Verify channel names, axes, position counts, dates, and well assignments.
4. Use fluorescence ND2 files for the pilot; do not default to BrightFocus/brightfield images.
5. Extract a single position/channel/timepoint from one mCherry reporter-control well and one primary experimental well.
6. Confirm the images visually before attempting stitching.
7. Only after the pilot succeeds, process additional days and replicates.

Do not convert all ND2 files into TIFF at once. That can duplicate the dataset and consume hundreds of gigabytes.

## Condition guardrails

The well conditions repeat in a four-row alphabetical cycle:

```text
C, G, K, ...: PLD3 only; no TMEM106B and no mCherry
D, H, L, ...: PLD3 + TMEM106B; no mCherry
E, I, M, ...: PLD3 + mCherry; mCherry reporter-control wells
F, J, N, ...: PLD3 + TMEM106B + mCherry; primary experimental wells
```

Only the E/I/M-phase and F/J/N-phase wells are valid for mCherry punctation-versus-diffusion analysis. Do not interpret C/G/K-phase or D/H/L-phase wells as zero-puncta mCherry samples.

It is fine that the files are ND2. Use the `nd2` package for lazy metadata inspection and Dask-backed indexed reads, then save only small pilot subsets as OME-TIFF or OME-Zarr when a downstream tool needs an interchange format.

## Reproduce the current E05/F05 pilot

From the repo root, after activating the environment:

```bash
python scripts/run_ef05_mcherry_pilot.py
```

This creates local interim previews and processed CSV/PNG summaries outside the repository. See `docs/PILOT_EF05_RESULTS.md` for the current preliminary values and limitations.
