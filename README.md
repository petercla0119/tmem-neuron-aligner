# TMEM Neuron Aligner

A lab-shareable starter codebase for turning Nikon ND2/tiled spinning-disk images into stitched, registered, neuron-centered month-long time series. The first goal is not a perfect automated pipeline; it is a reproducible workflow that lets the lab integrate images now, inspect the same neuron over roughly a month, and progressively add quantification.

## What this pipeline does

```text
ND2 or exported TIFF tiles
→ organize by plate / well / day / channel
→ stitch each well/day
→ register each well/day to a reference day
→ crop the same neuron ROI across days
→ locally re-align the neuron crop
→ export OME-TIFF and/or OME-Zarr
→ quantify punctate versus diffuse mCherry signal
```

## Why two-stage alignment?

For a month-long experiment, whole-well alignment alone is usually not enough. The pipeline uses:

1. **Well-level registration** to correct stage drift and make the same well comparable across days.
2. **Neuron-level local registration** to keep the same neuron centered across the month.

Do not use the mCherry phenotype channel as the main registration reference unless there is no alternative, because mCherry diffusion is part of the biology you want to measure. Prefer brightfield, nuclear, or a stable morphology channel.

## Quick start

### 1. Clone or upload this folder to GitHub

Create a new GitHub repository, upload the contents of this folder, and share the repository with lab members.

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[nd2,viewer,dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[nd2,viewer,dev]"
```

### 3. Copy and edit the experiment config

```bash
cp configs/template_experiment.yaml configs/my_experiment.yaml
```

Edit `configs/my_experiment.yaml` so it points to your raw/exported image folder and plate map.

### 4. Validate the config

```bash
tmem-align validate-config configs/my_experiment.yaml
```

### 5. Run a tiny first test

Start with one plate, one well, and two or three timepoints. Do not run the full month-long plate first.

```bash
tmem-align stitch configs/my_experiment.yaml --plate Plate001 --well A01

tmem-align register-well configs/my_experiment.yaml --plate Plate001 --well A01 --reference-day Day01

tmem-align make-roi-stack configs/my_experiment.yaml --plate Plate001 --well A01 --roi-id Neuron001

tmem-align quantify configs/my_experiment.yaml --plate Plate001 --well A01 --roi-id Neuron001
```

## Recommended folder organization

```text
data/
  raw/
    Plate001/
      Day01/
        Well_A01/
          tiles_or_nd2_files_here
      Day07/
      Day14/
      Day21/
      Day28/
  interim/
    stitched/
    registered_wells/
    neuron_rois/
  processed/
    ome_tiff/
    ome_zarr/
    measurements/
```

Raw ND2/TIFF/Zarr data are intentionally ignored by Git. Keep images on shared storage, Box, OneDrive, Google Drive, or an institutional server. Commit only code, configs, and small example metadata files.

## Plate map

Use `configs/plate_map_template.csv` as a starting point. Each row should describe one well/day/condition.

Minimum columns:

```text
plate,day,well,genotype,condition,replicate,raw_path
```

Optional but useful columns:

```text
channels,alignment_channel,pixel_size_um,z_step_um,notes
```

## Manual ROI annotations

For the first pass, manual neuron ROI selection is acceptable. Save ROIs in `configs/roi_annotations_template.csv` format:

```text
plate,well,roi_id,reference_day,x,y,width,height,notes
```

The pipeline crops the same region across aligned well images, then performs local registration inside the crop.

## Output files

Expected outputs:

```text
data/interim/stitched/Plate001/Day01/Well_A01_stitched.ome.tif

data/interim/registered_wells/Plate001/Well_A01/Day01_registered.ome.tif

data/interim/neuron_rois/Plate001/Well_A01/Neuron001/Neuron001_registered_timeseries.ome.tif

data/processed/ome_zarr/Plate001/Well_A01/Neuron001.ome.zarr

data/processed/measurements/Plate001_Well_A01_Neuron001_measurements.csv
```

## Rupture-like score

The starter quantification estimates:

```text
rupture_like_score = diffuse_mcherry_intensity / punctate_mcherry_intensity
```

This is only a screening metric. Stronger rupture validation should use Galectin-3/Galectin-8, p62/LC3, LAMP1/LAMP2 changes, LysoTracker loss, and LLOMe positive controls.

## Current limitations

- The ND2 converter is intentionally conservative because ND2 metadata structures vary by microscope/software version.
- Stitching works best when tile positions are in metadata or filenames. If not, provide grid dimensions in the config.
- Long-term neuron identity may still require manual review because neurons can migrate, change morphology, or die.
- This codebase is designed as a starting scaffold for lab collaboration, not a fully validated analysis package yet.
