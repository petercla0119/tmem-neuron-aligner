# Data folder

Place raw images outside Git when possible. This repository ignores ND2, TIFF, OME-TIFF, Zarr, and OME-Zarr files by default so large microscopy data are not accidentally committed.

Suggested structure:

```text
data/raw/Plate001/Day01/Well_A01/
data/interim/stitched/
data/interim/registered_wells/
data/interim/neuron_rois/
data/processed/ome_zarr/
data/processed/measurements/
```
