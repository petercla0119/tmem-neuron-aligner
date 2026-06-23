# Single-Neuron Time-Series Alignment Examples

These examples crop one automatically selected, compact 488-positive foreground component
from each well's globally registered stack, then perform local crop-level registration on
the 488 channel and apply the same transform to mCherry.

Use these as presentation examples of the alignment concept, not as final same-neuron
biological calls until the ROI identity is manually reviewed.

Report root: `reports/260213_pilot_20260623_125859`
Crop size: `160 x 160` pixels

Outputs:
- `figures/*_single_neuron_alignment_montage.png`
- `figures/*_single_neuron_mcherry.gif`
- `figures/*_single_neuron_overlay.gif`
- `figures/E05_vs_F05_single_neuron_mcherry.gif`
- `single_neuron_roi_selection.csv`
- `single_neuron_local_registration_qc.csv`
- `single_neuron_mcherry_metrics.csv`
- `ome_zarr/*single_neuron_registered_tcyx.ome.zarr`
- `ome_zarr/ome_zarr_export_manifest.csv`

Selected ROIs:
- E05: x=2228, y=172, area=372 px
- F05: x=1484, y=544, area=420 px
