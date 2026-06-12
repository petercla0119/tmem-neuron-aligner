from __future__ import annotations

from pathlib import Path

import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

from .io import read_image


def export_ome_zarr(
    image_path: str | Path,
    output_zarr: str | Path,
    axes: str | None = None,
    chunks: tuple[int, ...] | None = None,
) -> Path:
    """Export an OME-TIFF/TIFF image to OME-Zarr.

    This writes a basic OME-Zarr. After exporting, validate with an OME-NGFF validator or open in
    napari/Fiji-MoBIE to confirm axes and scale metadata.
    """
    image = np.asarray(read_image(image_path))
    output_zarr = Path(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    store = parse_url(str(output_zarr), mode="w").store
    axes = axes or _guess_axes(image.ndim)
    write_image(image=image, group=store, axes=axes, storage_options={"chunks": chunks} if chunks else None)
    return output_zarr


def _guess_axes(ndim: int) -> str:
    return {2: "yx", 3: "tyx", 4: "tcyx", 5: "tczyx"}.get(ndim, "yx")
