from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_image


def open_in_napari(image_path: str | Path) -> None:
    """Open a TIFF/OME-TIFF neuron stack in napari for manual inspection."""
    try:
        import napari
    except ImportError as exc:
        raise ImportError("Install viewer dependencies with `pip install -e .[viewer]`") from exc

    image = np.asarray(read_image(image_path))
    viewer = napari.Viewer()
    viewer.add_image(image, name=Path(image_path).stem)
    napari.run()
