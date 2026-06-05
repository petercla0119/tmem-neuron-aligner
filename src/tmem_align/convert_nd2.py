from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import write_ome_tiff


def convert_nd2_to_ometiff(nd2_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Convert a Nikon ND2 file into one or more OME-TIFF files.

    This is a cautious starter implementation. ND2 files vary depending on how NIS-Elements
    saved plates, wells, multipoints, z-stacks, and channels. For today's integration, the
    safest path is often to export TIFF/OME-TIFF from Nikon first. If direct conversion is
    needed, install the optional dependency with:

        pip install -e ".[nd2]"

    Then test on one small ND2 file and verify axes before batch conversion.
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import nd2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The optional 'nd2' package is not installed. Install with `pip install -e .[nd2]`, "
            "or export TIFF/OME-TIFF from Nikon/Fiji first."
        ) from exc

    outputs: list[Path] = []
    with nd2.ND2File(nd2_path) as f:
        arr = np.asarray(f.asarray())
        # Axis order depends on file; save as-is and inspect metadata after first run.
        out = output_dir / f"{nd2_path.stem}.ome.tif"
        write_ome_tiff(out, arr, axes=_guess_axes(arr.ndim))
        outputs.append(out)
    return outputs


def _guess_axes(ndim: int) -> str:
    guesses = {2: "YX", 3: "CYX", 4: "TCYX", 5: "TCZYX"}
    return guesses.get(ndim, "YX")
