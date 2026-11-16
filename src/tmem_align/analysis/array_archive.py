"""Lossless NumPy archives with inexpensive compression for quantitative images."""

from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def save_array_archive(path, **arrays):
    """Write ordinary NPZ files without spending GPU-idle time on level-6 DEFLATE."""
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
