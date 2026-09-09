"""Open a deconvolved CZYX OME-TIFF in napari with one layer per channel.

Usage:
    python scripts/view_deconvolved.py path/to/file.ome.tif
"""

import sys
import tifffile
import napari

CHANNEL_NAMES = ["MAP2 488", "LAMP1 640", "cl-TMEM 561", "DAPI 405"]

path = sys.argv[1]
arr = tifffile.imread(path)  # CZYX

viewer = napari.Viewer(title=path)
for i, name in enumerate(CHANNEL_NAMES):
    viewer.add_image(arr[i], name=name)

napari.run()
