"""Load IC->RL deconvolved OME-TIFFs and pair each FOV with its raw ND2.

Step 1 of the deconv threshold re-derivation
(03_contexts/2026_tmem/_artifacts/2026-09-09-deconv-threshold-rederivation-plan).

Deconv C-index order is the ACQUISITION order (verified 2026-09-09 by MIP
correlation of every deconv channel against the named raw ND2 channels:
0.87-0.98 on the diagonal, <0.1 off), NOT the filename token order:
    C0 = 488nm (MAP2)   C1 = 640nm (LAMP1)   C2 = 561nm (TMEM)   C3 = 405nm (DAPI)

Segment-on-raw: masks come from the raw ND2 (via if_spatial), measurement
channels come from the deconv OME-TIFF. Deconvolution preserves geometry, so
raw masks apply to deconv pixels 1:1 (asserted in _assert_aligned).

    PYTHONPATH=src python scripts/deconv_io.py    # self-check
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

DECONV_DIR = Path("/Users/pmihack/claire/tmem_2026/data/deconvolved_260821")
RAW_ROOT = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")

# deconv CZYX channel order (acquisition order, verified 2026-09-09)
DECONV_CH_INDEX = {"488nm": 0, "640nm": 1, "561nm": 2, "405nm": 3}

# filename condition prefix -> (label, raw condition subdir)
_COND = {
    "TMEMKO": ("KO", "TMEM_KO"),
    "PLD3Control": ("Control", "Z59_PLD_Control"),
    "PLD3TMEM106B": ("KI", "Z60_PLD_TMEMki"),
}


def _stem(path: str | Path) -> str:
    return Path(path).name.replace(".ome.tif", "").replace(".nd2", "")


def condition_from_stem(stem: str) -> str:
    for pref, (label, _) in _COND.items():
        if stem.startswith(pref):
            return label
    raise ValueError(f"unknown condition prefix: {stem}")


def timepoint_from_stem(stem: str) -> str:
    m = re.search(r"_d(\d+)_", stem)
    if not m:
        raise ValueError(f"no timepoint in {stem}")
    return f"d{m.group(1)}"


def load_deconv_fov_3d(tif_path: str | Path) -> dict[str, np.ndarray]:
    """{channel_name: ZYX float32} from a deconv CZYX OME-TIFF."""
    arr = tifffile.imread(str(tif_path))  # CZYX
    assert arr.ndim == 4 and arr.shape[0] == 4, f"expected CZYX C=4, got {arr.shape}"
    return {ch: np.asarray(arr[i], dtype=np.float32) for ch, i in DECONV_CH_INDEX.items()}


def raw_for_deconv(tif_path: str | Path) -> Path:
    """Raw ND2 path for a deconv OME-TIFF (segment-on-raw pairing)."""
    stem = _stem(tif_path)
    tp = timepoint_from_stem(stem)
    for pref, (_, subdir) in _COND.items():
        if stem.startswith(pref):
            return RAW_ROOT / tp / subdir / f"{stem}.nd2"
    raise ValueError(f"unknown condition prefix: {stem}")


def deconv_fovs(timepoint: str | None = None) -> list[Path]:
    fovs = sorted(DECONV_DIR.glob("*.ome.tif"))
    if timepoint:
        fovs = [f for f in fovs if timepoint_from_stem(_stem(f)) == timepoint]
    return fovs


def _assert_aligned(deconv_yx: tuple[int, int], raw_yx: tuple[int, int]) -> None:
    assert deconv_yx == raw_yx, f"raw/deconv YX mismatch: raw {raw_yx} vs deconv {deconv_yx}"


if __name__ == "__main__":
    # Self-check: pairing, condition/timepoint parse, raw<->deconv YX+z alignment.
    for tp in ("d7", "d14", "d28"):
        fovs = deconv_fovs(tp)
        print(f"{tp}: {len(fovs)} deconv FOVs")
        assert fovs, f"no deconv FOVs for {tp}"
    f = deconv_fovs("d7")[0]
    stem = _stem(f)
    chans = load_deconv_fov_3d(f)
    raw = raw_for_deconv(f)
    assert raw.exists(), f"raw missing for {stem}: {raw}"
    import nd2

    with nd2.ND2File(raw) as fh:
        raw_arr = fh.asarray()  # ZCYX
    raw_z, _, ry, rx = raw_arr.shape
    dz, dy, dx = chans["561nm"].shape
    _assert_aligned((dy, dx), (ry, rx))
    assert dz == raw_z, f"z mismatch raw {raw_z} vs deconv {dz}"
    print(f"OK  {stem}")
    print(f"    condition={condition_from_stem(stem)} timepoint={timepoint_from_stem(stem)}")
    print(f"    deconv 561nm ZYX={chans['561nm'].shape}  raw ZCYX={raw_arr.shape}  aligned")
    n = sum(len(deconv_fovs(t)) for t in ("d7", "d14", "d28"))
    print(f"total deconv FOVs d7+d14+d28: {n}")
