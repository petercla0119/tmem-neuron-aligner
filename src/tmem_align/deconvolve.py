"""Richardson-Lucy deconvolution of raw ND2 fixed-IF stacks.

Theoretical Gibson-Lanni PSF (``psfmodels``, vectorial model) per excitation line
plus 3-D Richardson-Lucy (``skimage.restoration.richardson_lucy``, which uses
scipy FFT convolution internally for stacks this size). Runs BEFORE stitch /
register. Design and parameter rationale live in ``docs/plan_deconvolution.md``.

Channel identity for this dataset (cleaved_tmem_pld3_260821) comes from the ND2
channel *name* (``'488nm'``, ``'640nm'``, ...), NOT from ``excitationLambdaNm`` /
``emissionLambdaNm`` — those metadata fields are misconfigured on the scope (they
report 405/515 for every channel). Names are clean and encode the laser line, so
mapping by name also auto-corrects the one d7 file whose C indices are swapped.
"""

from __future__ import annotations

import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import write_ome_tiff
from .preprocess import apply_ic_field, calculate_ic_fields_by_channel

# excitation line (nm) -> emission peak (nm) of the secondary on that line
EMISSION_NM: dict[int, int] = {405: 461, 488: 525, 561: 603, 640: 668}

# excitation line (nm) -> biological target (labelling / logging only)
TARGET: dict[int, str] = {405: "DAPI", 488: "MAP2", 561: "cleaved-TMEM", 640: "LAMP1"}

# canonical output channel order for this dataset; writing every file in this
# order makes the swapped-index d7 file line up with all the others downstream
CANON_ORDER: tuple[int, ...] = (488, 640, 561, 405)

_NAME_NM = re.compile(r"(\d{3})")
_D17 = re.compile(r"_D17_", re.IGNORECASE)


@dataclass(frozen=True)
class DeconvConfig:
    """Optical + algorithm parameters. Frozen so it is hashable for lru_cache."""

    na: float = 1.27          # Nikon Plan Apo IR 60x WI, NA 1.27
    ni: float = 1.333         # water immersion RI
    ns: float = 1.338         # ProLong Glass sample RI
    dxy_um: float = 0.1083    # lateral pixel size
    dz_um: float = 0.3        # axial step
    psf_xy_px: int = 61       # PSF lateral extent (odd)
    psf_z_px: int = 21        # PSF axial extent (odd)
    n_iter: int = 10          # Richardson-Lucy iterations
    filter_epsilon: float = 1e-6


# ---------------------------------------------------------------------------
# PSF
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def compute_psf(excitation_nm: int, config: DeconvConfig) -> np.ndarray:
    """Return a sum-normalized 3-D (Z, Y, X) PSF for one excitation line."""
    try:
        import psfmodels
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            'Deconvolution needs psfmodels. Run `pip install -e ".[deconv]"`.'
        ) from exc

    emission_um = EMISSION_NM[excitation_nm] / 1000.0
    psf = psfmodels.make_psf(
        z=config.psf_z_px,
        nx=config.psf_xy_px,
        dxy=config.dxy_um,
        dz=config.dz_um,
        NA=config.na,
        ni=config.ni,
        ni0=config.ni,
        ns=config.ns,
        wvl=emission_um,
        model="vectorial",
    ).astype(np.float32)
    total = psf.sum()
    if total <= 0:  # pragma: no cover - defensive
        raise ValueError(f"Degenerate PSF for {excitation_nm}nm (sum={total})")
    return psf / total  # RL preserves flux only with a sum-1 PSF


# ---------------------------------------------------------------------------
# Richardson-Lucy
# ---------------------------------------------------------------------------


def deconvolve_stack(
    stack: np.ndarray,
    psf: np.ndarray,
    n_iter: int,
    filter_epsilon: float = 1e-6,
) -> np.ndarray:
    """3-D RL on one (Z, Y, X) uint16 stack; returns uint16, same shape.

    Normalizes by the stack max into ~[0, 1] (keeps filter_epsilon meaningful and
    RL numerically well-behaved), then rescales back by the same factor.
    """
    from skimage.restoration import richardson_lucy

    f = stack.astype(np.float32)
    scale = float(f.max())
    if scale <= 0:  # blank channel
        return np.zeros(stack.shape, dtype=np.uint16)
    f /= scale
    dec = richardson_lucy(f, psf, num_iter=n_iter, clip=False, filter_epsilon=filter_epsilon)
    dec = np.clip(dec, 0.0, None) * scale
    return np.clip(np.rint(dec), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# IC field estimation
# ---------------------------------------------------------------------------


def estimate_ic_fields(
    nd2_paths: list[Path],
    channels: tuple[int, ...] | None = None,
    sample_n: int = 30,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Estimate per-channel 2-D IC fields from a sample of ND2 files.

    Extracts each channel ZYX by name (robust to the D20_F1 channel-swap),
    pools across sampled files, and returns one 2-D flatfield per channel nm.
    """
    import nd2

    wanted = channels or CANON_ORDER
    rng = random.Random(seed)
    sample = rng.sample(nd2_paths, min(sample_n, len(nd2_paths)))

    stacks_by_name: dict[str, list[np.ndarray]] = {str(nm): [] for nm in wanted}
    for path in sample:
        try:
            with nd2.ND2File(path) as im:
                ch_map = _channel_map(im)
                axes = list(im.sizes.keys())
                arr = np.asarray(im.asarray())
                c_axis = axes.index("C")
            for nm in wanted:
                if nm in ch_map:
                    stacks_by_name[str(nm)].append(np.take(arr, ch_map[nm], axis=c_axis))
        except Exception:
            continue

    fields = calculate_ic_fields_by_channel(stacks_by_name)
    return {int(k): v for k, v in fields.items() if v is not None}


# ---------------------------------------------------------------------------
# ND2 -> deconvolved OME-TIFF
# ---------------------------------------------------------------------------


def _channel_map(nd2file) -> dict[int, int]:
    """Map excitation line (nm) -> channel index, from channel names."""
    out: dict[int, int] = {}
    for idx, ch in enumerate(nd2file.metadata.channels):
        name = getattr(getattr(ch, "channel", ch), "name", "") or ""
        m = _NAME_NM.search(name)
        if m:
            out[int(m.group(1))] = idx
    return out


def is_d17_file(path: str | Path) -> bool:
    """True for the out-of-study d7 D17 files (excluded from processing)."""
    return bool(_D17.search(Path(path).name))


def deconvolve_nd2_file(
    nd2_path: str | Path,
    out_path: str | Path,
    config: DeconvConfig,
    channels: tuple[int, ...] | None = None,
    overwrite: bool = False,
    ic_fields: dict[int, np.ndarray] | None = None,
) -> Path:
    """Deconvolve the requested channels of one ND2 file -> OME-TIFF (CZYX).

    Channels are written in CANON_ORDER (intersected with ``channels``), so output
    channel indices are consistent across files regardless of stored ND2 order.
    If ``ic_fields`` is provided, each channel ZYX is IC-corrected before RL.
    """
    import nd2

    nd2_path = Path(nd2_path)
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path

    wanted = channels or CANON_ORDER

    with nd2.ND2File(nd2_path) as im:
        ch_map = _channel_map(im)
        axes = list(im.sizes.keys())  # e.g. ['Z', 'C', 'Y', 'X']
        arr = np.asarray(im.asarray())  # shape follows im.sizes order
        c_axis = axes.index("C")

        out_channels: list[np.ndarray] = []
        for nm in CANON_ORDER:
            if nm not in wanted:
                continue
            if nm not in ch_map:
                raise KeyError(f"{nd2_path.name}: no channel named ~{nm}nm (have {sorted(ch_map)})")
            zyx = np.take(arr, ch_map[nm], axis=c_axis)  # (Z, Y, X)
            if ic_fields and nm in ic_fields:
                zyx = apply_ic_field(zyx, ic_fields[nm])
            psf = compute_psf(nm, config)
            out_channels.append(deconvolve_stack(zyx, psf, config.n_iter, config.filter_epsilon))

    stacked = np.stack(out_channels, axis=0)  # (C, Z, Y, X)
    write_ome_tiff(out_path, stacked, axes="CZYX", pixel_size_um=config.dxy_um)
    return out_path


# ---------------------------------------------------------------------------
# Batch driver (process pool)
# ---------------------------------------------------------------------------

# Module-level IC fields set once per worker via pool initializer, so the
# ~128 MB dict is pickled 22 times (once/worker) rather than 209 times (once/job).
_IC_FIELDS: dict[int, np.ndarray] | None = None


def _init_worker(ic_fields: dict[int, np.ndarray] | None) -> None:
    global _IC_FIELDS
    _IC_FIELDS = ic_fields


def _worker(args):
    nd2_path, out_path, config, channels, overwrite = args
    try:
        deconvolve_nd2_file(nd2_path, out_path, config, channels, overwrite, _IC_FIELDS)
        return (str(nd2_path), "ok", None)
    except Exception as exc:  # keep one bad file from killing the batch
        return (str(nd2_path), "error", str(exc))


def discover_nd2(input_dir: str | Path) -> list[Path]:
    """All ND2 files under input_dir, excluding the D17 out-of-study files."""
    return sorted(p for p in Path(input_dir).rglob("*.nd2") if not is_d17_file(p))


def deconvolve_batch(
    input_dir: str | Path,
    out_dir: str | Path,
    config: DeconvConfig,
    channels: tuple[int, ...] | None = None,
    overwrite: bool = False,
    n_workers: int = 22,
    ic_fields: dict[int, np.ndarray] | None = None,
    estimate_ic: bool = True,
    ic_sample_n: int = 30,
) -> list[tuple[str, str, str | None]]:
    """Deconvolve every ND2 under ``input_dir`` -> ``out_dir/<stem>.ome.tif``.

    One job per file (209 files >> cores, so file-level parallelism already
    saturates the pool; loading each ND2 once and doing its 4 channels serially
    avoids the redundant I/O a per-channel split would cause).

    If ``ic_fields`` is None and ``estimate_ic`` is True, estimates IC fields
    from a random sample of ``ic_sample_n`` ND2 files before the batch runs.
    IC fields are sent once per worker via pool initializer (not per job).

    ponytail: file-level fan-out, not the plan's 836 file*channel jobs — same
    saturation, no re-reads. Split per channel only if file count ever drops
    below the core count.
    """
    # cap BLAS/FFT threads per worker BEFORE the pool spawns children, so N
    # workers * pocketfft threads don't oversubscribe the 32 cores
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_nd2(input_dir)
    if not files:
        raise FileNotFoundError(f"No ND2 files under {input_dir}")

    if ic_fields is None and estimate_ic:
        ic_fields = estimate_ic_fields(files, channels, ic_sample_n)

    jobs = [
        (f, out_dir / f"{f.stem}.ome.tif", config, channels, overwrite)
        for f in files
    ]

    if n_workers <= 1:
        _init_worker(ic_fields)
        return [_worker(j) for j in jobs]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(ic_fields,),
    ) as pool:
        return list(pool.map(_worker, jobs))
