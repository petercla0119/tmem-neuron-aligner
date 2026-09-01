#!/usr/bin/env python3
"""Save IC-corrected ND2 images as uint16 OME-TIFF (ZCYX).

Applies flat-field + darkfield correction to a directory of ND2 files
and writes one OME-TIFF per file preserving the full ZCYX z-stack.
Channel order in the output follows the ND2 metadata (name-keyed, so the
D20_F1 channel-order swap is handled correctly).

Usage:
    python scripts/save_ic_corrected.py \\
        --input-dir /path/to/nd2_directory \\
        --ic-npz /path/to/ic_fields.npz \\
        --output-dir /path/to/corrected_tiffs

    # Example: save all d7 KI FOVs
    python scripts/save_ic_corrected.py \\
        --input-dir data/cleaved_tmem_pld3_260821/d7/Z60_PLD_TMEMki \\
        --ic-npz data/ic_fields_260821_pooled.npz \\
        --output-dir data/ic_corrected/d7_ki
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

IC_FIELD_FLOOR = 0.1


def load_ic_fields(npz_path: str | Path) -> dict[str, tuple[np.ndarray, float]]:
    """Load {channel_name: (field_yx, darkfield_scalar)} from a pooled IC .npz."""
    data = np.load(npz_path)
    result = {}
    for key in data.files:
        if key.endswith("_darkfield"):
            continue
        dark_key = f"{key}_darkfield"
        dark = float(data[dark_key]) if dark_key in data.files else 0.0
        result[key] = (data[key].astype(np.float32), dark)
    return result


def apply_ic_zcyx(
    arr: np.ndarray,
    channel_names: list[str],
    ic_fields: dict[str, tuple[np.ndarray, float]],
) -> np.ndarray:
    """Apply IC to a ZCYX uint16 array. Returns uint16 ZCYX."""
    out = arr.astype(np.float32)
    for ci, ch in enumerate(channel_names):
        if ch not in ic_fields:
            continue
        flat, dark = ic_fields[ch]
        flat_c = np.clip(flat, IC_FIELD_FLOOR, None)
        out[:, ci] = np.clip(out[:, ci] - dark, 0.0, None) / flat_c[np.newaxis]
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def correct_nd2_to_tiff(
    nd2_path: Path,
    ic_fields: dict[str, tuple[np.ndarray, float]],
    output_path: Path,
) -> None:
    """Load one ND2, apply IC, save as uint16 OME-TIFF (ZCYX)."""
    import nd2

    with nd2.ND2File(nd2_path) as f:
        channel_names = [c.channel.name for c in f.metadata.channels]
        arr = f.asarray()  # ZCYX or CYX

    if arr.ndim == 3:  # CYX — add singleton Z
        arr = arr[np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected shape {arr.shape} in {nd2_path.name}")

    corrected = apply_ic_zcyx(arr, channel_names, ic_fields)

    # Write as OME-TIFF with channel metadata
    tifffile.imwrite(
        output_path,
        corrected,
        imagej=False,
        photometric="minisblack",
        metadata={
            "axes": "ZCYX",
            "channels": channel_names,
            "ic_npz": "ic_fields_260821_pooled.npz",
            "ic_sigma": "short_side/20 (102px)",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing .nd2 files to correct.")
    parser.add_argument("--ic-npz", type=Path,
                        default=Path("/Users/pmihack/claire/tmem_2026/data/ic_fields_260821_pooled.npz"),
                        help="Pre-computed IC fields .npz (from compute_ic_fields_for_if.py).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write corrected OME-TIFFs.")
    parser.add_argument("--pattern", default="*.nd2",
                        help="Glob pattern for input files (default: *.nd2).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default 1; increase for large batches).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading IC fields from {args.ic_npz}...", flush=True)
    ic_fields = load_ic_fields(args.ic_npz)
    for ch, (field, dark) in ic_fields.items():
        print(f"  {ch}: field {field.shape}  darkfield={dark:.1f} ADU", flush=True)

    nd2_paths = sorted(args.input_dir.glob(args.pattern))
    if not nd2_paths:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.input_dir}")
    print(f"\nFound {len(nd2_paths)} files to correct → {args.output_dir}\n", flush=True)

    def _process(path: Path) -> None:
        out = args.output_dir / path.with_suffix(".tiff").name
        if out.exists():
            print(f"  SKIP (exists): {out.name}", flush=True)
            return
        try:
            correct_nd2_to_tiff(path, ic_fields, out)
            print(f"  OK: {path.name} → {out.name}", flush=True)
        except Exception as e:
            print(f"  ERROR: {path.name}: {e}", flush=True)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_process, nd2_paths))
    else:
        for path in nd2_paths:
            _process(path)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
