#!/usr/bin/env python3
"""Compute pooled per-channel IC fields for the fixed-IF dataset and save as .npz.

Uses calculate_ic_fields_by_channel (channel-name-keyed, handles D20_F1 swap) with
the σ=102 default validated by the 2026-08-31 flatfield sweep. Pools d7+d14+d28
with 25% seeded sampling (identical to the sweep run).

Output .npz keys:
  '488nm', '561nm', '640nm', '405nm'             — 2-D YX float64 IC fields
  '488nm_darkfield', '561nm_darkfield', ...       — scalar camera offset (ADU)

Load with: tmem_align.analysis.if_spatial.load_ic_fields(npz_path)

Usage:
    python scripts/compute_ic_fields_for_if.py \\
        --data-root /path/to/cleaved_tmem_pld3_260821 \\
        --output data/ic_fields_260821_pooled.npz
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve worktree src so this script picks up the flatfield-branch preprocess
# (calculate_ic_fields_by_channel) whether or not it is installed as a package.
_WORKTREE_SRC = Path(__file__).resolve().parents[1] / ".claude/worktrees/flatfield/src"
if _WORKTREE_SRC.exists():
    sys.path.insert(0, str(_WORKTREE_SRC))

from tmem_align.preprocess import calculate_ic_fields_by_channel  # noqa: E402

CHANNELS = ["488nm", "561nm", "640nm", "405nm"]
WELL_DIRS = ["TMEM_KO", "Z59_PLD_Control", "Z60_PLD_TMEMki"]
TIMEPOINTS = ["d7", "d14", "d28"]
SAMPLE_FRACTION = 0.25
SEED = 0


def load_channel_means(path: Path) -> dict[str, np.ndarray]:
    """Load one ND2; return {channel_name: Z-mean YX float32}."""
    import nd2
    with nd2.ND2File(path) as f:
        ch_names = [c.channel.name for c in f.metadata.channels]
        arr = f.asarray().astype(np.float32)
    if arr.ndim == 4:
        return {name: arr[:, i].mean(axis=0) for i, name in enumerate(ch_names)}
    elif arr.ndim == 3:
        return {name: arr[i] for i, name in enumerate(ch_names)}
    raise ValueError(f"Unexpected shape {arr.shape} in {path}")


def build_pooled_images(data_root: Path) -> dict[str, list[np.ndarray]]:
    rng = random.Random(SEED)
    pooled: dict[str, list] = defaultdict(list)
    for tp in TIMEPOINTS:
        for well in WELL_DIRS:
            d = data_root / tp / well
            paths = sorted(d.glob("*.nd2")) if d.exists() else []
            sampled = rng.sample(paths, max(1, int(len(paths) * SAMPLE_FRACTION)))
            print(f"  {tp}/{well}: {len(sampled)}/{len(paths)} files", flush=True)
            for path in sampled:
                try:
                    for ch, yx in load_channel_means(path).items():
                        if ch in CHANNELS:
                            pooled[ch].append(yx)
                except Exception as e:
                    print(f"  WARNING: {path.name}: {e}", flush=True)
    return dict(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/pmihack/claire/tmem_2026/data/ic_fields_260821_pooled.npz"),
    )
    args = parser.parse_args()

    print(f"Data root: {args.data_root}", flush=True)
    print(f"Output:    {args.output}", flush=True)
    print(f"Sampling:  {SAMPLE_FRACTION*100:.0f}% seeded (seed={SEED})\n", flush=True)

    print("Loading ND2 files...", flush=True)
    images_by_channel = build_pooled_images(args.data_root)
    for ch, imgs in images_by_channel.items():
        print(f"  {ch}: {len(imgs)} images", flush=True)

    print("\nEstimating IC fields (σ=auto=short_side/20=102px, darkfield ON)...", flush=True)
    result = calculate_ic_fields_by_channel(
        images_by_channel,
        estimate_darkfield=True,  # ~91-100 ADU camera bias confirmed from raw ND2 metadata
    )

    npz_data: dict[str, np.ndarray] = {}
    print("\nResults:", flush=True)
    for ch, val in result.items():
        if isinstance(val, tuple):
            field, dark = val
            npz_data[ch] = field
            npz_data[f"{ch}_darkfield"] = np.array(dark)
            print(f"  {ch}: field shape={field.shape} range=[{field.min():.3f},{field.max():.3f}] darkfield={dark:.1f} ADU", flush=True)
        else:
            npz_data[ch] = val
            print(f"  {ch}: field shape={val.shape} range=[{val.min():.3f},{val.max():.3f}]", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **npz_data)
    print(f"\nSaved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
