"""Batch 3D TMEM-LAMP1 colocalization over a timepoint's FOVs.

    PYTHONPATH=src python scripts/run_if_coloc.py configs/if_coloc_d7_smoke.yaml

Writes <output_dir>/<timepoint>_coloc_percell.csv (one row per cell) and prints a
per-condition enrichment summary. TMEM_KO is the false-positive floor: real
colocalization in Control/KI must clear KO's mean enrichment.

Cell masks: read from cfg.mask_cache_dir (<stem>.npz key 'cells') when set -- reuses
the quantification run's segmentation, no model needed. Otherwise segments MAP2 cell
bodies on the fly (needs the HITL cellpose model; see if_spatial).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tmem_align.analysis.if_coloc import ColocConfig, analyze_fov_coloc  # noqa: E402
from tmem_align.analysis.if_features import parse_fov_metadata  # noqa: E402
from tmem_align.analysis.if_spatial import ICFields, load_fov_3d, load_ic_fields  # noqa: E402


def _cell_masks(nd2_path: Path, cfg: ColocConfig, ic_fields: ICFields | None = None) -> np.ndarray:
    """Cell-body labels for a FOV: from the mask cache if configured, else segment."""
    if cfg.mask_cache_dir:
        npz = Path(cfg.mask_cache_dir) / nd2_path.name.replace(".nd2", ".npz")
        if npz.exists():
            return np.load(npz)["cells"]
    from tmem_align.analysis.if_spatial import load_fov, segment_cell_bodies

    return segment_cell_bodies(load_fov(nd2_path, ic_fields=ic_fields)[cfg.map2_channel])


def _pick_fovs(paths: list[Path], cfg: ColocConfig) -> list[Path]:
    """Round-robin across conditions so a smoke cap still spans genotypes."""
    if cfg.max_fovs is None:
        return paths
    by_cond: dict[str, list[Path]] = {}
    for p in paths:
        by_cond.setdefault(parse_fov_metadata(p)["condition"], []).append(p)
    picked: list[Path] = []
    while len(picked) < cfg.max_fovs and any(by_cond.values()):
        for lst in by_cond.values():
            if lst and len(picked) < cfg.max_fovs:
                picked.append(lst.pop(0))
    return picked


_DEFAULT_IC_NPZ = Path(
    "/Users/pmihack/claire/tmem_2026/data/ic_fields_260821_pooled.npz"
)


def main() -> None:
    cfg = ColocConfig.from_yaml(sys.argv[1])
    root = Path(cfg.data_root) / cfg.timepoint
    paths = _pick_fovs(sorted(root.rglob("*.nd2")), cfg)

    # IC correction: load pre-computed fields if the .npz exists; skip silently if not.
    ic_fields: ICFields | None = None
    if _DEFAULT_IC_NPZ.exists():
        ic_fields = load_ic_fields(_DEFAULT_IC_NPZ)
        print(f"IC fields loaded from {_DEFAULT_IC_NPZ} ({len(ic_fields)} channels)")
    else:
        print(f"IC fields not found at {_DEFAULT_IC_NPZ} — running on raw images")

    print(f"coloc: {len(paths)} FOVs under {root}")

    rows: list[dict] = []
    for i, p in enumerate(paths):
        try:
            cells = _cell_masks(p, cfg, ic_fields=ic_fields)
            channels, sampling = load_fov_3d(p, ic_fields=ic_fields)
            meta = parse_fov_metadata(p)
            for r in analyze_fov_coloc(channels, sampling, cells, cfg, seed=cfg.seed + i):
                r.update(meta)
                rows.append(r)
            print(f"  [{i + 1}/{len(paths)}] {p.name}: {int(np.unique(cells).size - 1)} cells")
        except Exception as exc:  # noqa: BLE001 - keep going
            print(f"  SKIP {p.name}: {exc}")

    if not rows:
        print("no rows produced")
        return
    df = pd.DataFrame(rows)
    out = Path(cfg.output_dir) / f"{cfg.timepoint}_coloc_percell.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} cells)")

    have = df[df["n_puncta"] > 0]
    print(
        "\nper-condition mean enrichment (obs-null frac within "
        f"{cfg.threshold_um} µm), cells with >=1 punctum:"
    )
    summ = have.groupby("condition").agg(
        n_cells=("enrichment", "size"),
        mean_enrichment=("enrichment", "mean"),
        mean_obs_frac=("obs_frac_within", "mean"),
        mean_null_frac=("null_frac_within", "mean"),
    )
    print(summ.round(3))
    print(
        "\n^ KO row is the false-positive floor; Control/KI enrichment above it "
        "is candidate real coloc. n=1 replicate — descriptive only."
    )


if __name__ == "__main__":
    main()
