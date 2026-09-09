"""Check #2 — is the d14 LAMP1 dimness real (raw signal) or an artifact?

Loads RAW LAMP1 (640nm, pre-background-subtraction) from matched d7 vs d14
Control FOVs, compares intensity percentiles, checks acquisition metadata
(laser/exposure) so lower raw signal can be attributed to staining/biology
rather than a different exposure, and writes a shared-scale montage.

    PYTHONPATH=src python notebooks/check_d14_dimness.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tmem_align.analysis.if_spatial import CH_LAMP1, load_fov

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
OUT = Path("reports/if_segmentation_pilot/d14_vs_d7_control_lamp1_dimness.png")
WELLS = ["C17", "E17"]  # skip D17 (flagged for exclusion); Control wells


def control_fovs(tp: str) -> list[Path]:
    d = DATA / tp / "Z59_PLD_Control"
    return sorted(p for p in d.glob("*.nd2") if any(f"_{w}_" in p.name for w in WELLS))


def acquisition_meta(nd2_path: Path) -> str:
    """Best-effort pull of exposure / laser for the 640 channel."""
    try:
        import nd2
        with nd2.ND2File(str(nd2_path)) as f:
            chans = f.metadata.channels or []
            out = []
            for c in chans:
                name = getattr(c.channel, "name", "?")
                if "640" not in str(name):
                    continue
                exp = getattr(c.channel, "exposure_time", None) if hasattr(c, "channel") else None
                # exposure often lives under microscope/loops; fall back to text
                out.append(f"{name} exp={exp}")
            return "; ".join(out) or "(640 channel meta not found)"
    except Exception as e:  # noqa: BLE001
        return f"(meta unavailable: {e})"


def raw_lamp1(paths: list[Path]) -> tuple[list[np.ndarray], np.ndarray]:
    imgs, allpx = [], []
    for p in paths:
        ch = load_fov(p)
        img = ch[CH_LAMP1]
        imgs.append(img)
        allpx.append(img.ravel())
    return imgs, np.concatenate(allpx)


def main() -> None:
    d7p, d14p = control_fovs("d7"), control_fovs("d14")
    print(f"d7 Control FOVs: {len(d7p)} | d14 Control FOVs: {len(d14p)}")
    print("\nacquisition metadata (640/LAMP1 channel):")
    print(f"  d7 [{d7p[0].name}]:  {acquisition_meta(d7p[0])}")
    print(f"  d14[{d14p[0].name}]: {acquisition_meta(d14p[0])}")

    d7_imgs, d7_px = raw_lamp1(d7p)
    d14_imgs, d14_px = raw_lamp1(d14p)

    print("\nRAW LAMP1 (640nm) intensity percentiles — pooled pixels:")
    print(f"{'pct':>6} {'d7':>10} {'d14':>10} {'d7/d14':>8}")
    for q in [50, 90, 99, 99.9]:
        a, b = np.percentile(d7_px, q), np.percentile(d14_px, q)
        print(f"{q:>6} {a:>10.0f} {b:>10.0f} {a/max(b,1):>8.2f}")
    print(f"{'max':>6} {d7_px.max():>10.0f} {d14_px.max():>10.0f}")

    # Montage: 2 rows (d7 / d14), shared display range = 0..d7 p99.5
    vmax = np.percentile(d7_px, 99.5)
    n = min(4, len(d7_imgs), len(d14_imgs))
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for j in range(n):
        ax[0, j].imshow(d7_imgs[j], cmap="magma", vmin=0, vmax=vmax)
        ax[1, j].imshow(d14_imgs[j], cmap="magma", vmin=0, vmax=vmax)
        ax[0, j].set_title(d7p[j].name.split("_")[-2] + "_" + d7p[j].name.split("_")[-1][:2],
                           fontsize=8)
        for r in (0, 1):
            ax[r, j].axis("off")
    ax[0, 0].set_ylabel("d7", fontsize=12)
    ax[1, 0].set_ylabel("d14", fontsize=12)
    fig.suptitle(f"Control raw LAMP1 (640nm), shared scale 0–{vmax:.0f} DN — "
                 "d14 dimness check", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nmontage -> {OUT}")


if __name__ == "__main__":
    main()
