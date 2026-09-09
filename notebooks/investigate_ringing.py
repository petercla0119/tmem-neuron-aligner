"""Systematic investigation of RL deconvolution ringing on the 561 (cleaved-TMEM)
channel. Answers:

  Q1  what the ringing is + how much it moves the downstream puncta numbers
  Q2  is the moat REAL (in pixel values / RL overshoot) or a display/normalization
      artifact of deconvolve_stack's max-normalization?
  Q4  is the KI FOV worst-case? compare KO / Control conditions

Run: PYTHONPATH=src python notebooks/investigate_ringing.py
Writes reports/deconv_sweep/ringing_*.png and ringing_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
from skimage.feature import peak_local_max
from skimage.restoration import richardson_lucy

from tmem_align.deconvolve import DeconvConfig, compute_psf
from tmem_align.quantify import quantify_puncta_vs_diffuse_frame

OUT = Path("reports/deconv_sweep")
D7 = "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7"
FOVS = {
    "KI":      f"{D7}/Z60_PLD_TMEMki/PLD3TMEM106B_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C20_F1.nd2",
    "KO":      f"{D7}/Z04_PLD_TMEMko/TMEMKO_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C8_F1.nd2",
    "Control": f"{D7}/Z34_PLD_TMEMControl/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_C17_F1.nd2",
}
ITERS = (5, 10, 20)
NM = 561


def load_561(path):
    import glob
    if not Path(path).exists():
        # tolerate folder-name guesswork: match by basename anywhere under d7
        path = glob.glob(f"{D7}/**/{Path(path).name}", recursive=True)[0]
    with nd2.ND2File(path) as im:
        axes = list(im.sizes.keys())
        arr = np.asarray(im.asarray())
        cmap = {int(c.channel.name[:3]): i for i, c in enumerate(im.metadata.channels)}
    return np.take(arr, cmap[NM], axis=axes.index("C"))


def rl_float(stack, psf, n_iter, norm):
    """RL returning float32 WITHOUT the final 0-clip, so overshoot is visible.

    norm: 'max' (deconvolve_stack's scheme), '65535' (plan's scheme), or 'none'.
    """
    f = stack.astype(np.float32)
    if norm == "max":
        s = float(f.max()) or 1.0
    elif norm == "65535":
        s = 65535.0
    else:
        s = 1.0
    dec = richardson_lucy(f / s, psf, num_iter=n_iter, clip=False, filter_epsilon=1e-6)
    return dec * s  # back to original intensity units; NOT clipped at 0


def clip_u16(x):
    return np.clip(np.rint(np.clip(x, 0, None)), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------


def q2_real_vs_display(psf):
    """Line profile + moat depth under 3 normalizations on the KI brightest punctum."""
    zyx = load_561(FOVS["KI"])
    raw = zyx.max(0).astype(np.float32)
    # brightest isolated punctum
    yx = peak_local_max(raw, min_distance=15, num_peaks=1)[0]
    y, x = int(yx[0]), int(yx[1])
    half = 30
    sl = (slice(y - half, y + half + 1), slice(x - half, x + half + 1))

    profiles = {"raw": raw[sl][half]}
    results = {}
    for norm in ("max", "65535", "none"):
        dec = rl_float(zyx, psf, 10, norm)  # float, pre-clip
        mip = dec.max(0)
        profiles[f"decon ({norm}-norm)"] = mip[sl][half]
        # moat metrics relative to local background (scale-invariant)
        patch = mip[sl]
        peak = float(patch[half, half])
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
        r = np.sqrt(yy**2 + xx**2)
        bg = float(np.median(patch[(r > 22) & (r < 28)]))
        moat = float(np.min(patch[(r > 4) & (r < 10)]))  # ring just outside the peak
        results[norm] = {
            "peak": peak,
            "local_bg": bg,
            "moat_min": moat,
            "moat_below_bg_frac": (bg - moat) / bg if bg else None,
            "float_goes_negative": bool(moat < 0),
        }

    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, prof in profiles.items():
        ax.plot(prof, label=lab, lw=1.5 if lab == "raw" else 1.0)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_title("561 line profile through brightest punctum (absolute intensity)")
    ax.set_xlabel("pixel along horizontal line"); ax.set_ylabel("intensity (uint16 units)")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "ringing_lineprofile.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def q1q4_puncta_impact(psf):
    """Downstream puncta metrics: raw vs 5/10/20 iter, per condition."""
    table = {}
    for cond, path in FOVS.items():
        zyx = load_561(path)
        variants = {"raw": clip_u16(zyx.max(0).astype(np.float32))}
        for it in ITERS:
            variants[f"iter{it}"] = clip_u16(rl_float(zyx, psf, it, "max").max(0))
        rows = {}
        for name, mip in variants.items():
            m = quantify_puncta_vs_diffuse_frame(mip, min_size_pixels=6)
            rows[name] = {
                k: m[k] for k in (
                    "puncta_count", "mean_puncta_area_pixels", "punctate_mean",
                    "max_puncta_intensity", "rupture_like_score",
                )
            }
        table[cond] = rows
        print(f"\n=== {cond} (561) ===")
        base = rows["raw"]
        for name, r in rows.items():
            dc = 100 * (r["puncta_count"] - base["puncta_count"]) / max(base["puncta_count"], 1)
            print(f"  {name:7s} count={r['puncta_count']:5d} ({dc:+5.1f}%)  "
                  f"area={r['mean_puncta_area_pixels']:6.1f}  "
                  f"punctate_mean={r['punctate_mean']:8.1f}  "
                  f"rupture={r['rupture_like_score']:.3f}")
    return table


def q1_radial_profile(psf):
    """Radially-averaged intensity around top-K bright puncta (KI): raw vs 10-iter."""
    zyx = load_561(FOVS["KI"])
    raw = zyx.max(0).astype(np.float32)
    dec = clip_u16(rl_float(zyx, psf, 10, "max").max(0)).astype(np.float32)
    peaks = peak_local_max(raw, min_distance=20, num_peaks=25, threshold_rel=0.3)
    half = 20
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r = np.sqrt(yy**2 + xx**2).astype(int)
    rbins = np.arange(0, half + 1)

    def radial(img):
        accs = []
        for (y, x) in peaks:
            if y - half < 0 or y + half + 1 > img.shape[0] or x - half < 0 or x + half + 1 > img.shape[1]:
                continue
            patch = img[y - half:y + half + 1, x - half:x + half + 1]
            patch = patch / (patch.max() + 1e-6)  # per-punctum normalize so faint+bright combine
            accs.append([patch[r == rb].mean() for rb in rbins])
        return np.mean(accs, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rbins * 0.1083, radial(raw), "o-", label="raw", ms=3)
    ax.plot(rbins * 0.1083, radial(dec), "s-", label="10-iter RL", ms=3)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_title(f"Radial profile around {len(peaks)} brightest 561 puncta (KI)")
    ax.set_xlabel("radius (µm)"); ax.set_ylabel("normalized intensity")
    ax.legend()
    fig.savefig(OUT / "ringing_radial.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    psf = compute_psf(NM, DeconvConfig())
    print("Q2: real vs display / normalization ...")
    q2 = q2_real_vs_display(psf)
    for norm, r in q2.items():
        print(f"  {norm:6s}-norm: peak={r['peak']:.0f} bg={r['local_bg']:.1f} "
              f"moat_min={r['moat_min']:.1f} moat_below_bg={r['moat_below_bg_frac']:.1%} "
              f"float<0={r['float_goes_negative']}")
    print("\nQ1/Q4: puncta impact per condition ...")
    table = q1q4_puncta_impact(psf)
    print("\nQ1: radial profile ...")
    q1_radial_profile(psf)

    (OUT / "ringing_metrics.json").write_text(json.dumps({"q2_normalization": q2, "q1q4_puncta": table}, indent=2))
    print(f"\nWrote {OUT}/ringing_metrics.json + ringing_lineprofile.png + ringing_radial.png")


if __name__ == "__main__":
    main()
