import numpy as np
import pytest
from scipy.signal import fftconvolve

from tmem_align.deconvolve import (
    DeconvConfig,
    compute_psf,
    deconvolve_stack,
    is_d17_file,
)


def _sharpness(vol):
    """Peak-to-total ratio of the central bright object — higher = sharper."""
    return float(vol.max()) / float(vol.sum() + 1)


def test_rl_sharpens_a_blurred_point():
    """RL of a blurred point source must be sharper than the blurred input."""
    psf = compute_psf(561, DeconvConfig(n_iter=10))

    truth = np.zeros((21, 64, 64), dtype=np.float32)
    truth[10, 32, 32] = 1.0
    blurred = fftconvolve(truth, psf, mode="same")
    blurred_u16 = np.clip(np.rint(blurred / blurred.max() * 10000), 0, 65535).astype(np.uint16)

    out = deconvolve_stack(blurred_u16, psf, n_iter=10)

    assert out.shape == blurred_u16.shape
    assert out.dtype == np.uint16
    assert np.isfinite(out).all()
    # deconvolution concentrates the point back toward a peak
    assert _sharpness(out) > _sharpness(blurred_u16)


def test_blank_channel_returns_zeros():
    psf = compute_psf(405, DeconvConfig())
    blank = np.zeros((5, 32, 32), dtype=np.uint16)
    out = deconvolve_stack(blank, psf, n_iter=5)
    assert out.shape == blank.shape
    assert out.max() == 0


def test_psf_is_flux_normalized():
    psf = compute_psf(640, DeconvConfig())
    assert psf.shape == (21, 61, 61)
    assert pytest.approx(psf.sum(), rel=1e-5) == 1.0


def test_d17_detection():
    assert is_d17_file("PLD3Control_Plate1_d7_..._D17_F1.nd2")
    assert not is_d17_file("TMEMKO_Plate1_d7_..._C8_F1.nd2")
    assert not is_d17_file("something_D170_F1.nd2")  # not the D17 timepoint
