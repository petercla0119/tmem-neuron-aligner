"""Self-checks for fixed-IF cell-body expansion."""
import numpy as np

from tmem_align.analysis.if_spatial import cell_foreground_mask, expand_to_cell_bodies


def _synthetic():
    """Two nuclei in a MAP2 foreground blob, plus a background corner."""
    map2 = np.zeros((200, 200), dtype=np.float32)
    map2[20:180, 20:180] = 500.0  # bright foreground region (Otsu splits this from 0)
    nuclei = np.zeros((200, 200), dtype=np.int32)
    nuclei[50:60, 50:60] = 1  # cell 1 seed
    nuclei[140:150, 140:150] = 2  # cell 2 seed
    return nuclei, map2


def test_one_body_per_nucleus():
    nuclei, map2 = _synthetic()
    bodies = expand_to_cell_bodies(nuclei, map2, max_distance=60)
    assert set(np.unique(bodies)) == {0, 1, 2}, "one label per seed, plus background"


def test_bodies_enclose_their_nuclei():
    nuclei, map2 = _synthetic()
    bodies = expand_to_cell_bodies(nuclei, map2, max_distance=60)
    for label in (1, 2):
        seed = nuclei == label
        assert np.all(bodies[seed] == label), "cell body must cover its own nucleus"


def test_no_growth_into_background():
    nuclei, map2 = _synthetic()
    bodies = expand_to_cell_bodies(nuclei, map2, max_distance=200)
    background = map2 == 0
    assert bodies[background].max() == 0, "labels must not bleed into MAP2 background"


def test_expansion_cap_limits_radius():
    nuclei, map2 = _synthetic()
    # Tiny cap: bodies stay near their seeds, cannot fill the whole foreground.
    capped = expand_to_cell_bodies(nuclei, map2, max_distance=5)
    uncapped = expand_to_cell_bodies(nuclei, map2, max_distance=200)
    assert (capped > 0).sum() < (uncapped > 0).sum(), "smaller cap → smaller bodies"


def test_foreground_mask_separates_signal_from_black():
    _, map2 = _synthetic()
    fg = cell_foreground_mask(map2)
    assert fg[100, 100], "center of bright region is foreground"
    assert not fg[5, 5], "corner of black region is background"
