import numpy as np
import pytest


def test_archive_round_trip_preserves_camera_values_labels_and_validity(tmp_path):
    from tmem_align.analysis.array_archive import save_array_archive

    green = (np.arange(2048, dtype=np.float32).reshape(32, 64) / 4)[:, ::2]
    green[0, :4] = [0, -0.0, np.inf, np.nan]
    arrays = {
        "green": green,
        "red": np.full(green.shape, 115.25, dtype=np.float32),
        "valid": np.indices(green.shape).sum(axis=0) % 3 != 0,
        "labels": np.arange(green.size, dtype=np.uint32).reshape(green.shape),
    }
    original_bytes = {name: value.tobytes() for name, value in arrays.items()}
    path = tmp_path / "frame.npz"
    save_array_archive(path, **arrays)
    with np.load(path, allow_pickle=False) as actual:
        assert set(actual.files) == set(arrays)
        for name, expected in arrays.items():
            assert actual[name].dtype == expected.dtype
            assert actual[name].shape == expected.shape
            assert actual[name].tobytes() == original_bytes[name]
            assert expected.tobytes() == original_bytes[name]


def test_archive_rejects_object_arrays(tmp_path):
    from tmem_align.analysis.array_archive import save_array_archive

    with pytest.raises(ValueError, match="Object arrays"):
        save_array_archive(tmp_path / "objects.npz", value=np.array([object()], dtype=object))
