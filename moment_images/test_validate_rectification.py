import numpy as np

from moment_images.validate_rectification import load_rectified_reference


def test_load_rectified_reference_from_npy_stack(tmp_path):
    rectified_views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    path = tmp_path / "rectified_views.npy"
    np.save(path, rectified_views)

    loaded = load_rectified_reference(str(path))

    np.testing.assert_allclose(loaded, rectified_views.mean(axis=(0, 1)))


def test_load_rectified_reference_from_npz_rectified_views(tmp_path):
    rectified_views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    path = tmp_path / "rectified_views.npz"
    np.savez_compressed(path, rectified_views=rectified_views)

    loaded = load_rectified_reference(str(path))

    np.testing.assert_allclose(loaded, rectified_views.mean(axis=(0, 1)))


def test_load_rectified_reference_prefers_average_image(tmp_path):
    average = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "workspace.npz"
    np.savez_compressed(path, average_rectified_image=average)

    loaded = load_rectified_reference(str(path))

    np.testing.assert_allclose(loaded, average)
