import numpy as np

from moment_images.reliable_io import (
    read_json_first_valid,
    read_npy_first_valid,
    read_npz_first_valid,
    write_json_atomic,
    write_npy_atomic,
    write_npz_atomic,
)


def test_read_json_first_valid_falls_back_to_backup(tmp_path):
    path = tmp_path / "state.json"
    write_json_atomic(path, {"run": 1}, create_backup=True)
    write_json_atomic(path, {"run": 2}, create_backup=True)

    path.write_text("{corrupt", encoding="utf-8")
    payload, used_path, _ = read_json_first_valid(path, validator=lambda obj: isinstance(obj, dict))

    assert payload["run"] == 1
    assert str(used_path).endswith(".bak")


def test_read_npy_first_valid_falls_back_to_backup(tmp_path):
    path = tmp_path / "arr.npy"
    write_npy_atomic(path, np.array([1, 2, 3], dtype=np.int32), allow_pickle=False, create_backup=True)
    write_npy_atomic(path, np.array([4, 5, 6], dtype=np.int32), allow_pickle=False, create_backup=True)

    path.write_bytes(b"not-a-valid-npy")
    arr, used_path, _ = read_npy_first_valid(path, allow_pickle=False)

    np.testing.assert_array_equal(arr, np.array([1, 2, 3], dtype=np.int32))
    assert str(used_path).endswith(".bak")


def test_read_npz_first_valid_falls_back_to_backup(tmp_path):
    path = tmp_path / "bundle.npz"
    write_npz_atomic(path, {"a": np.array([1], dtype=np.int32)}, compressed=True, create_backup=True)
    write_npz_atomic(path, {"a": np.array([2], dtype=np.int32)}, compressed=True, create_backup=True)

    path.write_text("corrupt", encoding="utf-8")
    payload, used_path, _ = read_npz_first_valid(
        path,
        allow_pickle=False,
        validator=lambda obj: "a" in obj,
    )

    np.testing.assert_array_equal(payload["a"], np.array([1], dtype=np.int32))
    assert str(used_path).endswith(".bak")
