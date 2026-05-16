import numpy as np
import pytest

import moment_images.im_process as im_process
from moment_images.im_process import RectificationRunConfig, save_workspace_variables


def test_save_workspace_variables_adds_npz_suffix_when_missing(tmp_path):
    save_workspace_variables(str(tmp_path / "workspace"), {"arr": np.arange(3, dtype=np.int32)})

    assert (tmp_path / "workspace.npz").exists()
    assert (tmp_path / "workspace.mat").exists()
    assert not (tmp_path / "workspace").exists()


def test_save_workspace_variables_uses_npz_backup_path_on_repeat_save(tmp_path):
    save_path = tmp_path / "workspace"
    save_workspace_variables(str(save_path), {"arr": np.arange(3, dtype=np.int32)})
    save_workspace_variables(str(save_path), {"arr": np.arange(4, dtype=np.int32)})

    assert (tmp_path / "workspace.npz").exists()
    assert (tmp_path / "workspace.npz.bak").exists()
    assert (tmp_path / "workspace.mat").exists()
    assert not (tmp_path / "workspace.bak").exists()


def test_rectification_run_config_defaults_to_intensity_mode():
    config = RectificationRunConfig()

    assert config.displacement_metric_family == "intensity"
    assert config.displacement_alpha == 1.0
    assert config.displacement_component_weights is None
    assert config.metric_formulation_id == "intensity_default"
    assert config.combined_reference_mode == "global_average"


def test_rectification_run_config_rejects_non_default_intensity_settings():
    with pytest.raises(ValueError, match="requires alpha=1.0"):
        RectificationRunConfig(
            displacement_metric_family="intensity",
            displacement_alpha=0.25,
        )

    with pytest.raises(ValueError, match="does not accept component_weights"):
        RectificationRunConfig(
            displacement_metric_family="intensity",
            displacement_component_weights={"gradient": 1.0},
        )


def test_rectification_run_config_canonicalizes_legacy_metric_family_aliases():
    gradient_config = RectificationRunConfig(
        displacement_metric_family="ncc_gradient",
        displacement_alpha=0.25,
    )
    laplacian_config = RectificationRunConfig(
        displacement_metric_family="ncc_laplacian",
        displacement_alpha=0.25,
    )

    assert gradient_config.displacement_metric_family == "blurred_gradient_corr"
    assert gradient_config.displacement_alpha == 0.25
    assert gradient_config.displacement_component_weights is None
    assert laplacian_config.displacement_metric_family == "blurred_laplacian_corr"
    assert laplacian_config.displacement_alpha == 0.25
    assert laplacian_config.displacement_component_weights is None


def test_rectification_run_config_rejects_pair_family_component_weights():
    with pytest.raises(ValueError, match="does not accept component_weights"):
        RectificationRunConfig(
            displacement_metric_family="ncc_gradient",
            displacement_alpha=0.5,
            displacement_component_weights={"gradient": 1.0},
        )

    with pytest.raises(ValueError, match="does not accept component_weights"):
        RectificationRunConfig(
            displacement_metric_family="ncc_laplacian",
            displacement_alpha=0.5,
            displacement_component_weights={"laplacian": 1.0},
        )


def test_rectification_run_config_rejects_invalid_composite_weights():
    with pytest.raises(ValueError, match="only support"):
        RectificationRunConfig(
            displacement_metric_family="composite",
            displacement_alpha=0.5,
            displacement_component_weights={"gradnt": 1.0},
        )

    with pytest.raises(ValueError, match="requires positive non-intensity component weights"):
        RectificationRunConfig(
            displacement_metric_family="composite",
            displacement_alpha=0.5,
            displacement_component_weights={"gradient": 0.0, "laplacian": 0.0},
        )


def test_rectification_run_config_preserves_legacy_center_view_behavior():
    config = RectificationRunConfig(combined_use_global_avg_reference=False)

    assert config.combined_reference_mode == "center_view"
    assert config.combined_use_global_avg_reference is False


def test_rectify_views_from_config_combined_can_use_iterative_rebuild(monkeypatch):
    views = np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3)
    rebuilt_reference = np.full((3, 3), 7.0, dtype=np.float32)
    captured: dict[str, object] = {}

    class FakeRebuildResult:
        def __init__(self, reference_image: np.ndarray):
            self.reference_image = reference_image

    def fake_iterative_subtractive_max_rebuild(
        global_rectified_views: np.ndarray, **kwargs: object
    ) -> FakeRebuildResult:
        captured["rebuild_views"] = np.asarray(global_rectified_views)
        captured["rebuild_kwargs"] = kwargs
        return FakeRebuildResult(rebuilt_reference)

    def fake_compute_similarity_matrices(self, views: np.ndarray, ref_view_idx, **kwargs: object):
        captured["pixel_reference_view"] = kwargs.get("reference_view")
        return "pixel_similarity"

    def fake_rectify(self, views: np.ndarray, sim_mat=None, ref_view_idx=None):
        return np.mean(views, axis=(0, 1)), np.asarray(views)

    monkeypatch.setattr(
        im_process,
        "iterative_subtractive_max_rebuild",
        fake_iterative_subtractive_max_rebuild,
    )
    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "compute_similarity_matrices",
        fake_compute_similarity_matrices,
    )
    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "rectify",
        fake_rectify,
    )

    config = RectificationRunConfig(
        strategy="combined",
        refinement_window_size=1,
        max_displacement=0,
        gauss_sigma=1.0,
        gauss_size=5,
        disp_gauss_sigma=0.0,
        disp_gauss_size=1,
        global_threshold=0.1,
        pixel_threshold=0.05,
        sub_pix_factor=1,
        combined_reference_mode="iterative_subtractive_max_rebuild",
    )

    result = im_process.rectify_views_from_config(views, config)

    np.testing.assert_array_equal(captured["pixel_reference_view"], rebuilt_reference)
    rebuild_kwargs = captured["rebuild_kwargs"]
    assert isinstance(rebuild_kwargs, dict)
    np.testing.assert_array_equal(
        rebuild_kwargs["global_average"],
        np.mean(result.global_rectified_views, axis=(0, 1)),
    )
    assert rebuild_kwargs["metric_family"] == config.displacement_metric_family
    assert rebuild_kwargs["alpha"] == config.displacement_alpha
    assert result.pixel_similarity == "pixel_similarity"
