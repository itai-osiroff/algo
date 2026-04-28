import numpy as np
import pytest

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
