import warnings

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

import moment_images.im_process as im_process
from moment_images.im_process import RectificationRunConfig, save_workspace_variables
from moment_images.power_blend import PowerBlendFieldCache, PowerBlendFieldPair
from moment_images.rectification_strategy import PixelwiseSimilarityResult


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
    assert config.pixel_shift_power == 1.0
    assert config.pixel_shift_power_apply_to_reference is True


def test_rectification_run_config_can_preserve_legacy_directional_power_scoring():
    config = RectificationRunConfig(
        pixel_shift_power=2.0,
        pixel_shift_power_apply_to_reference=False,
    )

    assert config.pixel_shift_power_apply_to_reference is False


def test_rectification_run_config_normalizes_reference_power_boolean_text():
    config = RectificationRunConfig(pixel_shift_power_apply_to_reference="false")

    assert config.pixel_shift_power_apply_to_reference is False


def test_rectification_run_config_validates_power_blend_contract():
    config = RectificationRunConfig(
        strategy="combined",
        pixel_power_blend_enabled=True,
        pixel_power_blend_raised_power=2.0,
        pixel_power_blend_alpha=0.5,
        pixel_power_blend_delta_cap=1.5,
    )

    assert config.pixel_power_blend_raised_power == 2.0

    invalid_options = [
        {"strategy": "pixel"},
        {"pixel_sigma_pyramid": True},
        {"pixel_shift_power": 1.5},
        {"pixel_power_blend_raised_power": 1.0},
        {"pixel_power_blend_alpha": 1.1},
        {"pixel_power_blend_delta_cap": -0.1},
    ]
    for options in invalid_options:
        with pytest.raises(ValueError, match="pixel_power_blend"):
            kwargs = {
                "strategy": "combined",
                "pixel_power_blend_enabled": True,
                "pixel_power_blend_raised_power": 2.0,
            }
            kwargs.update(options)
            RectificationRunConfig(**kwargs)


@pytest.mark.parametrize("value", [0.0, -0.5, float("nan"), float("inf"), float("-inf")])
def test_rectification_run_config_rejects_invalid_pixel_shift_power(value):
    with pytest.raises(ValueError, match="pixel_shift_power"):
        RectificationRunConfig(pixel_shift_power=value)


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


def test_rectify_views_from_config_forwards_pixel_shift_power(monkeypatch):
    views = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    captured: dict[str, object] = {}

    def fake_compute_similarity_matrices(self, views: np.ndarray, ref_view_idx, **kwargs: object):
        captured["pixel_shift_power"] = kwargs.get("pixel_shift_power")
        return "pixel_similarity"

    def fake_rectify(self, views: np.ndarray, sim_mat=None, ref_view_idx=None):
        return np.mean(views, axis=(0, 1)), np.asarray(views)

    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "compute_similarity_matrices",
        fake_compute_similarity_matrices,
    )
    monkeypatch.setattr(im_process.PixelwisePatchRectificationStrategy, "rectify", fake_rectify)

    result = im_process.rectify_views_from_config(
        views,
        RectificationRunConfig(
            strategy="pixel",
            similarity_window_size=1,
            max_displacement=0,
            gauss_sigma=1.0,
            gauss_size=5,
            disp_gauss_sigma=0.0,
            disp_gauss_size=1,
            sub_pix_factor=1,
            pixel_shift_power=0.5,
        ),
    )

    assert captured["pixel_shift_power"] == 0.5
    assert result.pixel_similarity == "pixel_similarity"


def test_power_blend_combined_run_blends_baseline_and_raised_similarity_fields(monkeypatch):
    views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    powers: list[float] = []
    reference_power_modes: list[bool] = []
    rectification_similarity: list[PixelwiseSimilarityResult] = []

    def fake_global_similarity(self, stage_views, ref_view_idx, **kwargs):
        return np.zeros((2, 2, 2, 2, 2), dtype=np.int64)

    def fake_global_rectify(self, stage_views, sim_mat=None, ref_view_idx=None):
        return np.mean(stage_views, axis=(0, 1)), np.asarray(stage_views).copy()

    def fake_pixel_similarity(self, stage_views, ref_view_idx, **kwargs):
        power = float(kwargs["pixel_shift_power"])
        powers.append(power)
        reference_power_modes.append(bool(kwargs["pixel_shift_power_apply_to_reference"]))
        displacement = 0.0 if power == 1.0 else 2.0
        disps = np.full((2, 2, 2, 2), displacement, dtype=np.float32)
        return PixelwiseSimilarityResult(
            coordinates=np.zeros((2, 2, 2, 2, 2), dtype=np.float32),
            scores=np.full((2, 2, 2, 2), power, dtype=np.float64),
            disps_y=disps,
            disps_x=disps,
        )

    def fake_pixel_rectify(self, stage_views, sim_mat=None, ref_view_idx=None):
        rectification_similarity.append(sim_mat)
        return np.mean(stage_views, axis=(0, 1)), np.asarray(stage_views).copy()

    monkeypatch.setattr(im_process.GlobalTranslationRectificationStrategy, "compute_similarity_matrix", fake_global_similarity)
    monkeypatch.setattr(im_process.GlobalTranslationRectificationStrategy, "rectify", fake_global_rectify)
    monkeypatch.setattr(im_process.PixelwisePatchRectificationStrategy, "compute_similarity_matrices", fake_pixel_similarity)
    monkeypatch.setattr(im_process.PixelwisePatchRectificationStrategy, "rectify", fake_pixel_rectify)

    result = im_process.rectify_views_from_config(
        views,
        RectificationRunConfig(
            strategy="combined",
            refinement_window_size=1,
            max_displacement=0,
            gauss_sigma=1.0,
            gauss_size=5,
            disp_gauss_sigma=0.0,
            disp_gauss_size=1,
            sub_pix_factor=1,
            pixel_power_blend_enabled=True,
            pixel_power_blend_raised_power=2.0,
            pixel_power_blend_alpha=0.5,
            pixel_power_blend_delta_cap=2.0,
            pixel_shift_power_apply_to_reference=False,
        ),
    )

    assert powers == [1.0, 2.0]
    assert reference_power_modes == [False, False]
    assert result.pixel_similarity.power_blend_diagnostics["raised_power"] == 2.0
    np.testing.assert_allclose(rectification_similarity[0].disps_y, np.sqrt(0.5))


def test_power_blend_cache_reuses_fields_for_the_same_key(tmp_path):
    views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    config = RectificationRunConfig(
        strategy="combined",
        refinement_window_size=1,
        correlation_pad_width=1,
        max_displacement=0,
        gauss_sigma=1.0,
        gauss_size=5,
        disp_gauss_sigma=0.0,
        disp_gauss_size=1,
        sub_pix_factor=1,
        pixel_power_blend_enabled=True,
        pixel_power_blend_raised_power=2.0,
        pixel_power_blend_alpha=0.5,
        pixel_power_blend_delta_cap=1.0,
    )

    first = im_process.rectify_views_from_config(
        views, config, power_blend_cache_dir=tmp_path, power_blend_cache_key="same-fields"
    )
    second = im_process.rectify_views_from_config(
        views, config, power_blend_cache_dir=tmp_path, power_blend_cache_key="same-fields"
    )

    assert (tmp_path / "same-fields.npz").exists()
    assert first.pixel_similarity.power_blend_diagnostics["cache_hit"] is False
    assert second.pixel_similarity.power_blend_diagnostics["cache_hit"] is True


def test_power_blend_field_cache_preserves_runtime_warnings(tmp_path):
    calls = 0

    def make_result(warnings):
        disps = np.zeros((1, 1, 1, 1), dtype=np.float32)
        return PixelwiseSimilarityResult(
            coordinates=np.zeros((1, 1, 1, 1, 2), dtype=np.float32),
            scores=np.zeros((1, 1, 1, 1), dtype=np.float64),
            disps_y=disps,
            disps_x=disps,
            runtime_warnings=warnings,
        )

    def compute():
        nonlocal calls
        calls += 1
        return PowerBlendFieldPair(
            anchor=make_result([{"kind": "anchor_nonfinite"}]),
            challenger=make_result([{"kind": "challenger_nonfinite"}]),
        )

    cache = PowerBlendFieldCache(tmp_path)
    first, first_hit = cache.get_or_compute("shared-fields", compute)
    second, second_hit = cache.get_or_compute("shared-fields", compute)

    assert calls == 1
    assert first_hit is False
    assert second_hit is True
    assert first.anchor.runtime_warnings == second.anchor.runtime_warnings
    assert first.challenger.runtime_warnings == second.challenger.runtime_warnings


def test_power_blend_field_cache_runs_one_concurrent_producer(tmp_path):
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        time.sleep(0.1)
        disps = np.zeros((1, 1, 1, 1), dtype=np.float32)
        result = PixelwiseSimilarityResult(
            coordinates=np.zeros((1, 1, 1, 1, 2), dtype=np.float32),
            scores=np.zeros((1, 1, 1, 1), dtype=np.float64),
            disps_y=disps,
            disps_x=disps,
        )
        return PowerBlendFieldPair(anchor=result, challenger=result)

    cache = PowerBlendFieldCache(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: cache.get_or_compute("shared-fields", compute), range(2)))

    assert calls == 1
    assert sorted(cache_hit for _, cache_hit in results) == [False, True]


def test_powered_combined_run_preserves_raw_global_and_rectification_inputs(monkeypatch):
    views = np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3)
    views[0, 0, 0, 0] = np.nan
    views[1, 1, 0, 1] = np.inf
    views[0, 1, 0, 2] = -4.0
    captured: dict[str, np.ndarray] = {}

    def fake_global_similarity(self, global_views: np.ndarray, ref_view_idx, **kwargs: object):
        captured["global_views"] = np.asarray(global_views).copy()
        return np.zeros((2, 2, 2, 2, 2), dtype=np.int64)

    def fake_global_rectify(self, stage_views: np.ndarray, sim_mat=None, ref_view_idx=None):
        captured["global_rectification_views"] = np.asarray(stage_views).copy()
        return np.asarray(stage_views[0, 0]), np.asarray(stage_views).copy()

    def fake_pixel_rectify(self, stage_views: np.ndarray, sim_mat=None, ref_view_idx=None):
        captured["pixel_rectification_views"] = np.asarray(stage_views).copy()
        return np.asarray(stage_views[0, 0]), np.asarray(stage_views).copy()

    monkeypatch.setattr(
        im_process.GlobalTranslationRectificationStrategy,
        "compute_similarity_matrix",
        fake_global_similarity,
    )
    monkeypatch.setattr(
        im_process.GlobalTranslationRectificationStrategy,
        "rectify",
        fake_global_rectify,
    )
    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "rectify",
        fake_pixel_rectify,
    )

    with pytest.warns(RuntimeWarning, match="pixel_shift_power"):
        result = im_process.rectify_views_from_config(
            views,
            RectificationRunConfig(
                strategy="combined",
                refinement_window_size=1,
                max_displacement=0,
                gauss_sigma=1.0,
                gauss_size=5,
                disp_gauss_sigma=0.0,
                disp_gauss_size=1,
                sub_pix_factor=1,
                pixel_shift_power=0.5,
            ),
        )

    np.testing.assert_equal(captured["global_views"], views)
    np.testing.assert_equal(captured["global_rectification_views"], views)
    np.testing.assert_equal(captured["pixel_rectification_views"], views)
    np.testing.assert_equal(result.input_views, views)
    np.testing.assert_equal(result.global_rectified_views, views)
    np.testing.assert_equal(result.rectified_views, views)
    assert len(result.runtime_warnings) == 1
    assert result.pixel_similarity is not None
    assert result.runtime_warnings == result.pixel_similarity.runtime_warnings
    diagnostics = result.runtime_warnings[0]
    assert diagnostics["views"]["view_0_0"]["nan"] == 1
    assert diagnostics["views"]["view_1_1"]["positive_infinity"] == 1
    assert diagnostics["views"]["view_0_1"]["negative_finite"] == 1


def test_powered_pixel_run_preserves_raw_rectification_inputs(monkeypatch):
    views = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    views[0, 0, 0, 0] = np.nan
    views[0, 0, 0, 1] = np.inf
    views[0, 0, 0, 2] = -4.0
    captured: dict[str, np.ndarray] = {}

    def fake_pixel_rectify(self, stage_views: np.ndarray, sim_mat=None, ref_view_idx=None):
        captured["pixel_rectification_views"] = np.asarray(stage_views).copy()
        return np.asarray(stage_views[0, 0]), np.asarray(stage_views).copy()

    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "rectify",
        fake_pixel_rectify,
    )

    with pytest.warns(RuntimeWarning, match="pixel_shift_power"):
        result = im_process.rectify_views_from_config(
            views,
            RectificationRunConfig(
                strategy="pixel",
                similarity_window_size=1,
                max_displacement=0,
                gauss_sigma=1.0,
                gauss_size=5,
                disp_gauss_sigma=0.0,
                disp_gauss_size=1,
                sub_pix_factor=1,
                pixel_shift_power=0.5,
            ),
        )

    np.testing.assert_equal(captured["pixel_rectification_views"], views)
    np.testing.assert_equal(result.input_views, views)
    np.testing.assert_equal(result.rectified_views, views)
    assert result.pixel_similarity is not None
    assert result.runtime_warnings == result.pixel_similarity.runtime_warnings
    assert result.runtime_warnings[0]["views"]["view_0_0"]["nan"] == 1


def test_powered_hdr_run_diagnoses_only_the_scored_exposure(monkeypatch):
    first_exposure = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) + 1.0
    second_exposure = first_exposure.copy()
    second_exposure[0, 0, 0, 0] = np.nan
    views = np.stack([first_exposure, second_exposure], axis=0)
    rectification_inputs: list[np.ndarray] = []
    similarity_inputs: list[np.ndarray] = []

    original_compute_similarity = (
        im_process.PixelwisePatchRectificationStrategy.compute_similarity_matrices
    )

    def capture_similarity_inputs(self, scoring_views: np.ndarray, ref_view_idx, **kwargs: object):
        similarity_inputs.append(np.asarray(scoring_views).copy())
        return original_compute_similarity(self, scoring_views, ref_view_idx, **kwargs)

    def fake_pixel_rectify(self, stage_views: np.ndarray, sim_mat=None, ref_view_idx=None):
        rectification_inputs.append(np.asarray(stage_views).copy())
        return np.asarray(stage_views[0, 0]), np.asarray(stage_views).copy()

    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "rectify",
        fake_pixel_rectify,
    )
    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "compute_similarity_matrices",
        capture_similarity_inputs,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = im_process.rectify_views_from_config(
            views,
            RectificationRunConfig(
                strategy="pixel",
                similarity_window_size=1,
                max_displacement=0,
                gauss_sigma=1.0,
                gauss_size=5,
                disp_gauss_sigma=0.0,
                disp_gauss_size=1,
                sub_pix_factor=1,
                pixel_shift_power=0.5,
                hdr_enabled=True,
                hdr_exposure_times=[1.0, 0.5],
            ),
        )

    assert not [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, RuntimeWarning)
        and "pixel_shift_power" in str(warning.message)
    ]
    assert len(result.runtime_warnings) == 1
    diagnostics = result.runtime_warnings[0]
    assert set(diagnostics["views"]) == {"view_0_0"}
    assert diagnostics["views"]["view_0_0"]["nan"] == 0
    assert len(similarity_inputs) == 1
    np.testing.assert_equal(similarity_inputs[0], first_exposure)
    assert len(rectification_inputs) == 2
    np.testing.assert_equal(rectification_inputs[1], second_exposure)


def test_powered_combined_hdr_run_diagnoses_only_the_first_globally_rectified_exposure(
    monkeypatch,
):
    first_exposure = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) + 1.0
    second_exposure = first_exposure.copy()
    second_exposure[0, 0, 0, 0] = np.nan
    views = np.stack([first_exposure, second_exposure], axis=0)
    pixel_rectification_inputs: list[np.ndarray] = []
    similarity_inputs: list[np.ndarray] = []

    original_compute_similarity = (
        im_process.PixelwisePatchRectificationStrategy.compute_similarity_matrices
    )

    def fake_global_similarity(self, global_views: np.ndarray, ref_view_idx, **kwargs: object):
        return np.zeros((1, 1, 2, 2, 2), dtype=np.int64)

    def fake_global_rectify(self, stage_views: np.ndarray, sim_mat=None, ref_view_idx=None):
        return np.asarray(stage_views[0, 0]), np.asarray(stage_views).copy()

    def fake_pixel_rectify(self, stage_views: np.ndarray, sim_mat=None, ref_view_idx=None):
        pixel_rectification_inputs.append(np.asarray(stage_views).copy())
        return np.asarray(stage_views[0, 0]), np.asarray(stage_views).copy()

    def capture_similarity_inputs(self, scoring_views: np.ndarray, ref_view_idx, **kwargs: object):
        similarity_inputs.append(np.asarray(scoring_views).copy())
        return original_compute_similarity(self, scoring_views, ref_view_idx, **kwargs)

    monkeypatch.setattr(
        im_process.GlobalTranslationRectificationStrategy,
        "compute_similarity_matrix",
        fake_global_similarity,
    )
    monkeypatch.setattr(
        im_process.GlobalTranslationRectificationStrategy,
        "rectify",
        fake_global_rectify,
    )
    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "rectify",
        fake_pixel_rectify,
    )
    monkeypatch.setattr(
        im_process.PixelwisePatchRectificationStrategy,
        "compute_similarity_matrices",
        capture_similarity_inputs,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = im_process.rectify_views_from_config(
            views,
            RectificationRunConfig(
                strategy="combined",
                refinement_window_size=1,
                max_displacement=0,
                gauss_sigma=1.0,
                gauss_size=5,
                disp_gauss_sigma=0.0,
                disp_gauss_size=1,
                sub_pix_factor=1,
                pixel_shift_power=0.5,
                hdr_enabled=True,
                hdr_exposure_times=[1.0, 0.5],
            ),
        )

    assert not [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, RuntimeWarning)
        and "pixel_shift_power" in str(warning.message)
    ]
    assert len(result.runtime_warnings) == 1
    diagnostics = result.runtime_warnings[0]
    assert set(diagnostics["views"]) == {"view_0_0", "external_reference"}
    assert diagnostics["views"]["view_0_0"]["nan"] == 0
    assert len(similarity_inputs) == 1
    np.testing.assert_equal(similarity_inputs[0], first_exposure)
    assert len(pixel_rectification_inputs) == 2
    np.testing.assert_equal(pixel_rectification_inputs[0], first_exposure)
    np.testing.assert_equal(pixel_rectification_inputs[1], second_exposure)


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
