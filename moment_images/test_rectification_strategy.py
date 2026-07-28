import unittest
from unittest import mock
from typing import Tuple

import numpy as np

import moment_images.rectification_strategy as rectification_strategy
from moment_images.rectification_strategy import (
    BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
    BLURRED_GRADIENT_CORR_METRIC_FAMILY,
    BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
    GlobalTranslationRectificationStrategy,
    PixelwisePatchRectificationStrategy,
    PixelwiseSimilarityResult,
)


class GlobalTranslationRectificationStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = GlobalTranslationRectificationStrategy(prompt_for_all_views=False)

    def test_rectify_aligns_and_averages_views(self) -> None:
        views = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
        sim_mat = np.zeros((2, 2, 2, 2, 2), dtype=int)

        avg_image, rectified_views = self.strategy.rectify(views, sim_mat, (0, 0))

        np.testing.assert_allclose(rectified_views, views.astype(np.float64))
        np.testing.assert_allclose(avg_image, views.mean(axis=(0, 1)))

    def test_reference_view_passthrough(self) -> None:
        views = np.array(
            [
                [[[5, 5], [5, 5]], [[1, 2], [3, 4]]],
                [[[0, 0], [0, 0]], [[9, 9], [9, 9]]],
            ],
            dtype=np.float32,
        )
        sim_mat = np.zeros((2, 2, 2, 2, 2), dtype=int)

        _, rectified_views = self.strategy.rectify(views, sim_mat, (0, 1))

        np.testing.assert_array_equal(rectified_views[0, 1], views[0, 1])

    def test_translation_offsets_are_applied(self) -> None:
        views = np.zeros((2, 2, 3, 3), dtype=np.float32)
        views[1, 1, 1, 1] = 1.0

        sim_mat = np.zeros((2, 2, 2, 2, 2), dtype=int)
        sim_mat[0, 0, 1, 1] = [1, 1]

        avg_image, rectified_views = self.strategy.rectify(views, sim_mat, (0, 0))

        expected_shifted = np.zeros((3, 3), dtype=np.float64)
        expected_shifted[2, 2] = 1.0

        np.testing.assert_array_equal(rectified_views[1, 1], expected_shifted)
        np.testing.assert_equal(avg_image[2, 2], expected_shifted[2, 2] / 4)

    def test_select_reference_view_defaults_to_center(self) -> None:
        views = np.zeros((3, 3, 2, 2), dtype=np.float32)

        ref_idx = self.strategy.select_reference_view(views)

        self.assertEqual(ref_idx, (1, 1))

    def test_compute_similarity_matrix_matches_dimensions(self) -> None:
        views = np.zeros((2, 2, 2, 2), dtype=np.float32)
        strategy = GlobalTranslationRectificationStrategy(pad_width=1, prompt_for_all_views=False)

        sim_mat = strategy.compute_similarity_matrix(views, (0, 0))

        self.assertEqual(sim_mat.shape, (2, 2, 2, 2, 2))

    def test_rectify_with_internal_computation(self) -> None:
        views = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        strategy = GlobalTranslationRectificationStrategy(pad_width=1, prompt_for_all_views=False)

        avg_image, rectified_views = strategy.rectify(views)

        np.testing.assert_allclose(avg_image, views[0, 0])
        np.testing.assert_allclose(rectified_views[0, 0], views[0, 0])


class PixelwisePatchRectificationStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = PixelwisePatchRectificationStrategy(window_size=1)
        self.metric_strategy = PixelwisePatchRectificationStrategy(window_size=5, step_size=1)

    @staticmethod
    def _make_shifted_blob_case(
        shift_x: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], int]:
        height = width = 21
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        center_y = center_x = 10
        sigma = 1.2
        reference_view = np.exp(
            -(((yy - center_y) ** 2) + ((xx - center_x) ** 2)) / (2 * sigma**2)
        ).astype(np.float32)
        shifted_view = np.exp(
            -(((yy - center_y) ** 2) + ((xx - (center_x - shift_x)) ** 2)) / (2 * sigma**2)
        ).astype(np.float32)
        return shifted_view[None, None, :, :], reference_view, (center_y, center_x), shift_x

    def test_compute_similarity_matrix_produces_expected_shapes(self) -> None:
        views = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

        result = self.strategy.compute_similarity_matrices(views, (0, 0))

        self.assertIsInstance(result, PixelwiseSimilarityResult)
        self.assertEqual(result.coordinates.shape, (2, 2, 1, 1, 2))
        self.assertEqual(result.scores.shape, (2, 2, 1, 1))
        self.assertEqual(result.metric_family, "intensity")
        self.assertEqual(result.alpha, 1.0)
        self.assertIsNone(result.blended_score_maps)
        self.assertIsNone(result.component_score_maps)
        self.assertIsNone(result.component_score_maps_raw)
        self.assertIsNone(result.view_score_summary)

    def test_blend_pixelwise_similarity_results_caps_vector_residual_and_keeps_anchor_scores(self) -> None:
        anchor = PixelwiseSimilarityResult(
            coordinates=np.zeros((2, 2, 1, 1, 2), dtype=np.float32),
            scores=np.full((2, 2, 1, 1), 7.0, dtype=np.float64),
            disps_y=np.zeros((2, 2, 1, 1), dtype=np.float32),
            disps_x=np.zeros((2, 2, 1, 1), dtype=np.float32),
        )
        challenger = PixelwiseSimilarityResult(
            coordinates=np.zeros((2, 2, 1, 1, 2), dtype=np.float32),
            scores=np.full((2, 2, 1, 1), 3.0, dtype=np.float64),
            disps_y=np.full((2, 2, 1, 1), 3.0, dtype=np.float32),
            disps_x=np.full((2, 2, 1, 1), 4.0, dtype=np.float32),
        )

        blended, diagnostics = PixelwisePatchRectificationStrategy.blend_pixelwise_similarity_results(
            anchor,
            challenger,
            blend_alpha=1.0,
            delta_cap=2.5,
        )

        np.testing.assert_allclose(blended.disps_y, 1.5)
        np.testing.assert_allclose(blended.disps_x, 2.0)
        np.testing.assert_allclose(blended.scores, anchor.scores)
        np.testing.assert_allclose(blended.coordinates[0, 0, 0, 0], [-1.5, -2.0])
        self.assertEqual(diagnostics["scores_source"], "anchor_inherited")
        self.assertEqual(diagnostics["cap_hit_fraction"], 1.0)

    def test_pixel_shift_power_sanitizes_signed_nonfinite_inputs_and_reports_diagnostics(self) -> None:
        views = np.array(
            [[[[4.0, -4.0, 0.0], [np.nan, np.inf, -np.inf], [1.0, -1.0, 2.0]]]],
            dtype=np.float32,
        )
        reference = np.ones((3, 3), dtype=np.float32)

        with self.assertWarnsRegex(RuntimeWarning, "pixel_shift_power"):
            result = self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                max_disp=0,
                gauss_sigma=1.0,
                gauss_size=5,
                sub_pix_factor=1,
                reference_view=reference,
                pixel_shift_power=0.5,
            )

        self.assertTrue(np.isfinite(result.scores).all())
        self.assertTrue(np.isfinite(result.coordinates).all())
        self.assertEqual(len(result.runtime_warnings), 1)
        diagnostics = result.runtime_warnings[0]
        self.assertEqual(diagnostics["pixel_shift_power"], 0.5)
        self.assertEqual(diagnostics["views"]["view_0_0"]["negative_finite"], 2)
        self.assertEqual(diagnostics["views"]["view_0_0"]["zero"], 1)
        self.assertEqual(diagnostics["views"]["view_0_0"]["nan"], 1)
        self.assertEqual(diagnostics["views"]["view_0_0"]["positive_infinity"], 1)
        self.assertEqual(diagnostics["views"]["view_0_0"]["negative_infinity"], 1)

    def test_normalize_pixel_shift_power_validates_shared_contract(self) -> None:
        normalizer = rectification_strategy.normalize_pixel_shift_power

        self.assertEqual(normalizer("0.5"), 0.5)
        views = np.ones((1, 1, 3, 3), dtype=np.float32)
        for invalid_value in (0.0, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "pixel_shift_power"):
                    normalizer(invalid_value)
                with self.assertRaisesRegex(ValueError, "pixel_shift_power"):
                    self.strategy.compute_similarity_matrices(
                        views,
                        (0, 0),
                        max_disp=0,
                        pixel_shift_power=invalid_value,
                    )

    def test_reference_power_mode_normalizes_explicit_boolean_text(self) -> None:
        normalizer = rectification_strategy.normalize_pixel_shift_power_apply_to_reference

        for value in (True, "true", " 1 ", "yes"):
            with self.subTest(value=value):
                self.assertTrue(normalizer(value))
        for value in (False, "false", " 0 ", "no"):
            with self.subTest(value=value):
                self.assertFalse(normalizer(value))
        for value in ("enabled", 1, 0, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "pixel_shift_power_apply_to_reference"
                ):
                    normalizer(value)

    def test_pixel_shift_power_one_matches_legacy_similarity(self) -> None:
        views = np.array(
            [[[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]],
            dtype=np.float32,
        )
        reference = np.array(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=np.float32,
        )

        legacy = self.strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=1,
            gauss_sigma=1.0,
            gauss_size=5,
            sub_pix_factor=1,
            reference_view=reference,
        )
        explicit_default = self.strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=1,
            gauss_sigma=1.0,
            gauss_size=5,
            sub_pix_factor=1,
            reference_view=reference,
            pixel_shift_power=1.0,
        )

        np.testing.assert_array_equal(explicit_default.scores, legacy.scores)
        np.testing.assert_array_equal(explicit_default.disps_y, legacy.disps_y)
        np.testing.assert_array_equal(explicit_default.disps_x, legacy.disps_x)
        self.assertEqual(explicit_default.runtime_warnings, [])

    def test_pixel_shift_power_can_power_both_reference_and_shifted_intensity_inputs(self) -> None:
        views = np.array([[[[-4.0, 0.0], [9.0, 16.0]]]], dtype=np.float32)
        reference = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        filtered_inputs: list[np.ndarray] = []

        def identity_filter(values: np.ndarray, **kwargs: object) -> np.ndarray:
            filtered_inputs.append(np.asarray(values).copy())
            return np.asarray(values)

        original_filter = rectification_strategy.gaussian_filter
        rectification_strategy.gaussian_filter = identity_filter
        try:
            with self.assertWarnsRegex(RuntimeWarning, "pixel_shift_power"):
                self.strategy.compute_similarity_matrices(
                    views,
                    (0, 0),
                    max_disp=0,
                    gauss_sigma=1.0,
                    gauss_size=5,
                    sub_pix_factor=1,
                    reference_view=reference,
                    pixel_shift_power=0.5,
                    pixel_shift_power_apply_to_reference=True,
                )
        finally:
            rectification_strategy.gaussian_filter = original_filter

        expected_powered = np.array([[-2.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        expected_powered_reference = np.sqrt(reference)
        np.testing.assert_allclose(
            filtered_inputs[1], expected_powered_reference * expected_powered
        )
        np.testing.assert_allclose(filtered_inputs[2], expected_powered**2)

    def test_pixel_shift_power_can_preserve_legacy_shifted_only_intensity_scoring(self) -> None:
        views = np.array([[[[-4.0, 0.0], [9.0, 16.0]]]], dtype=np.float32)
        reference = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        filtered_inputs: list[np.ndarray] = []

        def identity_filter(values: np.ndarray, **kwargs: object) -> np.ndarray:
            filtered_inputs.append(np.asarray(values).copy())
            return np.asarray(values)

        original_filter = rectification_strategy.gaussian_filter
        rectification_strategy.gaussian_filter = identity_filter
        try:
            with self.assertWarnsRegex(RuntimeWarning, "pixel_shift_power"):
                self.strategy.compute_similarity_matrices(
                    views,
                    (0, 0),
                    max_disp=0,
                    gauss_sigma=1.0,
                    gauss_size=5,
                    sub_pix_factor=1,
                    reference_view=reference,
                    pixel_shift_power=0.5,
                    pixel_shift_power_apply_to_reference=False,
                )
        finally:
            rectification_strategy.gaussian_filter = original_filter

        expected_powered_shifted = np.array([[-2.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(filtered_inputs[1], reference * expected_powered_shifted)

    def test_pixel_shift_power_leaves_gradient_component_scores_unpowered(self) -> None:
        views, reference, _, _ = self._make_shifted_blob_case()
        common_kwargs = {
            "max_disp": 4,
            "gauss_sigma": 1.0,
            "gauss_size": 5,
            "normalized_conv": False,
            "sub_pix_factor": 1,
            "reference_view": reference,
            "metric_family": BLURRED_GRADIENT_CORR_METRIC_FAMILY,
            "alpha": 0.0,
        }

        baseline = self.metric_strategy.compute_similarity_matrices(
            views,
            (0, 0),
            **common_kwargs,
        )
        powered = self.metric_strategy.compute_similarity_matrices(
            views,
            (0, 0),
            **common_kwargs,
            pixel_shift_power=0.5,
        )

        assert baseline.component_score_maps_raw is not None
        assert powered.component_score_maps_raw is not None
        np.testing.assert_allclose(
            powered.component_score_maps_raw["gradient"],
            baseline.component_score_maps_raw["gradient"],
        )

    def test_pixel_shift_power_preserves_raw_auxiliary_inputs_with_nonfinite_values(self) -> None:
        views = np.array(
            [[[[np.nan, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]],
            dtype=np.float32,
        )
        reference = np.ones((3, 3), dtype=np.float32)
        for metric_family, component_name in (
            (BLURRED_GRADIENT_CORR_METRIC_FAMILY, "gradient"),
            (BLURRED_LAPLACIAN_CORR_METRIC_FAMILY, "laplacian"),
        ):
            with self.subTest(component_name=component_name):
                common_kwargs = {
                    "max_disp": 0,
                    "gauss_sigma": 1.0,
                    "gauss_size": 5,
                    "sub_pix_factor": 1,
                    "reference_view": reference,
                    "metric_family": metric_family,
                    "alpha": 0.0,
                    "emit_runtime_warning": False,
                }
                baseline = self.strategy.compute_similarity_matrices(
                    views,
                    (0, 0),
                    **common_kwargs,
                )
                powered = self.strategy.compute_similarity_matrices(
                    views,
                    (0, 0),
                    **common_kwargs,
                    pixel_shift_power=0.5,
                )

                assert baseline.component_score_maps_raw is not None
                assert powered.component_score_maps_raw is not None
                np.testing.assert_equal(
                    powered.component_score_maps_raw[component_name],
                    baseline.component_score_maps_raw[component_name],
                )

    def test_sigma_pyramid_forwards_pixel_shift_power_to_each_pass(self) -> None:
        views = np.ones((1, 1, 3, 3), dtype=np.float32)
        zero_coords = np.zeros((3, 3, 1, 1, 2), dtype=np.float32)
        zero_scores = np.zeros((3, 3, 1, 1), dtype=np.float64)
        zero_disps = np.zeros((3, 3, 1, 1), dtype=np.float32)
        fake_result = PixelwiseSimilarityResult(
            coordinates=zero_coords,
            scores=zero_scores,
            disps_y=zero_disps,
            disps_x=zero_disps,
        )
        with mock.patch.object(
            self.strategy,
            "_compute_similarity_matrices_single",
            return_value=fake_result,
        ) as compute:
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                gauss_sigma=2,
                use_sigma_pyramid=True,
                pixel_shift_power=0.5,
            )

        assert compute.call_count == 2
        assert all(call.kwargs["pixel_shift_power"] == 0.5 for call in compute.call_args_list)

    def test_rectify_uses_backward_mapping(self) -> None:
        views = np.zeros((2, 2, 3, 3), dtype=np.float32)
        views[0, 0, 1, 1] = 1.0  # reference bright pixel at center
        views[0, 1, 1, 0] = 1.0  # shifted left by one

        coords = np.zeros((3, 3, 2, 2, 2), dtype=int)
        scores = np.ones((3, 3, 2, 2), dtype=float)

        yy, xx = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        identity_coords = np.stack([yy, xx], axis=-1)

        coords[:, :, 0, 0, :] = identity_coords
        # destination (y,x) should sample source shifted left by 1 (x-1) to bring pixel at x=0 -> x=1
        coords[:, :, 0, 1, :] = np.stack([yy, np.clip(xx - 1, 0, 2)], axis=-1)
        coords[:, :, 1, 0, :] = identity_coords
        coords[:, :, 1, 1, :] = identity_coords
        disps_y = np.zeros((3, 3, 2, 2), dtype=int)
        disps_x = np.zeros((3, 3, 2, 2), dtype=int)
        # displacement is negative in x because the source pixel is left of center
        disps_x[:, :, 0, 1] = -1
        # sanity check: mapping for central pixel points to source (1,0)
        assert tuple(coords[1, 1, 0, 1]) == (1, 0)

        sim_result = PixelwiseSimilarityResult(
            coordinates=coords,
            scores=scores,
            disps_y=disps_y,
            disps_x=disps_x,
        )

        avg_image, rectified_views = self.strategy.rectify(views, sim_result, (0, 0))

        self.assertAlmostEqual(rectified_views[0, 1, 1, 1], 1.0)
        np.testing.assert_allclose(avg_image[1, 1], (rectified_views[:, :, 1, 1].sum()) / 4)

    def test_sigma_pyramid_stops_when_sigma_below_threshold(self) -> None:
        views = np.zeros((1, 1, 2, 2), dtype=np.float32)
        zero_coords = np.zeros((2, 2, 1, 1, 2), dtype=np.float32)
        zero_scores = np.zeros((2, 2, 1, 1), dtype=np.float64)
        zero_disps = np.zeros((2, 2, 1, 1), dtype=np.float32)
        fake_result = PixelwiseSimilarityResult(
            coordinates=zero_coords,
            scores=zero_scores,
            disps_y=zero_disps,
            disps_x=zero_disps,
        )
        with mock.patch.object(
            self.strategy,
            "_compute_similarity_matrices_single",
            return_value=fake_result,
        ) as mock_compute:
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                gauss_sigma=2,
                use_sigma_pyramid=True,
            )
            # 2 -> 1
            self.assertEqual(mock_compute.call_count, 2)

        with mock.patch.object(
            self.strategy,
            "_compute_similarity_matrices_single",
            return_value=fake_result,
        ) as mock_compute:
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                gauss_sigma=4,
                use_sigma_pyramid=True,
            )
            # 4 -> 2 -> 1
            self.assertEqual(mock_compute.call_count, 3)

    def test_blend_normalized_components_respects_alpha_semantics(self) -> None:
        intensity = np.array([[[[0.9, 0.1]]]], dtype=np.float32)
        gradient = np.array([[[[0.2, 0.8]]]], dtype=np.float32)
        stacks = {"intensity": intensity, "gradient": gradient}

        blended_intensity_only = self.strategy._blend_normalized_components(
            component_stacks_norm=stacks,
            alpha=1.0,
            normalized_component_weights={"gradient": 1.0},
        )
        blended_secondary_only = self.strategy._blend_normalized_components(
            component_stacks_norm=stacks,
            alpha=0.0,
            normalized_component_weights={"gradient": 1.0},
        )

        np.testing.assert_allclose(blended_intensity_only, intensity)
        np.testing.assert_allclose(blended_secondary_only, gradient)

    def test_composite_requires_non_intensity_mass_when_alpha_below_one(self) -> None:
        views = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

        with self.assertRaises(ValueError):
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                metric_family=BLURRED_COMPOSITE_CORR_METRIC_FAMILY,
                alpha=0.5,
            )

    def test_intensity_family_requires_alpha_one(self) -> None:
        views = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

        with self.assertRaises(ValueError):
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                metric_family="intensity",
                alpha=0.5,
            )

    def test_pair_families_reject_component_weights(self) -> None:
        views = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

        with self.assertRaises(ValueError):
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                metric_family=BLURRED_GRADIENT_CORR_METRIC_FAMILY,
                alpha=0.5,
                component_weights={"gradient": 1.0},
            )

        with self.assertRaises(ValueError):
            self.strategy.compute_similarity_matrices(
                views,
                (0, 0),
                metric_family=BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
                alpha=0.5,
                component_weights={"laplacian": 1.0},
            )

    def test_metric_mode_populates_component_diagnostics_when_requested(self) -> None:
        views = np.array(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[1.0, 2.0], [3.0, 4.0]],
                ],
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[3.0, 4.0], [5.0, 6.0]],
                ],
            ],
            dtype=np.float32,
        )

        result = self.strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=0,
            sub_pix_factor=1,
            metric_family="ncc_gradient",
            alpha=0.5,
        )

        self.assertEqual(result.metric_family, BLURRED_GRADIENT_CORR_METRIC_FAMILY)
        self.assertEqual(result.alpha, 0.5)
        self.assertIsNotNone(result.blended_score_maps)
        self.assertIsNotNone(result.component_score_maps)
        self.assertIn("gradient", result.component_score_maps or {})
        self.assertIsNotNone(result.view_score_summary)

    def test_gradient_metric_rectifies_shift_with_gradient_diagnostics(self) -> None:
        views, reference_view, center, expected_shift_x = self._make_shifted_blob_case()

        result = self.metric_strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=4,
            gauss_sigma=1.0,
            gauss_size=5,
            normalized_conv=False,
            sub_pix_factor=1,
            reference_view=reference_view,
            metric_family=BLURRED_GRADIENT_CORR_METRIC_FAMILY,
            alpha=0.0,
        )
        avg_image, rectified_views = self.metric_strategy.rectify(views, result, (0, 0))

        self.assertTrue(result.reference_is_external)
        self.assertEqual(result.metric_family, BLURRED_GRADIENT_CORR_METRIC_FAMILY)
        self.assertEqual(result.alpha, 0.0)
        self.assertIsNotNone(result.blended_score_maps)
        self.assertIsNotNone(result.component_score_maps)
        self.assertIsNotNone(result.component_score_maps_raw)
        self.assertIn("gradient", result.component_score_maps or {})
        self.assertNotIn("laplacian", result.component_score_maps or {})
        self.assertIn("view_0_0", result.view_score_summary or {})
        self.assertAlmostEqual(result.disps_x[center[0], center[1], 0, 0], expected_shift_x)
        self.assertAlmostEqual(result.disps_y[center[0], center[1], 0, 0], 0.0)
        self.assertEqual(np.unravel_index(np.argmax(rectified_views[0, 0]), rectified_views[0, 0].shape), center)
        self.assertAlmostEqual(rectified_views[0, 0, center[0], center[1]], 1.0, places=5)
        self.assertAlmostEqual(avg_image[center[0], center[1]], 1.0, places=5)

    def test_laplacian_metric_rectifies_shift_with_laplacian_diagnostics(self) -> None:
        views, reference_view, center, expected_shift_x = self._make_shifted_blob_case()

        result = self.metric_strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=4,
            gauss_sigma=1.0,
            gauss_size=5,
            normalized_conv=False,
            sub_pix_factor=1,
            reference_view=reference_view,
            metric_family=BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
            alpha=0.0,
        )
        avg_image, rectified_views = self.metric_strategy.rectify(views, result, (0, 0))

        self.assertTrue(result.reference_is_external)
        self.assertEqual(result.metric_family, BLURRED_LAPLACIAN_CORR_METRIC_FAMILY)
        self.assertEqual(result.alpha, 0.0)
        self.assertIsNotNone(result.blended_score_maps)
        self.assertIsNotNone(result.component_score_maps)
        self.assertIsNotNone(result.component_score_maps_raw)
        self.assertIn("laplacian", result.component_score_maps or {})
        self.assertNotIn("gradient", result.component_score_maps or {})
        self.assertIn("view_0_0", result.view_score_summary or {})
        self.assertAlmostEqual(result.disps_x[center[0], center[1], 0, 0], expected_shift_x)
        self.assertAlmostEqual(result.disps_y[center[0], center[1], 0, 0], 0.0)
        self.assertEqual(np.unravel_index(np.argmax(rectified_views[0, 0]), rectified_views[0, 0].shape), center)
        self.assertAlmostEqual(rectified_views[0, 0, center[0], center[1]], 1.0, places=5)
        self.assertAlmostEqual(avg_image[center[0], center[1]], 1.0, places=5)

    def test_alpha_one_metric_family_skips_metric_diagnostics(self) -> None:
        views = np.array(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[1.0, 2.0], [3.0, 4.0]],
                ],
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[3.0, 4.0], [5.0, 6.0]],
                ],
            ],
            dtype=np.float32,
        )

        result = self.strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=0,
            sub_pix_factor=1,
            metric_family="ncc_gradient",
            alpha=1.0,
        )

        self.assertEqual(result.metric_family, BLURRED_GRADIENT_CORR_METRIC_FAMILY)
        self.assertEqual(result.alpha, 1.0)
        self.assertIsNone(result.blended_score_maps)
        self.assertIsNone(result.component_score_maps)
        self.assertIsNone(result.view_score_summary)

    def test_alpha_one_gradient_and_laplacian_families_match_intensity_outputs(self) -> None:
        views, reference_view, _, _ = self._make_shifted_blob_case()

        intensity_result = self.metric_strategy.compute_similarity_matrices(
            views,
            (0, 0),
            max_disp=4,
            gauss_sigma=1.0,
            gauss_size=5,
            normalized_conv=False,
            sub_pix_factor=1,
            reference_view=reference_view,
            metric_family="intensity",
            alpha=1.0,
        )
        intensity_avg, intensity_rectified = self.metric_strategy.rectify(views, intensity_result, (0, 0))

        for family in (
            BLURRED_GRADIENT_CORR_METRIC_FAMILY,
            BLURRED_LAPLACIAN_CORR_METRIC_FAMILY,
        ):
            result = self.metric_strategy.compute_similarity_matrices(
                views,
                (0, 0),
                max_disp=4,
                gauss_sigma=1.0,
                gauss_size=5,
                normalized_conv=False,
                sub_pix_factor=1,
                reference_view=reference_view,
                metric_family=family,
                alpha=1.0,
            )
            avg_image, rectified_views = self.metric_strategy.rectify(views, result, (0, 0))

            self.assertEqual(result.metric_family, family)
            self.assertEqual(result.alpha, 1.0)
            self.assertIsNone(result.blended_score_maps)
            self.assertIsNone(result.component_score_maps)
            self.assertIsNone(result.component_score_maps_raw)
            self.assertIsNone(result.view_score_summary)
            np.testing.assert_allclose(result.disps_x, intensity_result.disps_x)
            np.testing.assert_allclose(result.disps_y, intensity_result.disps_y)
            np.testing.assert_allclose(result.scores, intensity_result.scores)
            np.testing.assert_allclose(avg_image, intensity_avg)
            np.testing.assert_allclose(rectified_views, intensity_rectified)


if __name__ == "__main__":
    unittest.main()
