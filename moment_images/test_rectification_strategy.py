import unittest
from unittest import mock

import numpy as np

from moment_images.rectification_strategy import (
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
        with self.assertRaises(ValueError):
            self.strategy._resolve_non_intensity_component_weights(
                metric_family="composite",
                component_weights=None,
                alpha=0.5,
            )

    def test_composite_weights_are_normalized(self) -> None:
        raw, normalized = self.strategy._resolve_non_intensity_component_weights(
            metric_family="composite",
            component_weights={"gradient": 2.0, "laplacian": 6.0},
            alpha=0.2,
        )
        self.assertEqual(raw["gradient"], 2.0)
        self.assertEqual(raw["laplacian"], 6.0)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertAlmostEqual(normalized["gradient"], 0.25)
        self.assertAlmostEqual(normalized["laplacian"], 0.75)

    def test_intensity_family_has_no_non_intensity_weights(self) -> None:
        raw, normalized = self.strategy._resolve_non_intensity_component_weights(
            metric_family="intensity",
            component_weights={"gradient": 2.0},
            alpha=1.0,
        )

        self.assertEqual(raw, {})
        self.assertEqual(normalized, {})

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

        self.assertEqual(result.metric_family, "ncc_gradient")
        self.assertEqual(result.alpha, 0.5)
        self.assertIsNotNone(result.blended_score_maps)
        self.assertIsNotNone(result.component_score_maps)
        self.assertIn("gradient", result.component_score_maps or {})
        self.assertIsNotNone(result.view_score_summary)

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

        self.assertEqual(result.metric_family, "ncc_gradient")
        self.assertEqual(result.alpha, 1.0)
        self.assertIsNone(result.blended_score_maps)
        self.assertIsNone(result.component_score_maps)
        self.assertIsNone(result.view_score_summary)


if __name__ == "__main__":
    unittest.main()
