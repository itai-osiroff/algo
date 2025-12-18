import unittest

import numpy as np

from moment_images.rectification_strategy import GlobalTranslationRectificationStrategy


class GlobalTranslationRectificationStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = GlobalTranslationRectificationStrategy()

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
        strategy = GlobalTranslationRectificationStrategy(pad_width=1)

        sim_mat = strategy.compute_similarity_matrix(views, (0, 0))

        self.assertEqual(sim_mat.shape, (2, 2, 2, 2, 2))

    def test_rectify_with_internal_computation(self) -> None:
        views = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        strategy = GlobalTranslationRectificationStrategy(pad_width=1)

        avg_image, rectified_views = strategy.rectify(views)

        np.testing.assert_allclose(avg_image, views[0, 0])
        np.testing.assert_allclose(rectified_views[0, 0], views[0, 0])


if __name__ == "__main__":
    unittest.main()
