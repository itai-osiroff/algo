import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from moment_images.rectification_strategy import PixelwisePatchRectificationStrategy, PixelwiseSimilarityResult

class TestRectificationStrategyPyramid(unittest.TestCase):

    def setUp(self):
        self.strategy = PixelwisePatchRectificationStrategy()
        self.h, self.w = 10, 10
        self.grid_rows, self.grid_cols = 1, 1

    def _create_result(self, val):
        shape = (self.h, self.w, self.grid_rows, self.grid_cols)
        disps_y = np.full(shape, val, dtype=np.float32)
        disps_x = np.full(shape, val, dtype=np.float32)
        scores = np.ones(shape, dtype=np.float64)
        coordinates = np.zeros((self.h, self.w, self.grid_rows, self.grid_cols, 2), dtype=np.float32)
        return PixelwiseSimilarityResult(
            coordinates=coordinates,
            scores=scores,
            disps_y=disps_y,
            disps_x=disps_x,
            sub_pix_factor=1
        )

    @patch.object(PixelwisePatchRectificationStrategy, '_compute_similarity_matrices_single')
    def test_sigma_pyramid_sequence_and_constraint(self, mock_compute):
        # Scenario:
        # Step 1 (Sigma 7): Return 0. Baseline = 0.
        # Step 2 (Sigma 4): Return 1.5.
        #   Diff vs Baseline(0) is 1.5. Valid (<=2). Current -> 1.5.
        #   Diff vs Old Current (0) is 1.5 (>1). Not converged.
        # Step 3 (Sigma 2): Return 5.
        #   Diff vs Baseline(0) is 5. Invalid (>2).
        #   Constraint logic:
        #     If reverts to baseline: new val is 0. Diff vs Old Current(1.5) is 1.5. Not converged?
        #     If keeps current: new val is 1.5. Diff vs Old Current(1.5) is 0. Converged.

        def side_effect(*args, **kwargs):
            sigma = kwargs.get('gauss_sigma')
            if sigma == 7:
                return self._create_result(0.0)
            elif sigma == 4:
                return self._create_result(1.5)
            elif sigma == 2:
                return self._create_result(5.0)
            return self._create_result(0.0)

        mock_compute.side_effect = side_effect

        views = np.zeros((1, 1, 10, 10))
        # gauss_sigma=7 triggers the 7 -> 4 -> 2 sequence
        res = self.strategy.compute_similarity_matrices(
            views, (0, 0),
            use_sigma_pyramid=True,
            gauss_sigma=7
        )

        # Verify call sequence
        calls = mock_compute.call_args_list
        sigmas = [c.kwargs['gauss_sigma'] for c in calls]
        self.assertEqual(sigmas, [7, 4.0, 2.0], "Should iterate through sigmas 7, 4, 2")

        # Verify result logic
        # If the code reverts to baseline (0) on error, this will be 0.
        # If the code keeps previous value (1.5) on error, this will be 1.5.
        self.assertEqual(res.disps_y[0,0,0,0], 1.5,
                         "Should keep the previous valid value (1.5) when the new value (5.0) deviates too much from baseline (0.0)")

    @patch.object(PixelwisePatchRectificationStrategy, '_compute_similarity_matrices_single')
    def test_sigma_pyramid_convergence(self, mock_compute):
        # Scenario: Convergence
        # Step 1 (Sigma 7): 0.
        # Step 2 (Sigma 4): 0.5. Diff vs 0 is 0.5 (<=1). Converged?
        # Check logic: converged = np.all(abs(constrained - current) <= 1)
        # constrained (0.5) - current (0) = 0.5 <= 1. True.
        # So it should stop after Sigma 4, and NOT run Sigma 2.

        def side_effect(*args, **kwargs):
            sigma = kwargs.get('gauss_sigma')
            if sigma == 7:
                return self._create_result(0.0)
            elif sigma == 4:
                return self._create_result(0.5)
            elif sigma == 2:
                return self._create_result(100.0) # Should not be reached
            return self._create_result(0.0)

        mock_compute.side_effect = side_effect

        views = np.zeros((1, 1, 10, 10))
        res = self.strategy.compute_similarity_matrices(
            views, (0, 0),
            use_sigma_pyramid=True,
            gauss_sigma=7
        )

        calls = mock_compute.call_args_list
        sigmas = [c.kwargs['gauss_sigma'] for c in calls]
        self.assertEqual(sigmas, [7, 4.0], "Should stop after sigma 4 due to convergence")
        self.assertEqual(res.disps_y[0,0,0,0], 0.5)
