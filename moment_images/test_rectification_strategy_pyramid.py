import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from moment_images.rectification_strategy import PixelwisePatchRectificationStrategy, PixelwiseSimilarityResult

class TestRectificationStrategyPyramid(unittest.TestCase):

    def setUp(self):
        self.strategy = PixelwisePatchRectificationStrategy(window_size=11, step_size=1)
        self.h, self.w = 20, 20
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
    def test_sigma_pyramid_sequence_and_chaining(self, mock_compute):
        # Scenario:
        # Step 1 (Sigma 7): Returns Res1.
        # Step 2 (Sigma 3): Called with prev_result=Res1. Returns Res2.
        # Step 3 (Sigma 1): Called with prev_result=Res2. Returns Res3.

        res1 = self._create_result(1.0)
        res2 = self._create_result(2.0)
        res3 = self._create_result(3.0)

        def side_effect(*args, **kwargs):
            sigma = kwargs.get('gauss_sigma')
            prev = kwargs.get('prev_result')

            if sigma == 7:
                # First call, prev should be None (or not passed)
                if prev is not None:
                     raise AssertionError("First call should have prev_result=None")
                return res1
            elif sigma == 3:
                if prev is not res1:
                    raise AssertionError("Second call should use res1 as prev_result")
                return res2
            elif sigma == 1:
                if prev is not res2:
                    raise AssertionError("Third call should use res2 as prev_result")
                return res3
            return self._create_result(0.0)

        mock_compute.side_effect = side_effect

        views = np.zeros((1, 1, 20, 20))
        # gauss_sigma=7 triggers 7 -> 3 -> 1
        res = self.strategy.compute_similarity_matrices(
            views, (0, 0),
            use_sigma_pyramid=True,
            gauss_sigma=7
        )

        calls = mock_compute.call_args_list
        sigmas = [c.kwargs.get('gauss_sigma') for c in calls]
        self.assertEqual(sigmas, [7, 3, 1], "Should iterate through sigmas 7, 3, 1")
        self.assertIs(res, res3)

    def test_masking_logic_in_single_step(self):
        # Use actual implementation logic (no mock) to test the masking
        H, W = 20, 20
        views = np.zeros((1, 1, H, W), dtype=np.float32)

        # Reference: dot at 10, 10
        ref_view = np.zeros((H, W), dtype=np.float32)
        ref_view[10, 10] = 1.0

        # View: dot at 10, 5. Correct displacement is (0, 5).
        views[0, 0, 10, 5] = 1.0

        # Run WITHOUT prev_result
        # max_disp=5
        # It should find (0, 5).
        res = self.strategy._compute_similarity_matrices_single(
            views, (0, 0), max_disp=5,
            reference_view=ref_view,
            sub_pix_factor=1,
            normalized_conv=False,
            gauss_sigma=0.1,
            gauss_size=1
        )
        self.assertEqual(res.disps_x[10,10,0,0], 5.0)

        # Now create a prev_result saying disp is (0, 0).
        # Window +/- 2. So range [-2, 2].
        # (0, 5) is outside.
        # It should NOT find (0, 5).
        prev_result = self._create_result(0.0) # all zeros

        res_constrained = self.strategy._compute_similarity_matrices_single(
            views, (0, 0), max_disp=5,
            reference_view=ref_view,
            sub_pix_factor=1,
            normalized_conv=False,
            gauss_sigma=0.1,
            gauss_size=1,
            prev_result=prev_result
        )

        self.assertNotEqual(res_constrained.disps_x[10,10,0,0], 5.0)
        self.assertLessEqual(abs(res_constrained.disps_x[10,10,0,0]), 2.0)
