"""
Interfaces for rectifying view grids extracted from light-field captures.

View stacks are expected as ``(N, N, H, W)`` arrays of floating-point grayscale
intensities. Global rectification strategies operate on translation similarity
matrices shaped ``(N, N, N, N, 2)``, while the pixelwise variant relies on a
``PixelwiseSimilarityResult`` containing dense coordinate and score maps. Both
approaches ultimately align every view to a selected reference and average the
results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter


DEFAULT_CORRELATION_PAD_WIDTH = int(125 * 0.45)


def resolve_reference_view(
    views: np.ndarray, ref_view_idx: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Validate ``views`` and resolve the effective reference view indices.

    This helper centralizes shape validation and defaulting to the center view
    so multi-stage rectification pipelines can share reference selection logic
    without re-prompting the user. Existing strategies retain their behavior by
    calling ``select_reference_view`` when desired.
    """

    views_arr, grid_rows, grid_cols, center_i, center_j = RectificationStrategy._default_reference_view(
        views
    )
    if ref_view_idx is None:
        return views_arr, (center_i, center_j)

    ref_i, ref_j = ref_view_idx
    if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
        raise ValueError("ref_view_idx must point to a valid view inside views")

    return views_arr, (ref_i, ref_j)


@dataclass
class PixelwiseSimilarityResult:
    """
    Container for pixelwise patch-matching outputs.

    Attributes mirror ``compute_similarity_matrices``: ``coordinates`` holds the
    best-match destination per pixel, ``scores`` stores the corresponding
    correlation values, and ``disps_y``/``disps_x`` cache the raw displacements
    selected for each view relative to the reference.
    """

    coordinates: np.ndarray
    scores: np.ndarray
    disps_y: np.ndarray
    disps_x: np.ndarray


class RectificationStrategy(ABC):
    """
    Abstract interface for aligning a grid of extracted views to a reference.

    Implementations accept a square view stack ``(N, N, H, W)`` and return both
    the averaged rectified image and the fully aligned stack. The similarity
    representation (global translations vs. per-pixel flows) is left to
    subclasses.
    """

    @abstractmethod
    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        """Pick a reference view index from a grid of views."""

    @staticmethod
    def _default_reference_view(views: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Validate ``views`` and return its normalized array form with center indices.

        This helper centralizes the shape/type checks shared by all rectification
        strategies without altering the existing default behavior (center view
        selection).
        """
        views_arr = np.asarray(views)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols = views_arr.shape[:2]
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        return views_arr, grid_rows, grid_cols, grid_rows // 2, grid_cols // 2

    @abstractmethod
    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[np.ndarray] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align ``views`` using a strategy-specific similarity description.

        Args:
            views: Grid of extracted grayscale patches with shape
                ``(N, N, H, W)``.
            sim_mat: Pre-computed similarity data for the strategy (shape/type
                defined by the implementation).
            ref_view_idx: Indices of the reference view inside ``views``. When
                omitted, the strategy chooses a default reference.

        Returns:
            A tuple ``(avg_image, rectified_views)`` where ``avg_image`` has
            shape ``(H, W)`` and ``rectified_views`` has shape ``(N, N, H, W)``,
            both aligned to the reference frame defined by ``ref_view_idx``.
        """


class GlobalTranslationRectificationStrategy(RectificationStrategy):
    """
    Apply global translations derived from a similarity matrix to align views.

    This mirrors the original light-field rectification behavior: each view is
    shifted according to its ``(dy, dx)`` offset relative to the chosen
    reference, and all rectified views are averaged to produce the aggregated
    output.
    """

    def __init__(
        self, pad_width: int = DEFAULT_CORRELATION_PAD_WIDTH, prompt_for_all_views: bool = True
    ) -> None:
        self.pad_width = int(pad_width)
        self.prompt_for_all_views = prompt_for_all_views

    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        _, _, _, center_i, center_j = self._default_reference_view(views)
        return center_i, center_j

    def compute_similarity_matrix(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int],
        thresh: Optional[float] = None,
    ) -> np.ndarray:
        """
        Correlate every view against every other to estimate global translations.

        The reference view indices control which correlations are computed when
        ``prompt_for_all_views`` is ``False``; otherwise all pairwise matches are
        evaluated. The result contains ``(dy, dx)`` offsets in pixels for each
        view pair, padded by ``pad_width`` to avoid edge effects.
        """
        views_arr = np.asarray(views, dtype=np.float64, copy=True)
        if thresh is not None:
            for i in range(views_arr.shape[0]):
                for j in range(views_arr.shape[1]):
                    views_arr[i, j, :, :][views_arr[i, j, :, :] < np.max(views_arr[i, j, :, :]) * thresh] = 0.0

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        num_views = views_arr.shape[0]
        sim_mat = self.pad_width * np.ones((num_views, num_views, num_views, num_views, 2))
        check_all_views = "n"
        if self.prompt_for_all_views:
            check_all_views = input("Check all views for similarity? (y/n): ").lower()

        ref_i, ref_j = ref_view_idx
        if not (0 <= ref_i < num_views and 0 <= ref_j < num_views):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        for i in range(num_views):
            if check_all_views != "y" and i != ref_view_idx[0]:
                continue
            for j in range(num_views):
                if check_all_views != "y" and j != ref_view_idx[1]:
                    continue
                for k in range(num_views):
                    for l in range(num_views):
                        view1 = views_arr[i, j]
                        view2 = views_arr[k, l]

                        padded_view1 = np.pad(view1, self.pad_width)

                        cor = correlate2d(padded_view1, view2, mode="valid")
                        max_cor_loc = np.unravel_index(np.argmax(cor), cor.shape)
                        sim_mat[i, j, k, l, :] = np.subtract(
                            max_cor_loc, (cor.shape[0] // 2, cor.shape[1] // 2)
                        )

        return sim_mat

    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[np.ndarray] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp each view by its global translation and average the aligned stack.

        If ``sim_mat`` is omitted the method computes pairwise translations on
        the fly using ``compute_similarity_matrix``. Warping is done with
        OpenCV's affine transform and the outputs retain the input dimensions.
        """
        views_arr = np.asarray(views, dtype=np.float64)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        ref_i, ref_j = ref_view_idx or self.select_reference_view(views_arr)

        sim_arr = (
            np.asarray(sim_mat)
            if sim_mat is not None
            else self.compute_similarity_matrix(views_arr, (ref_i, ref_j))
        )
        if sim_arr.shape != (grid_rows, grid_cols, grid_rows, grid_cols, 2):
            raise ValueError(
                "sim_mat must have shape (N, N, N, N, 2) matching views grid size"
            )

        if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        rectified_views = np.zeros((grid_rows, grid_cols, height, width), dtype=np.float64)

        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == ref_i and j == ref_j:
                    rectified_views[i, j] = views_arr[i, j]
                    continue

                dy, dx = sim_arr[ref_i, ref_j, i, j]
                warp_mat = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
                rectified_views[i, j] = cv.warpAffine(
                    views_arr[i, j], warp_mat, (width, height)
                )

        avg_image = np.mean(rectified_views, axis=(0, 1))
        return avg_image, rectified_views


class PixelwisePatchRectificationStrategy(RectificationStrategy):
    """
    Rectify views using per-pixel patch matching against the reference view.

    This strategy correlates a ``k x k`` patch around every pixel in each view
    with rolling windows in the reference view. For every source pixel, the
    correlation peak and its score are stored in a similarity matrix of shape
    ``(H, W, N, N, 2)`` (coordinates) alongside a score matrix of shape
    ``(H, W, N, N)``. A backward warp is then constructed per view to align all
    views to the reference grid.
    """

    def __init__(self, window_size: int = 41, step_size: Optional[int] = None) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive odd integer")
        elif window_size % 2 == 0:
            print("Warning: window_size should be odd, incrementing by 1")
            window_size += 1
        self.window_size = int(window_size)
        self.window_radius = self.window_size // 2
        self.step_size = step_size if step_size is not None else self.window_size // 10
        if self.step_size < 1:
            self.step_size = 1

    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        _, _, _, center_i, center_j = self._default_reference_view(views)
        return center_i, center_j

    def compute_similarity_matrices(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int], max_disp: Optional[int] = None,
        gauss_sigma: Optional[float] = None, gauss_size: Optional[int] = None,
        disp_gauss_sigma: Optional[float] = None, disp_gauss_size: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> PixelwiseSimilarityResult:
        """
        Perform per-pixel patch correlation between each view and the reference.

        For every non-reference view, the function rolls the image across a
        displacement grid and computes Gaussian-filtered correlation scores
        against the (optionally thresholded) reference. The best displacement
        per pixel is recorded along with raw scores and derived flow fields.

        Args:
            views: Square grid of grayscale patches with shape ``(N, N, H, W)``.
            ref_view_idx: ``(row, col)`` index of the reference view.
            max_disp: Maximum pixel displacement searched in each direction.
            gauss_sigma: Sigma for blurring the per-pixel correlation product.
            gauss_size: Kernel size used for the Gaussian blur.
            disp_gauss_sigma: Optional smoothing applied to the displacement
                fields.
            disp_gauss_size: Kernel size for displacement smoothing.
            thresh: Relative intensity threshold (0–1) below which reference
                pixels are ignored.

        Returns:
            ``PixelwiseSimilarityResult`` containing correlation scores,
            displacement maps, and target coordinates for each view.
        """
        views_arr = np.asarray(views, dtype=np.float64, copy=True)
        if thresh is not None:
            for i in range(views_arr.shape[0]):
                for j in range(views_arr.shape[1]):
                    views_arr[i, j, :, :][views_arr[i, j, :, :] < np.max(views_arr[i, j, :, :]) * thresh] = 0.0
        
        if max_disp is None:
            max_disp = self.window_radius
        if gauss_sigma is None:
            gauss_sigma = max_disp / 5.0
        if gauss_size is None:
            gauss_size = self.window_radius * 2 + 1
        if disp_gauss_sigma is None:
            disp_gauss_sigma = 0.0
        if disp_gauss_size is None:
            disp_gauss_size = self.window_radius * 2 + 1

        D = max_disp * 2 + 1
        disp_arr = np.arange(-max_disp, max_disp + 1)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")
        if self.window_size > height or self.window_size > width:
            raise ValueError("window_size cannot exceed view dimensions")

        ref_i, ref_j = ref_view_idx
        if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        # Setting up and thresholding the reference view according to percentages from maximal value
        ref_view = views_arr[ref_i, ref_j].astype(np.float32, copy=True)
        if thresh is not None:
            ref_view[ref_view < np.max(ref_view)*thresh] = 0.0

        corr_mats = np.zeros((height, width, grid_rows, grid_cols, D, D), dtype=np.float32)
        coordinates = np.zeros((height, width, grid_rows, grid_cols, 2), dtype=np.int32)
        scores = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float64)
        disps_y = np.zeros((height, width, grid_rows, grid_cols), dtype=np.int32)
        disps_x = np.zeros((height, width, grid_rows, grid_cols), dtype=np.int32)
        base_coords = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), -1)

        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == ref_i and j == ref_j:
                    continue
                view = views_arr[i, j].astype(np.float32, copy=False)
                for s_y_ind in range(0, D, self.step_size):
                    for s_x_ind in range(0, D, self.step_size):
                        s_y = s_y_ind - max_disp
                        s_x = s_x_ind - max_disp
                        shifted_view = np.roll(view, shift=(s_y, s_x), axis=(0, 1))
                        corr_mats[:, :, i, j, s_y_ind, s_x_ind] = gaussian_filter(
                            ref_view * shifted_view,
                            sigma=gauss_sigma,
                            truncate=(((gauss_size - 1) / 2) / gauss_sigma),
                            mode="constant",
                        )

                corr_stack = corr_mats[:, :, i, j, :, :]  # shape (H, W, D, D)
                max_val = np.max(corr_stack, axis=(2, 3))
                scores[:, :, i, j] = max_val

                flat = corr_stack.reshape((height, width, D * D))
                argmax_flat = np.argmax(flat, axis=2)
                idx_y, idx_x = np.unravel_index(argmax_flat, (D, D))

                # ensure pixels with zero max score keep zero displacement (center index)
                zero_mask = (max_val == 0)
                if np.any(zero_mask):
                    idx_y = idx_y.copy()
                    idx_x = idx_x.copy()
                    idx_y[zero_mask] = max_disp
                    idx_x[zero_mask] = max_disp

                # compute coordinates: base + displacement, apply gaussian filter if sigma is legal
                if disp_gauss_sigma is None or disp_gauss_sigma <= 0.0:
                    disps_y[:, :, i, j] = disp_arr[idx_y]
                    disps_x[:, :, i, j] = disp_arr[idx_x]
                else:
                    disps_y[:, :, i, j] = gaussian_filter(disp_arr[idx_y], sigma=disp_gauss_sigma, truncate=(((disp_gauss_size - 1) / 2) / disp_gauss_sigma), mode="constant") if disp_gauss_sigma is not None else disp_arr[idx_y]
                    disps_x[:, :, i, j] = gaussian_filter(disp_arr[idx_x], sigma=disp_gauss_sigma, truncate=(((disp_gauss_size - 1) / 2) / disp_gauss_sigma), mode="constant") if disp_gauss_sigma is not None else disp_arr[idx_x]
                coordinates[:, :, i, j, 0] = base_coords[:, :, 0] + disp_arr[idx_y]
                coordinates[:, :, i, j, 1] = base_coords[:, :, 1] + disp_arr[idx_x]

        return PixelwiseSimilarityResult(coordinates=coordinates, scores=scores, disps_y = disps_y, disps_x = disps_x)

    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[PixelwiseSimilarityResult] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward-warp each view using per-pixel displacement fields.

        The provided ``PixelwiseSimilarityResult`` supplies absolute coordinates
        for ``cv.remap``; when omitted, similarities are recomputed first. The
        reference view is copied directly into the rectified stack before the
        averaged image is calculated.
        """
        views_arr = np.asarray(views, dtype=np.float64)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        ref_i, ref_j = ref_view_idx or self.select_reference_view(views_arr)

        sim_result = (
            sim_mat
            if sim_mat is not None
            else self.compute_similarity_matrices(views_arr, (ref_i, ref_j))
        )
        if not isinstance(sim_result, PixelwiseSimilarityResult):
            raise TypeError("sim_mat must be a PixelwiseSimilarityResult for pixelwise rectification")

        coordinates = np.asarray(sim_result.coordinates)
        scores = np.asarray(sim_result.scores)
        disps_y = np.asarray(sim_result.disps_y)
        disps_x = np.asarray(sim_result.disps_x)

        if coordinates.shape != (height, width, grid_rows, grid_cols, 2):
            raise ValueError(
                "coordinates must have shape (H, W, N, N, 2) matching views dimensions"
            )
        if scores.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("scores must have shape (H, W, N, N) matching views dimensions")
        if disps_y.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("disps_y must have shape (H, W, N, N) matching views dimensions")
        if disps_x.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("disps_x must have shape (H, W, N, N) matching views dimensions")

        rectified_views = np.zeros((grid_rows, grid_cols, height, width), dtype=np.float64)

        base_mapping = self._identity_mapping(height, width)
        base_y = base_mapping[:, :, 0]
        base_x = base_mapping[:, :, 1]
        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == ref_i and j == ref_j:
                    rectified_views[i, j] = views_arr[i, j]
                    continue

                # map_y = (base_y - disps_y[:, :, i, j]).astype(np.float32)
                # map_x = (base_x - disps_x[:, :, i, j]).astype(np.float32)
                map_y = coordinates[:, :, i, j, 0].astype(np.float32)
                map_x = coordinates[:, :, i, j, 1].astype(np.float32)
                rectified_views[i, j] = cv.remap(
                    views_arr[i, j],
                    map_x,
                    map_y,
                    interpolation=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_CONSTANT,
                )

        avg_image = np.mean(rectified_views, axis=(0, 1))
        return avg_image, rectified_views

    def _build_backward_map(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        base_mapping: np.ndarray,
    ) -> np.ndarray:
        """
        Construct a reverse lookup map from destination coordinates and scores.

        For each candidate coordinate, the mapping keeps only the source pixel
        that achieved the highest score at that destination, falling back to the
        identity map when no valid mapping exists.
        """
        mapping = base_mapping.copy()
        height, width, _ = coordinates.shape
        best_scores = np.full((height, width), -np.inf, dtype=np.float64)

        for y in range(height):
            for x in range(width):
                target_y, target_x = coordinates[y, x]
                if not (0 <= target_y < height and 0 <= target_x < width):
                    continue

                if scores[y, x] > best_scores[target_y, target_x]:
                    best_scores[target_y, target_x] = scores[y, x]
                    mapping[target_y, target_x] = (y, x)

        return mapping

    @staticmethod
    def _identity_mapping(height: int, width: int) -> np.ndarray:
        """Return an ``(H, W, 2)`` array of pixel indices for identity remapping."""
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        return np.stack([yy, xx], axis=-1).astype(np.float32)
