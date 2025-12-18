"""
Interfaces for rectifying view grids extracted from light-field captures.

The default pipeline works with view stacks shaped as ``(N, N, H, W)`` where:

* ``N`` is the view grid size along each axis (e.g., 3 for a 3x3 grid).
* ``H``/``W`` are the per-view height and width, typically matching the
  extracted patch size ``d`` described in ``im_process.extract_views``.
* Values are expected to be floating-point grayscale intensities (``float64``)
  produced by the ROI/circle-based extraction step.

Similarity matrices are stored as ``np.ndarray`` of shape ``(N, N, N, N, 2)``,
where the last axis contains ``(dy, dx)`` integer translations from
``calculate_similarity_matrix``. Both arrays are consumed by rectification
strategies to align every view to the chosen reference and produce standardized
outputs.
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


@dataclass
class PixelwiseSimilarityResult:
    """Container for pixelwise similarity outputs."""

    coordinates: np.ndarray
    scores: np.ndarray


class RectificationStrategy(ABC):
    """
    Rectify a grid of extracted views against a reference and produce
    rectified outputs in a standardized format.

    Implementations should assume:

    * ``views`` has shape ``(N, N, H, W)`` with floating-point grayscale values.
    * ``sim_mat`` has shape ``(N, N, N, N, 2)`` containing translation offsets
      ``(dy, dx)`` relative to the correlation center.
    * ``ref_view_idx`` is a 2-tuple ``(row, col)`` pointing to the reference
      view inside ``views``.

    Returns:
        A tuple of:
        * ``avg_image``: ``np.ndarray`` with shape ``(H, W)`` representing the
          mean (or otherwise aggregated) rectified reference frame.
        * ``rectified_views``: ``np.ndarray`` with shape ``(N, N, H, W)``,
          aligned to the reference view using ``sim_mat`` semantics.
    """

    @abstractmethod
    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        """Pick a reference view index from a grid of views."""

    @abstractmethod
    def compute_similarity_matrix(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int]
    ) -> np.ndarray:
        """Calculate translation offsets between all view pairs."""

    @abstractmethod
    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[np.ndarray] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align ``views`` using ``sim_mat`` offsets and return the averaged image
        plus the full rectified stack.

        Args:
            views: Grid of extracted grayscale patches with shape
                ``(N, N, H, W)``.
            sim_mat: Translation field of shape ``(N, N, N, N, 2)`` produced by
                correlation, where the last axis stores ``(dy, dx)`` offsets.
            ref_view_idx: Indices of the reference view inside ``views``.

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
        views_arr = np.asarray(views)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols = views_arr.shape[:2]
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        return grid_rows // 2, grid_cols // 2

    def compute_similarity_matrix(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int]
    ) -> np.ndarray:
        views_arr = np.asarray(views, dtype=np.float64)

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

    def __init__(self, window_size: int = 5) -> None:
        if window_size % 2 == 0 or window_size < 1:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = int(window_size)
        self.window_radius = self.window_size // 2

    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        views_arr = np.asarray(views)

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols = views_arr.shape[:2]
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")

        return grid_rows // 2, grid_cols // 2

    def compute_similarity_matrix(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int], max_disp: Optional[int] = None,
        gauss_sigma: Optional[float] = None, gauss_size: Optional[int] = None,
    ) -> PixelwiseSimilarityResult:
        views_arr = np.asarray(views, dtype=np.float64)

        if max_disp is None:
            max_disp = self.window_size * 2
        if gauss_sigma is None:
            gauss_sigma = 1.0
        if gauss_size is None:
            gauss_size = self.window_size * 2 + 1

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

        ref_view = views_arr[ref_i, ref_j].astype(np.float32, copy=False)
        corr_mats = np.zeros((height, width, max_disp *2 + 1, max_disp *2 + 1), dtype=np.float32)
        coordinates = np.zeros((height, width, grid_rows, grid_cols, 2), dtype=np.int32)
        scores = np.full((height, width, grid_rows, grid_cols), -np.inf, dtype=np.float64)
        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == ref_i and j == ref_j:
                    continue
                view = views_arr[i, j].astype(np.float32, copy=False)
                for s_y in range(-max_disp, max_disp):
                    for s_x in range(-max_disp, max_disp):
                        shifted_view = np.roll(view, shift=(s_y, s_x), axis=(0,1))
                        corr_mats[:, :, s_y, s_x] = gaussian_filter(np.multiply(ref_view, shifted_view), sigma=gauss_sigma,
                                                truncate=(((gauss_size-1)/2)/gauss_sigma), mode="constant")
                max_val = np.max(corr_mats, axis=(2,3))
                scores[:, :, i, j] = max_val
                coordinates[:, :, i, j] = np.unravel_index(np.nonzero(corr_mats==max_val), corr_mats.shape[:2])

        return PixelwiseSimilarityResult(coordinates=coordinates, scores=scores)

    def rectify(
        self,
        views: np.ndarray,
        sim_mat: Optional[PixelwiseSimilarityResult] = None,
        ref_view_idx: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            else self.compute_similarity_matrix(views_arr, (ref_i, ref_j))
        )
        if not isinstance(sim_result, PixelwiseSimilarityResult):
            raise TypeError("sim_mat must be a PixelwiseSimilarityResult for pixelwise rectification")

        coordinates = np.asarray(sim_result.coordinates)
        scores = np.asarray(sim_result.scores)

        if coordinates.shape != (height, width, grid_rows, grid_cols, 2):
            raise ValueError(
                "coordinates must have shape (H, W, N, N, 2) matching views dimensions"
            )
        if scores.shape != (height, width, grid_rows, grid_cols):
            raise ValueError("scores must have shape (H, W, N, N) matching views dimensions")

        rectified_views = np.zeros((grid_rows, grid_cols, height, width), dtype=np.float64)

        base_mapping = self._identity_mapping(height, width)
        for i in range(grid_rows):
            for j in range(grid_cols):
                if i == ref_i and j == ref_j:
                    rectified_views[i, j] = views_arr[i, j]
                    continue

                mapping = self._build_backward_map(
                    coordinates[:, :, i, j], scores[:, :, i, j], base_mapping
                )
                map_x = mapping[:, :, 1].astype(np.float32)
                map_y = mapping[:, :, 0].astype(np.float32)
                rectified_views[i, j] = cv.remap(
                    views_arr[i, j],
                    map_x,
                    map_y,
                    interpolation=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REFLECT,
                )

        avg_image = np.mean(rectified_views, axis=(0, 1))
        return avg_image, rectified_views

    def _build_backward_map(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        base_mapping: np.ndarray,
    ) -> np.ndarray:
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
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        return np.stack([yy, xx], axis=-1).astype(np.float32)
