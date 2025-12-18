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
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from scipy.signal import correlate2d


DEFAULT_CORRELATION_PAD_WIDTH = int(125 * 0.45)


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

    def __init__(self, pad_width: int = DEFAULT_CORRELATION_PAD_WIDTH) -> None:
        self.pad_width = int(pad_width)

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

        ref_i, ref_j = ref_view_idx
        if not (0 <= ref_i < num_views and 0 <= ref_j < num_views):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        for i in range(num_views):
            for j in range(num_views):
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
