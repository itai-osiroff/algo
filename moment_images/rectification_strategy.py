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
from typing import Tuple

import numpy as np


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
    def rectify(
        self,
        views: np.ndarray,
        sim_mat: np.ndarray,
        ref_view_idx: Tuple[int, int],
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

