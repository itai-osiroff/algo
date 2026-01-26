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
    selected for each view relative to the reference. ``sub_pix_factor`` tracks
    the upsampling factor used to resolve sub-pixel shifts.
    """

    coordinates: np.ndarray
    scores: np.ndarray
    disps_y: np.ndarray
    disps_x: np.ndarray
    sub_pix_factor: int = 2
    reference_is_external: bool = False


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

        Accept either:
          - (N, N, H, W) : single view grid, or
          - (E, N, N, H, W): multiple exposures of a view grid (will be averaged
            across the exposure axis to produce a single (N, N, H, W) grid).

        This helper centralizes the shape/type checks shared by all rectification
        strategies without altering the existing default behavior (center view
        selection).
        """
        views_arr = np.asarray(views)

        # Accept either a 4D view grid or a 5D stack of exposures (E, N, N, H, W).
        if views_arr.ndim == 5:
            # Collapse exposures by averaging to produce a single (N, N, H, W) grid.
            views_arr = np.mean(views_arr, axis=0)
        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W) or (E, N, N, H, W)")

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
        self,
        pad_width: int = DEFAULT_CORRELATION_PAD_WIDTH,
        prompt_for_all_views: bool = True,
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
                    views_arr[i, j], warp_mat, (width, height), borderMode=cv.BORDER_CONSTANT
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

    def __init__(
        self,
        window_size: int = 41,
        step_size: Optional[int] = None,
        zero_fill_roll: bool = False,
    ) -> None:
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
        self.zero_fill_roll = zero_fill_roll

    def select_reference_view(self, views: np.ndarray) -> Tuple[int, int]:
        _, _, _, center_i, center_j = self._default_reference_view(views)
        return center_i, center_j

    def compute_similarity_matrices(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int], max_disp: Optional[int] = None,
        gauss_sigma: Optional[float] = None, gauss_size: Optional[int] = None,
        disp_gauss_sigma: Optional[float] = None, disp_gauss_size: Optional[int] = None,
        thresh: Optional[float] = None, normalized_conv: Optional[bool] = True,
        sub_pix_factor: int = 2,
        reference_view: Optional[np.ndarray] = None,
        use_sigma_pyramid: bool = False,
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
            use_sigma_pyramid: When ``True``, iteratively recompute similarities
                with smaller Gaussian sigmas while constraining updates to stay
                within two pixels of the initial displacement estimates.
            sub_pix_factor: Upsampling factor for sub-pixel displacement
                searches. ``1`` keeps full-pixel shifts, ``2`` enables half-pixel
                shifts, and so on.

        Returns:
            ``PixelwiseSimilarityResult`` containing correlation scores,
            displacement maps, and target coordinates for each view.
        """
        if not use_sigma_pyramid:
            return self._compute_similarity_matrices_single(
                views,
                ref_view_idx,
                max_disp=max_disp,
                gauss_sigma=gauss_sigma,
                gauss_size=gauss_size,
                disp_gauss_sigma=disp_gauss_sigma,
                disp_gauss_size=disp_gauss_size,
                thresh=thresh,
                normalized_conv=normalized_conv,
                sub_pix_factor=sub_pix_factor,
                reference_view=reference_view,
            )

        base_sigma = 7 if gauss_sigma is None else gauss_sigma
        base_size = int(base_sigma * 6) if gauss_size is None else gauss_size
        baseline_result = self._compute_similarity_matrices_single(
            views,
            ref_view_idx,
            max_disp=max_disp,
            gauss_sigma=base_sigma,
            gauss_size=base_size,
            disp_gauss_sigma=disp_gauss_sigma,
            disp_gauss_size=disp_gauss_size,
            thresh=thresh,
            normalized_conv=normalized_conv,
            sub_pix_factor=sub_pix_factor,
            reference_view=reference_view,
        )

        current_result = baseline_result
        current_sigma = float(base_sigma)
        while current_sigma >= 3:
            next_sigma = float(np.ceil(current_sigma / 2.0))
            if next_sigma == current_sigma or next_sigma < 1:
                break

            next_size = int(next_sigma * 6) if gauss_size is None else gauss_size
            candidate_result = self._compute_similarity_matrices_single(
                views,
                ref_view_idx,
                max_disp=max_disp,
                gauss_sigma=next_sigma,
                gauss_size=next_size,
                disp_gauss_sigma=disp_gauss_sigma,
                disp_gauss_size=disp_gauss_size,
                thresh=thresh,
                normalized_conv=normalized_conv,
                sub_pix_factor=sub_pix_factor,
                reference_view=reference_view,
            )

            current_result, converged = self._constrain_sigma_pyramid_update(
                baseline_result,
                current_result,
                candidate_result,
            )
            current_sigma = next_sigma
            if current_sigma < 3 or converged:
                break

        return current_result

    def _compute_similarity_matrices_single(
        self, views: np.ndarray, ref_view_idx: Tuple[int, int], max_disp: Optional[int] = None,
        gauss_sigma: Optional[float] = None, gauss_size: Optional[int] = None,
        disp_gauss_sigma: Optional[float] = None, disp_gauss_size: Optional[int] = None,
        thresh: Optional[float] = None, normalized_conv: Optional[bool] = True,
        sub_pix_factor: int = 2,
        reference_view: Optional[np.ndarray] = None,
    ) -> PixelwiseSimilarityResult:
        """Compute pixelwise similarity matrices for a single Gaussian sigma."""
        if sub_pix_factor < 1:
            raise ValueError("sub_pix_factor must be >= 1")
        sub_pix_factor = int(sub_pix_factor)

        views_arr = np.asarray(views, dtype=np.float64, copy=True)
        if thresh is not None:
            for i in range(views_arr.shape[0]):
                for j in range(views_arr.shape[1]):
                    views_arr[i, j, :, :][views_arr[i, j, :, :] < np.max(views_arr[i, j, :, :]) * thresh] = 0.0
        
        if max_disp is None:
            max_disp = self.window_radius
        if gauss_sigma is None:
            gauss_sigma = 7
        if gauss_size is None:
            gauss_size = int(gauss_sigma * 6)
        if disp_gauss_sigma is None:
            disp_gauss_sigma = 0.0

        if views_arr.ndim != 4:
            raise ValueError("views must have shape (N, N, H, W)")

        grid_rows, grid_cols, height, width = views_arr.shape
        if grid_rows != grid_cols:
            raise ValueError("views must represent a square grid of views")
        original_height, original_width = height, width

        if sub_pix_factor > 1:
            resized_views = np.zeros(
                (grid_rows, grid_cols, height * sub_pix_factor, width * sub_pix_factor),
                dtype=np.float64,
            )
            for i in range(grid_rows):
                for j in range(grid_cols):
                    resized_views[i, j] = cv.resize(
                        views_arr[i, j],
                        (width * sub_pix_factor, height * sub_pix_factor),
                        interpolation=cv.INTER_LINEAR,
                    )
            views_arr = resized_views
            height *= sub_pix_factor
            width *= sub_pix_factor

        effective_window_size = self.window_size * sub_pix_factor
        effective_window_radius = effective_window_size // 2
        effective_step_size = 1 if sub_pix_factor > 1 else self.step_size
        max_disp_scaled = max_disp * sub_pix_factor
        if disp_gauss_size is None:
            disp_gauss_size = effective_window_radius * 2 + 1

        if effective_window_size > height or effective_window_size > width:
            raise ValueError("window_size cannot exceed view dimensions")

        ref_i, ref_j = ref_view_idx
        if not (0 <= ref_i < grid_rows and 0 <= ref_j < grid_cols):
            raise ValueError("ref_view_idx must point to a valid view inside views")

        reference_is_external = reference_view is not None
        if reference_is_external:
            ref_view = np.asarray(reference_view, dtype=np.float32)
            if ref_view.ndim != 2:
                raise ValueError("reference_view must be a 2D array matching a single view")
            if ref_view.shape == (original_height, original_width) and sub_pix_factor > 1:
                ref_view = cv.resize(
                    ref_view,
                    (width, height),
                    interpolation=cv.INTER_LINEAR,
                )
            if ref_view.shape != (height, width):
                raise ValueError("reference_view must match the view dimensions")
        else:
            # Setting up and thresholding the reference view according to percentages from maximal value
            ref_view = views_arr[ref_i, ref_j].astype(np.float32, copy=True)
        if thresh is not None:
            ref_view[ref_view < np.max(ref_view)*thresh] = 0.0

        D = max_disp_scaled * 2 + 1
        disp_arr = np.arange(-max_disp_scaled, max_disp_scaled + 1, dtype=np.float32)

        corr_mats = np.zeros((height, width, grid_rows, grid_cols, D, D), dtype=np.float32)
        if normalized_conv:
            norm_mats = np.zeros((height, width, grid_rows, grid_cols, D, D), dtype=np.float32)
        coordinates = np.zeros((height, width, grid_rows, grid_cols, 2), dtype=np.float32)
        scores = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float64)
        disps_y = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float32)
        disps_x = np.zeros((height, width, grid_rows, grid_cols), dtype=np.float32)
        base_coords = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), -1)
        if normalized_conv:
            ref_view_blurred = gaussian_filter(
                                                ref_view,
                                                sigma=gauss_sigma,
                                                truncate=(gauss_size/gauss_sigma),
                                                mode="constant")
        for i in range(grid_rows):
            for j in range(grid_cols):
                if not reference_is_external and i == ref_i and j == ref_j:
                    continue
                view = views_arr[i, j].astype(np.float32, copy=False)
                for s_y_ind in range(0, D, effective_step_size):
                    for s_x_ind in range(0, D, effective_step_size):
                        s_y = s_y_ind - max_disp_scaled
                        s_x = s_x_ind - max_disp_scaled
                        shifted_view = self._roll_with_fill(
                            view, shift_y=s_y, shift_x=s_x, zero_fill=self.zero_fill_roll
                        )
                        corr_mats[:, :, i, j, s_y_ind, s_x_ind] = gaussian_filter(
                            ref_view * shifted_view,
                            sigma=gauss_sigma,
                            truncate=(gauss_size/ gauss_sigma),
                            mode="constant",
                        )
                        if normalized_conv:
                            norm_shifted_view = self._roll_with_fill(view, shift_y=s_y,
                                                                      shift_x=s_x,
                                                                        zero_fill=self.zero_fill_roll)
                            norm_mats[:, :, i, j, s_y_ind, s_x_ind] = gaussian_filter(
                                norm_shifted_view*2,
                                  sigma=gauss_sigma,
                                    truncate=(gauss_size/gauss_sigma),
                                      mode="constant")
                if normalized_conv:
                        norm_mats[: ,:, i, j, :, :] = np.sqrt(norm_mats[:, :, i, j, :, :]*np.tile(ref_view_blurred, [D, D, 1, 1]).T)
                        norm_mats[:, : ,i, j, :, :] = norm_mats[:, :, i, j, :, :] + 0.1*np.max(norm_mats[:, :, i, j, :, :])
                        corr_mats[:, :, i, j, :, :] = corr_mats[:, :, i, j, :, :] / norm_mats[:, : ,i ,j, : , :]
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
                    idx_y[zero_mask] = max_disp_scaled
                    idx_x[zero_mask] = max_disp_scaled

                # compute coordinates: base + displacement, apply gaussian filter if sigma is legal
                if disp_gauss_sigma is None or disp_gauss_sigma <= 0.0:
                    disps_y[:, :, i, j] = disp_arr[idx_y]
                    disps_x[:, :, i, j] = disp_arr[idx_x]
                else:
                    disps_y[:, :, i, j] = gaussian_filter(disp_arr[idx_y], sigma=disp_gauss_sigma, truncate=(((disp_gauss_size - 1) / 2) / disp_gauss_sigma), mode="constant") if disp_gauss_sigma is not None else disp_arr[idx_y]
                    disps_x[:, :, i, j] = gaussian_filter(disp_arr[idx_x], sigma=disp_gauss_sigma, truncate=(((disp_gauss_size - 1) / 2) / disp_gauss_sigma), mode="constant") if disp_gauss_sigma is not None else disp_arr[idx_x]
                coordinates[:, :, i, j, 0] = base_coords[:, :, 0] - disp_arr[idx_y]
                coordinates[:, :, i, j, 1] = base_coords[:, :, 1] - disp_arr[idx_x]

        if sub_pix_factor > 1:
            resized_scores = np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float64)
            resized_disps_y = np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float32)
            resized_disps_x = np.zeros((original_height, original_width, grid_rows, grid_cols), dtype=np.float32)
            for i in range(grid_rows):
                for j in range(grid_cols):
                    resized_scores[:, :, i, j] = cv.resize(
                        scores[:, :, i, j],
                        (original_width, original_height),
                        interpolation=cv.INTER_LINEAR,
                    )
                    resized_disps_y[:, :, i, j] = cv.resize(
                        disps_y[:, :, i, j],
                        (original_width, original_height),
                        interpolation=cv.INTER_LINEAR,
                    )
                    resized_disps_x[:, :, i, j] = cv.resize(
                        disps_x[:, :, i, j],
                        (original_width, original_height),
                        interpolation=cv.INTER_LINEAR,
                    )
            scores = resized_scores
            disps_y = resized_disps_y / sub_pix_factor
            disps_x = resized_disps_x / sub_pix_factor
            base_coords = np.stack(
                np.meshgrid(np.arange(original_height), np.arange(original_width), indexing="ij"),
                -1,
            ).astype(np.float32)
            coordinates = np.zeros((original_height, original_width, grid_rows, grid_cols, 2), dtype=np.float32)
            coordinates[:, :, :, :, 0] = base_coords[:, :, 0][..., None, None] - disps_y
            coordinates[:, :, :, :, 1] = base_coords[:, :, 1][..., None, None] - disps_x

        return PixelwiseSimilarityResult(
            coordinates=coordinates,
            scores=scores,
            disps_y=disps_y,
            disps_x=disps_x,
            sub_pix_factor=sub_pix_factor,
            reference_is_external=reference_is_external,
        )

    @staticmethod
    def _constrain_sigma_pyramid_update(
        baseline_result: PixelwiseSimilarityResult,
        current_result: PixelwiseSimilarityResult,
        candidate_result: PixelwiseSimilarityResult,
    ) -> Tuple[PixelwiseSimilarityResult, bool]:
        """Apply sigma pyramid constraints and report convergence status."""
        baseline_disps_y = np.asarray(baseline_result.disps_y)
        baseline_disps_x = np.asarray(baseline_result.disps_x)
        candidate_disps_y = np.asarray(candidate_result.disps_y)
        candidate_disps_x = np.asarray(candidate_result.disps_x)

        allowed_mask = (
            (np.abs(candidate_disps_y - baseline_disps_y) <= 2)
            & (np.abs(candidate_disps_x - baseline_disps_x) <= 2)
        )

        current_disps_y = np.asarray(current_result.disps_y)
        current_disps_x = np.asarray(current_result.disps_x)

        constrained_disps_y = np.where(allowed_mask, candidate_disps_y, current_disps_y)
        constrained_disps_x = np.where(allowed_mask, candidate_disps_x, current_disps_x)
        constrained_scores = np.where(
            allowed_mask,
            np.asarray(candidate_result.scores),
            np.asarray(current_result.scores),
        )

        height, width = constrained_disps_y.shape[:2]
        base_coords = np.stack(
            np.meshgrid(np.arange(height), np.arange(width), indexing="ij"),
            -1,
        ).astype(np.float32)
        coordinates = np.zeros(
            (height, width) + constrained_disps_y.shape[2:] + (2,),
            dtype=np.float32,
        )
        coordinates[:, :, :, :, 0] = base_coords[:, :, 0][..., None, None] - constrained_disps_y
        coordinates[:, :, :, :, 1] = base_coords[:, :, 1][..., None, None] - constrained_disps_x

        converged = np.all(np.abs(constrained_disps_y - current_disps_y) <= 1) and np.all(
            np.abs(constrained_disps_x - current_disps_x) <= 1
        )

        return (
            PixelwiseSimilarityResult(
                coordinates=coordinates,
                scores=constrained_scores,
                disps_y=constrained_disps_y,
                disps_x=constrained_disps_x,
                sub_pix_factor=candidate_result.sub_pix_factor,
                reference_is_external=candidate_result.reference_is_external,
            ),
            converged,
        )

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
        reference_is_external = getattr(sim_result, "reference_is_external", False)
        for i in range(grid_rows):
            for j in range(grid_cols):
                if not reference_is_external and i == ref_i and j == ref_j:
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

    @staticmethod
    def _roll_with_fill(
        array: np.ndarray, shift_y: int, shift_x: int, zero_fill: bool
    ) -> np.ndarray:
        """
        Shift ``array`` by ``(shift_y, shift_x)`` using ``np.roll`` with optional zero fill.

        When ``zero_fill`` is ``True``, regions that would wrap around the image edges
        are explicitly set to zero to avoid reflected artifacts from the roll
        operation.
        """

        shifted = np.roll(array, shift=(shift_y, shift_x), axis=(0, 1))
        if not zero_fill:
            return shifted

        rows_to_zero = min(abs(shift_y), shifted.shape[0])
        cols_to_zero = min(abs(shift_x), shifted.shape[1])

        if shift_y > 0:
            shifted[:rows_to_zero, :] = 0
        elif shift_y < 0:
            shifted[-rows_to_zero:, :] = 0

        if shift_x > 0:
            shifted[:, :cols_to_zero] = 0
        elif shift_x < 0:
            shifted[:, -cols_to_zero:] = 0

        return shifted
