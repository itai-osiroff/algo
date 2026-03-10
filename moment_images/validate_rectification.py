"""
Validation workflow for comparing rectification outputs to external camera images.

The module provides:
- A deterministic image-processing pipeline with explicit geometry/scale mode.
- Translation-ECC alignment that returns rich metadata.
- A single guided UI for geometry, intensity, scaling, alignment, and previews.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Literal

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image
from tkinter import Scale, filedialog, messagebox, ttk

MIN_TEMPLATE_MATCH_SCORE = 0.20
VALIDATION_MODE_PRESERVE = "preserve_scale"
VALIDATION_MODE_FIT = "fit_reference"
ALIGNMENT_MODEL_TRANSLATION_ECC = "translation_ecc"
PREVIEW_DEBOUNCE_MS = 140
SCALE_MIN = 0.001
SCALE_MAX = 8.0

ValidationMode = Literal["preserve_scale", "fit_reference"]
AlignmentModel = Literal["translation_ecc"]


# Ensure local modules can be imported
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import im_process
    from unpack_workspace import load_workspace_variables
else:
    from . import im_process
    from .unpack_workspace import load_workspace_variables


@dataclass(frozen=True)
class ValidationOptions:
    """Controls for how comparison and alignment are performed."""

    mode: ValidationMode = VALIDATION_MODE_PRESERVE
    auto_align: bool = False
    align_model: AlignmentModel = ALIGNMENT_MODEL_TRANSLATION_ECC
    joint_norm_percentiles: tuple[float, float] = (1.0, 99.0)
    template_score_threshold: float = MIN_TEMPLATE_MATCH_SCORE

    def normalized(self) -> "ValidationOptions":
        mode = self.mode
        if mode not in {VALIDATION_MODE_PRESERVE, VALIDATION_MODE_FIT}:
            mode = VALIDATION_MODE_PRESERVE
        align_model = self.align_model
        if align_model != ALIGNMENT_MODEL_TRANSLATION_ECC:
            align_model = ALIGNMENT_MODEL_TRANSLATION_ECC
        p_low, p_high = _sanitize_percentiles(self.joint_norm_percentiles)
        threshold = float(max(-1.0, min(1.0, self.template_score_threshold)))
        return ValidationOptions(
            mode=mode,  # type: ignore[arg-type]
            auto_align=bool(self.auto_align),
            align_model=align_model,  # type: ignore[arg-type]
            joint_norm_percentiles=(p_low, p_high),
            template_score_threshold=threshold,
        )


@dataclass
class AlignmentResult:
    """Alignment output including diagnostics and ROI mapping."""

    aligned_image: np.ndarray
    status: str
    mode: str
    match_found: bool
    match_score: float | None = None
    ecc_score: float | None = None
    warp_matrix: np.ndarray | None = None
    roi_reference: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    roi_input: tuple[int, int, int, int] | None = None  # (x, y, w, h)


@dataclass
class ComparisonData:
    """Prepared frames and metadata for visualization."""

    mode: str
    status: str
    match_found: bool
    score: float | None
    ecc_score: float | None
    as_is_reference: np.ndarray
    as_is_input: np.ndarray
    registered_reference: np.ndarray
    registered_input: np.ndarray
    diff_image: np.ndarray | None = None
    overlay_image: np.ndarray | None = None


@dataclass
class ValidationImageResult:
    """Final per-image result from the guided editor."""

    processed_image: np.ndarray
    options: ValidationOptions
    alignment_result: AlignmentResult | None
    scale_x: float
    scale_y: float
    intensity_min: float
    intensity_max: float
    crop_box: tuple[int, int, int, int] | None
    rotation_deg: int
    flip_h: bool
    flip_v: bool


def _average_rectified_from_stack(stack: np.ndarray) -> np.ndarray:
    """Convert saved rectified stack outputs into a 2D average rectified image."""
    if stack.ndim == 2:
        return stack
    if stack.ndim == 4:
        return stack.mean(axis=(0, 1))
    raise ValueError(
        "Expected rectified image with 2D shape or rectified views stack with 4D shape "
        f"(N, N, H, W), got {stack.shape}."
    )


def load_rectified_reference(path: str) -> np.ndarray:
    """Load a reference rectified image from workspace/sweep outputs (.npz/.npy)."""
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".npy":
        return _average_rectified_from_stack(np.load(path))

    if suffix == ".npz":
        with np.load(path) as data:
            if "average_rectified_image" in data:
                return data["average_rectified_image"]
            if "rectified_image" in data:
                return data["rectified_image"]
            if "rectified_views" in data:
                return _average_rectified_from_stack(data["rectified_views"])

        workspace = load_workspace_variables(path)
        if "average_rectified_image" in workspace:
            return workspace["average_rectified_image"]
        if "rectified_image" in workspace:
            return workspace["rectified_image"]
        if "rectified_views" in workspace:
            return _average_rectified_from_stack(workspace["rectified_views"])

        raise ValueError(
            "NPZ file does not contain 'average_rectified_image', 'rectified_image', or 'rectified_views'."
        )

    raise ValueError(
        "Unsupported file type. Please provide a .npz workspace/output or .npy rectified views file."
    )


def _sanitize_percentiles(percentiles: tuple[float, float]) -> tuple[float, float]:
    low = float(percentiles[0])
    high = float(percentiles[1])
    low = max(0.0, min(100.0, low))
    high = max(0.0, min(100.0, high))
    if high <= low:
        high = min(100.0, low + 1.0)
    return low, high


def _resolve_options(
    options: ValidationOptions | None = None,
    resize_to_reference: bool | None = None,
    prefer_smaller_pixels_contained: bool = False,  # backward-compat placeholder
) -> ValidationOptions:
    if options is not None:
        return options.normalized()

    if resize_to_reference is None:
        resize_to_reference = True

    mode: ValidationMode = VALIDATION_MODE_FIT if resize_to_reference else VALIDATION_MODE_PRESERVE
    threshold = MIN_TEMPLATE_MATCH_SCORE
    if prefer_smaller_pixels_contained:
        threshold = MIN_TEMPLATE_MATCH_SCORE
    return ValidationOptions(mode=mode, auto_align=False, template_score_threshold=threshold).normalized()


def _to_gray_float32(image: Image.Image | np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(np.float32)
            arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            arr = arr[..., 0]
    return np.asarray(arr, dtype=np.float32)


def _apply_camera_defaults(image: np.ndarray, camera_type: str | None) -> np.ndarray:
    # Keep preprocessing identical across camera sources; geometry is edited manually in the GUI.
    _ = camera_type
    return image


def _apply_geometry(
    image: np.ndarray,
    rotation_deg: int = 0,
    flip_h: bool = False,
    flip_v: bool = False,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32)

    rot = int(round(rotation_deg / 90.0)) % 4
    if rot:
        out = np.rot90(out, k=rot)
    if flip_h:
        out = np.fliplr(out)
    if flip_v:
        out = np.flipud(out)

    if crop_box is not None:
        out = _crop_image(out, crop_box)
    return out


def _crop_image(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop_box
    x0 = max(0, min(int(round(x0)), w))
    x1 = max(0, min(int(round(x1)), w))
    y0 = max(0, min(int(round(y0)), h))
    y1 = max(0, min(int(round(y1)), h))
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]


def _apply_intensity_window(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    lo = float(min_val)
    hi = float(max_val)
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo) * 255.0


def _scale_image(src_img: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    src_h, src_w = src_img.shape[:2]
    scaled_w = max(1, int(round(src_w * float(scale_x))))
    scaled_h = max(1, int(round(src_h * float(scale_y))))
    return cv.resize(src_img, (scaled_w, scaled_h), interpolation=cv.INTER_LINEAR)


def _joint_norm_bounds(
    img_a: np.ndarray, img_b: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    p_low, p_high = _sanitize_percentiles(percentiles)
    values = np.concatenate(
        [np.asarray(img_a, dtype=np.float32).ravel(), np.asarray(img_b, dtype=np.float32).ravel()]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    low = float(np.percentile(finite, p_low))
    high = float(np.percentile(finite, p_high))
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return low, high


def _normalize_with_bounds(image: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if high <= low:
        high = low + 1.0
    clipped = np.clip(arr, low, high)
    return (clipped - low) / (high - low)


def _joint_normalize_pair(
    img_a: np.ndarray, img_b: np.ndarray, percentiles: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    low, high = _joint_norm_bounds(img_a, img_b, percentiles)
    return _normalize_with_bounds(img_a, low, high), _normalize_with_bounds(img_b, low, high)


def _finite_display_bounds(image: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return low, high


def _imshow_with_colorbar(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    label: str | None = None,
):
    if vmin is None or vmax is None:
        auto_low, auto_high = _finite_display_bounds(image)
        if vmin is None:
            vmin = auto_low
        if vmax is None:
            vmax = auto_high
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label)
    return im


def _add_overlay_channel_colorbars(ax: plt.Axes, overlay_image: np.ndarray) -> None:
    ref_low, ref_high = _finite_display_bounds(overlay_image[..., 0])
    input_low, input_high = _finite_display_bounds(overlay_image[..., 1])

    ref_mappable = ScalarMappable(norm=Normalize(vmin=ref_low, vmax=ref_high), cmap="Reds")
    ref_mappable.set_array(overlay_image[..., 0])
    ref_cbar = ax.figure.colorbar(ref_mappable, ax=ax, fraction=0.046, pad=0.04)
    ref_cbar.set_label("Reference (Red)")

    input_mappable = ScalarMappable(norm=Normalize(vmin=input_low, vmax=input_high), cmap="Greens")
    input_mappable.set_array(overlay_image[..., 1])
    input_cbar = ax.figure.colorbar(input_mappable, ax=ax, fraction=0.046, pad=0.12)
    input_cbar.set_label("Input (Green)")


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high <= low:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = np.clip((arr - low) / (high - low), 0.0, 1.0) * 255.0
    return normalized.astype(np.uint8)


def _find_template_match(search_img: np.ndarray, template_img: np.ndarray) -> tuple[int, int, float]:
    """Find template location in search image using normalized cross-correlation."""
    search_u8 = _normalize_to_uint8(search_img)
    template_u8 = _normalize_to_uint8(template_img)
    match = cv.matchTemplate(search_u8, template_u8, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(match)
    score = float(max_val)
    if not np.isfinite(score):
        score = -1.0
    return int(max_loc[0]), int(max_loc[1]), score


def _find_containment_match(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[str, int, int, float] | None:
    """
    Decide containment direction and placement using template matching.

    Modes:
    - target_in_reference: target is contained in reference.
    - reference_in_target: reference is contained in target.
    """
    ref_h, ref_w = ref_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]

    candidates: list[tuple[str, int, int, float]] = []
    if tgt_h <= ref_h and tgt_w <= ref_w:
        x, y, score = _find_template_match(ref_img, target_img)
        candidates.append(("target_in_reference", x, y, score))

    if ref_h <= tgt_h and ref_w <= tgt_w:
        x, y, score = _find_template_match(target_img, ref_img)
        candidates.append(("reference_in_target", x, y, score))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[3])


def _comparison_mode_suffix(mode: str, score: float | None, ecc_score: float | None) -> str:
    score_text = f", template={score:.3f}" if score is not None else ""
    ecc_text = f", ecc={ecc_score:.3f}" if ecc_score is not None else ""
    if mode == "target_in_reference":
        return f" [input in reference{score_text}{ecc_text}]"
    if mode == "reference_in_target":
        return f" [reference in input{score_text}{ecc_text}]"
    if mode == VALIDATION_MODE_FIT:
        return f" [fit-to-reference{ecc_text}]"
    if mode == VALIDATION_MODE_PRESERVE:
        return f" [preserve-scale{ecc_text}]"
    return f" [{mode}{score_text}{ecc_text}]"


def _ecc_align_same_size(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Run ECC alignment for equal-shape images and return aligned image + score + warp."""
    ref_gray = np.asarray(ref_img, dtype=np.float32)
    target_gray = np.asarray(target_img, dtype=np.float32)
    if ref_gray.shape != target_gray.shape:
        raise ValueError("ECC requires equal-shape images.")

    warp_mode = cv.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 500, 1e-5)
    cc, warp_matrix = cv.findTransformECC(ref_gray, target_gray, warp_matrix, warp_mode, criteria)

    h, w = ref_gray.shape[:2]
    aligned_img = cv.warpAffine(
        target_gray,
        warp_matrix,
        (w, h),
        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
    )
    return aligned_img.astype(np.float32), float(cc), warp_matrix


def auto_align_images(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    options: ValidationOptions | None = None,
    *,
    resize_to_reference: bool | None = None,
    prefer_smaller_pixels_contained: bool = False,  # backward compatibility
) -> AlignmentResult:
    """
    Align target_img to ref_img using the requested strategy.

    The default strategy is translation ECC with either preserve-scale or
    fit-to-reference geometry mode.
    """
    opts = _resolve_options(
        options=options,
        resize_to_reference=resize_to_reference,
        prefer_smaller_pixels_contained=prefer_smaller_pixels_contained,
    )
    ref = _to_gray_float32(ref_img)
    target = _to_gray_float32(target_img)

    if not opts.auto_align:
        return AlignmentResult(
            aligned_image=target.copy(),
            status="auto_align_disabled",
            mode=opts.mode,
            match_found=ref.shape == target.shape,
            roi_reference=(0, 0, ref.shape[1], ref.shape[0]) if ref.shape == target.shape else None,
            roi_input=(0, 0, target.shape[1], target.shape[0]) if ref.shape == target.shape else None,
        )

    if opts.align_model != ALIGNMENT_MODEL_TRANSLATION_ECC:
        raise ValueError(f"Unsupported align model: {opts.align_model}")

    try:
        ref_h, ref_w = ref.shape[:2]
        tgt_h, tgt_w = target.shape[:2]

        if opts.mode == VALIDATION_MODE_FIT:
            target_resized = target if (tgt_h, tgt_w) == (ref_h, ref_w) else cv.resize(
                target, (ref_w, ref_h), interpolation=cv.INTER_LINEAR
            )
            aligned_img, cc, warp = _ecc_align_same_size(ref, target_resized)
            return AlignmentResult(
                aligned_image=aligned_img,
                status="aligned_fit_reference",
                mode=VALIDATION_MODE_FIT,
                match_found=True,
                ecc_score=cc,
                warp_matrix=warp,
                roi_reference=(0, 0, ref_w, ref_h),
                roi_input=(0, 0, ref_w, ref_h),
            )

        if (tgt_h, tgt_w) == (ref_h, ref_w):
            aligned_img, cc, warp = _ecc_align_same_size(ref, target)
            return AlignmentResult(
                aligned_image=aligned_img,
                status="aligned_preserve_same_shape",
                mode=VALIDATION_MODE_PRESERVE,
                match_found=True,
                ecc_score=cc,
                warp_matrix=warp,
                roi_reference=(0, 0, ref_w, ref_h),
                roi_input=(0, 0, tgt_w, tgt_h),
            )

        match_result = _find_containment_match(ref, target)
        if match_result is None:
            return AlignmentResult(
                aligned_image=target.copy(),
                status="no_containment_match",
                mode=VALIDATION_MODE_PRESERVE,
                match_found=False,
            )

        mode, x, y, score = match_result
        if score < opts.template_score_threshold:
            return AlignmentResult(
                aligned_image=target.copy(),
                status="low_confidence_match",
                mode=mode,
                match_found=False,
                match_score=score,
            )

        if mode == "target_in_reference":
            ref_roi = ref[y:y + tgt_h, x:x + tgt_w]
            aligned_img, cc, warp = _ecc_align_same_size(ref_roi, target)
            return AlignmentResult(
                aligned_image=aligned_img,
                status="aligned_target_in_reference",
                mode=mode,
                match_found=True,
                match_score=score,
                ecc_score=cc,
                warp_matrix=warp,
                roi_reference=(x, y, tgt_w, tgt_h),
                roi_input=(0, 0, tgt_w, tgt_h),
            )

        target_roi = target[y:y + ref_h, x:x + ref_w]
        aligned_roi, cc, warp = _ecc_align_same_size(ref, target_roi)
        aligned_full = target.copy()
        aligned_full[y:y + ref_h, x:x + ref_w] = aligned_roi
        return AlignmentResult(
            aligned_image=aligned_full,
            status="aligned_reference_in_target",
            mode=mode,
            match_found=True,
            match_score=score,
            ecc_score=cc,
            warp_matrix=warp,
            roi_reference=(0, 0, ref_w, ref_h),
            roi_input=(x, y, ref_w, ref_h),
        )

    except Exception as exc:  # includes cv.error from ECC on degenerate data
        return AlignmentResult(
            aligned_image=target.copy(),
            status=f"ecc_failed:{type(exc).__name__}",
            mode=opts.mode,
            match_found=False,
        )


def _crop_by_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    h_img, w_img = image.shape[:2]
    x0 = max(0, min(int(x), w_img))
    y0 = max(0, min(int(y), h_img))
    x1 = max(0, min(x0 + int(w), w_img))
    y1 = max(0, min(y0 + int(h), h_img))
    if x1 <= x0 or y1 <= y0:
        return image[:0, :0]
    return image[y0:y1, x0:x1]


def prepare_images_for_comparison(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    options: ValidationOptions | None = None,
    alignment_result: AlignmentResult | None = None,
    *,
    resize_to_reference: bool | None = None,
    prefer_smaller_pixels_contained: bool = False,  # backward compatibility
) -> ComparisonData:
    """Prepare as-is/registered/diff/overlay frames for comparison."""
    opts = _resolve_options(
        options=options,
        resize_to_reference=resize_to_reference,
        prefer_smaller_pixels_contained=prefer_smaller_pixels_contained,
    )
    ref = _to_gray_float32(ref_img)
    target = _to_gray_float32(target_img)

    status = "prepared_no_alignment"
    mode = opts.mode
    score: float | None = None
    ecc_score: float | None = None
    match_found = False

    if opts.mode == VALIDATION_MODE_FIT:
        if alignment_result is not None:
            reg_input = _to_gray_float32(alignment_result.aligned_image)
            status = alignment_result.status
            mode = alignment_result.mode
            score = alignment_result.match_score
            ecc_score = alignment_result.ecc_score
            match_found = alignment_result.match_found
        else:
            reg_input = cv.resize(target, (ref.shape[1], ref.shape[0]), interpolation=cv.INTER_LINEAR)
            status = "fit_reference_resized_without_alignment"
            match_found = True
        reg_ref = ref
    else:
        aligned_target = _to_gray_float32(alignment_result.aligned_image) if alignment_result is not None else target
        if alignment_result is not None:
            status = alignment_result.status
            mode = alignment_result.mode
            score = alignment_result.match_score
            ecc_score = alignment_result.ecc_score

        if alignment_result is not None and alignment_result.roi_reference and alignment_result.roi_input:
            reg_ref = _crop_by_roi(ref, alignment_result.roi_reference)
            reg_input = _crop_by_roi(aligned_target, alignment_result.roi_input)
            match_found = alignment_result.match_found
        elif aligned_target.shape == ref.shape:
            reg_ref = ref
            reg_input = aligned_target
            match_found = True
            if alignment_result is None:
                status = "preserve_same_shape_without_alignment"
                mode = VALIDATION_MODE_PRESERVE
        else:
            match_result = _find_containment_match(ref, aligned_target)
            if match_result is None:
                reg_ref = ref
                reg_input = aligned_target
                status = "no_containment_match"
                match_found = False
            else:
                match_mode, x, y, match_score = match_result
                score = match_score
                mode = match_mode
                if match_score < opts.template_score_threshold:
                    reg_ref = ref
                    reg_input = aligned_target
                    status = "low_confidence_match"
                    match_found = False
                elif match_mode == "target_in_reference":
                    h_t, w_t = aligned_target.shape[:2]
                    reg_ref = ref[y:y + h_t, x:x + w_t]
                    reg_input = aligned_target
                    status = "containment_match_target_in_reference"
                    match_found = True
                else:
                    h_r, w_r = ref.shape[:2]
                    reg_ref = ref
                    reg_input = aligned_target[y:y + h_r, x:x + w_r]
                    status = "containment_match_reference_in_target"
                    match_found = True

    if reg_ref.shape != reg_input.shape and match_found:
        min_h = min(reg_ref.shape[0], reg_input.shape[0])
        min_w = min(reg_ref.shape[1], reg_input.shape[1])
        reg_ref = reg_ref[:min_h, :min_w]
        reg_input = reg_input[:min_h, :min_w]

    diff_image: np.ndarray | None = None
    overlay_image: np.ndarray | None = None
    if match_found and reg_ref.size > 0 and reg_input.size > 0 and reg_ref.shape == reg_input.shape:
        reg_ref_norm, reg_input_norm = _joint_normalize_pair(reg_ref, reg_input, opts.joint_norm_percentiles)
        diff_image = np.abs(reg_ref_norm - reg_input_norm)
        overlay_image = np.zeros((reg_ref_norm.shape[0], reg_ref_norm.shape[1], 3), dtype=np.float32)
        overlay_image[..., 0] = reg_ref_norm
        overlay_image[..., 1] = reg_input_norm

    return ComparisonData(
        mode=mode,
        status=status,
        match_found=bool(match_found and reg_ref.shape == reg_input.shape and reg_ref.size > 0),
        score=score,
        ecc_score=ecc_score,
        as_is_reference=ref,
        as_is_input=target,
        registered_reference=reg_ref,
        registered_input=reg_input,
        diff_image=diff_image,
        overlay_image=overlay_image,
    )


def compare_images(
    rectified_img: np.ndarray,
    input_img: np.ndarray,
    input_name: str,
    options: ValidationOptions | None = None,
    alignment_result: AlignmentResult | None = None,
    *,
    resize_to_reference: bool | None = None,
    prefer_smaller_pixels_contained: bool = False,  # backward compatibility
) -> list[Figure]:
    """
    Compare input image to rectified image and return figure objects.

    Produces four figures:
    1) As-is pair
    2) Registered pair
    3) Difference map (or status panel)
    4) Overlay (or status panel)
    """
    opts = _resolve_options(
        options=options,
        resize_to_reference=resize_to_reference,
        prefer_smaller_pixels_contained=prefer_smaller_pixels_contained,
    )
    comp_data = prepare_images_for_comparison(
        rectified_img,
        input_img,
        options=opts,
        alignment_result=alignment_result,
    )
    mode_suffix = _comparison_mode_suffix(comp_data.mode, comp_data.score, comp_data.ecc_score)
    figures: list[Figure] = []

    # As-is
    fig_as_is, axes_as_is = plt.subplots(1, 2, figsize=(12, 6))
    # Keep each as-is panel independent so the loaded reference appearance is not
    # affected by edited-input scaling/intensity.
    ref_as_is_low, ref_as_is_high = _finite_display_bounds(comp_data.as_is_reference)
    input_as_is_low, input_as_is_high = _finite_display_bounds(comp_data.as_is_input)
    _imshow_with_colorbar(
        axes_as_is[0],
        comp_data.as_is_reference,
        cmap="gray",
        vmin=ref_as_is_low,
        vmax=ref_as_is_high,
    )
    axes_as_is[0].set_title(
        f"Reference As-Is [{comp_data.as_is_reference.shape[0]}x{comp_data.as_is_reference.shape[1]}]"
    )
    _imshow_with_colorbar(
        axes_as_is[1],
        comp_data.as_is_input,
        cmap="gray",
        vmin=input_as_is_low,
        vmax=input_as_is_high,
    )
    axes_as_is[1].set_title(
        f"{input_name} As-Edited [{comp_data.as_is_input.shape[0]}x{comp_data.as_is_input.shape[1]}]"
    )
    figures.append(fig_as_is)

    # Registered
    fig_reg, axes_reg = plt.subplots(1, 2, figsize=(12, 6))
    if comp_data.registered_reference.size > 0 and comp_data.registered_input.size > 0:
        reg_low, reg_high = _joint_norm_bounds(
            comp_data.registered_reference,
            comp_data.registered_input,
            opts.joint_norm_percentiles,
        )
        _imshow_with_colorbar(
            axes_reg[0],
            comp_data.registered_reference,
            cmap="gray",
            vmin=reg_low,
            vmax=reg_high,
        )
        _imshow_with_colorbar(
            axes_reg[1],
            comp_data.registered_input,
            cmap="gray",
            vmin=reg_low,
            vmax=reg_high,
        )
        axes_reg[0].set_title(f"Reference Registered{mode_suffix}")
        axes_reg[1].set_title(f"{input_name} Registered{mode_suffix}")
    else:
        for ax in axes_reg:
            ax.axis("off")
            ax.text(0.5, 0.5, "No registered ROI", ha="center", va="center")
    figures.append(fig_reg)

    # Difference
    fig_diff, ax_diff = plt.subplots(1, 1, figsize=(10, 8))
    if comp_data.diff_image is not None:
        _imshow_with_colorbar(ax_diff, comp_data.diff_image, cmap="viridis", label="|Reference - Input|")
        ax_diff.set_title(f"Difference |Reference - {input_name}|{mode_suffix}")
    else:
        ax_diff.axis("off")
        ax_diff.text(
            0.5,
            0.5,
            f"Difference unavailable\nstatus: {comp_data.status}",
            ha="center",
            va="center",
        )
        ax_diff.set_title(f"Difference (Unavailable){mode_suffix}")
    figures.append(fig_diff)

    # Overlay
    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(10, 8))
    if comp_data.overlay_image is not None:
        ax_overlay.imshow(comp_data.overlay_image)
        _add_overlay_channel_colorbars(ax_overlay, comp_data.overlay_image)
        ax_overlay.set_title(f"Overlay Red=Reference, Green={input_name}{mode_suffix}")
    else:
        ax_overlay.axis("off")
        ax_overlay.text(
            0.5,
            0.5,
            f"Overlay unavailable\nstatus: {comp_data.status}",
            ha="center",
            va="center",
        )
        ax_overlay.set_title(f"Overlay (Unavailable){mode_suffix}")
    figures.append(fig_overlay)

    return figures


class GuidedValidationEditor:
    """Single-window guided editor for geometry/intensity/scale/alignment/outputs."""

    def __init__(
        self,
        master: tk.Tk,
        image_path: str,
        reference_image: np.ndarray,
        title: str,
        camera_type: str | None = None,
    ):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title(title)
        self.top.geometry("1500x980")

        loaded = im_process.load_image(image_path)
        if loaded is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.original_image = _to_gray_float32(loaded)
        self.base_image = _apply_camera_defaults(self.original_image, camera_type)
        # Keep an isolated reference copy for preview rendering and alignment.
        self.reference_image = _to_gray_float32(reference_image).copy()
        self.reference_display_low, self.reference_display_high = _finite_display_bounds(self.reference_image)

        self.rotation_deg = 0
        self.flip_h = False
        self.flip_v = False
        self.crop_box: tuple[int, int, int, int] | None = None
        self.crop_active = False

        self.scale_x_value = 1.0
        self.scale_y_value = 1.0
        self.alignment_result: AlignmentResult | None = None
        self._alignment_dirty = True
        self._preview_job: str | None = None
        self._crop_selector: RectangleSelector | None = None
        self._result: ValidationImageResult | None = None

        raw_min = float(np.min(self.base_image))
        raw_max = float(np.max(self.base_image))
        if raw_max <= raw_min:
            raw_max = raw_min + 1.0
        self.raw_min_val = raw_min
        self.raw_max_val = raw_max
        self.intensity_min_value = raw_min
        self.intensity_max_value = raw_max

        self.mode_var = tk.StringVar(value=VALIDATION_MODE_PRESERVE)
        self.auto_align_var = tk.BooleanVar(value=False)
        self.scale_x_var = tk.StringVar(value=f"{self.scale_x_value:.4f}")
        self.scale_y_var = tk.StringVar(value=f"{self.scale_y_value:.4f}")
        self.scale_status_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.hover_status_var = tk.StringVar(value="Hover: move cursor over an image to inspect pixel values")
        self.reset_section_var = tk.StringVar(value="geometry")

        self.fig_as_is: Figure | None = None
        self.fig_registered: Figure | None = None
        self.fig_diff: Figure | None = None
        self.fig_overlay: Figure | None = None
        self.canvas_as_is: FigureCanvasTkAgg | None = None
        self.canvas_registered: FigureCanvasTkAgg | None = None
        self.canvas_diff: FigureCanvasTkAgg | None = None
        self.canvas_overlay: FigureCanvasTkAgg | None = None

        self._init_ui()
        self._schedule_preview()

    def _init_ui(self) -> None:
        controls = tk.Frame(self.top)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        # Split controls across multiple rows so action buttons stay visible on all displays.
        row_primary = tk.Frame(controls)
        row_primary.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        row_intensity = tk.Frame(controls)
        row_intensity.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        row_actions = tk.Frame(controls)
        row_actions.pack(side=tk.TOP, fill=tk.X)

        geo_frame = tk.LabelFrame(row_primary, text="Geometry")
        geo_frame.pack(side=tk.LEFT, padx=4, pady=2)
        tk.Button(geo_frame, text="Rot L", command=lambda: self._rotate(90)).pack(side=tk.LEFT, padx=2)
        tk.Button(geo_frame, text="Rot R", command=lambda: self._rotate(-90)).pack(side=tk.LEFT, padx=2)
        tk.Button(geo_frame, text="Flip H", command=self._flip_h).pack(side=tk.LEFT, padx=2)
        tk.Button(geo_frame, text="Flip V", command=self._flip_v).pack(side=tk.LEFT, padx=2)
        self.btn_crop = tk.Button(geo_frame, text="Crop OFF", command=self._toggle_crop)
        self.btn_crop.pack(side=tk.LEFT, padx=2)

        scale_frame = tk.LabelFrame(row_primary, text="Scale")
        scale_frame.pack(side=tk.LEFT, padx=4, pady=2)
        tk.Label(scale_frame, text="X").pack(side=tk.LEFT, padx=(4, 1))
        self.scale_x_entry = tk.Entry(scale_frame, textvariable=self.scale_x_var, width=8)
        self.scale_x_entry.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(scale_frame, text="Y").pack(side=tk.LEFT, padx=(0, 1))
        self.scale_y_entry = tk.Entry(scale_frame, textvariable=self.scale_y_var, width=8)
        self.scale_y_entry.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(scale_frame, text=f"[{SCALE_MIN:.3f}, {SCALE_MAX:.3f}]").pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(scale_frame, textvariable=self.scale_status_var, fg="red").pack(side=tk.LEFT)
        self.scale_x_entry.bind("<KeyRelease>", self._on_scale_entry_change)
        self.scale_x_entry.bind("<FocusOut>", self._on_scale_entry_commit)
        self.scale_x_entry.bind("<Return>", self._on_scale_entry_commit)
        self.scale_y_entry.bind("<KeyRelease>", self._on_scale_entry_change)
        self.scale_y_entry.bind("<FocusOut>", self._on_scale_entry_commit)
        self.scale_y_entry.bind("<Return>", self._on_scale_entry_commit)

        align_frame = tk.LabelFrame(row_primary, text="Alignment")
        align_frame.pack(side=tk.LEFT, padx=4, pady=2)
        tk.Label(align_frame, text="Mode").pack(side=tk.LEFT, padx=(4, 2))
        mode_menu = ttk.Combobox(
            align_frame,
            textvariable=self.mode_var,
            values=[VALIDATION_MODE_PRESERVE, VALIDATION_MODE_FIT],
            width=16,
            state="readonly",
        )
        mode_menu.pack(side=tk.LEFT, padx=2)
        mode_menu.bind("<<ComboboxSelected>>", self._on_mode_change)
        tk.Checkbutton(
            align_frame,
            text="Auto Align",
            variable=self.auto_align_var,
            command=self._on_auto_align_toggle,
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(align_frame, text="Run Alignment", command=self._run_alignment).pack(side=tk.LEFT, padx=4)

        intensity_frame = tk.LabelFrame(row_intensity, text="Intensity")
        intensity_frame.pack(side=tk.LEFT, padx=4, pady=2)
        tk.Label(intensity_frame, text="Min").pack(side=tk.LEFT)
        resolution = max((self.raw_max_val - self.raw_min_val) / 250.0, 1e-3)
        self.scale_intensity_min = Scale(
            intensity_frame,
            from_=self.raw_min_val,
            to=self.raw_max_val,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            command=self._on_intensity_change,
            length=180,
        )
        self.scale_intensity_min.set(self.raw_min_val)
        self.scale_intensity_min.pack(side=tk.LEFT, padx=2)
        tk.Label(intensity_frame, text="Max").pack(side=tk.LEFT)
        self.scale_intensity_max = Scale(
            intensity_frame,
            from_=self.raw_min_val,
            to=self.raw_max_val,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            command=self._on_intensity_change,
            length=180,
        )
        self.scale_intensity_max.set(self.raw_max_val)
        self.scale_intensity_max.pack(side=tk.LEFT, padx=2)

        outputs_frame = tk.LabelFrame(row_actions, text="Outputs")
        outputs_frame.pack(side=tk.LEFT, padx=4, pady=2)
        ttk.Combobox(
            outputs_frame,
            textvariable=self.reset_section_var,
            values=["geometry", "intensity", "scale", "alignment"],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(outputs_frame, text="Reset Section", command=self._reset_section).pack(side=tk.LEFT, padx=2)
        tk.Button(outputs_frame, text="Reset All", command=self._reset_all).pack(side=tk.LEFT, padx=2)

        done_frame = tk.Frame(row_actions)
        done_frame.pack(side=tk.RIGHT, padx=6)
        tk.Button(done_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=4)
        tk.Button(done_frame, text="Done", bg="green", fg="white", command=self._finish).pack(side=tk.RIGHT, padx=4)

        status_label = tk.Label(self.top, textvariable=self.status_var, anchor="w", fg="navy")
        status_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        hover_label = tk.Label(self.top, textvariable=self.hover_status_var, anchor="w", fg="darkgreen")
        hover_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        notebook = ttk.Notebook(self.top)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        tab_as_is = ttk.Frame(notebook)
        tab_registered = ttk.Frame(notebook)
        tab_diff = ttk.Frame(notebook)
        tab_overlay = ttk.Frame(notebook)
        notebook.add(tab_as_is, text="As-Is")
        notebook.add(tab_registered, text="Registered ROI")
        notebook.add(tab_diff, text="Difference")
        notebook.add(tab_overlay, text="Overlay")

        self.fig_as_is = Figure(figsize=(10, 7), dpi=100)
        self.canvas_as_is = FigureCanvasTkAgg(self.fig_as_is, master=tab_as_is)
        self.canvas_as_is.draw()
        self.canvas_as_is.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_registered = Figure(figsize=(10, 7), dpi=100)
        self.canvas_registered = FigureCanvasTkAgg(self.fig_registered, master=tab_registered)
        self.canvas_registered.draw()
        self.canvas_registered.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_diff = Figure(figsize=(10, 7), dpi=100)
        self.canvas_diff = FigureCanvasTkAgg(self.fig_diff, master=tab_diff)
        self.canvas_diff.draw()
        self.canvas_diff.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_overlay = Figure(figsize=(10, 7), dpi=100)
        self.canvas_overlay = FigureCanvasTkAgg(self.fig_overlay, master=tab_overlay)
        self.canvas_overlay.draw()
        self.canvas_overlay.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._bind_hover_readout()

        self.top.protocol("WM_DELETE_WINDOW", self._cancel)

    def _bind_hover_readout(self) -> None:
        for canvas in (self.canvas_as_is, self.canvas_registered, self.canvas_diff, self.canvas_overlay):
            if canvas is None:
                continue
            canvas.mpl_connect("motion_notify_event", self._on_canvas_hover)
            canvas.mpl_connect("figure_leave_event", self._on_canvas_leave)

    def _on_canvas_leave(self, _event: object) -> None:
        self.hover_status_var.set("Hover: move cursor over an image to inspect pixel values")

    def _on_canvas_hover(self, event: object) -> None:
        ax = getattr(event, "inaxes", None)
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if ax is None or xdata is None or ydata is None:
            self.hover_status_var.set("Hover: move cursor over an image to inspect pixel values")
            return
        images = getattr(ax, "images", [])
        if not images:
            self.hover_status_var.set("Hover: move cursor over an image to inspect pixel values")
            return

        arr = np.asarray(images[-1].get_array())
        if arr.ndim < 2:
            self.hover_status_var.set("Hover: move cursor over an image to inspect pixel values")
            return

        ix = int(np.rint(float(xdata)))
        iy = int(np.rint(float(ydata)))
        h, w = arr.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            self.hover_status_var.set("Hover: move cursor over an image to inspect pixel values")
            return

        pixel = arr[iy, ix]
        title = ax.get_title() or "Image"
        if np.ndim(pixel) == 0:
            if np.isfinite(float(pixel)):
                pixel_text = f"{float(pixel):.6g}"
            else:
                pixel_text = "nan"
            self.hover_status_var.set(f"{title} | x={ix}, y={iy}, value={pixel_text}")
            return

        channel_values = np.ravel(np.asarray(pixel, dtype=np.float32))
        if channel_values.size >= 3:
            labels = ("R", "G", "B")
            parts: list[str] = []
            for idx, channel in enumerate(channel_values[:3]):
                label = labels[idx]
                parts.append(f"{label}={float(channel):.6g}" if np.isfinite(channel) else f"{label}=nan")
            self.hover_status_var.set(f"{title} | x={ix}, y={iy}, " + ", ".join(parts))
            return

        values_text = ", ".join(
            f"{float(v):.6g}" if np.isfinite(v) else "nan" for v in channel_values
        )
        self.hover_status_var.set(f"{title} | x={ix}, y={iy}, value=[{values_text}]")

    def _parse_scale_value(self, raw_value: str, axis_label: str) -> tuple[float | None, str | None]:
        value_text = raw_value.strip()
        if value_text == "":
            return None, f"{axis_label} scale is empty"
        try:
            parsed = float(value_text)
        except ValueError:
            return None, f"{axis_label} scale must be numeric"
        if parsed < SCALE_MIN or parsed > SCALE_MAX:
            return None, f"{axis_label} scale out of range ({SCALE_MIN:.3f}-{SCALE_MAX:.3f})"
        return parsed, None

    def _apply_scale_entries(self, commit: bool = False) -> bool:
        x_val, x_err = self._parse_scale_value(self.scale_x_var.get(), "X")
        y_val, y_err = self._parse_scale_value(self.scale_y_var.get(), "Y")
        if x_err or y_err:
            self.scale_status_var.set(x_err if x_err else y_err or "")
            if commit:
                self.scale_x_var.set(f"{self.scale_x_value:.4f}")
                self.scale_y_var.set(f"{self.scale_y_value:.4f}")
                self.scale_status_var.set("")
            return False
        new_x = float(x_val)
        new_y = float(y_val)
        changed = not np.isclose([self.scale_x_value, self.scale_y_value], [new_x, new_y]).all()
        self.scale_x_value = new_x
        self.scale_y_value = new_y
        self.scale_status_var.set("")
        if changed:
            self._invalidate_alignment()
        return True

    def _on_scale_entry_change(self, _event: tk.Event) -> None:
        if self._apply_scale_entries(commit=False):
            self._schedule_preview()

    def _on_scale_entry_commit(self, _event: tk.Event) -> None:
        if self._apply_scale_entries(commit=True):
            self._schedule_preview()

    def _rotate(self, delta_deg: int) -> None:
        self.rotation_deg = (self.rotation_deg + delta_deg) % 360
        self._invalidate_alignment()
        self._schedule_preview()

    def _flip_h(self) -> None:
        self.flip_h = not self.flip_h
        self._invalidate_alignment()
        self._schedule_preview()

    def _flip_v(self) -> None:
        self.flip_v = not self.flip_v
        self._invalidate_alignment()
        self._schedule_preview()

    def _toggle_crop(self) -> None:
        self.crop_active = not self.crop_active
        self.btn_crop.config(text="Crop ON" if self.crop_active else "Crop OFF")
        if self._crop_selector is not None:
            self._crop_selector.set_active(self.crop_active)

    def _on_crop_select(self, eclick: object, erelease: object) -> None:
        if self._crop_selector is None or not self.crop_active:
            return
        xmin, xmax, ymin, ymax = self._crop_selector.extents
        geom_image = _apply_geometry(
            self.base_image,
            rotation_deg=self.rotation_deg,
            flip_h=self.flip_h,
            flip_v=self.flip_v,
            crop_box=None,
        )
        geom_h, geom_w = geom_image.shape[:2]
        sx = max(self.scale_x_value, 1e-6)
        sy = max(self.scale_y_value, 1e-6)
        x0 = int(round(min(xmin, xmax) / sx))
        x1 = int(round(max(xmin, xmax) / sx))
        y0 = int(round(min(ymin, ymax) / sy))
        y1 = int(round(max(ymin, ymax) / sy))
        x0 = max(0, min(x0, geom_w))
        x1 = max(0, min(x1, geom_w))
        y0 = max(0, min(y0, geom_h))
        y1 = max(0, min(y1, geom_h))
        if x1 > x0 and y1 > y0:
            self.crop_box = (x0, y0, x1, y1)
            self._invalidate_alignment()
            self._schedule_preview()

    def _on_intensity_change(self, _value: str) -> None:
        min_val = float(self.scale_intensity_min.get())
        max_val = float(self.scale_intensity_max.get())
        if min_val >= max_val:
            max_val = min(self.raw_max_val, min_val + max((self.raw_max_val - self.raw_min_val) / 250.0, 1e-3))
            self.scale_intensity_max.set(max_val)
        changed = not np.isclose([self.intensity_min_value, self.intensity_max_value], [min_val, max_val]).all()
        self.intensity_min_value = min_val
        self.intensity_max_value = max_val
        if changed:
            self._invalidate_alignment()
        self._schedule_preview()

    def _on_mode_change(self, _event: tk.Event) -> None:
        self._invalidate_alignment()
        self._schedule_preview()

    def _on_auto_align_toggle(self) -> None:
        self._schedule_preview()

    def _current_options(self, auto_align_override: bool | None = None) -> ValidationOptions:
        mode_text = self.mode_var.get()
        mode: ValidationMode = VALIDATION_MODE_PRESERVE
        if mode_text == VALIDATION_MODE_FIT:
            mode = VALIDATION_MODE_FIT
        auto_align = self.auto_align_var.get() if auto_align_override is None else auto_align_override
        return ValidationOptions(
            mode=mode,
            auto_align=bool(auto_align),
            align_model=ALIGNMENT_MODEL_TRANSLATION_ECC,
            joint_norm_percentiles=(1.0, 99.0),
            template_score_threshold=MIN_TEMPLATE_MATCH_SCORE,
        ).normalized()

    def _build_processed_image(self) -> np.ndarray:
        geom = _apply_geometry(
            self.base_image,
            rotation_deg=self.rotation_deg,
            flip_h=self.flip_h,
            flip_v=self.flip_v,
            crop_box=self.crop_box,
        )
        intensity = _apply_intensity_window(geom, self.intensity_min_value, self.intensity_max_value)
        return _scale_image(intensity, self.scale_x_value, self.scale_y_value)

    def _build_crop_preview_image(self, *, apply_scale: bool = True) -> np.ndarray:
        geom_no_crop = _apply_geometry(
            self.base_image,
            rotation_deg=self.rotation_deg,
            flip_h=self.flip_h,
            flip_v=self.flip_v,
            crop_box=None,
        )
        intensity = _apply_intensity_window(geom_no_crop, self.intensity_min_value, self.intensity_max_value)
        if apply_scale:
            return _scale_image(intensity, self.scale_x_value, self.scale_y_value)
        return intensity

    def _invalidate_alignment(self) -> None:
        self.alignment_result = None
        self._alignment_dirty = True

    def _run_alignment(self) -> None:
        if not self._apply_scale_entries(commit=True):
            return
        processed = self._build_processed_image()
        options = self._current_options(auto_align_override=True)
        self.alignment_result = auto_align_images(self.reference_image, processed, options=options)
        self._alignment_dirty = False
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._preview_job is not None:
            try:
                self.top.after_cancel(self._preview_job)
            except Exception:
                pass
            self._preview_job = None
        self._preview_job = self.top.after(PREVIEW_DEBOUNCE_MS, self._refresh_preview)

    def _refresh_preview(self) -> None:
        self._preview_job = None
        if not self._apply_scale_entries(commit=False):
            return

        options = self._current_options()
        processed = self._build_processed_image()

        alignment_for_preview: AlignmentResult | None = None
        if options.auto_align and not self._alignment_dirty and self.alignment_result is not None:
            alignment_for_preview = self.alignment_result

        comp_data = prepare_images_for_comparison(
            self.reference_image,
            processed,
            options=options,
            alignment_result=alignment_for_preview,
        )
        crop_preview = self._build_crop_preview_image(apply_scale=True)
        crop_preview_for_norm = self._build_crop_preview_image(apply_scale=False)

        self._render_as_is_tab(crop_preview, crop_preview_for_norm)
        self._render_registered_tab(comp_data, options)
        self._render_diff_tab(comp_data)
        self._render_overlay_tab(comp_data)

        status_parts = [
            f"Mode={options.mode}",
            f"Input={processed.shape[0]}x{processed.shape[1]}",
            f"Crop={'set' if self.crop_box is not None else 'none'}",
        ]
        if options.auto_align:
            if self._alignment_dirty or alignment_for_preview is None:
                status_parts.append("Alignment=stale (click Run Alignment)")
            else:
                status_parts.append(f"Alignment={alignment_for_preview.status}")
                if alignment_for_preview.match_score is not None:
                    status_parts.append(f"Template={alignment_for_preview.match_score:.3f}")
                if alignment_for_preview.ecc_score is not None:
                    status_parts.append(f"ECC={alignment_for_preview.ecc_score:.3f}")
        else:
            status_parts.append("Alignment=disabled")
        status_parts.append(f"Compare={comp_data.status}")
        self.status_var.set(" | ".join(status_parts))

    def _render_as_is_tab(
        self,
        crop_preview: np.ndarray,
        crop_preview_for_norm: np.ndarray | None = None,
    ) -> None:
        if self.fig_as_is is None or self.canvas_as_is is None:
            return
        self.fig_as_is.clf()
        ax_ref = self.fig_as_is.add_subplot(121)
        ax_edit = self.fig_as_is.add_subplot(122)

        # Keep reference display independent from input edits (scale/crop/intensity).
        ref_low = self.reference_display_low
        ref_high = self.reference_display_high
        norm_source = crop_preview if crop_preview_for_norm is None else crop_preview_for_norm
        input_low, input_high = _finite_display_bounds(norm_source)
        _imshow_with_colorbar(
            ax_ref,
            self.reference_image,
            cmap="gray",
            vmin=ref_low,
            vmax=ref_high,
        )
        ax_ref.set_title(
            f"Reference As-Is [{self.reference_image.shape[0]}x{self.reference_image.shape[1]}]"
        )
        _imshow_with_colorbar(
            ax_edit,
            crop_preview,
            cmap="gray",
            vmin=input_low,
            vmax=input_high,
        )
        ax_edit.set_title(
            f"Input Edit View (Crop Select) [{crop_preview.shape[0]}x{crop_preview.shape[1]}]"
        )

        if self.crop_box is not None:
            x0, y0, x1, y1 = self.crop_box
            rect = Rectangle(
                (x0 * self.scale_x_value, y0 * self.scale_y_value),
                (x1 - x0) * self.scale_x_value,
                (y1 - y0) * self.scale_y_value,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5,
            )
            ax_edit.add_patch(rect)

        if self._crop_selector is not None:
            try:
                self._crop_selector.set_active(False)
                self._crop_selector.disconnect_events()
            except Exception:
                pass
            self._crop_selector = None

        self._crop_selector = RectangleSelector(
            ax_edit,
            self._on_crop_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self._crop_selector.set_active(self.crop_active)
        self.canvas_as_is.draw()

    def _render_registered_tab(self, comp_data: ComparisonData, options: ValidationOptions) -> None:
        if self.fig_registered is None or self.canvas_registered is None:
            return
        self.fig_registered.clf()
        ax_ref = self.fig_registered.add_subplot(121)
        ax_input = self.fig_registered.add_subplot(122)
        suffix = _comparison_mode_suffix(comp_data.mode, comp_data.score, comp_data.ecc_score)

        if comp_data.registered_reference.size > 0 and comp_data.registered_input.size > 0:
            reg_low, reg_high = _joint_norm_bounds(
                comp_data.registered_reference, comp_data.registered_input, options.joint_norm_percentiles
            )
            _imshow_with_colorbar(
                ax_ref,
                comp_data.registered_reference,
                cmap="gray",
                vmin=reg_low,
                vmax=reg_high,
            )
            _imshow_with_colorbar(
                ax_input,
                comp_data.registered_input,
                cmap="gray",
                vmin=reg_low,
                vmax=reg_high,
            )
            ax_ref.set_title(f"Reference Registered{suffix}")
            ax_input.set_title(f"Input Registered{suffix}")
        else:
            ax_ref.axis("off")
            ax_input.axis("off")
            ax_ref.text(0.5, 0.5, "No registered ROI", ha="center", va="center")
            ax_input.text(0.5, 0.5, f"Status: {comp_data.status}", ha="center", va="center")
        self.canvas_registered.draw()

    def _render_diff_tab(self, comp_data: ComparisonData) -> None:
        if self.fig_diff is None or self.canvas_diff is None:
            return
        self.fig_diff.clf()
        ax = self.fig_diff.add_subplot(111)
        if comp_data.diff_image is not None:
            _imshow_with_colorbar(ax, comp_data.diff_image, cmap="viridis", label="|Reference - Input|")
            ax.set_title("Difference (Joint-Normalized)")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Difference unavailable\n{comp_data.status}", ha="center", va="center")
            ax.set_title("Difference Unavailable")
        self.canvas_diff.draw()

    def _render_overlay_tab(self, comp_data: ComparisonData) -> None:
        if self.fig_overlay is None or self.canvas_overlay is None:
            return
        self.fig_overlay.clf()
        ax = self.fig_overlay.add_subplot(111)
        if comp_data.overlay_image is not None:
            ax.imshow(comp_data.overlay_image)
            _add_overlay_channel_colorbars(ax, comp_data.overlay_image)
            ax.set_title("Overlay: Red=Reference, Green=Input")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Overlay unavailable\n{comp_data.status}", ha="center", va="center")
            ax.set_title("Overlay Unavailable")
        self.canvas_overlay.draw()

    def _reset_section(self) -> None:
        section = self.reset_section_var.get()
        if section == "geometry":
            self.rotation_deg = 0
            self.flip_h = False
            self.flip_v = False
            self.crop_box = None
        elif section == "intensity":
            self.scale_intensity_min.set(self.raw_min_val)
            self.scale_intensity_max.set(self.raw_max_val)
            self.intensity_min_value = self.raw_min_val
            self.intensity_max_value = self.raw_max_val
        elif section == "scale":
            self.scale_x_value = 1.0
            self.scale_y_value = 1.0
            self.scale_x_var.set("1.0000")
            self.scale_y_var.set("1.0000")
        elif section == "alignment":
            self.alignment_result = None
            self._alignment_dirty = True
            self._schedule_preview()
            return
        self._invalidate_alignment()
        self._schedule_preview()

    def _reset_all(self) -> None:
        self.rotation_deg = 0
        self.flip_h = False
        self.flip_v = False
        self.crop_box = None
        self.crop_active = False
        self.btn_crop.config(text="Crop OFF")
        self.scale_intensity_min.set(self.raw_min_val)
        self.scale_intensity_max.set(self.raw_max_val)
        self.intensity_min_value = self.raw_min_val
        self.intensity_max_value = self.raw_max_val
        self.scale_x_value = 1.0
        self.scale_y_value = 1.0
        self.scale_x_var.set("1.0000")
        self.scale_y_var.set("1.0000")
        self.mode_var.set(VALIDATION_MODE_PRESERVE)
        self.auto_align_var.set(False)
        self._invalidate_alignment()
        self._schedule_preview()

    def _finish(self) -> None:
        if not self._apply_scale_entries(commit=True):
            return
        options = self._current_options()
        processed = self._build_processed_image()
        alignment_result: AlignmentResult | None = None
        if options.auto_align:
            if self._alignment_dirty or self.alignment_result is None:
                alignment_result = auto_align_images(self.reference_image, processed, options=options)
            else:
                alignment_result = self.alignment_result
        self._result = ValidationImageResult(
            processed_image=processed.astype(np.float32),
            options=options,
            alignment_result=alignment_result,
            scale_x=self.scale_x_value,
            scale_y=self.scale_y_value,
            intensity_min=self.intensity_min_value,
            intensity_max=self.intensity_max_value,
            crop_box=self.crop_box,
            rotation_deg=self.rotation_deg,
            flip_h=self.flip_h,
            flip_v=self.flip_v,
        )
        self.top.destroy()

    def _cancel(self) -> None:
        self._result = None
        self.top.destroy()

    def run(self) -> ValidationImageResult | None:
        self.top.wait_window()
        return self._result


def _roi_to_array(roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return np.array([-1, -1, -1, -1], dtype=np.int32)
    x, y, w, h = roi
    return np.array([x, y, w, h], dtype=np.int32)


def _crop_box_to_array(crop_box: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop_box is None:
        return np.array([-1, -1, -1, -1], dtype=np.int32)
    return np.array(crop_box, dtype=np.int32)


def _options_to_json_array(options: ValidationOptions) -> np.ndarray:
    return np.array(json.dumps(asdict(options), sort_keys=True))


def _alignment_scores_array(alignment: AlignmentResult | None) -> np.ndarray:
    if alignment is None:
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array(
        [
            np.nan if alignment.match_score is None else alignment.match_score,
            np.nan if alignment.ecc_score is None else alignment.ecc_score,
        ],
        dtype=np.float32,
    )


def _alignment_status_array(alignment: AlignmentResult | None) -> np.ndarray:
    return np.array("not_run" if alignment is None else alignment.status)


def _alignment_mode_array(alignment: AlignmentResult | None) -> np.ndarray:
    return np.array("none" if alignment is None else alignment.mode)


def _save_workspace_data(
    save_path: str,
    rectified_img: np.ndarray,
    val_path: str,
    thor_path: str,
    val_result: ValidationImageResult,
    thor_result: ValidationImageResult,
) -> None:
    val_resize_to_reference = val_result.options.mode == VALIDATION_MODE_FIT
    thor_resize_to_reference = thor_result.options.mode == VALIDATION_MODE_FIT

    vars_to_save: dict[str, np.ndarray | np.bool_ | np.float32 | np.int32] = {
        # Backward-compatible keys
        "rectified_image": rectified_img,
        "validation_input_original": np.asarray(Image.open(val_path).convert("L")),
        "validation_processed": val_result.processed_image,
        "validation_resize_to_reference": np.array(val_resize_to_reference),
        "thorcam_input_original": np.asarray(Image.open(thor_path).convert("L")),
        "thorcam_processed": thor_result.processed_image,
        "thorcam_resize_to_reference": np.array(thor_resize_to_reference),
        # Extended metadata for reproducibility
        "validation_options_json": _options_to_json_array(val_result.options),
        "validation_mode": np.array(val_result.options.mode),
        "validation_auto_align": np.array(val_result.options.auto_align),
        "validation_scale_factors": np.array([val_result.scale_x, val_result.scale_y], dtype=np.float32),
        "validation_intensity_window": np.array([val_result.intensity_min, val_result.intensity_max], dtype=np.float32),
        "validation_crop_box": _crop_box_to_array(val_result.crop_box),
        "validation_rotation_deg": np.array(val_result.rotation_deg, dtype=np.int32),
        "validation_flip_h": np.array(val_result.flip_h),
        "validation_flip_v": np.array(val_result.flip_v),
        "validation_alignment_status": _alignment_status_array(val_result.alignment_result),
        "validation_alignment_mode": _alignment_mode_array(val_result.alignment_result),
        "validation_alignment_scores": _alignment_scores_array(val_result.alignment_result),
        "validation_alignment_roi_reference": _roi_to_array(
            None if val_result.alignment_result is None else val_result.alignment_result.roi_reference
        ),
        "validation_alignment_roi_input": _roi_to_array(
            None if val_result.alignment_result is None else val_result.alignment_result.roi_input
        ),
        "thorcam_options_json": _options_to_json_array(thor_result.options),
        "thorcam_mode": np.array(thor_result.options.mode),
        "thorcam_auto_align": np.array(thor_result.options.auto_align),
        "thorcam_scale_factors": np.array([thor_result.scale_x, thor_result.scale_y], dtype=np.float32),
        "thorcam_intensity_window": np.array([thor_result.intensity_min, thor_result.intensity_max], dtype=np.float32),
        "thorcam_crop_box": _crop_box_to_array(thor_result.crop_box),
        "thorcam_rotation_deg": np.array(thor_result.rotation_deg, dtype=np.int32),
        "thorcam_flip_h": np.array(thor_result.flip_h),
        "thorcam_flip_v": np.array(thor_result.flip_v),
        "thorcam_alignment_status": _alignment_status_array(thor_result.alignment_result),
        "thorcam_alignment_mode": _alignment_mode_array(thor_result.alignment_result),
        "thorcam_alignment_scores": _alignment_scores_array(thor_result.alignment_result),
        "thorcam_alignment_roi_reference": _roi_to_array(
            None if thor_result.alignment_result is None else thor_result.alignment_result.roi_reference
        ),
        "thorcam_alignment_roi_input": _roi_to_array(
            None if thor_result.alignment_result is None else thor_result.alignment_result.roi_input
        ),
    }
    im_process.save_workspace_variables(save_path, vars_to_save)


def main() -> None:
    root = tk.Tk()
    root.withdraw()  # Hide main window

    print("Please select a rectification output (.npz workspace/sweep output or .npy rectified views).")
    rectified_path = filedialog.askopenfilename(
        title="Select Rectification Output",
        filetypes=[
            ("Rectification Outputs", "*.npz;*.npy"),
            ("NPZ Files", "*.npz"),
            ("NPY Files", "*.npy"),
        ],
    )
    if not rectified_path:
        print("No rectification output selected. Exiting.")
        return

    try:
        rectified_img = load_rectified_reference(rectified_path)
    except Exception as exc:
        messagebox.showerror("Error", f"Failed to load rectification output: {exc}")
        return

    print("Please select the Validation Camera image.")
    val_path = filedialog.askopenfilename(
        title="Select Validation Image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")],
    )
    if not val_path:
        print("No validation image selected. Exiting.")
        return

    print("Please select the Thorcam image.")
    thor_path = filedialog.askopenfilename(
        title="Select Thorcam Image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")],
    )
    if not thor_path:
        print("No thorcam image selected. Exiting.")
        return

    print("Editing Validation Image...")
    editor_val = GuidedValidationEditor(
        root,
        val_path,
        reference_image=rectified_img,
        title="Validation Editor - Validation Camera",
        camera_type=None,
    )
    val_result = editor_val.run()
    if val_result is None:
        print("Validation image editing cancelled.")
        return

    print("Editing Thorcam Image...")
    editor_thor = GuidedValidationEditor(
        root,
        thor_path,
        reference_image=rectified_img,
        title="Validation Editor - Thorcam",
        camera_type=None,
    )
    thor_result = editor_thor.run()
    if thor_result is None:
        print("Thorcam image editing cancelled.")
        return

    print("Generating comparisons...")
    figs_val = compare_images(
        rectified_img,
        val_result.processed_image,
        "Validation",
        options=val_result.options,
        alignment_result=val_result.alignment_result,
    )
    figs_thor = compare_images(
        rectified_img,
        thor_result.processed_image,
        "Thorcam",
        options=thor_result.options,
        alignment_result=thor_result.alignment_result,
    )
    all_figures = figs_val + figs_thor

    plt.show()

    if messagebox.askyesno("Save", "Do you want to save the figures and workspace?"):
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            filename_changes = im_process.save_figures(all_figures, save_dir)
            if filename_changes:
                changes_message = "The following filenames were regularized for compatibility:\n\n"
                for original, regularized in sorted(filename_changes.items()):
                    changes_message += f"  {original}\n    -> {regularized}\n"
                messagebox.showinfo("Filename Changes", changes_message)

            save_path = os.path.join(save_dir, "validation_workspace.npz")
            _save_workspace_data(
                save_path=save_path,
                rectified_img=rectified_img,
                val_path=val_path,
                thor_path=thor_path,
                val_result=val_result,
                thor_result=thor_result,
            )
            messagebox.showinfo("Success", f"Saved to {save_dir}")

    root.destroy()


if __name__ == "__main__":
    main()
