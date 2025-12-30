import os
import sys
import PIL
from dataclasses import dataclass
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Any, cast
from itertools import combinations
from django.utils.text import slugify

if __package__ is None or __package__ == "":
    # Running as a script: make local package modules importable
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rectification_strategy import (
        GlobalTranslationRectificationStrategy,
        PixelwisePatchRectificationStrategy,
        resolve_reference_view,
    )
else:
    # Imported as a module within the package
    from .rectification_strategy import (
        GlobalTranslationRectificationStrategy,
        PixelwisePatchRectificationStrategy,
        resolve_reference_view,
    )

# --- Constants ---
DEFAULT_IMAGE_DIR = 'moment_images/images'
GAUSSIAN_BLUR_KERNEL = (51, 51)
GAUSSIAN_BLUR_SIGMA = 5
HOUGH_CIRCLES_PARAM1 = 50
HOUGH_CIRCLES_PARAM2 = 30
HOUGH_CIRCLES_MIN_DIST = 20
DEFAULT_VIEW_GRID_SIZE = 3
VIEW_DISTANCE_SCALE_FACTOR = 1.1
CORRELATION_PAD_WIDTH = int(125*0.45) # 30% of expected viewpoint size (125 pixels)
DEFAULT_SEARCH_RADIUS_MIN = 100
DEFAULT_SEARCH_RADIUS_MAX = 130


@dataclass
class RectificationRunConfig:
    """Configuration for running the rectification pipeline non-interactively.

    Attributes mirror the choices made during the interactive ``main`` flow so
    that external scripts can inject deterministic values when sweeping
    parameters.
    """

    strategy: str = "pixel"
    similarity_window_size: int = 41
    refinement_window_size: int = 14
    correlation_pad_width: int = CORRELATION_PAD_WIDTH
    max_displacement: Optional[int] = None
    gauss_sigma: Optional[float] = None
    gauss_size: Optional[int] = None
    disp_gauss_sigma: Optional[float] = None
    disp_gauss_size: Optional[int] = None
    global_threshold: float = 0.25
    pixel_threshold: float = 0.05
    zero_fill_roll: bool = True


@dataclass
class RectificationResult:
    """Container for automated rectification results."""

    avg_image: np.ndarray
    rectified_views: np.ndarray
    similarity_data: Any
    config: RectificationRunConfig
    ref_view_index: Tuple[int, int]

def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load a single image from disk, printing informative errors on failure.

    The function never raises when the file is missing or unreadable; it simply
    logs the reason and returns ``None`` so callers can skip the bad entry.

    Args:
        image_path: Path to the image file.

    Returns:
        The opened ``PIL.Image`` or ``None`` when the file does not exist or is
        not a recognizable image.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return None
    
    try:
        image = Image.open(image_path)
        return image
    except PIL.UnidentifiedImageError:
        print(f"Error: The file {image_path} is not a valid image.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the image: {e}")
        return None


def rectify_views_from_config(views: np.ndarray, config: RectificationRunConfig) -> RectificationResult:
    """Run rectification with injected parameters instead of interactive prompts.

    Args:
        views: Square grid of grayscale patches shaped ``(N, N, H, W)``.
        config: Rectification options to sweep over.

    Returns:
        ``RectificationResult`` containing the averaged rectified image, the
        rectified stack, and the similarity information produced by the chosen
        strategy.
    """

    views_arr, ref_view_idx = resolve_reference_view(views)

    strategy = config.strategy.lower()
    if strategy not in {"global", "pixel", "combined"}:
        raise ValueError("strategy must be one of 'global', 'pixel', or 'combined'")

    if strategy == "global":
        rectification_strategy = GlobalTranslationRectificationStrategy(
            pad_width=config.correlation_pad_width, prompt_for_all_views=False
        )
        similarity_data = rectification_strategy.compute_similarity_matrix(
            views_arr, ref_view_idx, thresh=config.global_threshold
        )
        avg_image, rectified_views = rectification_strategy.rectify(
            views_arr, similarity_data, ref_view_idx
        )
    elif strategy == "pixel":
        rectification_strategy = PixelwisePatchRectificationStrategy(
            config.similarity_window_size, zero_fill_roll=config.zero_fill_roll
        )
        similarity_data = rectification_strategy.compute_similarity_matrices(
            views_arr,
            ref_view_idx,
            max_disp=config.max_displacement,
            gauss_sigma=config.gauss_sigma,
            gauss_size=config.gauss_size,
            disp_gauss_sigma=config.disp_gauss_sigma,
            disp_gauss_size=config.disp_gauss_size,
            thresh=config.pixel_threshold,
        )
        avg_image, rectified_views = rectification_strategy.rectify(
            views_arr, similarity_data, ref_view_idx
        )
    else:
        rectification_strategy_global = GlobalTranslationRectificationStrategy(
            pad_width=config.correlation_pad_width, prompt_for_all_views=False
        )
        global_similarity = rectification_strategy_global.compute_similarity_matrix(
            views_arr, ref_view_idx, thresh=config.global_threshold
        )
        av_rect_global, rect_views_after_global = rectification_strategy_global.rectify(
            views_arr, global_similarity, ref_view_idx
        )

        rectification_strategy_pixel = PixelwisePatchRectificationStrategy(
            config.refinement_window_size, zero_fill_roll=config.zero_fill_roll
        )
        similarity_data = rectification_strategy_pixel.compute_similarity_matrices(
            rect_views_after_global,
            ref_view_idx,
            max_disp=config.max_displacement,
            gauss_sigma=config.gauss_sigma,
            gauss_size=config.gauss_size,
            disp_gauss_sigma=config.disp_gauss_sigma,
            disp_gauss_size=config.disp_gauss_size,
            thresh=config.pixel_threshold,
        )
        avg_image, rectified_views = rectification_strategy_pixel.rectify(
            rect_views_after_global, similarity_data, ref_view_idx
        )

    return RectificationResult(
        avg_image=avg_image,
        rectified_views=rectified_views,
        similarity_data=similarity_data,
        config=config,
        ref_view_index=ref_view_idx,
    )


def generate_rectification_figures(
    rectified_views: np.ndarray,
    avg_image: np.ndarray,
    ref_index: Optional[Tuple[int, int]] = None,
) -> Dict[str, List[Figure]]:
    """
    Build matplotlib figures that mirror the interactive workflow's outputs.

    The figures are created without calling ``plt.show`` so they can be saved
    non-interactively (e.g., during parameter sweeps).
    """

    figures: Dict[str, List[Figure]] = {
        "rectified_views": [],
        "overlays": [],
        "averages": [],
    }

    view_grid_size = rectified_views.shape[0]

    # Individual rectified view figures
    for i in range(view_grid_size):
        for j in range(view_grid_size):
            fig = plt.figure()
            plt.imshow(rectified_views[i, j], cmap="gray")
            plt.title(f"Rectified View {i+1}-{j+1}")
            figures["rectified_views"].append(fig)

    # Overlays for extreme views (and optional reference view)
    extreme_overlays = build_extreme_view_overlays(rectified_views, ref_index=ref_index)
    for overlay, (i1, j1), (i2, j2) in extreme_overlays:
        fig = plt.figure()
        plt.imshow(overlay)
        plt.title(f"Overlay ({i1 + 1}-{j1 + 1}) vs ({i2 + 1}-{j2 + 1})")
        figures["overlays"].append(fig)

    # Averaged rectified image and an equalized variant
    fig_avg = plt.figure()
    plt.imshow(avg_image, cmap="gray")
    plt.title("Averaged Rectified Image")
    figures["averages"].append(fig_avg)

    fig_eq = plt.figure()
    base_img = np.clip(avg_image, 0, 255).astype("uint8")
    eq_img = np.clip(cv.equalizeHist(base_img), 0.85 * 255, 255)
    plt.imshow(eq_img, cmap="gray")
    plt.title("Averaged Rectified Image - Equalized (clipped, 85%) for contrast")
    figures["averages"].append(fig_eq)

    return figures

def min_max_normalize(image: Image.Image, min_val: int = 0, max_val: int = 255) -> Image.Image:
    """
    Scale pixel values to the 0–255 range while optionally clamping extremes.

    ``min_val`` and ``max_val`` override the computed extrema when provided,
    allowing quick clipping of dark/bright outliers. A flat input image returns
    an unmodified copy to avoid division by zero.

    Args:
        image: Source ``PIL.Image``.
        min_val: Optional lower bound to use instead of the image minimum.
        max_val: Optional upper bound to use instead of the image maximum.

    Returns:
        A new ``PIL.Image`` normalized to ``uint8`` intensities.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object.")
    
    image_array = np.array(image)
    
    img_min = image_array.min()
    if min_val > 0:
        img_min = min_val
    
    img_max = image_array.max()
    if max_val < 255:
        img_max = max_val

    if img_min == img_max:
        print("Warning: The image has no variation in pixel values. Returning a blank image.")
        return image.copy()
    else:
        normalized_array = (image_array - img_min) / (img_max - img_min) * 255
        return Image.fromarray(normalized_array.astype('uint8'))

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """
    Sharpen an image with Gaussian blur subtraction and re-normalize to 0–255.

    The operation mirrors OpenCV's unsharp masking approach: blur the input,
    subtract a scaled version of the blur, and scale the result back to the
    original dtype range. Completely flat inputs return zeros to preserve
    downstream expectations.

    Args:
        image: Input ``numpy`` image array.
        sigma: Standard deviation for Gaussian blur.
        strength: Scaling factor applied to the high-frequency component.

    Returns:
        Sharpened image with the same dtype as ``image``.
    """
    image_float = image.astype(np.float32)

    blurred = cv.GaussianBlur(image_float, (0, 0), sigma)
    sharpened = cv.addWeighted(image_float, 1.0 + strength, blurred, -strength, 0)

    min_val = float(sharpened.min())
    max_val = float(sharpened.max())
    if np.isclose(min_val, max_val):
        normalized = np.zeros_like(sharpened)
    else:
        normalized = (sharpened - min_val) / (max_val - min_val) * 255.0

    return normalized.astype(image.dtype, copy=False)

def assemble_HDR(images: List[Image.Image], exposure_times: Optional[np.ndarray] = None) -> Image.Image:
    """
    Build and tone-map an HDR image using Debevec calibration and merging.

    Images are converted to BGR, ordered by exposure time (descending), and
    processed via OpenCV's Debevec calibrate/merge pipeline. The final HDR is
    tone-mapped to an 8-bit grayscale image using gamma 2.2.

    Args:
        images: List of ``PIL.Image`` instances (grayscale expected).
        exposure_times: Optional exposure times in seconds. When omitted,
            synthetic 1, 1/2, 1/4, ... values are generated.

    Returns:
        Tone-mapped HDR output as a ``PIL.Image``.
    """
    if not images:
        raise ValueError("No images provided for HDR assembly.")

    # Convert PIL Images to OpenCV BGR format and ensure consistent normalization
    cv_images = [cv.cvtColor(np.array(image), cv.COLOR_GRAY2BGR).astype('uint8') for image in images]

    # Normalize/validate exposure_times and make sure it's a 1D numpy array of float32
    if exposure_times is None:
        # Generate synthetic exposure times if none are provided (in seconds)
        exposure_times = np.array([1.0 / (2 ** i) for i in range(len(cv_images))], dtype=np.float32)
    else:
        exposure_times = np.array(exposure_times, dtype=np.float32)
        if exposure_times.ndim != 1 or len(exposure_times) != len(cv_images):
            raise ValueError("Exposure times must be a 1D array matching the number of images.")
        if np.any(exposure_times <= 0):
            raise ValueError("Exposure times must be positive.")

    # Order images and exposure times by exposure time descending (higher -> lower)
    sort_idx = np.argsort(-exposure_times)  # descending order
    exposure_times = exposure_times[sort_idx]
    cv_images = [cv_images[i] for i in sort_idx]

    calibrate = cv.createCalibrateDebevec()
    response = calibrate.process(cv_images, exposure_times)

    merge_debevec = cv.createMergeDebevec()
    hdr = merge_debevec.process(cv_images, exposure_times, response)

    # Tone map the HDR image to 8-bit for display and saving
    tonemap = cv.createTonemap(gamma=2.2)
    ldr = np.nan_to_num(tonemap.process(hdr.copy()))
    ldr_clipped = np.clip(ldr * 255, 0, 255).astype('uint8')

    return Image.fromarray(cv.cvtColor(ldr_clipped, cv.COLOR_BGR2GRAY))

def get_image_paths() -> Tuple[List[Image.Image], List[str], bool]:
    """
    Interactively collect images from a directory and report HDR intent.

    Users can choose the default image folder, supply a custom path, and mark
    the directory as HDR-capable. Invalid entries are skipped with console
    feedback rather than raising.

    Returns:
        ``(images, image_paths, hdr_flag)`` where ``images`` are loaded
        ``PIL.Image`` objects, ``image_paths`` are their file paths, and
        ``hdr_flag`` indicates whether the caller identified the directory as
        HDR content.
    """
    ims_dir = input(f"Enter path to images or enter 'default' to use '{DEFAULT_IMAGE_DIR}': ")
    if ims_dir.lower() == 'default':
        ims_dir = DEFAULT_IMAGE_DIR
    
    hdr_flag = input("Is this an HDR image directory? (y/n): ").lower() == "y"
    
    images, image_paths = [], []
    if not os.path.isdir(ims_dir):
        print(f"Error: Directory '{ims_dir}' not found.")
        return [], [], hdr_flag

    for f in sorted(os.listdir(ims_dir)):
        image_path = os.path.join(ims_dir, f)
        image = load_image(image_path)
        if image:
            images.append(image)
            image_paths.append(image_path)
            print(f"[{len(image_paths)}] Successfully loaded image: {f}")
        else:
            print(f"Failed to load image: {f}")

    if image_paths:
        print("\nLoaded images:")
        for idx, path in enumerate(image_paths, start=1):
            print(f"  {idx}: {os.path.basename(path)}")

    return images, image_paths, hdr_flag

def select_roi(image_np: np.ndarray, strategy: str, search_radius: int) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Extract a square ROI using one of several heuristic or manual strategies.

    The ROI is always ``(2*search_radius)**2`` pixels. Strategies include using
    the image center, the brightest Gaussian-smoothed pixel nearest the center,
    explicit coordinates provided by the user, or the full image without
    cropping.

    Args:
        image_np: Input grayscale image array.
        strategy: One of ``center``, ``max_pixel``, ``full``, or ``specific``.
        search_radius: Radius used to size the ROI square.

    Returns:
        ``(roi, offset)`` where ``roi`` is the extracted array and ``offset``
        is the ``(row, col)`` origin relative to the original image (``None``
        when ``strategy == "full"``).
    """
    h, w = image_np.shape
    roi_size = search_radius * 2
    
    if strategy == 'center':
        top = h // 2 - roi_size
        left = w // 2 - roi_size
        roi = image_np[top:top + roi_size*2, left:left + roi_size*2]
        return roi, (int(top), int(left))
    elif strategy == 'max_pixel':
        # Find all maximum intensity pixels after Gaussian blur
        blurred = cv.GaussianBlur(image_np, GAUSSIAN_BLUR_KERNEL, sigmaX=GAUSSIAN_BLUR_SIGMA, sigmaY=GAUSSIAN_BLUR_SIGMA)
        max_val = blurred.max()
        max_coords = np.argwhere(blurred == max_val)
        
        # Calculate geometric center (handles even/odd dimensions)
        center_y, center_x = h // 2, w // 2
        
        # Find closest maximum to center using Euclidean distance
        distances = np.sqrt((max_coords[:,0] - center_y)**2 + (max_coords[:,1] - center_x)**2)
        min_dist_idx = np.argmin(distances)
        selected_y, selected_x = max_coords[min_dist_idx]
        
        # Calculate ROI boundaries centered on selected pixel
        top = selected_y - roi_size
        left = selected_x - roi_size
        
        # Ensure ROI stays within image bounds
        top = max(0, min(top, h - roi_size*2))
        left = max(0, min(left, w - roi_size*2))
        
        roi = image_np[top:top + roi_size*2, left:left + roi_size*2]
        return roi, (int(top), int(left))
    elif strategy == 'specific':
        # Prompt user for specific ROI coordinates
        center_y = int(input(f"Enter center row (0 to {h - roi_size*2}): "))
        center_x = int(input(f"Enter center column (0 to {w - roi_size*2}): "))
        
        if center_x < roi_size or center_x + roi_size > h or center_y < roi_size or center_y + roi_size > w:
            raise ValueError("Specified ROI is out of bounds.")
        
        roi = image_np[center_x - roi_size:center_x + roi_size, center_y - roi_size:center_y + roi_size]
        return roi, (int(center_x - roi_size), int(center_y - roi_size))
    else: # 'full'
        return image_np, None

def detect_and_adjust_circles(roi: np.ndarray, roi_offset: Optional[Tuple[int, int]], min_radius: int, max_radius: int) -> Optional[np.ndarray]:
    """
    Run Hough circle detection inside an ROI and rebase results to the full image.

    Args:
        roi: Grayscale ROI used for detection.
        roi_offset: ``(row, col)`` offset locating the ROI within the original
            image; ``None`` when the ROI is the full image.
        min_radius: Minimum circle radius in pixels.
        max_radius: Maximum circle radius in pixels.

    Returns:
        ``np.ndarray`` of detected ``(x, y, r)`` circles or ``None`` when no
        detections are produced.
    """
    circles = cv.HoughCircles(roi, cv.HOUGH_GRADIENT, dp=1, minDist=HOUGH_CIRCLES_MIN_DIST, 
                              param1=HOUGH_CIRCLES_PARAM1, param2=HOUGH_CIRCLES_PARAM2, 
                              minRadius=min_radius, maxRadius=max_radius)
    
    if circles is None:
        return None
        
    circles = np.around(circles).astype(np.int64)
    
    if roi_offset:
        circles[0, :, 0] += roi_offset[1]  # Adjust x-coordinates
        circles[0, :, 1] += roi_offset[0]  # Adjust y-coordinates
        
    return circles

def display_circles(image: Image.Image, circles: np.ndarray):
    """
    Visualize up to the first two detected circles on top of an image.

    Args:
        image: The PIL image to draw on.
        circles: A NumPy array of circles to display.
    """
    cimg = cv.cvtColor(np.array(image), cv.COLOR_GRAY2BGR)
    for i in circles[0, :2]:  # Draw the top 2 circles
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Outer circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)    # Center
    
    plt.figure()
    plt.imshow(cimg)
    plt.show()

def extract_views(image_np: np.ndarray, base_circle: np.ndarray, num_views: int, view_dist: int) -> np.ndarray:
    """
    Slice a square grid of equally sized patches centered around a base circle.

    Each grid element is centered ``view_dist`` pixels apart; patches that would
    exceed image bounds are left as zeroed arrays and logged to stdout.

    Args:
        image_np: Grayscale source image.
        base_circle: Reference ``(x, y, radius)`` used as the grid origin.
        num_views: Grid dimension (e.g., ``3`` for ``3x3``).
        view_dist: Pixel distance between adjacent view centers.

    Returns:
        Array of shape ``(num_views, num_views, view_dist, view_dist)`` with
        extracted (or zero-filled) patches.
    """
    d = view_dist
    views = np.zeros((num_views, num_views, d, d))
    
    for i in range(num_views):
        for j in range(num_views):
            x = int(base_circle[0] + (i - num_views // 2) * d)
            y = int(base_circle[1] + (j - num_views // 2) * d)
            
            half_d = d // 2
            if half_d * 2 != d:
                even_odd_flag = 1
            else:
                even_odd_flag = 0
            if x - half_d < 0 or y - half_d < 0 or x + half_d >= image_np.shape[1] or y + half_d >= image_np.shape[0]:
                print(f"Skipping view {i+1}-{j+1} due to out-of-bounds coordinates.")
                continue
            
            views[i, j, :, :] = image_np[y - half_d:y + half_d + even_odd_flag, x - half_d:x + half_d + even_odd_flag]
            
    return views


def build_extreme_view_overlays(rect_images: np.ndarray, ref_index: Optional[Tuple[int, int]] = None) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """
    Create normalized RGB overlays comparing each pair of corner rectified views.

    The four extreme grid positions are paired in all combinations; each overlay
    stores one view in the red channel and the other in green to highlight
    misalignments. Inputs are scaled to [0, 1] to avoid brightness imbalance.

    If ``ref_index`` is provided, also add overlays comparing each extreme
    view with the reference view (one overlay per extreme).
    """
    grid_size = rect_images.shape[0]
    if grid_size < 2:
        return []

    extreme_indices = [
        (0, 0),
        (0, grid_size - 1),
        (grid_size - 1, 0),
        (grid_size - 1, grid_size - 1),
    ]

    overlays: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]] = []
    pairs_added = set()
    for (i1, j1), (i2, j2) in combinations(extreme_indices, 2):
        view_1 = rect_images[i1, j1].astype(np.float32)
        view_2 = rect_images[i2, j2].astype(np.float32)

        max_val = float(max(np.max(view_1), np.max(view_2), 1e-8))
        view_1_norm = view_1 / max_val
        view_2_norm = view_2 / max_val

        overlay = np.zeros((*view_1.shape, 3), dtype=np.float32)
        overlay[..., 0] = view_1_norm
        overlay[..., 1] = view_2_norm

        overlays.append((overlay, (i1, j1), (i2, j2)))
        pairs_added.add(frozenset({(i1, j1), (i2, j2)}))

    # Add overlays that compare each extreme view with the reference view (if provided)
    if ref_index is not None:
        for (ei, ej) in extreme_indices:
            ref_pair = frozenset({(ei, ej), ref_index})
            # Skip if same view or already added above
            if ref_pair in pairs_added or (ei, ej) == ref_index:
                continue

            view_ext = rect_images[ei, ej].astype(np.float32)
            view_ref = rect_images[ref_index].astype(np.float32)

            max_val = float(max(np.max(view_ext), np.max(view_ref), 1e-8))
            view_ext_norm = view_ext / max_val
            view_ref_norm = view_ref / max_val

            overlay = np.zeros((*view_ext.shape, 3), dtype=np.float32)
            overlay[..., 0] = view_ext_norm
            overlay[..., 1] = view_ref_norm

            overlays.append((overlay, (ei, ej), ref_index))
            pairs_added.add(ref_pair)

    return overlays

def save_figures(figures: List[Figure], save_dir: str, dpi: int = 1000):
    """
    Persist matplotlib figures to disk with sanitized titles as filenames.

    Figures without titles fall back to ``figure_<index>.jpg``. Non-fatal save
    errors are reported to stdout so the rest of the batch can continue.
    """
    if not figures:
        print("No figures to save.")
        return
    os.makedirs(save_dir, exist_ok=True)
    for idx, fig in enumerate(figures, start=1):
        # Get the figure title in a type-safe way (avoid accessing private attributes directly)
        title_obj = getattr(cast(Any, fig), "_suptitle", None)
        title = slugify(title_obj.get_text()) if title_obj is not None else None
        if not title:
            # Try to get title from axes
            if fig.axes and fig.axes[0].get_title():
                title = fig.axes[0].get_title()
        
        # Use title if available, otherwise use default naming
        if title:
            filename = os.path.join(save_dir, f"{title}.jpg")
        else:
            filename = os.path.join(save_dir, f"figure_{idx}.jpg")
        
        try:
            fig.savefig(filename, format='jpg', dpi=dpi, bbox_inches='tight')
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")

def main():
    """Run the interactive light-field rectification workflow."""
    images, image_paths, hdr_flag = get_image_paths()

    if not images:
        print("No images loaded. Exiting.")
        return

    if hdr_flag:
        exp_time = input("Do you want to enter exposure times for the images or generate them automatically? (manual/auto): ").lower()
        if exp_time == 'auto':
            exp_times = None
        else:
            exp_times = np.zeros(len(images))
            for i in range(len(images)):
                exp_times[i] = float(input(f"Enter exposure time (in seconds) for image {image_paths[i]}: "))
        print("Assembling HDR image...")
        hdr_image = assemble_HDR(images, exp_times)
        images = [hdr_image]
        print("HDR image assembled successfully.")
    else:
        image_choice = input("Enter image number or name for processing (or press Enter for all): ").strip()
        if image_choice:
            if image_choice.isdigit():
                chosen_idx = int(image_choice)
                if 1 <= chosen_idx <= len(images):
                    images = [images[chosen_idx - 1]]
                else:
                    print(f"Invalid selection. Number must be between 1 and {len(images)}.")
                    images = []
            else:
                images = [im for i, im in enumerate(images) if image_choice in image_paths[i]]

    if not images:
        print("No images found for processing. Exiting.")
        return

    processed_images = [(im, min_max_normalize(im)) for im in images]
    
    # New: collect created matplotlib figures so we can optionally save them later
    created_figures: List[Figure] = []
    
    circles_chosen = None
    norm_im_chosen = None

    for orig_im, norm_im in processed_images:
        im_np = np.array(norm_im)
        
        # give user choice between using circle-detection to find viewpoints, or manually setting the viewpoints
        viewpoint_method = input("Select viewpoint selection method - circle-detection or manual? (c/m)").lower()
        if viewpoint_method == 'm':
            fig = plt.figure(1)
            created_figures.append(fig)
            plt.imshow(im_np, cmap='gray')
            fig = plt.figure(2)
            created_figures.append(fig)
            plt.imshow(np.clip(cv.equalizeHist(im_np), 0.85*255, 255), cmap='gray')
            plt.show()
            center_x = int(input(f"Enter center x coordinate of viewpoint: "))
            center_y = int(input(f"Enter center y coordinate of viewpoint: "))
            radius = int(input(f"Enter radius of viewpoint: ")) #Expected size of each viewpoint is 250 pixels->radius~125 pixels
            circles = np.array([[[center_x, center_y, radius]]])
            display_circles(norm_im, circles)
            if input("Use this image for further processing? (y/n): ").lower() == 'y':
                circles_chosen = circles
                norm_im_chosen = im_np
                break
            plt.close("all")
        else:
            roi_strategy = input("Enter ROI strategy (full, center, max_pixel, specific): ").lower()
            if roi_strategy == 'specific':
                fig = plt.figure()
                created_figures.append(fig)
                plt.imshow(im_np, cmap='gray')
                plt.show()
            roi, roi_offset = select_roi(im_np, roi_strategy, DEFAULT_SEARCH_RADIUS_MAX)

            circles = detect_and_adjust_circles(roi, roi_offset, DEFAULT_SEARCH_RADIUS_MIN, DEFAULT_SEARCH_RADIUS_MAX)
            
            if circles is None:
                print("No circles found in this image. ")
                continue
                
            display_circles(norm_im, circles)
            
            if input("Use this image for further processing? (y/n): ").lower() == 'y':
                circles_chosen = circles
                norm_im_chosen = im_np
                break
            plt.close("all")

    if circles_chosen is None or norm_im_chosen is None:
        print("No image selected for processing/Problem with constructing ROI structure. Exiting.")
        return
    
    windows = circles_chosen[0, :2]
    d = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * np.max(windows[:, 2])))
    
    view_grid_size = int(input(f"Enter view grid size *default {DEFAULT_VIEW_GRID_SIZE}*: ") or DEFAULT_VIEW_GRID_SIZE)
    views = extract_views(norm_im_chosen, windows[0], view_grid_size, d)
    
    # Display extracted views
    for i in range(view_grid_size):
        for j in range(view_grid_size):
            fig = plt.figure()
            created_figures.append(fig)
            plt.imshow(views[i, j], cmap="gray")
            plt.title(f"View {i+1}-{j+1}")
    plt.show()
    plt.close("all")

    rect_strat = int(input("Start rectification process. Choose the rectification strategy:\n1: Global Translation Rectification\n2: Pixel-wise Patch Rectification\n3: Combined (Global followed by Pixel-wise refinement)\nEnter your choice:"))
    if rect_strat == 1:
        rectification_strategy = GlobalTranslationRectificationStrategy(CORRELATION_PAD_WIDTH)
        rectification_strategy_global = None
        rectification_strategy_pixel = None
    elif rect_strat == 2:
        sim_window_size = int(input("Enter similarity window size (odd integer, default 41): ") or 41)
        rectification_strategy = PixelwisePatchRectificationStrategy(sim_window_size)
        rectification_strategy_global = None
        rectification_strategy_pixel = None
    else:
        rectification_strategy = None
        rectification_strategy_global = GlobalTranslationRectificationStrategy(CORRELATION_PAD_WIDTH)
        refinement_window_size = int(input("Enter similarity window size for pixel-wise refinement (odd integer, default 7): ") or 14)
        rectification_strategy_pixel = PixelwisePatchRectificationStrategy(refinement_window_size, zero_fill_roll=True)
    
    views_for_rect = np.asarray(views)
    _, ref_view_choice = resolve_reference_view(views_for_rect)

    # Cleanup of reference view for better correlation
    # view_mask = np.array([val if np.linalg.norm(np.array(np.unravel_index(i, views[ref_view_choice].shape)) - np.array(views[ref_view_choice].shape)//2) < windows[0, 2] else 0 for i, val in enumerate(views[ref_view_choice].flatten())])
    # view_mask = np.reshape(view_mask, views[ref_view_choice].shape)
    # views[ref_view_choice] = unsharp_mask(view_mask)

    #calculate similarity matrix
    if rect_strat == 1:
        assert isinstance(rectification_strategy, GlobalTranslationRectificationStrategy)
        sim_mat = rectification_strategy.compute_similarity_matrix(views, ref_view_choice, thresh=0.25)
    elif rect_strat == 2:
        assert isinstance(rectification_strategy, PixelwisePatchRectificationStrategy)
        sim_mat_pixelwise = rectification_strategy.compute_similarity_matrices(views, ref_view_choice, thresh=0.05)
    else:
        assert rectification_strategy is None
        assert isinstance(rectification_strategy_global, GlobalTranslationRectificationStrategy)
        assert isinstance(rectification_strategy_pixel, PixelwisePatchRectificationStrategy)

        # Compute global similarity and perform global rectification ONCE.
        sim_mat = rectification_strategy_global.compute_similarity_matrix(views, ref_view_choice, thresh=0.25)
        av_rect_global, rect_views_after_global = rectification_strategy_global.rectify(views, sim_mat, ref_view_choice)
        
        # Prepare pixel-wise similarity on the global-rectified views (refinement)
        refined_ref_view_choice = ref_view_choice
        sim_mat_pixelwise = rectification_strategy_pixel.compute_similarity_matrices(
            rect_views_after_global, refined_ref_view_choice, thresh=0.05
        )

    # Assemble and display similarity matrix
    # Display of similarity matrices
    if rect_strat == 1:
        assert isinstance(rectification_strategy, GlobalTranslationRectificationStrategy)
        fig_x, ax_x = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_x)
        fig_y, ax_y = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_y)
        for idx in range(view_grid_size):    
            for jdx in range(view_grid_size): 
                im_x = ax_x[idx,jdx].imshow(sim_mat[idx, jdx, :, :, 1], cmap="gray")
                im_y = ax_y[idx,jdx].imshow(sim_mat[idx, jdx, :, :, 0], cmap="gray")
                ax_x[idx, jdx].set_title(f"View {idx+1}-{jdx+1} X translation similarity")
                ax_y[idx, jdx].set_title(f"View {idx+1}-{jdx+1} Y translation similarity")
                fig_x.colorbar(im_x, ax=ax_x[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
                fig_y.colorbar(im_y, ax=ax_y[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
        fig_x.suptitle("X translation similarity")
        fig_y.suptitle("Y translation similarity")

        av_rect, rect_images = rectification_strategy.rectify(views, sim_mat, ref_view_choice)
    elif rect_strat == 2:
        assert isinstance(rectification_strategy, PixelwisePatchRectificationStrategy)
        disps_y = sim_mat_pixelwise.disps_y
        disps_x = sim_mat_pixelwise.disps_x
        # sim_mat_scores = sim_mat_pixelwise.scores
        fig_x, ax_x = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_x)
        fig_y, ax_y = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_y)
        for idx in range(view_grid_size):    
            for jdx in range(view_grid_size): 
                im_x = ax_x[idx,jdx].imshow(disps_x[:,:,idx, jdx], cmap="gray")
                im_y = ax_y[idx,jdx].imshow(disps_y[:,:,idx, jdx], cmap="gray")
                ax_x[idx, jdx].set_title(f"View {idx+1}-{jdx+1} X translation similarity")
                ax_y[idx, jdx].set_title(f"View {idx+1}-{jdx+1} Y translation similarity")
                fig_x.colorbar(im_x, ax=ax_x[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
                fig_y.colorbar(im_y, ax=ax_y[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
        fig_x.suptitle("X translation similarity")
        fig_y.suptitle("Y translation similarity")

        av_rect, rect_images = rectification_strategy.rectify(views, sim_mat_pixelwise, ref_view_choice)
    else:
        # --- Global similarity visualisation (uses sim_mat computed earlier) ---
        disps_y_global = sim_mat[:, :, :, :, 0]
        disps_x_global = sim_mat[:, :, :, :, 1]
        fig_x_global, ax_x_global = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_x_global)
        fig_y_global, ax_y_global = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_y_global)
        for idx in range(view_grid_size):
            for jdx in range(view_grid_size):
                im_x = ax_x_global[idx, jdx].imshow(disps_x_global[idx, jdx, :, :,], cmap="gray")
                im_y = ax_y_global[idx, jdx].imshow(disps_y_global[idx, jdx, :, :,], cmap="gray")
                ax_x_global[idx, jdx].set_title(f"Global: View {idx+1}-{jdx+1} X translation similarity")
                ax_y_global[idx, jdx].set_title(f"Global: View {idx+1}-{jdx+1} Y translation similarity")
                fig_x_global.colorbar(im_x, ax=ax_x_global[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
                fig_y_global.colorbar(im_y, ax=ax_y_global[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
        fig_x_global.suptitle("Global X translation similarity")
        fig_y_global.suptitle("Global Y translation similarity")

        # --- Display global-rectified views and overlays (collected for saving) ---
        for i in range(view_grid_size):
            for j in range(view_grid_size):
                fig = plt.figure()
                created_figures.append(fig)
                manager = plt.get_current_fig_manager()
                _fullscreen = getattr(manager, "full_screen_toggle", None)
                if callable(_fullscreen):
                    _fullscreen()
                plt.imshow(rect_views_after_global[i, j], cmap="gray")
                plt.title(f"Global Rectified View {i+1}-{j+1}")

        # Pass reference view to include reference comparisons
        extreme_overlays_global = build_extreme_view_overlays(rect_views_after_global, ref_index=ref_view_choice)
        for overlay, (i1, j1), (i2, j2) in extreme_overlays_global:
            fig = plt.figure()
            created_figures.append(fig)
            plt.imshow(overlay)
            plt.title(f"Global Extreme View Overlay ({i1 + 1}-{j1 + 1}) vs ({i2 + 1}-{j2 + 1})")

        fig = plt.figure()
        created_figures.append(fig)
        plt.imshow(av_rect_global, cmap="gray")
        manager = plt.get_current_fig_manager()
        _fullscreen = getattr(manager, "full_screen_toggle", None)
        if callable(_fullscreen):
            _fullscreen()
        plt.title("Averaged Global Rectified Image")

        fig = plt.figure()
        created_figures.append(fig)
        plt.imshow(np.clip(cv.equalizeHist(av_rect_global.astype('uint8')), 0.85*255, 255), cmap="gray")
        manager = plt.get_current_fig_manager()
        _fullscreen = getattr(manager, "full_screen_toggle", None)
        if callable(_fullscreen):
            _fullscreen()
        plt.title("Averaged Global Rectified Image - Equalized (clipped, 85%) for contrast")

        # --- Pixel-wise similarity visualisation (refinement stage) ---
        disps_y = sim_mat_pixelwise.disps_y
        disps_x = sim_mat_pixelwise.disps_x
        fig_x, ax_x = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_x)
        fig_y, ax_y = plt.subplots(nrows=view_grid_size, ncols=view_grid_size, sharex=True, sharey=True)
        created_figures.append(fig_y)
        for idx in range(view_grid_size):
            for jdx in range(view_grid_size):
                im_x = ax_x[idx, jdx].imshow(disps_x[:, :, idx, jdx], cmap="gray")
                im_y = ax_y[idx, jdx].imshow(disps_y[:, :, idx, jdx], cmap="gray")
                ax_x[idx, jdx].set_title(f"View {idx+1}-{jdx+1} X translation similarity")
                ax_y[idx, jdx].set_title(f"View {idx+1}-{jdx+1} Y translation similarity")
                fig_x.colorbar(im_x, ax=ax_x[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
                fig_y.colorbar(im_y, ax=ax_y[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
        fig_x.suptitle("Pixel-Wise X translation similarity")
        fig_y.suptitle("Pixel-Wise Y translation similarity")

        # Perform pixel-wise rectification ONCE on the global-rectified views
        assert isinstance(rectification_strategy_pixel, PixelwisePatchRectificationStrategy)
        av_rect, rect_images = rectification_strategy_pixel.rectify(rect_views_after_global, sim_mat_pixelwise, refined_ref_view_choice)
    
    # plt.show()    
    # Sharpen image (?)
    sharpen_sets = [(2.0, 2.0), (3.0, 2.0), (2.0, 3.0), (3.0, 3.0)]
    av_rect_sharp = []
    for i in range(len(sharpen_sets)):
        sigma, strength = sharpen_sets[i]
        av_rect_sharp.append(unsharp_mask(av_rect, sigma=sigma, strength=strength))

    # Display rectified images views
    for i in range(view_grid_size):
        for j in range(view_grid_size):
            fig = plt.figure()
            created_figures.append(fig)
            manager = plt.get_current_fig_manager()
            _fullscreen = getattr(manager, "full_screen_toggle", None)
            if callable(_fullscreen):
                _fullscreen()
            plt.imshow(rect_images[i, j], cmap="gray")
            plt.title(f"Rectified View {i+1}-{j+1}")
    # plt.show()
    # plt.close("all")

    # Build and display RGB overlays for the most extreme rectified views
    # + comparison to displacement maps (flag)
    disp_comp_flag = input("Do you want to see displacement comparison overlays for extreme views? (y/n): ").lower() == 'y'
    extreme_overlays = build_extreme_view_overlays(rect_images, ref_index=ref_view_choice)
    for overlay, (i1, j1), (i2, j2) in extreme_overlays:
        fig = plt.figure()
        created_figures.append(fig)
        plt.imshow(overlay)
        plt.title(
            "Extreme View Overlay "
            f"({i1 + 1}-{j1 + 1}) vs ({i2 + 1}-{j2 + 1})"
        )
        if disp_comp_flag:
            # Ensure displacement maps exist in this run; skip gracefully otherwise
            if 'disps_x' not in locals() or 'disps_y' not in locals():
                print("Displacement maps unavailable for this run — skipping displacement comparison overlays.")
                continue

            # Extract views and build boolean masks for X/Y displacements
            view1 = rect_views_after_global[i1, j1, :, :].astype(np.float32)
            view2 = rect_views_after_global[i2, j2, :, :].astype(np.float32)

            mask_x1 = np.abs(disps_x[:, :, i1, j1]) > 0
            mask_y1 = np.abs(disps_y[:, :, i1, j1]) > 0
            mask_x2 = np.abs(disps_x[:, :, i2, j2]) > 0
            mask_y2 = np.abs(disps_y[:, :, i2, j2]) > 0

            masked_x_view1 = mask_x1 * view1
            masked_y_view1 = mask_y1 * view1
            masked_x_view2 = mask_x2 * view2
            masked_y_view2 = mask_y2 * view2

            # Single, organized figure: two rows (view1 / view2), three columns (original, masked X, masked Y),
            # and a full-width overlay below.
            fig_all = plt.figure(constrained_layout=True, figsize=(12, 10))
            gs = fig_all.add_gridspec(3, 3)
            ax0 = fig_all.add_subplot(gs[0, 0])
            ax1 = fig_all.add_subplot(gs[0, 1])
            ax2 = fig_all.add_subplot(gs[0, 2])
            ax3 = fig_all.add_subplot(gs[1, 0])
            ax4 = fig_all.add_subplot(gs[1, 1])
            ax5 = fig_all.add_subplot(gs[1, 2])
            ax_overlay = fig_all.add_subplot(gs[2, :])

            im0 = ax0.imshow(view1, cmap="gray")
            ax0.set_title(f"View {i1+1}-{j1+1} — Original")
            im1 = ax1.imshow(masked_x_view1, cmap="gray")
            ax1.set_title("Masked X × Original")
            im2 = ax2.imshow(masked_y_view1, cmap="gray")
            ax2.set_title("Masked Y × Original")

            im3 = ax3.imshow(view2, cmap="gray")
            ax3.set_title(f"View {i2+1}-{j2+1} — Original")
            im4 = ax4.imshow(masked_x_view2, cmap="gray")
            ax4.set_title("Masked X × Original")
            im5 = ax5.imshow(masked_y_view2, cmap="gray")
            ax5.set_title("Masked Y × Original")

            ax_overlay.imshow(overlay)
            ax_overlay.set_title(f"Overlay (Red=({i1+1}-{j1+1}), Green=({i2+1}-{j2+1}))")

            # Add colorbars for the masked maps to emphasise displacement regions
            fig_all.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.035, pad=0.02)
            fig_all.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.035, pad=0.02)
            fig_all.colorbar(im4, ax=ax4, orientation='vertical', fraction=0.035, pad=0.02)
            fig_all.colorbar(im5, ax=ax5, orientation='vertical', fraction=0.035, pad=0.02)

            fig_all.suptitle(f"Displacement masks and original views: ({i1+1}-{j1+1}) vs ({i2+1}-{j2+1})")
            created_figures.append(fig_all)

    # Display the averaged rectified image
    fig = plt.figure()
    created_figures.append(fig)
    plt.imshow(av_rect, cmap="gray")
    manager = plt.get_current_fig_manager()
    _fullscreen = getattr(manager, "full_screen_toggle", None)
    if callable(_fullscreen):
        _fullscreen()
    plt.title("Averaged Rectified Image")
    fig = plt.figure()
    created_figures.append(fig)
    plt.imshow(np.clip(cv.equalizeHist(av_rect.astype('uint8')), 0.85*255, 255), cmap="gray")
    manager = plt.get_current_fig_manager()
    _fullscreen = getattr(manager, "full_screen_toggle", None)
    if callable(_fullscreen):
        _fullscreen()
    plt.title("Averaged Rectified Image - Equalized (clipped, 85%) for contrast")
    for i in range(len(sharpen_sets)):
        fig = plt.figure()
        created_figures.append(fig)
        plt.imshow(av_rect_sharp[i], cmap="gray")
        manager = plt.get_current_fig_manager()
        _fullscreen = getattr(manager, "full_screen_toggle", None)
        if callable(_fullscreen):
            _fullscreen()
        plt.title(f"Averaged Rectified Image (Sharpened) with parameters - sigma={sharpen_sets[i][0]}, strength={sharpen_sets[i][1]}")
    plt.show()

    # New: Ask user whether to save displayed figures
    if created_figures:
        save_opt = input("Do you want to save the displayed figures as high-resolution JPG files? (y/n): ").lower()
        if save_opt == 'y':
            save_dir = input("Enter directory path to save figures (will be created if not exists): ").strip()
            if save_dir:
                save_figures(created_figures, save_dir, dpi=1000)
            else:
                print("No directory entered. Skipping save.")
    # Close all figures afterwards
    plt.close("all")

if __name__ == "__main__":
    main()
