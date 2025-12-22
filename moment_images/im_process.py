import os
import PIL
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, List, Optional, Any, cast
from itertools import combinations
from django.utils.text import slugify
try:
    # Prefer absolute import when running as a script
    from rectification_strategy import GlobalTranslationRectificationStrategy, PixelwisePatchRectificationStrategy
except Exception:
    # When imported as a package, use relative import
    from .rectification_strategy import GlobalTranslationRectificationStrategy, PixelwisePatchRectificationStrategy

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

def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load an image from the specified path.
    
    :param image_path: Path to the image file.
    :return: PIL Image object or None if the image cannot be loaded.
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

def min_max_normalize(image: Image.Image, min_val: int = 0, max_val: int = 255) -> Image.Image:
    """
    Normalize the image using min-max normalization. 
    
    :param image: PIL Image object.
    :return: Normalized PIL Image object.
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
    Apply unsharp masking while keeping the output normalized to the same
    range as the other views (0-255).
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
    Assemble a High Dynamic Range (HDR) image from a list of images using Debevec's method.

    :param images: List of PIL Image objects.
    :param exposure_times: Optional array of exposure times corresponding to the images.
    :return: Tone-mapped HDR image as a PIL Image object.
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
    Prompt the user for an image directory, load images, and handle HDR assembly.

    Returns:
        A tuple containing:
        - A list of loaded PIL Image objects.
        - A list of corresponding image file paths.
        - A boolean indicating if the images were from an HDR directory.
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
    Select a Region of Interest (ROI) from the image.

    Args:
        image_np: The input image as a NumPy array.
        strategy: The ROI selection strategy ('center', 'max_pixel', 'full', or 'specific').
        search_radius: The search radius to define the ROI size.

    Returns:
        A tuple containing:
        - The selected ROI as a NumPy array.
        - The top-left coordinates (row, col) of the ROI in the original image, or None if the full image is used.
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
    Detect circles in a given ROI and adjust their coordinates to the full image frame.

    Args:
        roi: The Region of Interest (ROI) to search for circles in.
        roi_offset: The (row, col) offset of the ROI relative to the full image.
        min_radius: The minimum radius of circles to detect.
        max_radius: The maximum radius of circles to detect.

    Returns:
        A NumPy array of detected circles (x, y, radius), or None if no circles are found.
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
    Draw detected circles on an image and display it.

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
    Extract a grid of square views from an image based on a reference circle.

    Args:
        image_np: The input image as a NumPy array.
        base_circle: The reference circle (x, y, radius) to build the grid around.
        num_views: The number of views to extract in each dimension (e.g., 3 for a 3x3 grid).
        view_dist: The distance between the centers of adjacent views.

    Returns:
        A NumPy array containing the extracted views.
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


def build_extreme_view_overlays(rect_images: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """
    Create RGB overlays comparing each pair of extreme rectified views.

    The red channel is populated by the first view in the pair and the green
    channel by the second, enabling pixel-by-pixel inspection of differences.
    Values are normalized to [0, 1] to ensure consistent display regardless of
    the original image scale.

    Args:
        rect_images: Rectified grayscale views arranged in a square grid of
            shape (N, N, H, W).

    Returns:
        A list of tuples containing the overlay image and the paired indices.
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

    return overlays

def save_figures(figures: List[Figure], save_dir: str, dpi: int = 600):
    """
    Save a list of matplotlib Figure objects as high-resolution JPG files.
    The filename is based on the figure's title.
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
    """Main function to run the image processing pipeline."""
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

    rect_strat = int(input("Start rectification process. Choose the rectification strategy:\n1: Global Translation Rectification\n2: Pixel-wise Patch Rectification\nEnter your choice:"))
    if rect_strat == 1:
        rectification_strategy = GlobalTranslationRectificationStrategy(CORRELATION_PAD_WIDTH)
    else:
        sim_window_size = int(input("Enter similarity window size (odd integer, default 61): ") or 61)
        rectification_strategy = PixelwisePatchRectificationStrategy(sim_window_size)
    
    ref_view_choice = rectification_strategy.select_reference_view(views)

    # Cleanup of reference view for better correlation
    view_mask = np.array([val if np.linalg.norm(np.array(np.unravel_index(i, views[ref_view_choice].shape)) - np.array(views[ref_view_choice].shape)//2) < windows[0, 2] else 0 for i, val in enumerate(views[ref_view_choice].flatten())])
    view_mask = np.reshape(view_mask, views[ref_view_choice].shape)
    views[ref_view_choice] = unsharp_mask(view_mask)

    #calculate similarity matrix
    if rect_strat == 1:
        assert isinstance(rectification_strategy, GlobalTranslationRectificationStrategy)
        sim_mat = rectification_strategy.compute_similarity_matrix(views, ref_view_choice)
    else:
        assert isinstance(rectification_strategy, PixelwisePatchRectificationStrategy)
        sim_mat_pixelwise = rectification_strategy.compute_similarity_matrices(views, ref_view_choice)
    
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
    else:
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
    extreme_overlays = build_extreme_view_overlays(rect_images)
    for overlay, (i1, j1), (i2, j2) in extreme_overlays:
        fig = plt.figure()
        created_figures.append(fig)
        plt.imshow(overlay)
        plt.title(
            "Extreme View Overlay "
            f"({i1 + 1}-{j1 + 1}) vs ({i2 + 1}-{j2 + 1})"
        )
    
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
                save_figures(created_figures, save_dir, dpi=600)
            else:
                print("No directory entered. Skipping save.")
    # Close all figures afterwards
    plt.close("all")

if __name__ == "__main__":
	main()
