import os
import PIL
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from typing import Tuple, List, Optional

# --- Constants ---
DEFAULT_IMAGE_DIR = 'moment_images/images'
GAUSSIAN_BLUR_KERNEL = (51, 51)
GAUSSIAN_BLUR_SIGMA = 5
HOUGH_CIRCLES_PARAM1 = 50
HOUGH_CIRCLES_PARAM2 = 30
HOUGH_CIRCLES_MIN_DIST = 20
VIEW_GRID_SIZE = 3
VIEW_DISTANCE_SCALE_FACTOR = 1.05
CORRELATION_PAD_WIDTH = 30
DEFAULT_SEARCH_RADIUS_MIN = 90
DEFAULT_SEARCH_RADIUS_MAX = 120

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

def assemble_HDR(images: List[Image.Image], weights: Optional[np.ndarray] = None) -> Image.Image:
    """
    Assemble a High Dynamic Range (HDR) image from a list of images.
    
    :param images: List of PIL Image objects.
    :param weights: Optional weights for each image as a NumPy array.
    :return: Assembled HDR image as a PIL Image object.
    """
    if not images:
        raise ValueError("No images provided for HDR assembly.")
    
    image_arrays = [np.array(min_max_normalize(image)) for image in images]
    
    if weights is None:
        hdr_weights = np.ones(len(images)) / len(images)
    else:
        if weights.ndim != 1 or len(weights) != len(images):
            raise ValueError("Weights must be a 1D array matching the number of images.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        hdr_weights = weights / np.sum(weights)

    stacked_images = np.stack(image_arrays, axis=-1)
    hdr_image_array = np.average(stacked_images, axis=-1, weights=hdr_weights).astype(np.uint8)
    
    return Image.fromarray(hdr_image_array)

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
            print(f"Successfully loaded image: {f}")
        else:
            print(f"Failed to load image: {f}")
            
    return images, image_paths, hdr_flag

def select_roi(image_np: np.ndarray, strategy: str, search_radius: int) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Select a Region of Interest (ROI) from the image.

    Args:
        image_np: The input image as a NumPy array.
        strategy: The ROI selection strategy ('center', 'max_pixel', or 'full').
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

def calculate_similarity_matrix(views: np.ndarray, ref_view_idx: Tuple[int, int], pad_width: int) -> np.ndarray:
    """
    Calculate the translational similarity between all pairs of views.

    Args:
        views: A NumPy array of image views.
        pad_width: The padding to add to each view for correlation.

    Returns:
        A 5D NumPy array representing the similarity matrix with translation offsets.
    """
    num_views = views.shape[0]
    sim_mat = pad_width * np.ones((num_views, num_views, num_views, num_views, 2))
    check_all_views = input("Check all views for similarity? (y/n): ").lower()
    for i in range(num_views):
        if check_all_views != "y" and i != ref_view_idx[0]:
            continue
        for j in range(num_views):
            if check_all_views != "y" and j != ref_view_idx[1]:
                continue
            for k in range(num_views):
                for l in range(num_views):
                    view1 = views[i, j]
                    view2 = views[k, l]
                    
                    # Pad view1 for correlation
                    padded_view1 = np.pad(view1, pad_width, mode='wrap')
                    
                    # Calculate correlation
                    cor = correlate2d(padded_view1, view2, mode='valid')
                    max_cor_loc = np.unravel_index(np.argmax(cor), cor.shape)
                    
                    sim_mat[i, j, k, l, :] = max_cor_loc
                    
    return sim_mat

def rectify_and_average_views(views: np.ndarray, sim_mat: np.ndarray, ref_view_idx: Tuple[int, int], cor_shape: Tuple[int, int]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align views to a reference view and compute their average.

    Args:
        views: The array of image views.
        sim_mat: The similarity matrix containing translation offsets.
        ref_view_idx: The (row, col) index of the reference view.
        cor_shape: The shape of the correlation matrix.

    Returns:
        The averaged image of the aligned views as a NumPy array.
    """
    num_views = views.shape[0]
    rect_images = np.zeros_like(views)
    
    ref_i, ref_j = ref_view_idx
    
    for i in range(num_views):
        for j in range(num_views):
            if i == ref_i and j == ref_j:
                rect_images[i, j] = views[i, j]
                continue
            
            dx = sim_mat[ref_i, ref_j, i, j, 1] - cor_shape[1] // 2
            dy = sim_mat[ref_i, ref_j, i, j, 0] - cor_shape[0] // 2
            
            warp_mat = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            rect_images[i, j] = cv.warpAffine(views[i, j], warp_mat, (views.shape[3], views.shape[2]))

    return (np.mean(rect_images, axis=(0, 1)), rect_images)

def main():
    """Main function to run the image processing pipeline."""
    images, image_paths, hdr_flag = get_image_paths()

    if not images:
        print("No images loaded. Exiting.")
        return

    if hdr_flag:
        print("Assembling HDR image...")
        hdr_image = assemble_HDR(images)
        images = [hdr_image]
        print("HDR image assembled successfully.")
    else:
        image_name = input("Enter image name for processing (or press Enter for all): ")
        if image_name:
            images = [im for i, im in enumerate(images) if image_name in image_paths[i]]

    if not images:
        print("No images found for processing. Exiting.")
        return

    processed_images = [(im, min_max_normalize(im)) for im in images]
    
    circles_chosen = None
    norm_im_chosen = None

    for orig_im, norm_im in processed_images:
        im_np = np.array(norm_im)
        
        roi_strategy = input("Enter ROI strategy (full, center, max_pixel): ").lower()
        roi, roi_offset = select_roi(im_np, roi_strategy, DEFAULT_SEARCH_RADIUS_MAX)
        
        circles = detect_and_adjust_circles(roi, roi_offset, DEFAULT_SEARCH_RADIUS_MIN, DEFAULT_SEARCH_RADIUS_MAX)
        
        if circles is None:
            print("No circles found in this image.")
            continue
            
        display_circles(norm_im, circles)
        
        if input("Use this image for further processing? (y/n): ").lower() == 'y':
            circles_chosen = circles
            norm_im_chosen = im_np
            break
        plt.close("all")

    if circles_chosen is None or norm_im_chosen is None:
        print("No image selected for processing. Exiting.")
        return

    windows = circles_chosen[0, :2]
    d = int(np.floor(VIEW_DISTANCE_SCALE_FACTOR * 2 * np.max(windows[:, 2])))
    
    views = extract_views(norm_im_chosen, windows[0], VIEW_GRID_SIZE, d)
    
    # Display extracted views
    for i in range(VIEW_GRID_SIZE):
        for j in range(VIEW_GRID_SIZE):
            plt.figure()
            plt.imshow(views[i, j], cmap="gray")
            plt.title(f"View {i+1}-{j+1}")
    plt.show()
    plt.close("all")

    ref_view_choice = (1, 1) # Default to center view as reference
    sim_mat = calculate_similarity_matrix(views, ref_view_choice, CORRELATION_PAD_WIDTH)
    
    # Assemble and dispaly similarity matrix
    # Display of similarity matrices
    fig_x, ax_x = plt.subplots(nrows=VIEW_GRID_SIZE, ncols=VIEW_GRID_SIZE, sharex=True, sharey=True)
    fig_y, ax_y = plt.subplots(nrows=VIEW_GRID_SIZE, ncols=VIEW_GRID_SIZE, sharex=True, sharey=True)
    for idx in range(VIEW_GRID_SIZE):    
        for jdx in range(VIEW_GRID_SIZE): 
            im_x = ax_x[idx,jdx].imshow(sim_mat[idx, jdx, :, :, 1], cmap="gray")
            im_y = ax_y[idx,jdx].imshow(sim_mat[idx, jdx, :, :, 0], cmap="gray")
            ax_x[idx, jdx].set_title(f"View {idx+1}-{jdx+1} X translation similarity")
            ax_y[idx, jdx].set_title(f"View {idx+1}-{jdx+1} Y translation similarity")
            fig_x.colorbar(im_x, ax=ax_x[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
            fig_y.colorbar(im_y, ax=ax_y[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
    fig_x.suptitle("X translation similarity")
    fig_y.suptitle("Y translation similarity")
    # plt.show()

    # This is needed for the warp affine later
    padded_view_for_corr_shape = np.pad(views[ref_view_choice], CORRELATION_PAD_WIDTH, mode='wrap')
    cor_shape = correlate2d(padded_view_for_corr_shape, views[ref_view_choice], mode='valid').shape

    av_rect, rect_images = rectify_and_average_views(views, sim_mat, ref_view_choice, cor_shape)
    
    # Display rectified images views
    for i in range(VIEW_GRID_SIZE):
        for j in range(VIEW_GRID_SIZE):
            plt.figure()
            plt.imshow(rect_images[i, j], cmap="gray")
            plt.title(f"Rectified View {i+1}-{j+1}")
    # plt.show()
    # plt.close("all")
    
    # Display the averaged rectified image
    plt.figure()
    plt.imshow(av_rect, cmap="gray")
    plt.title("Averaged Rectified Image")
    plt.show()
    plt.close("all")

if __name__ == "__main__":
    main()
