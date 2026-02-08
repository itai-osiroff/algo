"""
Script to validate rectification results against new images from other cameras.
Allows interactive processing (rotation, flipping, cropping) of input images.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, Scale
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2 as cv
from PIL import Image

# Ensure local modules can be imported
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import im_process
    from unpack_workspace import load_workspace_variables
else:
    from . import im_process
    from .unpack_workspace import load_workspace_variables

class InteractiveImageAdjuster:
    """GUI to adjust an image (rotate, flip, crop) before processing."""

    def __init__(self, master, image_path, reference_image=None, title="Adjust Image", camera_type=None):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title(title)
        self.top.geometry("1100x800") # Increased size for controls

        self.original_image_pil = im_process.load_image(image_path)
        if self.original_image_pil is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Geometric state (rotated/flipped)
        # We work with the original PIL image for geometric transforms to handle high bit depth if possible
        # but PIL rotation might convert 16-bit to something else depending on version.
        # Safest is to keep it in PIL for now.
        self.geo_image_pil = self.original_image_pil.copy()
        
        # Apply default operations based on camera type
        self._apply_default_operations(camera_type)

        # Calculate raw min/max for the sliders
        # We use numpy for this
        raw_arr = np.asarray(self.original_image_pil)
        self.raw_min_val = float(raw_arr.min())
        self.raw_max_val = float(raw_arr.max())

        # Display range state (min/max intensity)
        self.display_min = self.raw_min_val
        self.display_max = self.raw_max_val

        # Reference image (numpy array, likely grayscale)
        self.reference_image = reference_image

        # Transformations state
        self.rotation = 0 # 0, 90, 180, 270
        self.flip_h = False
        self.flip_v = False
        self.crop_box = None # (left, upper, right, lower)

        self.show_ref_var = tk.BooleanVar(value=False)

        self.result_image = None
        self.is_done = False

        self._init_ui()
        self._update_display()

    def _apply_default_operations(self, camera_type):
        """Apply default geometric operations based on camera type."""
        if camera_type == "validation":
            # Right rotation -> flip vertical
            self.geo_image_pil = self.geo_image_pil.rotate(-90, expand=True)  # Right rotation
            self.geo_image_pil = self.geo_image_pil.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertical
        elif camera_type == "thorcam":
            # Right rotation
            self.geo_image_pil = self.geo_image_pil.rotate(-90, expand=True)  # Right rotation

    def _init_ui(self):
        # Toolbar Frame
        toolbar_frame = tk.Frame(self.top)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Geometric controls
        geo_frame = tk.LabelFrame(toolbar_frame, text="Geometry")
        geo_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        tk.Button(geo_frame, text="Rot L", command=lambda: self._rotate(90)).pack(side=tk.LEFT, padx=2)
        tk.Button(geo_frame, text="Rot R", command=lambda: self._rotate(-90)).pack(side=tk.LEFT, padx=2)
        tk.Button(geo_frame, text="Flip H", command=self._flip_h).pack(side=tk.LEFT, padx=2)
        tk.Button(geo_frame, text="Flip V", command=self._flip_v).pack(side=tk.LEFT, padx=2)
        self.btn_crop = tk.Button(geo_frame, text="Crop: OFF", command=self._toggle_crop)
        self.btn_crop.pack(side=tk.LEFT, padx=2)

        # Enhancement controls (Min/Max Intensity)
        enh_frame = tk.LabelFrame(toolbar_frame, text="Range")
        enh_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # We need a range that makes sense.
        # If the image is 16-bit, values can be 0-65535.
        # Resolution needs to be granular enough.

        tk.Label(enh_frame, text="Min:").pack(side=tk.LEFT)
        self.scale_min = Scale(enh_frame, from_=self.raw_min_val, to=self.raw_max_val,
                               resolution=(self.raw_max_val - self.raw_min_val)/200.0 if self.raw_max_val != self.raw_min_val else 1,
                               orient=tk.HORIZONTAL, command=self._on_scale_change)
        self.scale_min.set(self.raw_min_val)
        self.scale_min.pack(side=tk.LEFT, padx=2)

        tk.Label(enh_frame, text="Max:").pack(side=tk.LEFT)
        self.scale_max = Scale(enh_frame, from_=self.raw_min_val, to=self.raw_max_val,
                               resolution=(self.raw_max_val - self.raw_min_val)/200.0 if self.raw_max_val != self.raw_min_val else 1,
                               orient=tk.HORIZONTAL, command=self._on_scale_change)
        self.scale_max.set(self.raw_max_val)
        self.scale_max.pack(side=tk.LEFT, padx=2)

        # Reference checkbox
        if self.reference_image is not None:
            tk.Checkbutton(toolbar_frame, text="Show Ref", variable=self.show_ref_var, command=self._update_display).pack(side=tk.LEFT, padx=10)

        tk.Button(toolbar_frame, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar_frame, text="Done", command=self._finish, bg="green", fg="white").pack(side=tk.RIGHT, padx=5)

        # Plot Area
        self.fig = Figure(figsize=(10, 8), dpi=100)
        # self.ax will be created in _update_display
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Rectangle Selector for cropping (initialized in _update_display)
        self.rs = None
        self.crop_active = False

    def _on_scale_change(self, _):
        self.display_min = self.scale_min.get()
        self.display_max = self.scale_max.get()

        # Ensure min < max
        if self.display_min >= self.display_max:
             # Prevent crossing
             # Either clamp or ignore?
             # If user drags min past max, maybe push max?
             pass

        self._update_display()

    def _rotate(self, angle):
        self.geo_image_pil = self.geo_image_pil.rotate(angle, expand=True)
        self.crop_box = None
        self._update_display()

    def _flip_h(self):
        self.geo_image_pil = self.geo_image_pil.transpose(Image.FLIP_LEFT_RIGHT)
        self.crop_box = None
        self._update_display()

    def _flip_v(self):
        self.geo_image_pil = self.geo_image_pil.transpose(Image.FLIP_TOP_BOTTOM)
        self.crop_box = None
        self._update_display()

    def _toggle_crop(self):
        self.crop_active = not self.crop_active
        if self.crop_active:
            self.btn_crop.config(text="Crop: ON", bg="yellow")
            if self.rs: self.rs.set_active(True)
        else:
            self.btn_crop.config(text="Crop: OFF", bg="SystemButtonFace")
            if self.rs: self.rs.set_active(False)

    def _on_select(self, eclick, erelease):
        extent = self.rs.extents # xmin, xmax, ymin, ymax
        self.crop_box = (int(extent[0]), int(extent[2]), int(extent[1]), int(extent[3]))
        print(f"Crop selected: {self.crop_box}")

    def _reset(self):
        self.geo_image_pil = self.original_image_pil.copy()
        self.scale_min.set(self.raw_min_val)
        self.scale_max.set(self.raw_max_val)
        self.display_min = self.raw_min_val
        self.display_max = self.raw_max_val
        self.crop_box = None
        self.crop_active = False
        self.btn_crop.config(text="Crop: OFF", bg="SystemButtonFace")
        self._update_display()

    def _get_display_image(self):
        # Convert geometric PIL image to numpy
        img_arr = np.asarray(self.geo_image_pil).astype(np.float32)

        # Apply normalization based on display_min / display_max
        # Clip to range
        clipped = np.clip(img_arr, self.display_min, self.display_max)

        # Scale to 0-255
        divisor = (self.display_max - self.display_min)
        if divisor <= 1e-8:
            divisor = 1.0 # avoid div by zero, output will be 0

        normalized = (clipped - self.display_min) / divisor * 255.0

        return normalized.astype(np.uint8)

    def _update_display(self):
        if not hasattr(self, 'fig'): return

        self.fig.clf()

        show_ref = self.show_ref_var.get() and self.reference_image is not None

        ax_edit = None

        if show_ref:
            ax_ref = self.fig.add_subplot(121)
            ax_ref.imshow(self.reference_image, cmap='gray')
            ax_ref.set_title("Reference (Global Avg)")

            ax_edit = self.fig.add_subplot(122)
        else:
            ax_edit = self.fig.add_subplot(111)

        display_img = self._get_display_image()

        ax_edit.imshow(display_img, cmap='gray')
        ax_edit.set_title("Adjust Image (Min/Max Scaling)")

        # Re-initialize selector
        self.rs = RectangleSelector(
            ax_edit, self._on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        if self.crop_active:
            self.rs.set_active(True)
        else:
            self.rs.set_active(False)

        self.canvas.draw()

    def _finish(self):
        # We need to return the final image with geometry AND the intensity scaling applied.
        # First, we need to handle crop on the geometric image if it exists.

        final_pil = self.geo_image_pil
        if self.crop_box:
            xmin, xmax, ymin, ymax = self.rs.extents
            w, h = final_pil.size
            xmin = max(0, min(xmin, w))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h))
            ymax = max(0, min(ymax, h))

            if xmax > xmin and ymax > ymin:
                final_pil = final_pil.crop((xmin, ymin, xmax, ymax))

        # Now convert to numpy and apply the intensity scaling
        img_arr = np.asarray(final_pil).astype(np.float32)

        # Apply normalization based on display_min / display_max
        clipped = np.clip(img_arr, self.display_min, self.display_max)
        divisor = (self.display_max - self.display_min)
        if divisor <= 1e-8:
            divisor = 1.0

        normalized = (clipped - self.display_min) / divisor * 255.0

        self.result_image = normalized.astype(np.uint8)
        self.is_done = True
        self.top.destroy()
        self.master.quit()

    def run(self):
        # Wait for window to close
        self.top.wait_window()
        return self.result_image

def auto_align_images(ref_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Align target_img to ref_img using ECC (Translation).
    Returns aligned_img.
    """
    try:
        # Resize target to match ref size if different (for initial estimation)
        # But warpAffine needs same size output.
        h, w = ref_img.shape

        # Ensure target is same size before alignment?
        # Typically rectification produces same size images.
        # But if the user cropped the target, it might be different size.
        # If sizes differ, we should probably resize target to match ref first?
        # Or place target in center?
        # The simplest is resize.
        target_resized = cv.resize(target_img, (w, h), interpolation=cv.INTER_LINEAR)

        ref_gray = ref_img.astype(np.float32)
        target_gray = target_resized.astype(np.float32)

        warp_mode = cv.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 500, 1e-5)

        (cc, warp_matrix) = cv.findTransformECC(ref_gray, target_gray, warp_matrix, warp_mode, criteria)

        # Use WARP_INVERSE_MAP as ECC finds the map from template (ref) to input (target)
        aligned_img = cv.warpAffine(target_resized, warp_matrix, (w, h), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

        print(f"Auto-alignment successful. Correlation: {cc:.4f}")
        return aligned_img.astype(target_img.dtype)

    except Exception as e:
        print(f"Auto-alignment failed: {e}")
        # Fallback to resized version
        if 'target_resized' in locals():
             return target_resized
        return cv.resize(target_img, (ref_img.shape[1], ref_img.shape[0]))


def compare_images(rectified_img: np.ndarray, input_img: np.ndarray, input_name: str) -> list[Figure]:
    """
    Compare input image to rectified image.
    Resizes input_img to match rectified_img shape.
    Returns list of figures.
    """
    figures = []

    # 1. Resize input to match rectified (if not already aligned)
    h, w = rectified_img.shape

    if input_img.shape != (h, w):
        input_resized = cv.resize(input_img, (w, h), interpolation=cv.INTER_LINEAR)
    else:
        input_resized = input_img

    # Ensure float for processing
    rect_norm = im_process.min_max_normalize(rectified_img).astype(np.float32) / 255.0
    input_norm = im_process.min_max_normalize(input_resized).astype(np.float32) / 255.0

    # 2. Overlay (Anaglyph)
    # Red = Rectified, Green = Input.
    overlay2 = np.zeros((h, w, 3), dtype=np.float32)
    overlay2[..., 0] = rect_norm
    overlay2[..., 1] = input_norm

    fig1 = plt.figure(figsize=(10, 8))
    plt.imshow(overlay2)
    plt.title(f"Overlay: Red=Rectified, Green={input_name}")
    # Add a custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='green', lw=4)]
    plt.legend(custom_lines, ['Rectified', input_name], loc='upper right')
    figures.append(fig1)

    # 3. Difference Map
    diff = np.abs(rect_norm - input_norm)
    fig2 = plt.figure(figsize=(10, 8))
    im = plt.imshow(diff, cmap='viridis') # viridis is good for intensity
    im_process.add_colorbar(im)
    plt.title(f"Absolute Difference: |Rectified - {input_name}|")
    figures.append(fig2)

    # 4. Side by Side
    fig3, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rect_norm, cmap='gray')
    axes[0].set_title("Rectified Image")
    axes[1].imshow(input_norm, cmap='gray')
    axes[1].set_title(f"{input_name}")
    figures.append(fig3)

    return figures

def main():
    root = tk.Tk()
    root.withdraw() # Hide main window

    # 1. Load Workspace
    print("Please select the im_process workspace file (.npz).")
    workspace_path = filedialog.askopenfilename(title="Select Workspace File", filetypes=[("NPZ Files", "*.npz")])
    if not workspace_path:
        print("No workspace selected. Exiting.")
        return

    try:
        workspace = load_workspace_variables(workspace_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load workspace: {e}")
        return

    # Extract rectified image
    if 'average_rectified_image' not in workspace:
        messagebox.showerror("Error", "Workspace does not contain 'average_rectified_image'.")
        return

    rectified_img = workspace['average_rectified_image']

    # 2. Load Validation Image
    print("Please select the Validation Camera image.")
    val_path = filedialog.askopenfilename(title="Select Validation Image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")])
    if not val_path:
        print("No validation image selected. Exiting.")
        return

    # 3. Load Thorcam Image
    print("Please select the Thorcam image.")
    thor_path = filedialog.askopenfilename(title="Select Thorcam Image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")])
    if not thor_path:
        print("No thorcam image selected. Exiting.")
        return

    # 4. Process Validation Image
    print("Adjusting Validation Image...")
    adj_val = InteractiveImageAdjuster(root, val_path, reference_image=rectified_img, title="Adjust Validation Image", camera_type="validation")
    val_processed = adj_val.run()
    if val_processed is None:
        print("Validation image processing cancelled.")
        return

    # 5. Process Thorcam Image
    print("Adjusting Thorcam Image...")
    adj_thor = InteractiveImageAdjuster(root, thor_path, reference_image=rectified_img, title="Adjust Thorcam Image", camera_type="thorcam")
    thor_processed = adj_thor.run()
    if thor_processed is None:
        print("Thorcam image processing cancelled.")
        return

    # 6. Auto Align
    if messagebox.askyesno("Auto Align", "Do you want to automatically align the images to the rectified reference?"):
        print("Aligning Validation Image...")
        val_processed = auto_align_images(rectified_img, val_processed)
        print("Aligning Thorcam Image...")
        thor_processed = auto_align_images(rectified_img, thor_processed)

    # 7. Compare
    print("Generating comparisons...")
    figs_val = compare_images(rectified_img, val_processed, "Validation")
    figs_thor = compare_images(rectified_img, thor_processed, "Thorcam")

    all_figures = figs_val + figs_thor

    # Show figures
    plt.show()

    # 8. Save
    if messagebox.askyesno("Save", "Do you want to save the figures and workspace?"):
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            # Save Figures
            im_process.save_figures(all_figures, save_dir)

            # Save Workspace
            save_path = os.path.join(save_dir, "validation_workspace.npz")
            vars_to_save = {
                "rectified_image": rectified_img,
                "validation_input_original": np.asarray(Image.open(val_path).convert('L')),
                "validation_processed": val_processed,
                "thorcam_input_original": np.asarray(Image.open(thor_path).convert('L')),
                "thorcam_processed": thor_processed
            }
            im_process.save_workspace_variables(save_path, vars_to_save)
            messagebox.showinfo("Success", f"Saved to {save_dir}")

    root.destroy()

if __name__ == "__main__":
    main()
