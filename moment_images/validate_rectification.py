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


class InteractiveScaleAdjuster:
    """GUI to anisotropically scale an image against an optional reference."""

    def __init__(self, master, input_image: np.ndarray, reference_image=None, title="Scale Correct Image"):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title(title)
        self.top.geometry("1100x800")

        self.input_image = np.asarray(input_image)
        if self.input_image.ndim == 3:
            self.input_image = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)

        self.reference_image = reference_image
        if self.reference_image is not None and self.reference_image.ndim == 3:
            self.reference_image = cv.cvtColor(self.reference_image, cv.COLOR_BGR2GRAY)

        self.scale_x_var = tk.DoubleVar(value=1.0)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        self.show_ref_var = tk.BooleanVar(value=False)
        self.resize_for_downstream_var = tk.BooleanVar(value=True)
        self.resize_for_downstream = True

        self.result_image = None
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self._init_ui()
        self._update_display()

    def _init_ui(self):
        toolbar_frame = tk.Frame(self.top)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        scale_frame = tk.LabelFrame(toolbar_frame, text="Scale")
        scale_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        tk.Label(scale_frame, text="X Scale:").pack(side=tk.LEFT, padx=(4, 0))
        self.scale_x = Scale(
            scale_frame,
            from_=0.2,
            to=3.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.scale_x_var,
            command=self._on_scale_change,
            length=220,
        )
        self.scale_x.pack(side=tk.LEFT, padx=4)

        tk.Label(scale_frame, text="Y Scale:").pack(side=tk.LEFT, padx=(8, 0))
        self.scale_y = Scale(
            scale_frame,
            from_=0.2,
            to=3.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.scale_y_var,
            command=self._on_scale_change,
            length=220,
        )
        self.scale_y.pack(side=tk.LEFT, padx=4)

        if self.reference_image is not None:
            tk.Checkbutton(
                toolbar_frame,
                text="Show Ref",
                variable=self.show_ref_var,
                command=self._update_display,
            ).pack(side=tk.LEFT, padx=10)

        tk.Checkbutton(
            toolbar_frame,
            text="Resize To Ref Downstream",
            variable=self.resize_for_downstream_var,
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(toolbar_frame, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar_frame, text="Done", command=self._finish, bg="green", fg="white").pack(side=tk.RIGHT, padx=5)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_scale_change(self, _):
        self._update_display()

    @staticmethod
    def _scale_image(
        src_img: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        src_h, src_w = src_img.shape[:2]

        scaled_w = max(1, int(round(src_w * scale_x)))
        scaled_h = max(1, int(round(src_h * scale_y)))
        return cv.resize(src_img, (scaled_w, scaled_h), interpolation=cv.INTER_LINEAR)

    def _build_scaled_output(self) -> np.ndarray:
        return self._scale_image(
            self.input_image,
            float(self.scale_x_var.get()),
            float(self.scale_y_var.get()),
        )

    def _update_display(self):
        if not hasattr(self, "fig"):
            return

        self.fig.clf()
        scaled_img = self._build_scaled_output()
        display_img = im_process.min_max_normalize(scaled_img)

        show_ref = self.show_ref_var.get() and self.reference_image is not None
        if show_ref:
            ax_ref = self.fig.add_subplot(121)
            ax_ref.imshow(self.reference_image, cmap="gray")
            ax_ref.set_title("Reference (Global Avg)")
            ax_edit = self.fig.add_subplot(122)
        else:
            ax_edit = self.fig.add_subplot(111)

        ax_edit.imshow(display_img, cmap="gray")
        ax_edit.set_title(
            f"Scale Corrected (X={self.scale_x_var.get():.2f}, Y={self.scale_y_var.get():.2f})"
        )
        self.canvas.draw()

    def _reset(self):
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)
        self._update_display()

    def _finish(self):
        self.result_image = self._build_scaled_output()
        self.resize_for_downstream = bool(self.resize_for_downstream_var.get())
        self.top.destroy()
        self.master.quit()

    def _cancel(self):
        self.result_image = None
        self.resize_for_downstream = bool(self.resize_for_downstream_var.get())
        self.top.destroy()
        self.master.quit()

    def run(self):
        self.top.wait_window()
        return self.result_image


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert arbitrary-range grayscale image to uint8 for robust matching."""
    norm = im_process.min_max_normalize(img)
    return np.asarray(norm).astype(np.uint8)


def _find_template_match(search_img: np.ndarray, template_img: np.ndarray) -> tuple[int, int, float]:
    """Find template location in search image using normalized cross-correlation."""
    search_u8 = _normalize_to_uint8(search_img)
    template_u8 = _normalize_to_uint8(template_img)
    match = cv.matchTemplate(search_u8, template_u8, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(match)
    return int(max_loc[0]), int(max_loc[1]), float(max_val)


def prepare_images_for_comparison(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    resize_to_reference: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, float | None]:
    """
    Build shape-compatible image pair for downstream overlay/diff comparison.
    """
    ref_h, ref_w = ref_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]

    if resize_to_reference:
        if (tgt_h, tgt_w) == (ref_h, ref_w):
            return ref_img, target_img, "same_shape", None
        target_resized = cv.resize(target_img, (ref_w, ref_h), interpolation=cv.INTER_LINEAR)
        return ref_img, target_resized, "resized_to_reference", None

    # Containment mode: compare overlapping ROI without global resize.
    if tgt_h <= ref_h and tgt_w <= ref_w:
        x, y, score = _find_template_match(ref_img, target_img)
        ref_roi = ref_img[y:y + tgt_h, x:x + tgt_w]
        return ref_roi, target_img, "target_in_reference", score

    if ref_h <= tgt_h and ref_w <= tgt_w:
        x, y, score = _find_template_match(target_img, ref_img)
        tgt_roi = target_img[y:y + ref_h, x:x + ref_w]
        return ref_img, tgt_roi, "reference_in_target", score

    # Mixed dimensions (one wider, the other taller): fallback to resize.
    target_resized = cv.resize(target_img, (ref_w, ref_h), interpolation=cv.INTER_LINEAR)
    return ref_img, target_resized, "mixed_dims_fallback_resize", None


def _ecc_align_same_size(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Run ECC alignment for equal-shape images and return aligned image + score + warp."""
    ref_gray = ref_img.astype(np.float32)
    target_gray = target_img.astype(np.float32)

    warp_mode = cv.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 500, 1e-5)
    cc, warp_matrix = cv.findTransformECC(ref_gray, target_gray, warp_matrix, warp_mode, criteria)

    h, w = ref_img.shape[:2]
    aligned_img = cv.warpAffine(
        target_img,
        warp_matrix,
        (w, h),
        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
    )
    return aligned_img.astype(target_img.dtype), float(cc), warp_matrix


def auto_align_images(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    resize_to_reference: bool = True,
) -> np.ndarray:
    """
    Align target_img to ref_img using ECC (Translation).
    Returns aligned_img.
    """
    try:
        ref_h, ref_w = ref_img.shape[:2]
        tgt_h, tgt_w = target_img.shape[:2]

        if resize_to_reference:
            target_resized = cv.resize(target_img, (ref_w, ref_h), interpolation=cv.INTER_LINEAR)
            aligned_img, cc, _ = _ecc_align_same_size(ref_img, target_resized)
            print(f"Auto-alignment successful (resized mode). Correlation: {cc:.4f}")
            return aligned_img

        # Containment mode: align without forcing global resize.
        if tgt_h <= ref_h and tgt_w <= ref_w:
            x, y, score = _find_template_match(ref_img, target_img)
            ref_roi = ref_img[y:y + tgt_h, x:x + tgt_w]
            aligned_img, cc, _ = _ecc_align_same_size(ref_roi, target_img)
            print(
                f"Auto-alignment successful (containment: target in reference). "
                f"Template score={score:.4f}, ECC={cc:.4f}"
            )
            return aligned_img

        if ref_h <= tgt_h and ref_w <= tgt_w:
            x, y, score = _find_template_match(target_img, ref_img)
            target_roi = target_img[y:y + ref_h, x:x + ref_w]
            _, cc, warp_matrix = _ecc_align_same_size(ref_img, target_roi)
            aligned_full = cv.warpAffine(
                target_img,
                warp_matrix,
                (tgt_w, tgt_h),
                flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
            )
            print(
                f"Auto-alignment successful (containment: reference in target). "
                f"Template score={score:.4f}, ECC={cc:.4f}"
            )
            return aligned_full.astype(target_img.dtype)

        print("Auto-alignment containment mode fell back to resize mode due to mixed dimensions.")
        target_resized = cv.resize(target_img, (ref_w, ref_h), interpolation=cv.INTER_LINEAR)
        aligned_img, cc, _ = _ecc_align_same_size(ref_img, target_resized)
        print(f"Auto-alignment successful (fallback resized mode). Correlation: {cc:.4f}")
        return aligned_img

    except Exception as e:
        print(f"Auto-alignment failed: {e}")
        if resize_to_reference:
            return cv.resize(target_img, (ref_img.shape[1], ref_img.shape[0]))
        return target_img


def compare_images(
    rectified_img: np.ndarray,
    input_img: np.ndarray,
    input_name: str,
    resize_to_reference: bool = True,
) -> list[Figure]:
    """
    Compare input image to rectified image.
    In containment mode, compares best matching ROIs without global resize.
    Returns list of figures.
    """
    figures = []

    ref_comp, input_comp, mode, score = prepare_images_for_comparison(
        rectified_img,
        input_img,
        resize_to_reference=resize_to_reference,
    )

    # Ensure float for processing
    rect_norm = im_process.min_max_normalize(ref_comp).astype(np.float32) / 255.0
    input_norm = im_process.min_max_normalize(input_comp).astype(np.float32) / 255.0
    h, w = rect_norm.shape

    # 2. Overlay (Anaglyph)
    # Red = Rectified, Green = Input.
    overlay2 = np.zeros((h, w, 3), dtype=np.float32)
    overlay2[..., 0] = rect_norm
    overlay2[..., 1] = input_norm

    fig1 = plt.figure(figsize=(10, 8))
    plt.imshow(overlay2)
    title_suffix = ""
    if mode == "target_in_reference":
        title_suffix = f" [Containment: {input_name} in Rectified"
        if score is not None:
            title_suffix += f", score={score:.3f}"
        title_suffix += "]"
    elif mode == "reference_in_target":
        title_suffix = f" [Containment: Rectified in {input_name}"
        if score is not None:
            title_suffix += f", score={score:.3f}"
        title_suffix += "]"
    elif mode == "resized_to_reference":
        title_suffix = " [Resized to reference]"
    elif mode == "mixed_dims_fallback_resize":
        title_suffix = " [Mixed dims fallback resize]"

    plt.title(f"Overlay: Red=Rectified, Green={input_name}{title_suffix}")
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
    plt.title(f"Absolute Difference: |Rectified - {input_name}|{title_suffix}")
    figures.append(fig2)

    # 4. Side by Side
    fig3, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rect_norm, cmap='gray')
    axes[0].set_title("Rectified Image (comparison frame)")
    axes[1].imshow(input_norm, cmap='gray')
    axes[1].set_title(f"{input_name} (comparison frame)")
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

    val_resize_to_reference = True
    thor_resize_to_reference = True

    # 6. Optional Manual Scale Correction
    if messagebox.askyesno(
        "Manual Scale Correction",
        "Do you want to open a second window to manually scale images (X/Y) against the reference?",
    ):
        print("Scale-correcting Validation Image...")
        scale_val = InteractiveScaleAdjuster(
            root,
            val_processed,
            reference_image=rectified_img,
            title="Scale Validation Image",
        )
        val_scaled = scale_val.run()
        if val_scaled is not None:
            val_processed = val_scaled
            val_resize_to_reference = scale_val.resize_for_downstream
        else:
            print("Validation scale correction skipped/cancelled.")

        print("Scale-correcting Thorcam Image...")
        scale_thor = InteractiveScaleAdjuster(
            root,
            thor_processed,
            reference_image=rectified_img,
            title="Scale Thorcam Image",
        )
        thor_scaled = scale_thor.run()
        if thor_scaled is not None:
            thor_processed = thor_scaled
            thor_resize_to_reference = scale_thor.resize_for_downstream
        else:
            print("Thorcam scale correction skipped/cancelled.")

    # 7. Auto Align (optional, after all manual edits)
    if messagebox.askyesno("Auto Align", "Do you want to automatically align the images to the rectified reference?"):
        print("Aligning Validation Image...")
        val_processed = auto_align_images(
            rectified_img,
            val_processed,
            resize_to_reference=val_resize_to_reference,
        )
        print("Aligning Thorcam Image...")
        thor_processed = auto_align_images(
            rectified_img,
            thor_processed,
            resize_to_reference=thor_resize_to_reference,
        )

    # 8. Compare
    print("Generating comparisons...")
    figs_val = compare_images(
        rectified_img,
        val_processed,
        "Validation",
        resize_to_reference=val_resize_to_reference,
    )
    figs_thor = compare_images(
        rectified_img,
        thor_processed,
        "Thorcam",
        resize_to_reference=thor_resize_to_reference,
    )

    all_figures = figs_val + figs_thor

    # Show figures
    plt.show()

    # 9. Save
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
                "validation_resize_to_reference": np.array(val_resize_to_reference),
                "thorcam_input_original": np.asarray(Image.open(thor_path).convert('L')),
                "thorcam_processed": thor_processed,
                "thorcam_resize_to_reference": np.array(thor_resize_to_reference),
            }
            im_process.save_workspace_variables(save_path, vars_to_save)
            messagebox.showinfo("Success", f"Saved to {save_dir}")

    root.destroy()

if __name__ == "__main__":
    main()
