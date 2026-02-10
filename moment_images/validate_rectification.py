"""
Script to validate rectification results against new images from other cameras.
Allows interactive processing (rotation, flipping, cropping) of input images.
"""

import os
import sys
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox, Scale
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2 as cv
from PIL import Image

MIN_TEMPLATE_MATCH_SCORE = 0.20

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

    def run(self):
        # Wait for window to close
        self.top.wait_window()
        return self.result_image


class InteractiveScaleAdjuster:
    """GUI to anisotropically scale an image against an optional reference."""

    def __init__(
        self,
        master,
        input_image: np.ndarray,
        reference_image=None,
        title="Scale Correct Image",
        default_scale_x: float = 1.0,
        default_scale_y: float = 1.0,
    ):
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

        self.scale_min = 0.001
        self.scale_max = 8.00
        self.default_scale_x = float(np.clip(default_scale_x, self.scale_min, self.scale_max))
        self.default_scale_y = float(np.clip(default_scale_y, self.scale_min, self.scale_max))
        self.scale_x_value = self.default_scale_x
        self.scale_y_value = self.default_scale_y
        self.scale_x_var = tk.StringVar(value=f"{self.scale_x_value:.4f}")
        self.scale_y_var = tk.StringVar(value=f"{self.scale_y_value:.4f}")
        self.scale_status_var = tk.StringVar(value="")
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

        tk.Label(scale_frame, text="X Scale:").pack(side=tk.LEFT, padx=(4, 2))
        self.scale_x_entry = tk.Entry(scale_frame, textvariable=self.scale_x_var, width=8)
        self.scale_x_entry.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(scale_frame, text="Y Scale:").pack(side=tk.LEFT, padx=(0, 2))
        self.scale_y_entry = tk.Entry(scale_frame, textvariable=self.scale_y_var, width=8)
        self.scale_y_entry.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            scale_frame,
            text=f"Limits: {self.scale_min:.3f} to {self.scale_max:.3f}",
            fg="gray25",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.scale_status_label = tk.Label(
            scale_frame,
            textvariable=self.scale_status_var,
            fg="red",
        )
        self.scale_status_label.pack(side=tk.LEFT, padx=(0, 4))

        self.scale_x_entry.bind("<KeyRelease>", self._on_scale_entry_change)
        self.scale_x_entry.bind("<FocusOut>", self._on_scale_entry_commit)
        self.scale_x_entry.bind("<Return>", self._on_scale_entry_commit)
        self.scale_y_entry.bind("<KeyRelease>", self._on_scale_entry_change)
        self.scale_y_entry.bind("<FocusOut>", self._on_scale_entry_commit)
        self.scale_y_entry.bind("<Return>", self._on_scale_entry_commit)

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

    def _parse_scale_value(self, raw_value: str, axis_label: str) -> tuple[float | None, str | None]:
        value_text = raw_value.strip()
        if value_text == "":
            return None, f"{axis_label} scale is empty"

        try:
            parsed = float(value_text)
        except ValueError:
            return None, f"{axis_label} scale must be numeric"

        if parsed < self.scale_min or parsed > self.scale_max:
            return None, (
                f"{axis_label} scale out of range "
                f"({self.scale_min:.3f}-{self.scale_max:.3f})"
            )

        return parsed, None

    def _apply_scale_entries(self, commit: bool = False) -> bool:
        x_val, x_err = self._parse_scale_value(self.scale_x_var.get(), "X")
        y_val, y_err = self._parse_scale_value(self.scale_y_var.get(), "Y")

        if x_err or y_err:
            error_text = x_err if x_err else y_err
            self.scale_status_var.set(error_text)
            if commit:
                self.scale_x_var.set(f"{self.scale_x_value:.4f}")
                self.scale_y_var.set(f"{self.scale_y_value:.4f}")
                self.scale_status_var.set("")
            return False

        self.scale_x_value = float(x_val)
        self.scale_y_value = float(y_val)
        self.scale_status_var.set("")
        self._update_display()
        return True

    def _on_scale_entry_change(self, _event):
        self._apply_scale_entries(commit=False)

    def _on_scale_entry_commit(self, _event):
        self._apply_scale_entries(commit=True)

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
            self.scale_x_value,
            self.scale_y_value,
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
            f"Scale Corrected (X={self.scale_x_value:.2f}, Y={self.scale_y_value:.2f})"
        )
        self.canvas.draw()

    def _reset(self):
        self.scale_x_value = self.default_scale_x
        self.scale_y_value = self.default_scale_y
        self.scale_x_var.set(f"{self.scale_x_value:.4f}")
        self.scale_y_var.set(f"{self.scale_y_value:.4f}")
        self.scale_status_var.set("")
        self._update_display()

    def _finish(self):
        self._apply_scale_entries(commit=True)
        self.result_image = self._build_scaled_output()
        self.resize_for_downstream = bool(self.resize_for_downstream_var.get())
        self.top.destroy()

    def _cancel(self):
        self.result_image = None
        self.resize_for_downstream = bool(self.resize_for_downstream_var.get())
        self.top.destroy()

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
    score = float(max_val)
    if not np.isfinite(score):
        score = -1.0
    return int(max_loc[0]), int(max_loc[1]), score


def _find_containment_match(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    prefer_smaller_pixels_contained: bool = False,
) -> tuple[str, int, int, float] | None:
    """
    Decide containment direction and placement using AS-IS template matching.

    Modes:
    - ``target_in_reference``: target is contained in reference.
    - ``reference_in_target``: reference is contained in target.
    """
    ref_h, ref_w = ref_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]

    target_is_smaller = tgt_h <= ref_h and tgt_w <= ref_w and (tgt_h < ref_h or tgt_w < ref_w)
    if prefer_smaller_pixels_contained and target_is_smaller:
        x, y, score = _find_template_match(ref_img, target_img)
        return "target_in_reference", x, y, score

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


@dataclass
class ComparisonData:
    """Prepared comparison frames and metadata for visualization."""

    mode: str
    score: float | None
    match_found: bool
    side_reference: np.ndarray
    side_input: np.ndarray
    diff_containing: np.ndarray | None = None
    diff_contained: np.ndarray | None = None
    containing_label: str | None = None
    contained_label: str | None = None


def _comparison_mode_suffix(mode: str, score: float | None) -> str:
    if mode == "target_in_reference":
        return f" [Match: input in reference, score={score:.3f}]" if score is not None else " [Match: input in reference]"
    if mode == "reference_in_target":
        return f" [Match: reference in input, score={score:.3f}]" if score is not None else " [Match: reference in input]"
    if mode == "resized_to_reference":
        return " [Resized to reference]"
    if mode == "same_shape_no_resize":
        return " [Same size, no resize]"
    if mode == "same_shape":
        return " [Same size]"
    if mode == "low_confidence_match":
        return f" [No good match, best score={score:.3f}]" if score is not None else " [No good match]"
    if mode == "no_containment_match":
        return " [No containment scenario]"
    return ""


def prepare_images_for_comparison(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    resize_to_reference: bool = True,
    prefer_smaller_pixels_contained: bool = False,
) -> ComparisonData:
    """
    Prepare visualization frames for comparison.

    If a good containment match is found, side-by-side uses the contained image
    and the corresponding crop from the containing image. Otherwise, side-by-side
    falls back to original scales and diff/overlay are disabled.
    """
    ref_h, ref_w = ref_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]

    if resize_to_reference:
        target_resized = target_img if (tgt_h, tgt_w) == (ref_h, ref_w) else cv.resize(
            target_img, (ref_w, ref_h), interpolation=cv.INTER_LINEAR
        )
        mode = "same_shape" if (tgt_h, tgt_w) == (ref_h, ref_w) else "resized_to_reference"
        return ComparisonData(
            mode=mode,
            score=None,
            match_found=True,
            side_reference=ref_img,
            side_input=target_resized,
            diff_containing=ref_img,
            diff_contained=target_resized,
            containing_label="Reference",
            contained_label="Input",
        )

    if (tgt_h, tgt_w) == (ref_h, ref_w):
        return ComparisonData(
            mode="same_shape_no_resize",
            score=None,
            match_found=True,
            side_reference=ref_img,
            side_input=target_img,
            diff_containing=ref_img,
            diff_contained=target_img,
            containing_label="Reference",
            contained_label="Input",
        )

    match_result = _find_containment_match(
        ref_img,
        target_img,
        prefer_smaller_pixels_contained=prefer_smaller_pixels_contained,
    )
    if match_result is None:
        return ComparisonData(
            mode="no_containment_match",
            score=None,
            match_found=False,
            side_reference=ref_img,
            side_input=target_img,
        )

    mode, x, y, score = match_result
    if score < MIN_TEMPLATE_MATCH_SCORE:
        return ComparisonData(
            mode="low_confidence_match",
            score=score,
            match_found=False,
            side_reference=ref_img,
            side_input=target_img,
        )

    if mode == "target_in_reference":
        reference_crop = ref_img[y:y + tgt_h, x:x + tgt_w]
        return ComparisonData(
            mode=mode,
            score=score,
            match_found=True,
            side_reference=reference_crop,
            side_input=target_img,
            diff_containing=reference_crop,
            diff_contained=target_img,
            containing_label="Reference Crop",
            contained_label="Input",
        )

    input_crop = target_img[y:y + ref_h, x:x + ref_w]
    return ComparisonData(
        mode=mode,
        score=score,
        match_found=True,
        side_reference=ref_img,
        side_input=input_crop,
        diff_containing=input_crop,
        diff_contained=ref_img,
        containing_label="Input Crop",
        contained_label="Reference",
    )


def _ecc_align_same_size(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Run ECC alignment for equal-shape images and return aligned image + score + warp."""
    ref_gray = ref_img.astype(np.float32)
    target_gray = target_img.astype(np.float32)

    warp_mode = cv.MOTION_TRANSLATION
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
    prefer_smaller_pixels_contained: bool = False,
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

        if (tgt_h, tgt_w) == (ref_h, ref_w):
            aligned_img, cc, _ = _ecc_align_same_size(ref_img, target_img)
            print(f"Auto-alignment successful (same-size non-resize mode). Correlation: {cc:.4f}")
            return aligned_img

        # Containment mode: align without forcing global resize.
        match_result = _find_containment_match(
            ref_img,
            target_img,
            prefer_smaller_pixels_contained=prefer_smaller_pixels_contained,
        )
        if match_result is not None:
            mode, x, y, score = match_result
            if score < MIN_TEMPLATE_MATCH_SCORE:
                print(
                    "Auto-alignment skipped in containment mode due to low match score "
                    f"({score:.4f} < {MIN_TEMPLATE_MATCH_SCORE:.2f})."
                )
                return target_img

            if mode == "target_in_reference":
                ref_roi = ref_img[y:y + tgt_h, x:x + tgt_w]
                aligned_img, cc, _ = _ecc_align_same_size(ref_roi, target_img)
                print(
                    f"Auto-alignment successful (containment: target in reference). "
                    f"Template score={score:.4f}, ECC={cc:.4f}"
                )
                return aligned_img

            target_roi = target_img[y:y + ref_h, x:x + ref_w]
            aligned_roi, cc, _ = _ecc_align_same_size(ref_img, target_roi)
            aligned_full = target_img.copy()
            aligned_full[y:y + ref_h, x:x + ref_w] = aligned_roi
            print(
                f"Auto-alignment successful (containment: reference in target). "
                f"Template score={score:.4f}, ECC={cc:.4f}"
            )
            return aligned_full.astype(target_img.dtype)

        print("Auto-alignment skipped: could not determine containment without resizing.")
        return target_img

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
    prefer_smaller_pixels_contained: bool = False,
) -> list[Figure]:
    """
    Compare input image to rectified image.
    In containment mode, compares best matching ROIs without global resize.
    Returns list of figures.
    """
    figures = []

    # Always show an as-is comparison (original reference scale vs processed input scale),
    # even when containment matching is weak or fails.
    ref_as_is_norm = im_process.min_max_normalize(rectified_img).astype(np.float32) / 255.0
    input_as_is_norm = im_process.min_max_normalize(input_img).astype(np.float32) / 255.0
    fig_as_is, axes_as_is = plt.subplots(1, 2, figsize=(12, 6))
    axes_as_is[0].imshow(ref_as_is_norm, cmap='gray')
    axes_as_is[0].set_title(f"Reference (As-Is) [{rectified_img.shape[0]}x{rectified_img.shape[1]}]")
    axes_as_is[1].imshow(input_as_is_norm, cmap='gray')
    axes_as_is[1].set_title(f"{input_name} (As-Edited As-Is) [{input_img.shape[0]}x{input_img.shape[1]}]")
    figures.append(fig_as_is)

    comp_data = prepare_images_for_comparison(
        rectified_img,
        input_img,
        resize_to_reference=resize_to_reference,
        prefer_smaller_pixels_contained=prefer_smaller_pixels_contained,
    )

    side_ref_norm = im_process.min_max_normalize(comp_data.side_reference).astype(np.float32) / 255.0
    side_input_norm = im_process.min_max_normalize(comp_data.side_input).astype(np.float32) / 255.0
    title_suffix = _comparison_mode_suffix(comp_data.mode, comp_data.score)

    # A. Side-by-side view (always shown)
    fig_side, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(side_ref_norm, cmap='gray')
    axes[0].set_title(f"Reference View{title_suffix}")
    axes[1].imshow(side_input_norm, cmap='gray')
    axes[1].set_title(f"{input_name} View{title_suffix}")
    figures.append(fig_side)

    # B/C only if a good containment/resize match exists.
    if not comp_data.match_found or comp_data.diff_containing is None or comp_data.diff_contained is None:
        print(f"{input_name}: no good containment match for diff/overlay; showing as-is + side-by-side only.")
        return figures

    containing_norm = im_process.min_max_normalize(comp_data.diff_containing).astype(np.float32) / 255.0
    contained_norm = im_process.min_max_normalize(comp_data.diff_contained).astype(np.float32) / 255.0

    # B. Difference map at contained-image scale.
    diff = np.abs(containing_norm - contained_norm)
    fig_diff = plt.figure(figsize=(10, 8))
    im = plt.imshow(diff, cmap='viridis')
    im_process.add_colorbar(im)
    plt.title(f"Difference |Containing Crop - Contained|{title_suffix}")
    figures.append(fig_diff)

    # C. Overlay of contained over containing crop.
    overlay = np.zeros((contained_norm.shape[0], contained_norm.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = containing_norm
    overlay[..., 1] = contained_norm
    fig_overlay = plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.title(f"Overlay: Red=Containing, Green=Contained{title_suffix}")
    from matplotlib.lines import Line2D
    label_containing = comp_data.containing_label or "Containing"
    label_contained = comp_data.contained_label or "Contained"
    custom_lines = [Line2D([0], [0], color='red', lw=4), Line2D([0], [0], color='green', lw=4)]
    plt.legend(custom_lines, [label_containing, label_contained], loc='upper right')
    figures.append(fig_overlay)

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
    manual_scale_opt_in = False

    # 6. Optional Manual Scale Correction
    if messagebox.askyesno(
        "Manual Scale Correction",
        "Do you want to open a second window to manually scale images (X/Y) against the reference?",
    ):
        manual_scale_opt_in = True
        print("Scale-correcting Validation Image...")
        scale_val = InteractiveScaleAdjuster(
            root,
            val_processed,
            reference_image=rectified_img,
            title="Scale Validation Image",
            default_scale_x=0.038,
            default_scale_y=0.035,
        )
        val_scaled = scale_val.run()
        val_resize_to_reference = scale_val.resize_for_downstream
        if val_scaled is not None:
            val_processed = val_scaled
        else:
            print("Validation scale correction skipped/cancelled.")

        print("Scale-correcting Thorcam Image...")
        scale_thor = InteractiveScaleAdjuster(
            root,
            thor_processed,
            reference_image=rectified_img,
            title="Scale Thorcam Image",
            default_scale_x=0.07,
            default_scale_y=0.07,
        )
        thor_scaled = scale_thor.run()
        thor_resize_to_reference = scale_thor.resize_for_downstream
        if thor_scaled is not None:
            thor_processed = thor_scaled
        else:
            print("Thorcam scale correction skipped/cancelled.")

    # 7. Auto Align (optional, after all manual edits)
    if messagebox.askyesno("Auto Align", "Do you want to automatically align the images to the rectified reference?"):
        print("Aligning Validation Image...")
        val_processed = auto_align_images(
            rectified_img,
            val_processed,
            resize_to_reference=val_resize_to_reference,
            prefer_smaller_pixels_contained=manual_scale_opt_in,
        )
        print("Aligning Thorcam Image...")
        thor_processed = auto_align_images(
            rectified_img,
            thor_processed,
            resize_to_reference=thor_resize_to_reference,
            prefer_smaller_pixels_contained=manual_scale_opt_in,
        )

    # 8. Compare
    print("Generating comparisons...")
    figs_val = compare_images(
        rectified_img,
        val_processed,
        "Validation",
        resize_to_reference=val_resize_to_reference,
        prefer_smaller_pixels_contained=manual_scale_opt_in,
    )
    figs_thor = compare_images(
        rectified_img,
        thor_processed,
        "Thorcam",
        resize_to_reference=thor_resize_to_reference,
        prefer_smaller_pixels_contained=manual_scale_opt_in,
    )

    all_figures = figs_val + figs_thor

    # Show figures
    plt.show()

    # 9. Save
    if messagebox.askyesno("Save", "Do you want to save the figures and workspace?"):
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if save_dir:
            # Save Figures
            filename_changes = im_process.save_figures(all_figures, save_dir)

            # Notify user of any filename changes
            if filename_changes:
                changes_message = "The following filenames were regularized for compatibility:\n\n"
                for original, regularized in sorted(filename_changes.items()):
                    changes_message += f"  {original}\n    → {regularized}\n"
                messagebox.showinfo("Filename Changes", changes_message)

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
