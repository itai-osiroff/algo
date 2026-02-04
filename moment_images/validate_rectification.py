
"""
Script to validate rectification results against new images from other cameras.
Allows interactive processing (rotation, flipping, cropping) of input images.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
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

    def __init__(self, master, image_path, title="Adjust Image"):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title(title)
        self.top.geometry("900x700")

        self.original_image_pil = im_process.load_image(image_path)
        if self.original_image_pil is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale if not already
        if self.original_image_pil.mode != 'L':
             self.original_image_pil = self.original_image_pil.convert('L')

        self.current_image_pil = self.original_image_pil.copy()

        # Transformations state
        self.rotation = 0 # 0, 90, 180, 270
        self.flip_h = False
        self.flip_v = False
        self.crop_box = None # (left, upper, right, lower)

        self.result_image = None
        self.is_done = False

        self._init_ui()
        self._update_display()

    def _init_ui(self):
        # Toolbar Frame
        toolbar_frame = tk.Frame(self.top)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(toolbar_frame, text="Rotate Left", command=lambda: self._rotate(90)).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Rotate Right", command=lambda: self._rotate(-90)).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Flip Horiz", command=self._flip_h).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Flip Vert", command=self._flip_v).pack(side=tk.LEFT, padx=2)
        self.btn_crop = tk.Button(toolbar_frame, text="Crop Mode: OFF", command=self._toggle_crop)
        self.btn_crop.pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar_frame, text="Done", command=self._finish, bg="green", fg="white").pack(side=tk.RIGHT, padx=5)

        # Plot Area
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Rectangle Selector for cropping (initialized in _update_display)
        self.rs = None
        self.crop_active = False
        self._update_display() # Initialize display and selector

    def _rotate(self, angle):
        # Rotate logic on the PIL image directly?
        # Better to keep a pipeline: Original -> Rotate -> Flip -> Crop
        # But for UI responsiveness, applying incrementally is fine.
        # Let's modify the current state variable self.current_image_pil
        self.current_image_pil = self.current_image_pil.rotate(angle, expand=True)
        # Reset crop if rotated (simplification)
        self.crop_box = None
        self._update_display()

    def _flip_h(self):
        self.current_image_pil = self.current_image_pil.transpose(Image.FLIP_LEFT_RIGHT)
        self.crop_box = None
        self._update_display()

    def _flip_v(self):
        self.current_image_pil = self.current_image_pil.transpose(Image.FLIP_TOP_BOTTOM)
        self.crop_box = None
        self._update_display()

    def _toggle_crop(self):
        self.crop_active = not self.crop_active
        if self.crop_active:
            self.btn_crop.config(text="Crop Mode: ON", bg="yellow")
            if self.rs: self.rs.set_active(True)
        else:
            self.btn_crop.config(text="Crop Mode: OFF", bg="SystemButtonFace")
            if self.rs: self.rs.set_active(False)

    def _on_select(self, eclick, erelease):
        # x1, y1 = eclick.xdata, eclick.ydata
        # x2, y2 = erelease.xdata, erelease.ydata
        extent = self.rs.extents # xmin, xmax, ymin, ymax
        # Convert to integer
        self.crop_box = (int(extent[0]), int(extent[2]), int(extent[1]), int(extent[3]))
        print(f"Crop selected: {self.crop_box}")

    def _reset(self):
        self.current_image_pil = self.original_image_pil.copy()
        self.crop_box = None
        self.crop_active = False
        self.btn_crop.config(text="Crop Mode: OFF", bg="SystemButtonFace")
        self._update_display()

    def _update_display(self):
        self.ax.clear()

        im_array = np.asarray(self.current_image_pil)
        self.ax.imshow(im_array, cmap="gray")
        self.ax.set_title("Adjust Image (Close crop mode to apply crop implicitly on Done)")

        # Re-initialize selector because ax.clear() removed it
        self.rs = RectangleSelector(
            self.ax, self._on_select,
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
        # Apply crop if selected
        final_img = self.current_image_pil
        if self.crop_box:
            # crop_box is (left, upper, right, lower)
            # Extents from RectangleSelector are (xmin, xmax, ymin, ymax) which is (left, right, upper, lower)
            # PIL crop expects (left, upper, right, lower)

            # Wait, self.rs.extents returns (xmin, xmax, ymin, ymax)
            xmin, xmax, ymin, ymax = self.rs.extents
            # Ensure coords are within bounds
            w, h = final_img.size
            xmin = max(0, min(xmin, w))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h))
            ymax = max(0, min(ymax, h))

            if xmax > xmin and ymax > ymin:
                final_img = final_img.crop((xmin, ymin, xmax, ymax))

        self.result_image = np.asarray(final_img)
        self.is_done = True
        self.top.destroy()
        self.master.quit() # Stop the local mainloop if we started one

    def run(self):
        # Wait for window to close
        self.top.wait_window()
        return self.result_image

def compare_images(rectified_img: np.ndarray, input_img: np.ndarray, input_name: str) -> list[Figure]:
    """
    Compare input image to rectified image.
    Resizes input_img to match rectified_img shape.
    Returns list of figures.
    """
    figures = []

    # 1. Resize input to match rectified
    h, w = rectified_img.shape

    # Ensure input is float for processing
    rect_norm = im_process.min_max_normalize(rectified_img).astype(np.float32) / 255.0

    # Resize input
    input_resized = cv.resize(input_img, (w, h), interpolation=cv.INTER_LINEAR)
    input_norm = im_process.min_max_normalize(input_resized).astype(np.float32) / 255.0

    # 2. Overlay (Anaglyph)
    # Red = Rectified, Cyan (G+B) = Input
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[..., 0] = rect_norm  # Red channel
    overlay[..., 1] = input_norm # Green
    overlay[..., 2] = input_norm # Blue (making it cyan)

    # Or maybe Green vs Magenta?
    # Let's do Red=Rectified, Green=Input.
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
    axes[1].set_title(f"{input_name} (Resized)")
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
    # We need to create a dummy root or use Toplevel
    # Since root is withdrawn, Toplevel is fine.
    adj_val = InteractiveImageAdjuster(root, val_path, title="Adjust Validation Image")
    val_processed = adj_val.run()
    if val_processed is None:
        print("Validation image processing cancelled.")
        return

    # 5. Process Thorcam Image
    print("Adjusting Thorcam Image...")
    adj_thor = InteractiveImageAdjuster(root, thor_path, title="Adjust Thorcam Image")
    thor_processed = adj_thor.run()
    if thor_processed is None:
        print("Thorcam image processing cancelled.")
        return

    # 6. Compare
    print("Generating comparisons...")
    figs_val = compare_images(rectified_img, val_processed, "Validation")
    figs_thor = compare_images(rectified_img, thor_processed, "Thorcam")

    all_figures = figs_val + figs_thor

    # Show figures
    plt.show()

    # 7. Save
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
