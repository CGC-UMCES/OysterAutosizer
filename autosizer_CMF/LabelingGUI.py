"""
Image Labeling Tool with Optional Overlays and New Calibration Format
======================================================================

Purpose:
    This script implements an interactive image labeling tool using Tkinter for the GUI,
    Pillow for image processing, and OpenCV/NumPy for overlay annotations.
    
    In addition to placing manual annotation points on images, the tool checks for two types
    of optional overlay files in subfolders:
      - Binary_Mask_Step_2: An image file (same name and extension as the current image)
      - Calibration_Results: A TXT file with the following two-line format:
            pxPerMM: 9.460737937559129
            Detection Info: {'mode': 'text_mode', 'detection_line_pts': ((467, 199), (1467, 201)), 'bounding_box': {'x1': 467, 'y1': 199, 'x2': 1467, 'y2': 201}}
    
    If the corresponding calibration file exists, its detection line is loaded and can be toggled on/off.
    
Key Functionalities:
    - Load images from a user-selected folder.
    - Check for optional annotation files in subfolders:
         * Binary_Mask_Step_2: for binary mask overlay.
         * Calibration_Results: for calibration overlay (using the new file format).
    - Toggle buttons allow the user to show/hide each overlay.
    - Interactive annotation: add points (left-click), undo (Ctrl+Z), and delete selected points.
    - Zooming and panning (mouse wheel, right-click drag, and dedicated buttons).
    - Resizable interface using a PanedWindow to adjust both the canvas and spreadsheet pane.
    - Manual annotations are saved/loaded as CSV files in a "results" folder within the image directory.
    
Usage:
    Run this script with Python 3.x. Ensure Tkinter, Pillow, OpenCV, NumPy, and ast (standard library) are installed.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import csv
import cv2
import numpy as np
import ast

def overlay_binary_mask(img, mask, alpha=0.5):
    """
    Overlays a binary mask onto an image.

    Args:
        img (numpy.ndarray): Image in BGR format.
        mask (numpy.ndarray): Grayscale or binary mask.
        alpha (float): Transparency factor.

    Returns:
        numpy.ndarray: Image with red mask overlay.
    """
    colored_mask = np.zeros_like(img)
    colored_mask[:, :] = (0, 0, 255)  # Red in BGR
    overlaid = img.copy()
    # Blend only pixels where the mask is non-zero.
    overlaid[mask != 0] = cv2.addWeighted(img[mask != 0], 1 - alpha,
                                           colored_mask[mask != 0], alpha, 0)
    return overlaid

def overlay_calibration(img, detectionLinePoints):
    """
    Draws a detection line on the image.

    Args:
        img (numpy.ndarray): Image in BGR format.
        detectionLinePoints (dict): Dictionary with key "detection_line_pts" containing two (x,y) tuples.

    Returns:
        numpy.ndarray: Image with the detection line drawn.
    """
    overlaid = img.copy()
    (dx1, dy1), (dx2, dy2) = detectionLinePoints["detection_line_pts"]
    cv2.line(overlaid, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)  # Green line
    return overlaid

def load_calibration(calibration_path):
    """
    Reads the calibration file and returns the pixel-per-millimeter value and detection info.

    Expected file format:
        pxPerMM: 9.460737937559129
        Detection Info: {'mode': 'text_mode', 'detection_line_pts': ((467, 199), (1467, 201)), 'bounding_box': {'x1': 467, 'y1': 199, 'x2': 1467, 'y2': 201}}

    Args:
        calibration_path (str): Path to the calibration txt file.

    Returns:
        tuple: (pxPerMM (float), detection_info (dict))
    """
    with open(calibration_path, 'r') as f:
        lines = f.read().splitlines()

    try:
        # Parse pxPerMM value.
        px_line = lines[0].strip()  # e.g., "pxPerMM: 9.460737937559129"
        pxPerMM = float(px_line.split(":", 1)[1].strip())
    except Exception as e:
        raise ValueError("Invalid format for pxPerMM in calibration file.") from e

    # Parse detection info.
    det_line = lines[1].strip()  # e.g., "Detection Info: { ... }"
    prefix = "Detection Info:"
    if det_line.startswith(prefix):
        detection_str = det_line[len(prefix):].strip()
        try:
            detection_info = ast.literal_eval(detection_str)
        except Exception as e:
            raise ValueError("Invalid format for Detection Info in calibration file.") from e
    else:
        detection_info = None

    return pxPerMM, detection_info

class ImageLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Labeling Tool with Optional Overlays")

        # Create a horizontal PanedWindow for resizable canvas and control pane.
        self.paned = tk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.paned.pack(expand=True, fill=tk.BOTH)

        # Left frame for the image canvas.
        self.left_frame = tk.Frame(self.paned, bg="black")
        self.paned.add(self.left_frame, stretch="always")

        # Right frame for controls and the spreadsheet (Treeview).
        self.right_frame = tk.Frame(self.paned)
        self.paned.add(self.right_frame, stretch="always")

        # Initial canvas dimensions.
        self.canvas_width = 800
        self.canvas_height = 600

        self.canvas = tk.Canvas(self.left_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)      # Windows/macOS
        self.canvas.bind("<Button-4>", self.on_mousewheel)        # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)        # Linux scroll down
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.left_frame.bind("<Configure>", self.on_left_frame_configure)

        # Bind Ctrl+Z for undo.
        self.master.bind("<Control-z>", lambda event: self.undo_last())

        # Spreadsheet pane (Treeview) for annotation points.
        self.tree = ttk.Treeview(self.right_frame, columns=("Index", "X", "Y"), show="headings")
        self.tree.heading("Index", text="Point #")
        self.tree.heading("X", text="X")
        self.tree.heading("Y", text="Y")
        self.tree.pack(expand=True, fill=tk.BOTH, pady=5)

        # Helper text above the buttons.
        self.helper_label = tk.Label(self.right_frame, 
                                     text="Helper: Left-click to add a point; select a row and click 'Delete Selected' to remove it; "
                                          "use Ctrl+Z to undo; scroll to zoom; right-click & drag to pan.",
                                     wraplength=300, justify="left")
        self.helper_label.pack(pady=(5, 0))

        # Buttons frame for annotation management.
        btn_frame = tk.Frame(self.right_frame)
        btn_frame.pack(pady=5)
        self.undo_button = tk.Button(btn_frame, text="Undo Last", command=self.undo_last)
        self.undo_button.grid(row=0, column=0, padx=5, pady=5)
        self.delete_button = tk.Button(btn_frame, text="Delete Selected", command=self.delete_selected)
        self.delete_button.grid(row=0, column=1, padx=5, pady=5)

        # Navigation buttons.
        nav_frame = tk.Frame(self.right_frame)
        nav_frame.pack(pady=5)
        self.prev_button = tk.Button(nav_frame, text="Previous Image", command=self.prev_image)
        self.prev_button.grid(row=0, column=0, padx=5, pady=5)
        self.next_button = tk.Button(nav_frame, text="Next Image", command=self.next_image)
        self.next_button.grid(row=0, column=1, padx=5, pady=5)

        # Zoom controls.
        zoom_frame = tk.Frame(self.right_frame)
        zoom_frame.pack(pady=5)
        self.zoom_in_button = tk.Button(zoom_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=0, column=0, padx=5, pady=5)
        self.zoom_out_button = tk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.grid(row=0, column=1, padx=5, pady=5)
        self.reset_zoom_button = tk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom)
        self.reset_zoom_button.grid(row=0, column=2, padx=5, pady=5)

        # Annotation toggle buttons.
        opt_frame = tk.Frame(self.right_frame)
        opt_frame.pack(pady=5)
        self.toggle_mask_button = tk.Button(opt_frame, text="Toggle Binary Mask", command=self.toggle_binary_mask, state=tk.DISABLED)
        self.toggle_calibration_button = tk.Button(opt_frame, text="Toggle Calibration", command=self.toggle_calibration, state=tk.DISABLED)
        self.toggle_mask_button.grid(row=0, column=0, padx=5, pady=5)
        self.toggle_calibration_button.grid(row=0, column=1, padx=5, pady=5)

        # Ask for the image directory.
        self.image_dir = filedialog.askdirectory(title="Select the Folder Containing Your Images")
        if not self.image_dir:
            messagebox.showerror("Error", "No directory selected. Exiting.")
            master.destroy()
            return
        self.cropped_image_dir = os.path.join(self.image_dir, "Cropped_Images")
        self.image_files = [f for f in os.listdir(self.cropped_image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not self.image_files:
            messagebox.showerror("Error", "No image files found in the selected directory.")
            master.destroy()
            return

        self.image_index = 0

        # Create a "results" folder INSIDE the selected image directory.
        self.results_dir = os.path.join(self.image_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Annotation points stored as (orig_x, orig_y) in original image coordinates.
        self.points = []

        # Zoom and pan settings.
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.init_zoom = 1.0
        self.init_pan_x = 0
        self.init_pan_y = 0

        #Predeclare overlays.
        self.binary_mask = None           # Will hold the binary mask image (grayscale).
        self.detectionLinePoints = None     # Will hold calibration detection info (dictionary).
        self.show_binary_mask = False
        self.show_calibration = False
        self.pxPerMM = None  # Store the pixel-per-mm value if needed. Mostly a placeholder for now, unless future functionality is added!

        self.load_image()

    def on_left_frame_configure(self, event):
        """Update canvas dimensions when the left frame is resized."""
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)
        self.update_canvas()

    def load_image(self):
        """Load the current image, compute initial zoom/pan, load saved annotations, and check for optional overlays."""
        self.canvas.delete("all")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.points = []

        image_path = os.path.join(self.cropped_image_dir, self.image_files[self.image_index])
        try:
            self.current_image = Image.open(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            return

        # Compute initial zoom so the image fits within the canvas.
        iw, ih = self.current_image.width, self.current_image.height
        self.zoom_factor = min(self.canvas_width / iw, self.canvas_height / ih, 1.0)
        new_width = int(iw * self.zoom_factor)
        new_height = int(ih * self.zoom_factor)
        self.pan_x = (self.canvas_width - new_width) // 2
        self.pan_y = (self.canvas_height - new_height) // 2

        # Save initial values for resetting zoom/pan.
        self.init_zoom = self.zoom_factor
        self.init_pan_x = self.pan_x
        self.init_pan_y = self.pan_y

        # Load saved annotation CSV if available.
        csv_filename = os.path.splitext(self.image_files[self.image_index])[0] + ".csv"
        csv_path = os.path.join(self.results_dir, csv_filename)
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.points.append((float(row["X"]), float(row["Y"])))
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to load annotations from CSV: {e}")

        # Check for binary mask overlay.
        binary_mask_path = os.path.join(self.image_dir, "Binary_Mask_Step_2", f"step_2_binary_mask_{self.image_files[self.image_index]}")
        if os.path.exists(binary_mask_path):
            self.binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
            self.show_binary_mask = False  # Default is off.
            self.toggle_mask_button.config(state=tk.NORMAL)
        else:
            self.binary_mask = None
            self.show_binary_mask = False
            self.toggle_mask_button.config(state=tk.DISABLED)

        # Check for calibration information to draw ruler if available.
        calib_filename = os.path.splitext(self.image_files[self.image_index])[0] + ".txt"
        calibration_path = os.path.join(self.image_dir, "Calibration_Results", f"calibration_{calib_filename}")

        if os.path.exists(calibration_path):
            try:
                self.pxPerMM, detection_info = load_calibration(calibration_path)
                self.detectionLinePoints = detection_info  # detection_info should include "detection_line_pts"
                self.show_calibration = False  # Default is off.
                self.toggle_calibration_button.config(state=tk.NORMAL)
            except Exception as e:
                print(f"Error loading calibration: {e}")
                self.detectionLinePoints = None
                self.show_calibration = False
                self.toggle_calibration_button.config(state=tk.DISABLED)
        else:
            self.detectionLinePoints = None
            self.show_calibration = False
            self.toggle_calibration_button.config(state=tk.DISABLED)

        self.update_canvas()

    def update_canvas(self):
        """Function that will redraw the canvas with the current image, manual annotations, and optional overlays."""
        self.canvas.delete("all")
        new_width = int(self.current_image.width * self.zoom_factor)
        new_height = int(self.current_image.height * self.zoom_factor)
        self.resized_image = self.current_image.resize((new_width, new_height), Image.LANCZOS)
        # Convert image to NumPy array in BGR for OpenCV processing.
        img_np = cv2.cvtColor(np.array(self.resized_image), cv2.COLOR_RGB2BGR)
        
        # Apply binary mask overlay if toggled.
        if self.show_binary_mask and self.binary_mask is not None:
            mask_resized = cv2.resize(self.binary_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            img_np = overlay_binary_mask(img_np, mask_resized)
        
        # Apply calibration overlay if toggled.
        if self.show_calibration and self.detectionLinePoints is not None:
            # Scale the detection line points by the zoom factor.
            pt1, pt2 = self.detectionLinePoints["detection_line_pts"]
            scaled_pts = ((int(pt1[0] * self.zoom_factor), int(pt1[1] * self.zoom_factor)),
                          (int(pt2[0] * self.zoom_factor), int(pt2[1] * self.zoom_factor)))
            detection_line_scaled = {"detection_line_pts": scaled_pts}
            img_np = overlay_calibration(img_np, detection_line_scaled)
        
        # Convert processed image back to RGB and then to a PIL image.
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(img_np)
        self.tk_image = ImageTk.PhotoImage(annotated_image)
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_image)

        # Update manual annotation points on the canvas and treeview.
        for item in self.tree.get_children():
            self.tree.delete(item)
        r = 3  # Marker radius for manual annotations.
        for idx, (orig_x, orig_y) in enumerate(self.points):
            scaled_x = orig_x * self.zoom_factor + self.pan_x
            scaled_y = orig_y * self.zoom_factor + self.pan_y
            self.canvas.create_oval(scaled_x - r, scaled_y - r, scaled_x + r, scaled_y + r, fill="red")
            self.canvas.create_text(scaled_x, scaled_y - 10, text=str(idx+1), fill="yellow")
            self.tree.insert("", "end", values=(idx+1, orig_x, orig_y))

        self.master.title(f"Image Labeling Tool - {self.image_files[self.image_index]} (Zoom: {self.zoom_factor:.2f})")

    def on_canvas_click(self, event):
        """Add a new annotation point by converting canvas coordinates back to original image coordinates."""
        orig_x = (event.x - self.pan_x) / self.zoom_factor
        orig_y = (event.y - self.pan_y) / self.zoom_factor
        self.points.append((orig_x, orig_y))
        self.update_canvas()

    def undo_last(self):
        """Undo the last added annotation point."""
        if self.points:
            self.points.pop()
            self.update_canvas()
        else:
            messagebox.showinfo("Info", "No points to undo.")

    def delete_selected(self):
        """Delete selected annotation points."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Info", "No point selected.")
            return
        indices = sorted([int(self.tree.item(sel, "values")[0]) - 1 for sel in selected], reverse=True)
        for idx in indices:
            if 0 <= idx < len(self.points):
                self.points.pop(idx)
        self.update_canvas()

    def save_annotations(self):
        """Save manual annotations to a CSV file in the results folder."""
        csv_filename = os.path.splitext(self.image_files[self.image_index])[0] + ".csv"
        csv_path = os.path.join(self.results_dir, csv_filename)
        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Point", "X", "Y"])
                for idx, (x, y) in enumerate(self.points, start=1):
                    writer.writerow([idx, x, y])
            print(f"Annotations saved to {csv_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")

    def next_image(self):
        """Save annotations and load the next image."""
        self.save_annotations()
        if self.image_index < len(self.image_files) - 1:
            self.image_index += 1
            self.load_image()
        else:
            messagebox.showinfo("Info", "This is the last image.")

    def prev_image(self):
        """Save annotations and load the previous image."""
        self.save_annotations()
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()
        else:
            messagebox.showinfo("Info", "This is the first image.")

    def zoom_in(self):
        """Zoom in (centered at canvas center) and update display."""
        center_x = self.canvas_width // 2
        center_y = self.canvas_height // 2
        self.adjust_zoom(1.2, center_x, center_y)

    def zoom_out(self):
        """Zoom out (centered at canvas center) and update display."""
        center_x = self.canvas_width // 2
        center_y = self.canvas_height // 2
        self.adjust_zoom(1/1.2, center_x, center_y)

    def reset_zoom(self):
        """Reset zoom and pan to the initial values computed when the image was loaded."""
        self.zoom_factor = self.init_zoom
        self.pan_x = self.init_pan_x
        self.pan_y = self.init_pan_y
        self.update_canvas()

    def on_mousewheel(self, event):
        """Handle mouse wheel events for zooming centered on the mouse pointer."""
        if hasattr(event, 'delta'):
            factor = 1.2 if event.delta > 0 else 1/1.2
        elif event.num == 4:
            factor = 1.2
        elif event.num == 5:
            factor = 1/1.2
        else:
            factor = 1.0
        self.adjust_zoom(factor, event.x, event.y)

    def adjust_zoom(self, factor, center_x, center_y):
        """Adjust the zoom factor and pan offsets so that zoom is centered at (center_x, center_y)."""
        img_center_x = (center_x - self.pan_x) / self.zoom_factor
        img_center_y = (center_y - self.pan_y) / self.zoom_factor
        self.zoom_factor *= factor
        self.pan_x = center_x - img_center_x * self.zoom_factor
        self.pan_y = center_y - img_center_y * self.zoom_factor
        self.update_canvas()

    def start_pan(self, event):
        """Record the starting positions for panning."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.orig_pan_x = self.pan_x
        self.orig_pan_y = self.pan_y

    def do_pan(self, event):
        """Update pan offsets based on mouse dragging."""
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_x = self.orig_pan_x + dx
        self.pan_y = self.orig_pan_y + dy
        self.update_canvas()

    def toggle_binary_mask(self):
        """Toggle the binary mask overlay on/off."""
        self.show_binary_mask = not self.show_binary_mask
        self.update_canvas()

    def toggle_calibration(self):
        """Toggle the calibration (detection line) overlay on/off."""
        self.show_calibration = not self.show_calibration
        self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()
