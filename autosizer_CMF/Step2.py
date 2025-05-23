import os
import cv2
import numpy as np
import pandas as pd
import math
from datetime import datetime

# Input folders
input_folder = "/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1/Binary_Mask_Step_2"
red_dot_csv_folder = "/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1/results"

# Output folders and CSV file
output_folder = "/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1/Outline"
output_csv = "/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1/particle_analysis_results.csv"
debug_folder = "/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1/Debug_RedDots"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

min_size_threshold = 500
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
warning_file = "/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1/warnings.txt"

def log_warning(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(warning_file, "a") as wf:
        wf.write(f"[{current_time}] {message}\n")

with open(warning_file, "a") as wf:
    wf.write("WARNING, PLEASE CHECK THE FOLLOWING RED DOT FILES!")

results = []

for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(tuple(IMAGE_EXTENSIONS)):
        continue

    base_name = os.path.splitext(file_name)[0].replace("step_2_binary_mask_", "")
    binary_mask_path = os.path.join(input_folder, file_name)
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

    if binary_mask is None:
        log_warning(f"Could not load binary mask at {binary_mask_path}. Skipping.")
        continue

    kernel = np.ones((3, 3), np.uint8)
    processed_mask = cv2.erode(binary_mask, kernel, iterations=1)
    processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_size_threshold]
    output_image = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)

    csv_path = os.path.join(red_dot_csv_folder, f"{base_name}.csv")
    red_dot_coords = []
    if os.path.exists(csv_path):
        try:
            red_df = pd.read_csv(csv_path)
            if {"Point", "X", "Y"}.issubset(red_df.columns):
                red_df = red_df.sort_values("Point")
                red_dot_coords = list(zip(red_df["X"], red_df["Y"]))
            else:
                log_warning(f"{base_name}: Missing required columns in CSV.")
        except Exception as e:
            log_warning(f"{base_name}: Error reading CSV: {e}")
    else:
        log_warning(f"No red dot CSV found for {base_name}.")

    debug_image = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
    for idx, (rx, ry) in enumerate(red_dot_coords):
        cv2.circle(debug_image, (int(rx), int(ry)), 10, (0, 0, 255), 2)
        cv2.putText(debug_image, f"RedID {idx + 1}", (int(rx) + 10, int(ry) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    debug_output_path = os.path.join(debug_folder, f"Debug_{base_name}.jpeg")
    cv2.imwrite(debug_output_path, debug_image)

    if len(red_dot_coords) != len(contours):
        log_warning(f"Warning for {base_name}: Found {len(contours)} particles, {len(red_dot_coords)} red dots.")

    particle_id = 1
    for red_x, red_y in red_dot_coords:
        red_x, red_y = int(red_x), int(red_y)
        best_contour = None
        best_dist = float("inf")

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = math.hypot(red_x - cx, red_y - cy)
            if dist < best_dist:
                best_dist = dist
                best_contour = contour

        if best_contour is None:
            results.append({
                "Pseudo_ID": str(particle_id),
                "Label": base_name,
                "Area": -999,
                "X": -999,
                "Y": -999,
                "Major": -999,
                "Minor": -999,
                "Angle": -999,
                "Red_X": red_x,
                "Red_Y": red_y,
                "Farthest_X": -999,
                "Farthest_Y": -999,
                "Farthest Length": -999,
            })
            continue

        if len(best_contour) >= 5:
            ellipse = cv2.fitEllipse(best_contour)
            center, axes, orientation_angle = ellipse
            x, y = int(center[0]), int(center[1])
            major_axis, minor_axis = max(axes), min(axes)
            if axes[0] < axes[1]:
                orientation_angle += 90
            angle = orientation_angle
        else:
            x, y, w, h = cv2.boundingRect(best_contour)
            major_axis = max(w, h)
            minor_axis = min(w, h)
            angle = 0

        max_distance = 0
        farthest_point = None
        for point in best_contour:
            px, py = point[0]
            dist = math.hypot(red_x - px, red_y - py)
            if dist > max_distance:
                max_distance = dist
                farthest_point = (px, py)
        farthest_x, farthest_y = farthest_point if farthest_point else (-999, -999)

        cv2.drawContours(output_image, [best_contour], -1, (0, 255, 0), 2)
        cv2.putText(output_image, str(particle_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append({
            "Pseudo_ID": str(particle_id),
            "Label": base_name,
            "Area": cv2.contourArea(best_contour),
            "X": x,
            "Y": y,
            "Major": major_axis,
            "Minor": minor_axis,
            "Angle": angle,
            "Red_X": red_x,
            "Red_Y": red_y,
            "Farthest_X": farthest_x,
            "Farthest_Y": farthest_y,
            "Farthest Length": max_distance,
        })
        particle_id += 1

    output_file_name = file_name.replace("step_2_binary_mask", "Outline")
    output_path = os.path.join(output_folder, output_file_name)
    cv2.imwrite(output_path, output_image)
    print(f"Saved outlined image to {output_path}")

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"Particle analysis results saved to {output_csv}")
