"""
Module for Image Segmentation and Ruler Calibration

This script performs image segmentation on images located in a specified input folder using the SAM (Segment Anything Model).
It also optionally calibrates images by detecting a pixel-to-millimeter ruler using OCR (pytesseract) on predefined phrases/lengths.

The processing pipeline includes:
    1. Preprocessing:
       - Crops the staging area from each image using edge detection and morphological operations.
       - Converts images to grayscale, applies Gaussian blur, Sobel edge detection, and morphological closing/opening.
    2. Mask Generation:
       - Uses the SAM model to generate segmentation masks.
       - Saves intermediate binary masks (if save_intermediate is True) for the raw SAM output and after area filtering.
    3. Mask Filtering:
       - Applies an area threshold to remove large background objects.
       - Applies a mean pixel value threshold to exclude low-mean-value objects (e.g., water).
    4. Ruler Calibration (Optional):
       - If calibrateRuler is enabled, calibrates the image by detecting either a specific text phrase or two numeric words on a ruler.
       - Uses pytesseract for OCR and computes the pixels-per-millimeter ratio.
       - Outputs calibration results to a text file.
    5. Annotation and Saving:
       - Overlays the final binary mask on the original image.
       - Draws the detection line (from calibration) on the image if available.
       - Saves cropped, annotated images and all intermediate outputs to designated sub-folders within the input folder.

Input Parameters:
    - input_folder (str): Folder containing original images. Output folders will be created within this folder.
    - save_intermediate (bool): Flag to save intermediate masks (raw SAM output and post-area threshold).
    - SAM Model Parameters:
        - model_type (str): Type of SAM model to use ('vit_t', 'vit_b', 'vit_l', or 'vit_h').
        - checkpoint_path (str): Path to the SAM model checkpoint.
    - Segmentation Thresholds:
        - mean_threshold (int): Minimum mean pixel value for including an object.
        - area_threshold (float): Relative area threshold to exclude large background objects.
    - Ruler Calibration Parameters:
        - calibrateRuler (bool): Whether to perform pixel/mm calibration using a ruler in the image.
        - pyTesseractPath (str): Path to the Tesseract executable.
        - useTextMode (bool): Mode for calibration. If True, searches for a text phrase; if False, searches for two numeric values such as on a ruler.
        - phrase (str): Text phrase (or two numbers separated by space in numeric mode) to detect on the ruler.
        - phrase_length_mm (float): The known physical length (in mm) of the phrase or spacing between numbers.

Module Dependencies:
    - OpenCV (cv2), NumPy, os, and torch.
    - PIL and IPython for image display.
    - pytesseract and tesseract for OCR (when ruler calibration is enabled). Follow https://github.com/UB-Mannheim/tesseract/wiki for instructions. 
    - The SAM model and automatic mask generator from the mobile_sam module.

Usage:
    1. Set the input parameters and file paths at the top of the script.
    2. Ensure that all dependencies (SAM model, pytesseract, etc.) are properly installed and configured.
    3. Run the script. Processed images, masks, calibration results, and annotated images will be saved in sub-folders within the input folder.

"""


####INPUT PARAMETERS ######


# Input folder containing original images, output folder will be created within this folder
input_folder = r"/Users/chelseafowler/Documents/PhD_oysters/autosize/autosize2/length_Spring2025_tp1"  # Replace with your input folder

# Set level of output verbosity
save_intermediate = False #save intermediate masks from the 3 steps. 



###### IMAGE SEGMENTATION PARAMETERS ######
# SAM Model parameters/settings
model_type = "vit_t"  # Change this to 'vit_b', 'vit_l', 'vit_h', or 'vit_t' based on your use case
checkpoint_path = r"/Users/chelseafowler/Documents/PhD_oysters/autosize/MobileSAM/weights/mobile_sam.pt"  # Replace with your checkpoint path

# Thresholds for object filtering
mean_threshold = 15  # Mean pixel value threshold for excluding low-mean-value objects (such as water)
area_threshold = 0.5  # Relative size threshold to exclude large background objects
##### END IMAGE SEGMENTATION PARAMETERS ######



###### RULER CALIBRATION PARAMETERS ######
# Do we want to calibrate the px/mm ruler? If True, the ruler will be detected and the user will be prompted to enter the px/mm value
calibrateRuler = True
pyTesseractPath = r'/opt/homebrew/bin/tesseract'  # Replace with your pytesseract path

useTextMode = True ##do we use numeric or text mode for the px/mm calibration. 
                    ###if true, will search for the phrase on the ruler to calibrate the ruler
                    ###If false, will search for 2 numbers on ruler as defined in phrase

phrase= "THE OYSTER ALLIANCE OYSTER RULER"
##in text mode, will search for this text to calibrate the ruler
##in numeric mode, will search for the numbers seperated by a space to calibrate the ruler, e.g. "13 14"

phrase_length_mm = 105.7 
##In text mode, is the length of the phrase to search for.  In numeric mode, is spacing between two numbers

###END RULER CALIBRATION PARAMETERS ######

#### END INPUT PARAMETERS ######



### MODULE IMPORTS ####
import cv2
import os
import numpy as np
from mobile_sam import build_sam, sam_model_registry, SamAutomaticMaskGenerator
import torch
from PIL import Image
from IPython.display import display
### END MODULE IMPORTS ###

#### FUNCTION DEFINITIONS ####
def imshow(im):
    """Helper function to display images, just for conveneince."""
    display(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))

def diamond_kernel(radius):
    """
    Creates a diamond-shaped structuring element with given radius.
    The diamond shape is for the removal of the ruler tick lines and image cleanup 
    """
    
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if abs(i - radius) + abs(j - radius) <= radius:
                kernel[i, j] = 1
    return kernel

def cropstage(im):
    """
    Crops the staging area from an image
    Works when the box covers all 4 edges of the image.  
    Tries to use edge detection to get the edge of the box, then crops the image to that edge.
    Fills in holes from center, so has to have all 4 edges.  
    """

    # 1. Convert to grayscale and apply Gaussian filtering.
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_im = cv2.GaussianBlur(gray_im, (0, 0), sigmaX=3)

    # 2. Edge detection using Sobel operator. Should get the box edges
    # Compute gradient magnitude.
    sobel_x = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize to range 0-255 and convert to uint8.
    grad_mag = np.uint8(255 * grad_mag / grad_mag.max())

    # Use Otsu's threshold to convert gradient magnitude to binary edge image. 2&3 equivalte matlab edge()
    _, edge_im = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3. Morphological closing with a diamond-shaped kernel of radius 20.
    diamond = diamond_kernel(20)
    edge_im_closed = cv2.morphologyEx(edge_im, cv2.MORPH_CLOSE, diamond)
    
    
    # 4. Fill holes, by things that don't touch the edge
    # Approach: flood-fill from the border to find the background, invert it, then combine.
    im_floodfill = edge_im_closed.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image to get the holes.
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine to fill holes.
    big_obj = edge_im_closed | im_floodfill_inv
    #imshow(big_obj)
    # 5. Keep only the largest connected component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(big_obj, connectivity=8)
    if num_labels > 1:
        # Ignore background (label 0) and find the largest component.
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        big_obj = np.zeros_like(big_obj)
        big_obj[labels == largest_label] = 255
    else:
        big_obj = np.zeros_like(big_obj)

    # 6. Morphological opening with a disk-shaped kernel (radius 50).  Cause big kernel for small object removal
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 50 + 1, 2 * 50 + 1))
    big_obj = cv2.morphologyEx(big_obj, cv2.MORPH_OPEN, disk_kernel)

    # 7. Find the bounding box of the largest object to crop the stage area.
    contours, _ = cv2.findContours(big_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Select the largest contour by area.
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        # Crop the stage area from the original image.
        stage = im[y:y+h_box, x:x+w_box]
    else:
        stage = im.copy()
    #display(Image.fromarray(cv2.cvtColor(stage, cv2.COLOR_BGR2RGB)))
    return stage

def calibrate_image(
    img,
    useTextMode: bool = True,
    # Oyster text parameters
    phrase: str = "THE OYSTER ALLIANCE OYSTER RULER",
    phrase_length_mm: float = 105.7,
    # Tesseract config
    tesseract_config: str = "--psm 11"
):
    """
    Detects either:
      1) The Oyster Alliance text bounding box (if useTextMode=True), or
      2) The distance between two numeric words (e.g., '13' and '14').
    Returns:
      pxPerMM (float or None):  Pixels per mm if calibration was successful, else None.
      detection_info (dict):    Details of the detection, including bounding boxes or midpoints,
                                plus a 'detection_line_pts' that shows how we measured the text.
    """
    # Convert to grayscale and threshold (invert + Otsu)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # For numeric mode, remove thin lines with morphological open, this is likely to be the ruler ticks
    if not useTextMode:
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # OCR
    ocr_data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=tesseract_config)

    pxPerMM = None
    detection_info = {
        "mode": "text_mode" if useTextMode else "numeric_mode",
        "detection_line_pts": None,  # We'll fill this in once we find the text
    }

    if useTextMode:
        # --- OYSTER ALLIANCE TEXT MODE ---
        lines = {}
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            txt = ocr_data['text'][i].strip()
            if not txt:
                continue
            block_num = ocr_data['block_num'][i]
            line_num = ocr_data['line_num'][i]

            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]

            line_key = (block_num, line_num)
            if line_key not in lines:
                lines[line_key] = {
                    "words": [],
                    "lefts": [],
                    "tops": [],
                    "rights": [],
                    "bottoms": []
                }
            lines[line_key]["words"].append(txt)
            lines[line_key]["lefts"].append(x)
            lines[line_key]["tops"].append(y)
            lines[line_key]["rights"].append(x + w)
            lines[line_key]["bottoms"].append(y + h)

        # Search for the text phrase in each line
        for line_key, info in lines.items():
            line_text = " ".join(info["words"]).upper()
            if phrase.upper() not in line_text:
                continue

            # Found the line containing the phrase
            x1 = info["lefts"][0]
            y1 = info["bottoms"][0]
            x2 = info["rights"][-1]
            y2 = info["bottoms"][-1]
            # Measure the top line for the bounding box width
            line_length_px = abs(x2 - x1)

            # Convert to pxPerMM
            pxPerMM = line_length_px / phrase_length_mm

            # Save detection info
            detection_info["bounding_box"] = {
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2
            }
            # The detection line is the top edge of that bounding box
            detection_info["detection_line_pts"] = ((x1, y1), (x2, y2))

            break  # Stop after first match

    else:
        # --- NUMERIC MODE (e.g. '13 14') ---
        words = phrase.split()
        if len(words) != 2:
            raise ValueError("numeric_phrase must have exactly two words, e.g. '13 14'.")

        word1, word2 = words
        found1 = None
        found2 = None
        n_boxes = len(ocr_data["text"])

        for i in range(n_boxes):
            txt = ocr_data["text"][i].strip()
            if not txt:
                continue

            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]

            if txt == word1:
                found1 = (x, y, w, h)
            elif txt == word2:
                found2 = (x, y, w, h)

        if found1 and found2:
            x1, y1, w1, h1 = found1
            mid1 = (x1 + w1 / 2.0, y1 + h1 / 2.0)

            x2, y2, w2, h2 = found2
            mid2 = (x2 + w2 / 2.0, y2 + h2 / 2.0)

            dist_px = np.hypot(mid2[0] - mid1[0], mid2[1] - mid1[1])
            pxPerMM = dist_px / phrase_length_mm

            detection_info["found_words"] = {word1: found1, word2: found2}
            # The detection line is the line connecting midpoints
            detection_info["detection_line_pts"] = (
                (int(mid1[0]), int(mid1[1])),
                (int(mid2[0]), int(mid2[1]))
            )

    return pxPerMM, detection_info
    
def annotateImage(img, mask, detectionLinePoints=None):
    ##helper function to take img, mask, and detectionLinePoints, 
    #and return annotated image that has the binary mask overlayed on the image,
    #and the detection line drawn on the image
    ##if detection not done or failed, skips the detection line drawing
    
    colored_mask = np.zeros_like(img)
    colored_mask[:, :] = (0, 0, 255)  # Red color in BGR

    # Create an overlay image by blending only in the masked areas
    alpha = 0.5  # Transparency factor

    # Make a copy of the original image to apply the overlay
    overlaid = img.copy()
    # For pixels where mask is non-zero, blend the original image with the colored mask
    overlaid[mask != 0] = cv2.addWeighted(img[mask != 0], 1 - alpha, 
                                        colored_mask[mask != 0], alpha, 0)
    
    if detectionLinePoints is not None:
        # Draw the detection line on the overlay image
        (dx1, dy1), (dx2, dy2) = detectionLinePoints["detection_line_pts"]
        cv2.line(overlaid, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
    return overlaid

### END FUNCTION DEFINITIONS ###


## pre-loop initialization

# Determine if cuda is available (for speed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Output folders for intermediate and final results, created within input folder
if save_intermediate:
    output_folder_step_0 = os.path.join(input_folder, "Binary_Mask_Step_0")  # SAM output
    output_folder_step_1 = os.path.join(input_folder, "Binary_Mask_Step_1")  # After area threshold
    os.makedirs(output_folder_step_0, exist_ok=True)
    os.makedirs(output_folder_step_1, exist_ok=True)
    
if calibrateRuler:
    calib_folder = os.path.join(input_folder, "Calibration_Results")  # Calibration data saved here
    os.makedirs(calib_folder, exist_ok=True)
    
output_folder_step_2 = os.path.join(input_folder, "Binary_Mask_Step_2")  # Final processed masks
crop_folder = os.path.join(input_folder, "Cropped_Images")  # Cropped images
annotated_folder = os.path.join(input_folder, "Annotated_Images")  # Annotated images
os.makedirs(output_folder_step_2, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)


# Initialize the SAM model
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

# Initialize the automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

if calibrateRuler:
    import pytesseract
    from pytesseract import Output
    # set pytesseract path
    pytesseract.pytesseract.tesseract_cmd = pyTesseractPath

#### BEGIN FILE PROCESSING LOOP ####
# Process each file in the input folder
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Load the original image
    image_path = os.path.join(input_folder, file_name)
    rawImage = cv2.imread(image_path)

    if rawImage is None:
        print(f"Error: Could not load image at {image_path}. Skipping.")
        continue
    
    # Crop the staging area for the image, otherwise the mask will be faulty
    image = cropstage(rawImage)
    
    #write the cropped image to outputFolder
    cv2.imwrite(os.path.join(crop_folder, file_name), image)
    
    # Generate masks using SAM
    try:
        masks = mask_generator.generate(image)
    except Exception as e:
        print(f"Error during mask generation for {file_name}: {e}")
        continue

    # Step 0: Save the SAM output directly
    if masks:
        binary_mask_step_0 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask in masks:
            binary_mask_step_0 = np.maximum(binary_mask_step_0, mask["segmentation"].astype(np.uint8))
        # Save the raw SAM output
        if save_intermediate:
            output_file_name_step_0 = f"step_0_binary_mask_{file_name}"
            output_path_step_0 = os.path.join(output_folder_step_0, output_file_name_step_0)
            cv2.imwrite(output_path_step_0, binary_mask_step_0 * 255)
    else:
        print(f"No masks were generated for {file_name}.")
        continue

    # Step 1: Apply area threshold to exclude large background objects
    binary_mask_step_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    total_pixels = image.shape[0] * image.shape[1]  # Total pixels in the image

    for mask in masks:
        segmentation = mask["segmentation"].astype(np.uint8)
        mask_area = mask["area"]  # Area of the mask

        if mask_area / total_pixels < area_threshold:  # Exclude large masks
            binary_mask_step_1 = np.maximum(binary_mask_step_1, segmentation)

    if save_intermediate:
        # Save the Step 1 mask
        output_file_name_step_1 = f"step_1_binary_mask_{file_name}"
        output_path_step_1 = os.path.join(output_folder_step_1, output_file_name_step_1)
        cv2.imwrite(output_path_step_1, binary_mask_step_1 * 255)

    # Step 2: Apply mean pixel value threshold
    binary_mask_step_2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for mask in masks:
        segmentation = mask["segmentation"].astype(np.uint8)
        mask_area = mask["area"]
        mean_pixel_value = cv2.mean(image, mask=segmentation)[0]  # Mean pixel value in the object

        # Include the object if it passes both thresholds
        if mask_area / total_pixels < area_threshold and mean_pixel_value > mean_threshold:
            binary_mask_step_2 = np.maximum(binary_mask_step_2, segmentation)
        
    # Save the Step 2 mask
    output_file_name_step_2 = f"step_2_binary_mask_{file_name}"
    output_path_step_2 = os.path.join(output_folder_step_2, output_file_name_step_2)
    cv2.imwrite(output_path_step_2, binary_mask_step_2 * 255)
    #print(f"Segmentation done for {file_name}")
    
    if calibrateRuler:
        pxPerMM, det_info = calibrate_image(image, useTextMode=useTextMode, phrase=phrase, phrase_length_mm=phrase_length_mm)

        
        #dump calibration information
        
        ##extract filename from file_name
        calib_file_name = f"calibration_{os.path.splitext(file_name)[0]}.txt"
        calib_file_path = os.path.join(calib_folder, calib_file_name)
        
        if pxPerMM is None:
            ##hos img and detection info if detection failed.  
            ##TODO maybe have an info dump of detected words annotated?  
            print("Calibration failed.")
            print("Detection Info:", det_info)
            with open(calib_file_path, "w") as f:
                f.write("Calibration failed.\n")
                f.write(f"Detection Info: {det_info}\n")
        else:
            with open(calib_file_path, "w") as f:
                f.write(f"pxPerMM: {pxPerMM}\n")
                f.write(f"Detection Info: {det_info}\n")
        
        annotated_img = annotateImage(image, binary_mask_step_2, det_info)
    else:
        annotated_img = annotateImage(image, binary_mask_step_2)
    cv2.imwrite(os.path.join(annotated_folder, file_name), annotated_img)
    print(f"Processing complete for {file_name}")
print("Processing complete!")
#### END FILE PROCESSING LOOP ####

