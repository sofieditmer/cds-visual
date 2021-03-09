#!/usr/bin/env python
"""
Load image, draw ROI on the image, crop the image to only contain ROI, apply canny edge detection, draw contous around detected letters, save output. 
Parameters:
    image_path: str <path-to-image-dir>
    ROI_coordinates: int x1 y1 x2 y2
    output_path: str <filename-to-save-results>
Usage:
    edge_detection.py --image_path <path-to-image> --ROI_coordinates x1 y1 x2 y2 --output_path <filename-to-save-results>
Example:
    $ python edge_detection.py --image_path data/img/jefferson_memorial.jpeg --ROI_coordinates 1400 880 2900 2800 --output_path output/
Output:
    image_with_ROI.jpg
    image_cropped.jpg
    image_letters.jpg 
"""

# Import dependencies
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import argparse

# Define a function that automatically finds the upper and lower thresholds to be used when performing canny edge detection.
def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # Return the edged image
    return edged

# Define main function
def main():
    
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    # Argument 1: the path to the image
    ap.add_argument("-i", "--image_path", required = True, help = "Path to image")
    # Argument 2: the coordinates of the ROI
    ap.add_argument("-r", "--ROI_coordinates", required = True, help = "Coordinates of ROI in the image", nargs='+')
    # Argument 3: the path to the output directory
    ap.add_argument("-o", "--output_path", required = True, help = "Path to output directory")
    # Parse arguments
    args = vars(ap.parse_args())

    # Output path
    output_path = args["output_path"]
    # Create output directory if it doesn't exist already
    if not os.path.exists("output_path"):
        os.mkdir("output_path")

    # Image path
    image_path = args["image_path"]
    # Read image
    image = cv2.imread(image_path)
    # Extract filename from image path to give image a unique name 
    image_name, _ = os.path.splitext(os.path.basename(image_path))
    
    ## DRAW ROI AND CROP IMAGE ##
    
    # Define ROI coordinates
    ROI_coordinates = args["ROI_coordinates"]
    # Define top left point of ROI
    top_left = (int(ROI_coordinates[0]), int(ROI_coordinates[1]))
    # Define bottom right point of ROI
    bottom_right = (int(ROI_coordinates[2]), int(ROI_coordinates[3]))
    # Draw green ROI rectangle on image
    ROI_image = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 255, 0), (2))
    # Save image with ROI
    cv2.imwrite(os.path.join(output_path, f"{image_name}_with_ROI.jpg"), ROI_image)
    
    # Crop image to only include ROI
    image_cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # Save the cropped image
    cv2.imwrite(os.path.join(output_path, f"{image_name}_cropped.jpg"), image_cropped)
    
    ## CANNY EDGE DETECTION ##
    
    # Convert the croppe image to greyscale
    grey_cropped_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    # Perform Gaussian blurring
    blurred_image = cv2.GaussianBlur(grey_cropped_image, (3,3), 0) # I use a 3x3 kernel
    # Perform canny edge detection with the auto_canny function
    canny = auto_canny(blurred_image)
    
    ## FIND AND DRAW CONTOURS ##
    
    # Find contours 
    (contours, _) = cv2.findContours(canny.copy(), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw green contours on the cropped image
    image_letters = cv2.drawContours(image_cropped.copy(), contours, -1, (0,255,0), 2)
    # Save cropped image with contours
    cv2.imwrite(os.path.join(output_path, f"{image_name}_letters.jpg"), image_letters)
    
    # Message to user
    print(f"\nThe output images are now saved in {output_path}.\n")

                
# Define behaviour when called from command line
if __name__=="__main__":
    main()