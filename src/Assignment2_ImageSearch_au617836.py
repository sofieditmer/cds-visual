#!/usr/bin/env python

"""
Compare 3D color histogram for target image to images in a corpus one by one.
Parameters:
    path: str <path-to-image-directory>
    target image: str <filename-of-target-image>
Usage:
    Assignment2_ImageSearch.py --p <path-to-image-directory> -t <filename-of-target-image>
Example:
    $ python Assignment2_ImageSearch.py -p data/img/flowers -t image0001.jpg
"""

# Load Libaries
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse

# Define Main Function # 
def main():
    
    # First I want to define the arguments that the function requires in order to be run from the command line 
    # I do this using the argparse module
 
    # Define function arguments 
    ap = argparse.ArgumentParser()
    # Argument 1: the first argument is the path to the image directory
    ap.add_argument("-p", "--path", required = True, help= "Path to directory of images")
    # Argument 2: the second argument is the filename of the particular target image
    ap.add_argument("-t", "--target_image", required = True, help= "Filename of the target image")
    # Create a variable containing the argument parameters defined above
    args = vars(ap.parse_args())
    
    # Define path to image directory
    image_directory = args["path"]
    # Define the filename of the target image
    target_name = args["target_image"]
    
    # Create an empty dataframe to save the data in
    data = pd.DataFrame(columns=["filename", "distance"])
    
    # Read the target image with imread()
    target_image = cv2.imread(os.path.join(image_directory, target_name))
    # Create a histogram for the target image for all 3 color channels
    target_hist = cv2.calcHist([target_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    # Normalizae the histogram for the target image using the NORM_MINMAX normalization 
    target_hist_norm = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
    # For each image that ends with .jpg in the directory
    for image_path in Path(image_directory).glob("*.jpg"):
        # Take the second part of the path and save it as "image" 
        _, image = os.path.split(image_path)
        # If the image is not the target image then the loop can continue
        if image != target_name:
            # Read the image with imread() and save it as comparison image
            comparison_image = cv2.imread(os.path.join(image_directory, image))
            # Create a histogram for the comparison image
            comparison_hist = cv2.calcHist([comparison_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
            # Mormalize the histogram with the NORM_MINMAX normalization
            comparison_hist_norm = cv2.normalize(comparison_hist, comparison_hist, 0,255, cv2.NORM_MINMAX)    
            # Calculate the distance between the target image histogram and the comparison image histogram
            # The distance between the two distributions is calculated as the chi-square distance
            distance = round(cv2.compareHist(target_hist_norm, comparison_hist_norm, cv2.HISTCMP_CHISQR), 2)
            # Append the information to the empty data frame created previosuly
            data = data.append({"filename": image, 
                                "distance": distance}, ignore_index = True)
    
    # Save the results as a csv-file in the current directory
    data.to_csv(f"{target_name}_comparison.csv")
    
    # Print a message to the user saying that the file has now been saved
    print(f"output file is saved in current directory as {target_name}_comparison.csv")
    
    # Print the filename of the image which is closest to the target image
    print(data[data.distance == data.distance.min()])
            
# Define behaviour when called from command line
if __name__=="__main__":
    main()