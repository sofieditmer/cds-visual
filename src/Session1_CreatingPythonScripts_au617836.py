# Required libraries
import sys
sys.path.append(os.path.join(".."))  
import cv2 
import numpy as np
from utils.imutils import jimshow

# Translate function              
def translate(image, x, y):
    # Define translation matrix
    M = np.float64([[1, 0, x],
                  [0, 1, y]])
    # Perform translation on the chosen image
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # Return the translated image
    return shifted   

# Then define a main() function. A main function is something you make in order to be able to run your script from a bash (terminal)
def main():
    """
    In a function called main(), you should include the 'core logic' or
    your script.
    """
    return

# Declare namespace. These lines need to be present in your script if you want to be able to run the script from bash. 
if __name__=="__main__":
    main()
