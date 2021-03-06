# Assignment 3: Edge Detection

### Description of task: Finding text using edge detection <br>
The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation.

### Running the script <br>
Step-by-step-guide:

1. Clone the repository
```
git clone https://github.com/sofieditmer/cds-visual.git cds-visual-sd
```

2. Navigate to the newly created directory
```
cd cds-visual-sd
```

3. Create and activate virtual environment, cv101, by running the bash script create_vision_venv.sh
```
bash create_vision_venv.sh
source cv101/bin/activate
```

4. Navigate to the assignment folder containing the edge_detection.py script
```
cd assignments/assignment3_EdgeDetection
```

5. Run the edge_detection.py script within the cv101 environment. You can run the script on the provided image in the data folder called "jefferson_memorial.jpeg" and specify the following parameters:

`-i:` path to the image <br>
`-r:` the x and y coordinates of the ROI (region of interest). You should specify these as four integers, x1 y1 x2 y2, separated by whitespace. x1 y1 are the coordinates of the top-left corner of the ROI, while x2 y2 are coordinates of the bottom-right corner of the ROI. <br>
`-o:` path to output directory

Example: <br>
```
python3 edge_detection.py -i ../../data/img/jefferson_memorial.jpeg -r 1400 880 2900 2800 -o output/
```

### Output <br>
When running the edge_detection.py script you will get three outputs saved in the specified output directory:
1. {image_name}_with_ROI.jpg which is the original image with a green rectangle (ROI)
2. {image_name}_cropped.jpg which is the original image cropped to only contain the ROI.
3. {image_name}_letters.jpg which is the cropped image with green contours around the detected letters. 
