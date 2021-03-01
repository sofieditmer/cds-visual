# Assignment 2: Simple Image Search
This script compares the 3D color histogram of a target image in a given directory to all other images in the given directory one-by-one. The chi-square distance is used as a measure to compare the images. The results are saved in a CSV-file containing the filename and distance. The filename and distance of the image which is closest to the target image is printed in the console.

For this assignment the following dataset was used: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

The dataset "flowers" can be downloaded as a zip-file, but will then have to be unzipped in order to use.

### How to run the script: ###

Clone the repository:
```
git clone https://github.com/sofieditmer/VisualAnalytics.git cds-visual-sofie
```
From the terminal, navigate to the directory:
```
cd cds-visual-sofie
```
If you want to use the provided data, you can unzip it by executing the following:
```
cd cds-visual-sofie/data

unzip flowers.zip
```
Now you can create a virtual environment and activate it in order to be able to run the script:
```
bash create_vision_venv.sh

source cv101/bin/activate
```
Run the script, by specifying the required parameters:

-p: path to the directory of images
-t: name of the target image

Example:
```
python3 src/Assignment2_ImageSearch.py -p data/flowers/ -t image_0001.jpg
