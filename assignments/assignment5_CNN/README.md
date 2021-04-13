## Assignment 5 - CNNs on cultural image data

__Task__ <br>
Build a deep learning model using convolutional neural networks which classify paintings by their respective artists.

__Data__ <br>
You can find the full dataset used for this here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data. A subset of the dataset is provided in the data-folder.

__Running the script__ <br>
1. Clone the repository
```
git clone https://github.com/sofieditmer/cds-visual.git cds-visual-sd
```

2. Navigate to the newly created directory
```
cd cds-visual-sd/assignments/assignment5_CNN
```

3. Create and activate virtual environment, "ass5", by running the bash script create_ass5_venv.sh. This will install the required dependencies listed in requirements.txt 

```
bash create_ass5_venv.sh
source ass5/bin/activate
```

4. Now you have activated the virtual environment in which you can run script *cnn-artists.py*. First you need to navigate to the src folder in which the script is located.

```
cd src
```

Example: <br>
```
$ python cnn-artists.py
```

Optional parameters can be set if one wishes to tweak the hyperparameters of the LeNet model. However, default parameters are specified which means that the script can simple be run with the above-mentioned command. 

__Output__ <br>
The following files will be saved in the out/ folder: 
1. LeNet_model.png: visualization of the LeNet model structure.
2. model_summary.txt: summary of the LeNet model structure.
3. model_history.png: visualization showing loss/accuracy of the model during training.
4. classification_report.txt: classification report.