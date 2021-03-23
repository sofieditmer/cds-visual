# Assignment 4: Classification Benchmarks

### Description of task: Classifier benchmarks using Logistic Regression and a Neural Network

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal. <br>

### Running the script <br>
Step-by-step-guide:

1. Clone the repository
```
git clone https://github.com/sofieditmer/cds-visual.git cds-visual-sd
```

2. Navigate to the newly created directory
```
cd cds-visual-sd/assignments/assignment4_Classification
```

3. Create and activate virtual environment, "ass4", by running the bash script create_ass4_venv.sh. This will install the required dependencies listed in requirements.txt 

```
bash create_ass4_venv.sh
source ass4/bin/activate
```

4. Now you have activated the virtual environment in which you can run the two scripts, lr-mnist.py and nn-mnist.py. First you need to navigate to the scr folder where the scripts are located, and then you can run the scripts individually. 

```
cd src
```

Example: <br>
```
$ python lr-mnist.py --input_dataset 'mnist_784'  --test_size 0.2

$ python lr-mnist.py --test_size 0.2 --epochs 1000
```

### Output <br>
When running the lr-mnist.py script the classification metrics are printed in the terminal as well as saved in the output directory as lr_classification_metrics.csv. The confusion matrix is also saved in the output directory as confusion_matrix.png. When you run the nn-mnist.py script the classifcation metrics are also both printed in the terminal as well as saved in the output directory. 