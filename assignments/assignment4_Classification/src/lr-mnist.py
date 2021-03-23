#!/usr/bin/env python

"""
Given the full MNIST data set, a Logistic Regression Classifier is trained, and evaluation metrics are printed to the terminal and saved in output folder as a CSV-file.

Parameters:
    input_dataset: str <path-to-dataset>
    test_size: num <size-of-test-data>
    
Usage:
    lr-mnist.py --input_dataset <path-to-dataset> --test_size <size-of-test-data>

Example:
    $ python lr-mnist.py --input_dataset 'mnist_784'  --test_size 0.2

Output:
    lr_classification_metrics.csv: Logistic regression classification metrics
    confusion_matrix.png: Normalized confusion matrix 
"""

### DEPENDENCIES ###
import os
import sys
sys.path.append(os.path.join(".."))
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import utils.classifier_utils as clf_util
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse

### MAIN FUNCTION ###

def main():
   
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: the path to the edgelist
    ap.add_argument("-i", "--input_dataset", 
                    type = str,
                    required = False, 
                    help = "Define the path to the dataset", 
                    default = 'mnist_784')

    # Argument 2: size of test dataset
    ap.add_argument("-ts", "--test_size", 
                    type = float,
                    required = False, 
                    help = "Define the size of the test dataset", 
                    default = 0.2)
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    dataset = args["input_dataset"]
    size_test = args["test_size"]
    
    # Create output directory
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Start message to user
    print("\n[INFO] Initializing logistic regression classification...")
    
    # Load data
    X, y = fetch_openml(dataset, version=1, return_X_y=True)
        
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create train and test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=9,
                                                        test_size=size_test)
    # Min-Max regularization
    X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
    X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())
    
    # Perform logistic regression
    clf = LogisticRegression(penalty='none', 
                                 tol=0.1,
                                 solver='saga',
                                 multi_class='multinomial').fit(X_train_scaled, y_train)
    
    # Evaluation of logistic regression classifier 
    # Extract predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Compute the classification metrics
    classification_metrics = metrics.classification_report(y_test, y_pred)
    
    # Print metrics to the terminal
    print("\n[INFO] Below are the classification metrics. These can also be found as CSV-file in output folder\n")
    print(classification_metrics)
    
    # Save classification metrics report to output folder as csv-file
    # Convert string to dataframe with io and pandas
    classification_metrics_df = pd.read_csv(io.StringIO(classification_metrics), sep=",")
    # Convert dataframe to csv with pandas
    classification_metrics_df.to_csv(os.path.join("..", "output", "lr_classification_metrics.csv"))

    # Create confusion matrix to visualize performance of the model
    confusion_matrix = clf_util.plot_cm(y_test, y_pred, normalized=True)
    
    # Save confusion matrix as .png to output folder
    plt.savefig(os.path.join("..", "output", "confusion_matrix.png"), dpi = 300, bbox_inches = "tight")
    
    # User message
    print("\n[INFO] Done! You have now performed the logistic regression classification. Classification metrics are saved in output directory together with the confusion matrix\n")
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()