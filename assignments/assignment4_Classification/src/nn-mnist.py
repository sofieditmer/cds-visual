#!/usr/bin/env python

"""
Given the full MNIST data set, a neural network classifier is trained, and the evaluation metrics are printed to the terminal.

Parameters:
    test_size: num <size-of-test-data>
    epochs: num <number-of-epochs>
    
Usage:
    nn-mnist.py --test_size <size-of-test-data> --epochs <number-of-epochs>

Example:
    $ python lr-mnist.py --test_size 0.2 --epochs 1000

Output:
    nn_classification_metrics.csv: Neural network classification metrics
"""

### DEPENDENCIES ###
import sys,os
sys.path.append(os.path.join(".."))
import pandas as pd
import numpy as np
import io
from utils.neuralnetwork import NeuralNetwork 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import fetch_openml
import argparse

### MAIN FUNCTION ###

def main():
    
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: size of the test data
    ap.add_argument("-ts", "--test_size", 
                    type = float,
                    required = False, 
                    help = "Define the size of the test data", 
                    default = 0.2)
    
    # Argument 2: number of epochs
    ap.add_argument("-e", "--epochs", 
                    type = int,
                    required = False, 
                    help = "Define the number of epochs", 
                    default = 1000)
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    test_size = args["test_size"]
    epochs = args["epochs"]
    
    # Create output directory
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
        
    # Start message to user
    print("\n[INFO] Initializing neural network classification...\n")
    
    # Load data
    digits = datasets.load_digits()
    
    # Convert to floats
    data = digits.data.astype("float")
    
    # Perform min-max regularization of data
    data = (data - data.min())/(data.max() - data.min())
    
    # Print dimensions of data
    print(f"[INFO] samples: {data.shape[0]}, dim: {data.shape[1]}\n")
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                  digits.target, 
                                                  test_size=test_size)
    
    # Binarize training labels
    y_train = LabelBinarizer().fit_transform(y_train)
    
    # Binarize test labels
    y_test = LabelBinarizer().fit_transform(y_test)
    
    # Train the neural network
    print("[INFO] Training network...")
    nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
    
    # Fit the neural network
    nn.fit(X_train, y_train, epochs=epochs)
    
    # Evaluate the neural network
    print("\n[INFO] Evaluating network...\n")
    
    # Compute predictions
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    
    # Classification report
    print("\n[INFO] Below are the classification metrics. These can also be found in output directory\n")
    classification_metrics = classification_report(y_test.argmax(axis = 1), predictions)
        
    # Print classification report to terminal
    print(classification_metrics)
    
    # Save classification metrics to output directory
    # Convert string to dataframe with io and pandas
    classification_metrics_df = pd.read_csv(io.StringIO(classification_metrics), sep=",")
    # Convert dataframe to csv with pandas
    classification_metrics_df.to_csv(os.path.join("..", "output", "nn_classification_metrics.csv"))

    # Message to user
    print("\n[INFO] Done! You have now performed the neural network classification.\n")
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()