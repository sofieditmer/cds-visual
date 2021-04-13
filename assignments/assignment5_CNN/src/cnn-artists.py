#!/usr/bin/env python

"""
This script builds a deep learning model using LeNet as the convolutional neural network architecture. This network is used to classify impressionist paintings by their artists. 

Usage:
    $ python cnn-artists.py
"""

### DEPENDENCIES ###

# Data tools
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import argparse
from contextlib import redirect_stdout


# Sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to training data
    ap.add_argument("-t", "--train_data",
                    type = str,
                    required = False,
                    help = "Path to the training data",
                    default = "../../data/subset_impressionist_classifier_data/training/training")
    
    # Argument 2: Path to test data
    ap.add_argument("-te", "--test_data",
                    type = str,
                    required = False,
                    help = "Path to the test/validation data",
                    default = "../../data/subset_impressionist_classifier_data/validation/validation")
    
    # Argument 3: Number of epochs
    ap.add_argument("-e", "--n_epochs",
                    type = int,
                    required = False,
                    help = "The number of epochs to train the model on",
                    default = 20)
    
    # Argument 4: Batch size
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False,
                    help = "The size of the batch on which to train the model",
                    default = 32)
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    train_data = args["train_data"]
    test_data = args["test_data"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    
    # Create output directory
    if not os.path.exists(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))

    # Start message to user
    print("\n[INFO] Initializing the construction of a LeNet convolutional neural network model...")
    
    # Create list of label names
    label_names = listdir_nohidden(train_data)
    
    # Find the optimal dimensions to resize the images 
    print("\n[INFO] Estimating the optimal image dimensions to resize images...")
    min_height, min_width = find_image_dimensions(train_data, test_data, label_names)
    print(f"\n[INFO] Input images are resized to dimensions of height = {min_height} and width = {min_width}...")
    
    # Create trainX and trainY
    print("\n[INFO] Resizing training images and creating training data, trainX, and labels, trainY...")
    trainX, trainY = create_trainX_trainY(train_data, min_height, min_width, label_names)
    
    # Create testX and testY
    print("\n[INFO] Resizing validation images and creating validation data, testX, and labels, testY...")
    testX, testY = create_testX_testY(test_data, min_height, min_width, label_names)
    
    # Normalize data and binarize labels
    print("\n[INFO] Normalize training and validation data and binarizing training and validation labels...")
    trainX, trainY, testX, testY = normalize_binarize(trainX, trainY, testX, testY)
    
    # Define model
    print("\n[INFO] Defining LeNet model architecture...")
    model = define_LeNet_model(min_width, min_height)
    
    # Train model
    print("\n[INFO] Training model...")
    H = train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size)
    
    # Plot loss/accuracy history of the model
    plot_history(H, n_epochs)
    
    # Evaluate model
    print("\n[INFO] Evaluating model... Below is the classification report. This can also be found in the out folder.\n")
    evaluate_model(model, testX, testY, batch_size, label_names)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained a convolutional neural network on impressionist paintings that can classify paintings by their artists\n")
    
### FUNCTIONS USED WITHIN MAIN FUNCTION ###

def listdir_nohidden(path):
    """
    Define the label names by listing the names of the folders within training directory without listing hidden files. 
    """
    # Create empty list
    label_names = []
    
    # For every name in training directory
    for name in os.listdir(path):
        # If it does not start with . (which hidden files do)
        if not name.startswith('.'):
            label_names.append(name)
            
    return label_names

def find_image_dimensions(train_data, test_data, label_names):
    """
    Function that estimates the optimal dimensions (height and width) of the input images to be used when the images are to be size normalized. Since all of the images are of different shapes and sizes, we need to resize them to be a uniform, smaller shape. Hence, finding the smallest image and resizing all other images to fit the dimensions of that will do.
    """
    # Create empty lists
    heights_train = []
    widths_train = []
    heights_test = []
    widths_test = []
    
    # Loop through directories for each painter
    for name in label_names:
        
        # Take images in train data
        train_images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # Loop through images in training data
        for image in train_images:
            # Load image
            loaded_img = cv2.imread(image)
            
            # Find dimensions of each image
            height, width, _ = loaded_img.shape
        
            # Append to lists
            heights_train.append(height)
            widths_train.append(width)
        
        # Take images in test data
        test_images = glob.glob(os.path.join(test_data, name, "*.jpg"))
        
        # Loop through images in test data
        for image in test_images:
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Find dimensions of each image
            height, width, _ = loaded_img.shape
        
            # Append to lists
            heights_test.append(height)
            widths_test.append(width)
            
    # Find the smallest image dimensions among all images 
    min_height = min(heights_train + heights_test + widths_train + widths_test)
    min_width = min(heights_train + heights_test + widths_train + widths_test)
    
    return min_height, min_width


def create_trainX_trainY(train_data, min_height, min_width, label_names):
    """
    This function creates the trainX and trainY which contain the training data and its labels respectively. 
    """
    # Create empty array and list
    trainX = np.empty((0, min_height, min_width, 3))
    trainY = []
    
    # Loop through images in training data
    for name in label_names:
        images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image with the specified dimensions
            resized_img = cv2.resize(loaded_img, (min_width, min_height), interpolation = cv2.INTER_AREA)
        
            # Create array of image
            image_array = np.array([np.array(resized_img)])
        
            # Append
            trainX = np.vstack((trainX, image_array))
            trainY.append(name)
        
    return trainX, trainY


def create_testX_testY(test_data, min_height, min_width, label_names):
    """
    This function creates testX and testY which contain the test/validation data and its labels respectively. 
    """
    # Create empty array and list
    testX = np.empty((0, min_height, min_width, 3))
    testY = []
    
    # Loop through images in test data
    for name in label_names:
        images = glob.glob(os.path.join(test_data, name, "*.jpg"))
    
    # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image
            resized_img = cv2.resize(loaded_img, (min_width, min_height), interpolation = cv2.INTER_AREA)
        
            # Create array
            image_array = np.array([np.array(resized_img)])
        
            # Append
            testX = np.vstack((testX, image_array))
            testY.append(name)
        
    return testX, testY


def normalize_binarize(trainX, trainY, testX, testY):
    """
    This function normalizes the training and test data and binarizes the training and test labels. 
    """
    
    # Normalize training and test data
    trainX_norm = trainX.astype("float") / 255.
    testX_norm = testX.astype("float") / 255.
    
    # Binarize training and test labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    return trainX_norm, trainY, testX_norm, testY


def define_LeNet_model(min_width, min_height):
    """
    This function defines the LeNet model architecture.
    """
    # Define model
    model = Sequential()

    # Add first set of convolutional layer, ReLu activation function, and pooling layer
    # Convolutional layer
    model.add(Conv2D(32, (3, 3), 
                     padding="same", 
                     input_shape=(min_height, min_width, 3)))
    
    # Activation function
    model.add(Activation("relu"))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2))) # stride of 2 horizontal, 2 vertical
    
    # Add second set of convolutional layer, ReLu activation function, and pooling layer
    # Convolutional layer
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    
    # Activation function
    model.add(Activation("relu"))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    
    # Add fully-connected layer
    model.add(Flatten()) # flattening layer
    model.add(Dense(500)) # dense network with 500 nodes
    model.add(Activation("relu")) # activation function
    
    # Add output layer
    # softmax classifier
    model.add(Dense(10)) # dense layer of 10 nodes used to classify the images
    model.add(Activation("softmax"))

    # Define optimizer 
    opt = SGD(lr=0.01)
    
    # Compile model
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt, 
                  metrics=["accuracy"])
    
    # Model summary
    model_summary = model.summary()
    
    # Save model summary
    with open('../out/model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    # Visualization of model
    plot_LeNet_model = plot_model(model,
                                  to_file = "../out/LeNet_model.png",
                                  show_shapes=True,
                                  show_layer_names=True)
    
    return model


def train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size):
    """
    This function trains the LeNet model on the training data and validates it on the test data.
    """
    # Train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=batch_size, 
                  epochs=n_epochs, verbose=1)
    
    return H
    
    
def plot_history(H, n_epochs):
    """
    Function that plots the loss/accuracy of the model during training.
    """
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../out/model_history.png")
    
    
def evaluate_model(model, testX, testY, batch_size, label_names):
    """
    This function evaluates the trained model and saves the classification report in output directory. 
    """
    # Predictions
    predictions = model.predict(testX, batch_size=batch_size)
    
    # Classification report
    classification = classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names)
            
    # Print classification report
    print(classification)
    
    # Save classification report
    with open("../out/classification_report.txt", 'w', encoding='utf-8') as f:
        f.writelines(classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names))

# Define behaviour when called from command line
if __name__=="__main__":
    main()