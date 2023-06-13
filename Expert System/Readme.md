# MedMNIST Breast MNIST Expert System

This code provides an expert system for classifying breast pathology using the MedMNIST Breast MNIST dataset. The expert system consists of the following steps:

## Step 1: Instructions and Help

Displays welcome message and provides an overview of the system.

## Step 2: Background Information

Provides information about the MedMNIST Breast MNIST dataset, including image dimensions and class labels.

## Step 3: Load Dataset

Downloads and loads the MedMNIST Breast MNIST dataset using the provided URL. It retrieves the training and testing images and labels.

## Step 4: Preview Image Data

Displays a sample image from the training set along with its corresponding label. The image is shown in grayscale.

## Step 5: Montage of Images

Creates a montage of multiple images from the training set. It displays a specified number of images concatenated horizontally.

## Step 6: Generate Histogram of Dataset

Creates a histogram of the dataset, showing the frequency of each class (benign and malignant) in the training set.

## Step 7: View Separate RGB Color Channels

Converts the sample grayscale image to RGB format and displays its separate red, green, and blue channels.

## Step 8: Train Neural Network

Prepares the dataset for training by reshaping the images and converting them to tensors. It creates data loaders for both the training and testing sets.

Defines a Multi-Layer Perceptron (MLP) model using PyTorch's `nn.Module` class. The model consists of three fully connected layers.

## Step 9: Train and Evaluate the Model

Defines functions for training and evaluating the model. The training function performs forward and backward propagation, updates the model's parameters using the Adam optimizer, and calculates the training loss and accuracy.

The evaluation function calculates the loss and accuracy on the testing set, without updating the model's parameters.

## Step 10: Train the Model

Trains the model for a specified number of epochs using the training and evaluation functions. Displays the training loss and testing accuracy for each epoch.

## Step 11: Plot Training and Testing Accuracy

Plots the training and testing accuracy over the epochs using Matplotlib.

## Step 12: Expert System Style Q/A for Patient Follow-up

Defines an expert system function that takes an input image and classifies the tumor as benign or malignant. It uses the trained model to make predictions based on the input image.

Provides an example input image from the testing set and prints the expert system's classification result.

---

**Note**: Make sure to install the required dependencies (such as PyTorch, NumPy, and Matplotlib) before running the code.
