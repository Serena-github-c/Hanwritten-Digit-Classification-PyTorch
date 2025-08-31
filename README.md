# MNIST Handwritten Digits Classifier with PyTorch

Purpose of this project : developing, training and evaluating a neural netowrk to classify ahndwritten digits from the MNIST dataset using PyTorch.
The final model achieved an accuracy of 97.96% on the test set.
Project Overview

## Steps
- Load and preprocess the MNIST dataset.
- Design a neural network architecture suitable for image classification.
- Train the model using PyTorch, monitoring loss and accuracy.
- Experiment with different optimizers, learning rates, and a learning rate scheduler to improve performance.
- Evaluate the trained model on the test set.
- Save the best-performing model.

## Dataset
MNIST dataset:
- A training set of 60,000 28x28 grayscale images of handwritten digits (0-9).
- A test set of 10,000 28x28 grayscale images.
The data was normalized using the standard MNIST mean (0.1307) and standard deviation (0.3081).


## Tech 
- Python 3
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

### Model Architecture & Training

The primary model architecture consists of:

* A Flatten layer to convert 28x28 images into a 784-feature vector.
* Two hidden fully connected (Linear) layers with ReLU activations.
    - layer_1: 784 input features -> 256 output features
    - layer_2: 256 input features -> 128 output features
* An output fully connected layer with 10 output features (for digits 0-9).
    - layer_3: 128 input features -> 10 output features

- Optimizer : Adam
- Loss : CrossEntropyLoss

### Results

The best model achieved an accuracy of 97.96% on the 10,000 MNIST test images. Training and validation loss/accuracy were plotted over epochs to monitor performance and ensure the model was learning effectively without significant overfitting.
