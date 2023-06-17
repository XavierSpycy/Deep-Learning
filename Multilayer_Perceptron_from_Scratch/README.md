# Repository Contents
‘**MLPClassifier.py**’: This Python script houses the implementation of our multilayer perceptron classifier leveraging NumPy.

‘**main.py**’: This Python script provides an exemplary demonstration of our multilayer perceptron classifier in action.

‘**main.ipynb**’: This Jupyter Notebook contains the same content as main.py, but additionally captures the output for interactive exploration and review.

‘**pytorch.ipynb**’: To demonstrate the efficiency of our model, this Jupyter Notebook features a similar architecture to the multilayer perceptron implemented using PyTorch, following a similar training procedure.

**extend.ipynb**: To demonstrate the flexibility of our model, this Jupyter Notebook showcases the ease with which users can tailor their own unique architectures, adjust hyperparameters, and select different optimizers.

‘**data**’: This directory contains the dataset used for our exemplary demonstration.

You can download the datasets [here](https://www.kaggle.com/datasets/xavierspycy/multilayer-perceptron-datasets).

'**train_data.npy**', a NumPy file which stores the training inputs.

'**train_label.npy**', a NumPy file which stores the training labels.

'**test_data.npy**', a NumPy file which stores the test inputs.

'**test_label.npy**', a NumPy file which stores the test labels.

# Brief introduction of modules in MLPClassifier
## Prerequisite:
'**Scaler**', a class designed to scale data through normalization or standardization, although there is room for further optimization.

'**Dataset**', a class designed to load our dataset, featuring several preprocessing methods. These methods include splitting of training and validation sets, centering, normalization, and addressing missing values, among others.

## Core modules:
'**Activation**', a class which implements a range of activation functions and their corresponding derivatives. Specifically, these functions include sigmoid, hyperbolic tangent, ReLU, Leaky ReLU, ELU, Swish, Softplus functions, and more.

'**HiddenLayer**', a class which realizes the implementation of a hidden layer, which can also serve as a batch normalization layer. This class encompasses functionalities such as weight initialization, dropout, and both forward and backward propagation between layers.

'**MLP**': a class which implements the multilayer perceptron. It integrates the hidden layers (including batch normalization layers), thus defining the propagation mechanism between layers. This class has an embedded criterion that indicates the loss function, such as cross entropy loss. It includes several optimizers like SGD, NAG, Adagrad, Adadelta, Adam, and more. Lastly, it encapsulates the training procedure, prediction method, and performance metric—specifically, accuracy.

# Requirements
To install and run this project, you will need:

Python: Our project uses Python 3.9.16. You can download Python [here](https://www.python.org/downloads/).

NumPy: Our multilayer perceptron implementation relies on NumPy 1.24.2. You can install it with pip:
```
pip install numpy
```
