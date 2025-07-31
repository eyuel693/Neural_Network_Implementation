# Neural Network Implementation from Scratch
## Overview
---
This repository implements a modular, from-scratch neural network framework using NumPy and pandas. It includes activation functions such as Sigmoid, ReLU, and Softmax, and supports both forward and backward propagation, as well as gradient descent for training.

In a standard neural network workflow, data is typically split into three parts: training data for learning, validation data to monitor overfitting, and test data to evaluate the model's generalization. To handle this, the framework includes a custom data splitter.

The model uses gradient descent to update learnable parameters during backpropagation. To ensure numerical stability and prevent undefined behaviors like log(0), gradient clipping is applied. The cross-entropy loss function is used for measuring model performance.

Data is fed into the model in batches, and hyperparameters such as batch size, number of epochs, number of layers, neurons per layer, input size, and output size can all be configured by the user.

To evaluate results, the framework provides accuracy and precision metrics. It also includes an early stopping mechanism that monitors validation loss to prevent overfitting. Additionally, all input features are normalized for better training efficiency.

The framework supports both Adam and Stochastic Gradient Descent (SGD) optimizers, allowing users to choose based on their needs.
---

## Key Features

- **Activation Functions** sigmoid, ReLU, and softmax.
- **Forward and Backward Propagation**  parameter updates using gradient descent.
- **Loss Function: Uses cross-entropy** loss for classification tasks.
- **Data Splitting** Includes a dataset splitter to divide the data into:
        - **Training set** for learning the parameters.
        - **Validation set**  to prevent overfitting (used for early stopping).
        - **Test set** for final evaluation.
- **Optimization Algorithms**
        - **Stochastic Gradient Descent (SGD)**
        - **Adam Optimizer**

### Requirements
- **Python 3.7+**
- **NumPy**

### Install dependencies

```
bash

pip install -r requirements.txt

```

### Train the Model

```
bash

python neuralnet.py
```