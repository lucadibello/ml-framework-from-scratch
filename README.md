# Neural Network Framework (From Scratch)

This project presents a simple neural network framework implemented entirely from scratch in Python. The framework is designed to help users understand the foundational concepts behind neural networks and machine learning while providing a user-friendly API inspired by popular deep learning libraries such as Keras and PyTorch.

### Features

- **Custom Implementation**: No external machine learning libraries are used, making this framework educational and lightweight.
- **Layer Support**: Fully connected (dense) layers are supported.
- **Modular Design**: The framework is modular, making it easy to define and experiment with different network architectures.
- **Basic API**: Inspired by Keras and PyTorch, the API is intuitive and minimalistic.

### Implemented Components

#### Layers

- **`Layer`**: Base class for all layers in the neural network.
- **`Linear`**: Implements a fully connected layer.
- **`Sequential`**: A container for stacking layers sequentially.
- **`Activation`**: Implements activation functions for layers.

#### Activation Functions

- **`ReLU`**: Rectified Linear Unit activation function.

#### Loss Functions

- **`SoftmaxCrossEntropy`**: Combines softmax activation and cross-entropy loss for classification tasks.

### Example Usage

The following code demonstrates how to define a simple feedforward neural network for a classification task with 2 input features, 3 hidden layers, and 2 output units.

```python
# Import model components
from framework.layers import Sequential, Linear
from framework.activations import ReLU
from framework.loss import SoftmaxCrossEntropy
from framework.network import train_one_step

# Define the model
model = Sequential([
    Linear(2, 20), ReLU(),    # Input layer (2D data) with a larger hidden layer
    Linear(20, 10), ReLU(),   # Second hidden layer
    Linear(10, 5), ReLU(),    # Third hidden layer
    Linear(5, 2),             # Output layer for classification
])

# Define the loss function
loss = SoftmaxCrossEntropy()

# Define hyperparameters
learning_rate = 0.02 # Adjust as needed
weight_decay = 0.0005 # L2 regularization parameter
epochs = 10

# Example training step
inputs = [[0.5, -1.2], [1.3, 3.5]]  # Replace with your data
labels = [0, 1]  # Replace with your labels

for i in range(epochs):
    curr_loss = train_one_step(
        model, # reference to the model object
        loss, # reference to the loss object
        learning_rate, # learning rate
        inputs, # training data
        labels, # training labels
        weight_decay # L2 regularization parameter
    )
    print(f"Epoch {epoch + 1}, Loss: {loss_value}")
```

This repository provides two example:

1. **Non-linear binary classification**: A simple example demonstrating how to train a neural network for binary classification using synthetic data (two ellipse shapes).

2. **MNIST classification**: A more complex example demonstrating how to train a neural network for classifying handwritten digits from the MNIST dataset.

### Key Features in Development

1. **Backpropagation**: Each layer implements methods for calculating gradients for weights and biases.
2. **Weight Updates**: Gradient-based weight update rules using optimizers (e.g., SGD, Adam).
3. **Training Loop**: Simple training loop with forward and backward propagation steps.

### Installation

Clone this repository and ensure you have Python 3.6 or later installed. No additional libraries are required except for basic dependencies like `numpy`.

```bash
git clone <repository-url>
cd neural-network-framework
pip install -r requirements.txt  # Optional, if any external libraries are used
```
