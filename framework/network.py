from typing import List

import numpy as np

from framework.layer import Layer, Loss
from framework.layers.sequential import Sequential


def train_one_step(model: Layer, loss: Loss, learning_rate: float, input: np.ndarray, target: np.ndarray, weight_decay=0.0005) -> float:
    # Forward pass
    pred = model.forward(input)

    # Compute loss value (using loss function passed as parameter)
    loss_value = loss.forward(pred, target)

    # Compute loss function gradient: dL/dy
    loss_gradients = loss.backward()

    # Compute gradient of each learnable parameter (weights and biases)
    back_results = model.backward(loss_gradients)
    variable_gradients = back_results.variable_grads

    # Update the weights and biases 
    for key, grad in variable_gradients.items():
        # Apply weight decay
        grad += weight_decay * model.var[key]
        # Update the weights and biases
        model.var[key] -= learning_rate * grad

    return loss_value

def create_network(layers: List[Layer]) -> Layer:
    # Create a network that is a sequence of layers
    return Sequential(layers)

def train_one_step(model: Layer, loss: Loss, learning_rate: float, input: np.ndarray, target: np.ndarray, weight_decay=0.0005) -> float:
    # Forward pass
    pred = model.forward(input)

    # Compute loss value (using loss function passed as parameter)
    loss_value = loss.forward(pred, target)

    # Compute loss function gradient: dL/dy
    loss_gradients = loss.backward()

    # Compute gradient of each learnable parameter (weights and biases)
    back_results = model.backward(loss_gradients)
    variable_gradients = back_results.variable_grads

    # Update the weights and biases 
    for key, grad in variable_gradients.items():
        # Apply weight decay
        grad += weight_decay * model.var[key]
        # Update the weights and biases
        model.var[key] -= learning_rate * grad

    return loss_value
