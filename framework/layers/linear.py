import numpy as np

from framework.layer import Layer


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.var = {
            "W": np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size)),
            "b": np.zeros((output_size), dtype=np.float32)
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        W = self.var['W'] # weights
        b = self.var['b'] # bias

        # Compute the linear transformation of x
        y =  np.dot(x, W) + b

        # Save the input for the backward pass
        self.saved_variables = {
            "input": x,
        }

        # return the result
        return y

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        W = self.var['W'] # weights
        x = self.saved_variables["input"]

        ## Implement

        # Gradient of the loss function L with respect to the weights W
        # --- Math resoning ---
        # dL/dW = dL/dy * dy/dW
        # where:
        #   - dy/dW = x (input transposed for alignment)
        #   - dL/dy = grad_in (gradient from the next layer)
        # For a batch of inputs, the gradient is computed as:
        # dL/dW = x^T * grad_in
        dW = x.T @ grad_in

        # Gradient of the loss function L with respect to the bias b
        # --- Math resoning ---
        # dL/db = dL/dy * dy/db
        # where:
        #   - dL/dy = grad_in (gradient from the next layer)
        #   - dy/db = 1 (bias only shifts the output, so its derivative is 1)
        # Therefore, for each element of the bias vector b:
        #   dL/db[j] = sum of all gradients dL/dy[i, j] over the batch dimension i
        # For a batch of inputs, we compute the sum along axis 0 to aggregate gradients for each bias term:
        db = grad_in.sum(axis=0)

        # Gradient of the loss function L with respect to the input x
        # --- Math resoning ---
        # dL/dx = dL/dy * dy/dx
        # where:
        #   - dL/dy = grad_in (gradient from the next layer)
        #   - dy/dx = W (weights) 
        d_inputs = grad_in @ W.T

        # ensure dimensions match
        assert d_inputs.shape == x.shape, "Input: grad shape differs: %s %s" % (d_inputs.shape, x.shape)
        assert dW.shape == self.var["W"].shape, "W: grad shape differs: %s %s" % (dW.shape, self.var["W"].shape)
        assert db.shape == self.var["b"].shape, "b: grad shape differs: %s %s" % (db.shape, self.var["b"].shape)

        self.saved_variables = None
        updates = {"W": dW,
                   "b": db}
        return Layer.BackwardResult(updates, d_inputs)