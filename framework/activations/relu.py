import numpy as np

from framework.layer import Layer


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        result = np.maximum(0, x)
        self.saved_variables = {"result": result}
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        # result of forward pass, needed to compute the derivative of ReLu(X) -> ReLu'(X)
        relu_x = self.saved_variables["result"]
        # Derviative of ReLU: f'(x) = 1 if x > 0 else 0
        d_x = np.where(relu_x > 0, 1, 0) * grad_in

        # ensure dimensions match
        assert d_x.shape == relu_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, relu_x.shape)
        
        # Clean up internal state + return backward result
        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)