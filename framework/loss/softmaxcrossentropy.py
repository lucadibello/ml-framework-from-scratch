from framework.layer import Loss

import numpy as np


class SoftmaxCrossEntropy(Loss):
    def forward(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        y = inputs
        t = targets
        n = inputs.shape[0]
        num_classes = y.shape[1]

        ## Remember that the here targets are not one-hot vectors, but integers specifying the class.
        ## The loss function has to return a single scalar, so we have to take the mean over the batch dimension.
        ## Don't forget to save your variables needed for backward to self.saved_variables.

        # Convert the target indices to one-hot vectors

        # Matrix (n, num_classes) with all zeros
        one_hot_t = np.zeros((n, num_classes))

        # For each class, we set the corresponding index to 1
        for i in range(n):
            one_hot_t[i, t[i]] = 1

        # Numerically stable softmax computation
        # Source: https://www.parasdahal.com/softmax-crossentropy

        # Compute the softmax
        shifted_logits = y - np.max(y, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted_logits)
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # Compute the cross-entropy loss
        # H(y, softmax) = - sum(t * log(softmax))
        mean_crossentropy = -np.sum(one_hot_t * np.log(softmax)) / n

        self.saved_variables = {
            "S": softmax,
            "t": one_hot_t,
            "n": n
        }

        return mean_crossentropy

    def backward(self) -> np.ndarray:
        softmax = self.saved_variables["S"]
        one_hot_t = self.saved_variables["t"]
        n = self.saved_variables["n"]

        ## Implement

        # To compute the gradient of the cross-entropy loss with respect to the input of the softmax layer,
        # we can use the following formula:
        # dL/dy = softmax - t

        d_inputs = (softmax - one_hot_t) / n

        ## End

        assert d_inputs.shape == softmax.shape, f"Error shape doesn't match prediction: {d_inputs.shape} {softmax.shape}"
        self.saved_variables = None
        return d_inputs