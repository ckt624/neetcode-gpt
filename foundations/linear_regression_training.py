import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64]):
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        for _ in range(num_iterations):
            y_hat = self.get_model_prediction(X, initial_weights)
            initial_weights -= self.learning_rate * self.get_derivative(y_hat, Y, len(Y), X)

#            initial_weights[0] -= self.learning_rate * self.get_derivative(y_hat, Y, len(Y), X, 0)
#            initial_weights[1] -= self.learning_rate * self.get_derivative(y_hat, Y, len(Y), X, 1)
#            initial_weights[2] -= self.learning_rate * self.get_derivative(y_hat, Y, len(Y), X, 2)
        
        return np.round(initial_weights, 5)

