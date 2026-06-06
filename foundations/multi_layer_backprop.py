import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        n = len(y_true)
        x = np.array(x).reshape(1, -1)
        b1 = np.array(b1)
        W1 = np.array(W1)
        b2 = np.array(b2)
        W2 = np.array(W2)
        y_true = np.array(y_true)

        z1 = x @ W1.T + b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ W2.T + b2
        
        loss = np.average((y_true - z2) ** 2)
        db2 = (z2 - y_true) * 2 / n
        dW2 = db2.T @ a1
        db1 = db2 @ W2 * (z1 > 0)
        dW1 = db1.T @ x

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1.reshape(-1), 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2.reshape(-1), 4).tolist(),
        } 


