import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            mu = np.mean(x, axis=0)
            s2 = np.var(x, axis=0)
            
            y = ((x - mu) * (s2 + eps) ** -0.5) * gamma + beta
            running_mean = running_mean * (1 - momentum) + momentum * mu
            running_var = running_var * (1 - momentum) + momentum * s2
        else:
            y = ((x - running_mean) * (running_var + eps) ** -0.5) * gamma + beta

        return (np.round(y, 4).tolist(), np.round(running_mean, 4).tolist(), np.round(running_var, 4).tolist())
