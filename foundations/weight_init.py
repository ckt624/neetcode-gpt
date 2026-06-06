import torch
import torch.nn as nn
import math
import numpy as np
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        return torch.round(torch.randn(fan_out, fan_in) * (2 / (fan_out + fan_in)) ** 0.5, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        return torch.round(torch.randn(fan_out, fan_in) * (2 / fan_in) ** 0.5, decimals=4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        weights = []
        torch.manual_seed(0)
        fan_in = lambda i: input_dim if i == 0 else hidden_dim
        for i in range(num_layers):
            if init_type == 'xavier':
                std = (2 / (hidden_dim + fan_in(i))) ** 0.5
            elif init_type == 'kaiming':
                std = (2 / fan_in(i)) ** 0.5
            else:
                std = 1
            
            weights.append(torch.randn(hidden_dim, fan_in(i)) * std)

        x = torch.randn(1, input_dim)
        ans = []
        for w in weights:
            x @= w.T
            x = torch.relu(x)
            ans.append(round(x.std().item(), 2))
        return ans


