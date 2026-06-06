import torch
import torch.nn as nn
from typing import List, Dict


class Solution:
    def _format(self, x):
        return round(x.item(), 4)

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        ans = []
        with torch.no_grad():
            for child in model.children():
                x = child(x)
                if isinstance(child, nn.Linear):
                    mean = x.mean()
                    std = x.std()
                    dead_fraction = (x <= 0.).all(dim=0).float().mean()

                    ans.append({'mean': self._format(mean), 'std': self._format(std), 'dead_fraction': self._format(dead_fraction)})

        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()

        ans = []
        for child in model.children():
            if isinstance(child, nn.Linear):
                x = child.weight.grad
                mean = x.mean()
                std = x.std()
                norm = torch.norm(x)

                ans.append({'mean': self._format(mean), 'std': self._format(std), 'norm': self._format(norm)})

        return ans

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for activ in activation_stats:
            if activ['dead_fraction'] > 0.5:
                return 'dead_neurons'

        for grad in gradient_stats:
            if grad['norm'] > 1000:
                return 'exploding_gradients'

        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        
        for activ in activation_stats:
            if activ['std'] > 10:
                return 'exploding_gradients'
            if activ['std'] < 0.1:
                return 'vanishing_gradients'

        return 'healthy'
            
