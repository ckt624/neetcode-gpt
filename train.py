import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr)

        for epoch in range(epochs):
            # 1 epoch = 1 batch = 1 step. 
            torch.manual_seed(epoch)
            # random sample start index for a sequence
            starts = torch.randint(len(data) - context_length, (batch_size,))
            # stack: [tensor]-> tensor. tensor: [[]] -> tensor
            X = torch.stack([data[s:s+context_length] for s in starts])
            Y = torch.stack([data[s+1:s+1+context_length] for s in starts])

            Y_hat = model(X)
            # flattern BT as 1d for loss.
            loss = F.cross_entropy(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1))

            # Always. or model.zero_grad().
            optimizer.zero_grad()
            # Calculate grad
            loss.backward()
            # Update weights by grad
            optimizer.step()

        # item: tensor to float. tolist: tensor to list.
        return round(loss.item(), 4)
            
