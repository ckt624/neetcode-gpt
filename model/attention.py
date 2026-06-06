import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.k(embedded) # B, T, attention_dim
        q = self.q(embedded)
        v = self.v(embedded)

        y = q @ k.transpose(-1, -2) * k.shape[2] ** -0.5 # B, T, T
        mask = torch.tril(torch.ones(k.shape[1], k.shape[1]))
        y = y.masked_fill(mask == 0, float('-inf'))
        y = nn.functional.softmax(y, dim=-1) @ v

        return torch.round(y, decimals=4)

