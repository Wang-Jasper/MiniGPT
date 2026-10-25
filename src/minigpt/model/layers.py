import math

import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    '''
    input x -> (batch_size, seq_length, embedding_dim)

    Linear 1
    W1 -> (4 * embedding_dim, embedding_dim)
    b1 -> (4 * embedding_dim)
    output W1 @ x + b1 -> (4 * embedding_dim, seq_length)
    by increasing the dimension, the features can learn many combinations and form new features

    After GELU
    Linear 2
    W2 -> (embedding_dim, 4 * embedding_dim)
    b2 -> (4 * embedding_dim)
    output W2 @ x + b2 -> (embedding_dim, seq_length)
    '''
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
            GELU(),
                nn.Linear(4 * config["embedding_dim"], config["embedding_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

