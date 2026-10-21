import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_size, embedding_dim)

    def forward(self, token_ids):
        sample_length = token_ids.size(1)
        positions = torch.arange(sample_length).to(token_ids.device)
        pos_embeddings = self.pos_embedding(positions)

        token_embeddings = self.token_embedding(token_ids)

        return token_embeddings + pos_embeddings