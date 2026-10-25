import torch
from torch import nn
from .layers import LayerNorm
from .transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )
        self.position_embedding = nn.Embedding(
            config["context_size"], config["embedding_dim"]
        )
        self.drop_embedding = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["transformer_block_num"])]
        )

        self.final_norm = LayerNorm(config["embedding_dim"])
        self.output_head = nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False
        )

    def forward(self, x_embedding_idx):
        batch_size, seq_len = x_embedding_idx.shape

        token_embedding = self.token_embedding(x_embedding_idx)
        position_embedding = self.position_embedding(torch.arange(seq_len, device=x_embedding_idx.device))

        x = token_embedding + position_embedding
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits
