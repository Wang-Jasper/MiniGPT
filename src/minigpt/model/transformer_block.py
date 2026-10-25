from torch import nn
from .attention import MutiHeadAttention
from .layers import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["embedding_dim"] % config["head_num"] != 0:
            raise ValueError("embedding_dim must be divisible by head_num")

        self.attention = MutiHeadAttention(
            head_num=config["head_num"],
            qkv_dim=config["embedding_dim"] // config["head_num"],
            embedding_dim=config["embedding_dim"],
            context_size=config["context_size"],
        )
        self.feed_forward = FeedForward(config)
        self.norm_attention = LayerNorm(config["embedding_dim"])
        self.norm_feed_forward = LayerNorm(config["embedding_dim"])
        self.drop_residual = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Residual Connection: y = x + F(x)
        # 1. Easier optimization: Learns residual corrections instead of the full mapping.
        # 2. Better gradient flow: Provides an identity gradient path (I + J_F), mitigating vanishing gradients without completely preventing them.
        # 3. Feature preservation: Allows original features to bypass transformations and contribute directly to the output.

        # residual for attention block
        residual = x
        x = self.norm_attention(x)
        x = self.attention(x)
        x = x.dropout(x)
        x = x + residual

        # residual for feed forward network
        residual = x
        x = self.norm_feed_forward(x)
        x = self.feed_forward(x)
        x = self.drop_residual(x)
        x = x + residual

        return x



