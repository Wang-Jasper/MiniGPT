GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_size": 256,
    "embedding_dim": 768,
    "head_num": 12,
    "transformer_block_num": 12,
    "drop_rate": 0.1,
}

# GPT-2 124M pretrained architecture
GPT2_PRETRAINED_CONFIG_124M = {
    **GPT_CONFIG_124M,
    "context_size": 1024,
    "qkv_bias": True,
    "output_proj_bias": True,
    "tie_embeddings": True,
}
