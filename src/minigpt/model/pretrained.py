"""Load Hugging Face GPT-2 weights into the local MiniGPT implementation."""

import torch

from configs.gpt_config import GPT2_PRETRAINED_CONFIG_124M
from .gpt import MiniGPT


def _model_config(source, context_size):
    """Reject GPT-2 variants whose forward pass the local model cannot reproduce."""
    if source.model_type != "gpt2":
        raise ValueError("The pretrained model must use the GPT-2 architecture.")
    if source.vocab_size != 50257:
        raise ValueError("The pretrained model must use the standard GPT-2 vocabulary (50257 tokens).")
    if source.activation_function != "gelu_new" or source.layer_norm_epsilon != 1e-5:
        raise ValueError("GPT-2 loading requires gelu_new and layer_norm_epsilon=1e-5.")
    if source.n_inner not in (None, 4 * source.n_embd):
        raise ValueError("GPT-2 loading requires a feed-forward dimension of 4 * n_embd.")
    if (
        source.add_cross_attention
        or not source.scale_attn_weights
        or source.scale_attn_by_inverse_layer_idx
        or source.reorder_and_upcast_attn
        or not source.tie_word_embeddings
    ):
        raise ValueError("Unsupported GPT-2 attention settings or untied output embeddings.")
    if not source.embd_pdrop == source.attn_pdrop == source.resid_pdrop:
        raise ValueError("MiniGPT requires matching embedding, attention, and residual dropout rates.")

    context_size = source.n_positions if context_size is None else context_size
    if context_size > source.n_positions:
        raise ValueError(f"context_size cannot exceed the pretrained capacity ({source.n_positions}).")
    return {
        **GPT2_PRETRAINED_CONFIG_124M,
        "vocab_size": source.vocab_size,
        "context_size": context_size,
        "embedding_dim": source.n_embd,
        "head_num": source.n_head,
        "transformer_block_num": source.n_layer,
        "drop_rate": source.embd_pdrop,
    }


def _convert_weights(weights, model, config):
    """Map every trainable parameter explicitly; keep only locally generated masks."""
    converted = {
        "token_embedding.weight": weights["transformer.wte.weight"],
        "position_embedding.weight": weights["transformer.wpe.weight"][:config["context_size"]],
        "final_norm.scale": weights["transformer.ln_f.weight"],
        "final_norm.shift": weights["transformer.ln_f.bias"],
        "output_head.weight": weights["lm_head.weight"],
    }
    for index, block in enumerate(model.transformer_blocks):
        source = f"transformer.h.{index}"
        target = f"transformer_blocks.{index}"
        # HF Conv1D stores [input, output]; nn.Linear stores [output, input].
        qkv_weights = weights[f"{source}.attn.c_attn.weight"].split(config["embedding_dim"], dim=1)
        qkv_biases = weights[f"{source}.attn.c_attn.bias"].split(config["embedding_dim"], dim=0)
        if len(qkv_weights) != 3 or len(qkv_biases) != 3:
            raise ValueError(f"Invalid combined Q/K/V dimensions in layer {index}.")
        for name, weight, bias in zip(("W_query", "W_key", "W_value"), qkv_weights, qkv_biases):
            converted[f"{target}.attention.{name}.weight"] = weight.T
            converted[f"{target}.attention.{name}.bias"] = bias

        for source_name, target_name in (
            ("attn.c_proj", "attention.output_proj"),
            ("mlp.c_fc", "feed_forward.layers.0"),
            ("mlp.c_proj", "feed_forward.layers.2"),
        ):
            # The square attention projection must also be transposed.
            converted[f"{target}.{target_name}.weight"] = weights[f"{source}.{source_name}.weight"].T
            converted[f"{target}.{target_name}.bias"] = weights[f"{source}.{source_name}.bias"]
        for source_name, target_name in (
            ("ln_1", "norm_attention"),
            ("ln_2", "norm_feed_forward"),
        ):
            converted[f"{target}.{target_name}.scale"] = weights[f"{source}.{source_name}.weight"]
            converted[f"{target}.{target_name}.shift"] = weights[f"{source}.{source_name}.bias"]
        converted[f"{target}.attention.mask"] = block.attention.mask
    return converted


def load_pretrained_gpt2(
    model_name="gpt2", *, context_size=None, device="cpu", cache_dir=None,
    local_files_only=False,
):
    """Return (model, model_config), ready for inference or subsequent fine-tuning.

    model_name accepts a Hugging Face GPT-2 repository or local model directory.
    context_size defaults to the source capacity; a shorter size slices its
    position embeddings. Training seq_length is configured independently.
    The returned model uses float32, shared input/output embeddings, and eval
    mode. Call model.train() before fine-tuning, then create the optimizer.
    Transformers is optional for existing MiniGPT checkpoint loading.
    """
    if context_size is not None and (
        not isinstance(context_size, int) or isinstance(context_size, bool) or context_size < 1
    ):
        raise ValueError("context_size must be a positive integer or None.")
    device = torch.device(device)
    try:
        from transformers import AutoConfig, GPT2LMHeadModel
    except ImportError as error:
        raise ImportError(
            'GPT-2 loading requires Transformers. Install it with: '
            'python -m pip install "transformers>=4.40,<5"'
        ) from error

    model_name = "openai-community/gpt2" if str(model_name) == "gpt2" else model_name
    options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    source_config = AutoConfig.from_pretrained(model_name, **options)
    config = _model_config(source_config, context_size)
    reference, loading_info = GPT2LMHeadModel.from_pretrained(
        model_name, config=source_config, torch_dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True, **options,
    )
    if any(loading_info.get(name) for name in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete GPT-2 pretrained weights: {loading_info}")

    model = MiniGPT(config)
    try:
        converted = _convert_weights(reference.state_dict(), model, config)
    except KeyError as error:
        raise ValueError(f"Missing GPT-2 pretrained parameter: {error.args[0]}") from error
    model.load_state_dict(converted, strict=True)
    # Release the reference model and its tensors before moving MiniGPT to GPU.
    del converted, reference
    model.to(device)
    model.eval()
    return model, config
