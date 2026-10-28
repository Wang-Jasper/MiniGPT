"""Generate text using temperature and top-k sampling or greedy decoding."""

import torch


def text_to_token_ids(text, tokenizer):
    """Encode text as an integer tensor of shape (1, seq_length)"""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """Decode a single token sequence with its leading batch dimension"""
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate_token_ids(
    model,
    token_ids,
    max_new_tokens,
    context_size,
    temperature=0.8,
    top_k=50,
    eos_id=None,
    progress_callback=None,
):
    """Append next-token predictions and restore the original model mode.

    token_ids - Prompt IDs shaped (batch_size, sequence_length).
    max_new_tokens - Number of tokens to append.
    context_size - Maximum number of recent tokens passed to the model.
    temperature - Sampling temperature; 0 uses greedy decoding.
    top_k - Number of highest-scoring tokens to retain; None keeps all.
    eos_id - End token ID, or None to generate the full requested length.
    progress_callback - Optional callback receiving completed and requested token counts.

    Returned IDs include EOS. Finished rows receive EOS padding until all
    rows finish or max_new_tokens is reached.
    """
    if temperature < 0:
        raise ValueError("temperature must be nonnegative.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive or None.")

    token_ids = token_ids.to(next(model.parameters()).device)
    finished = torch.zeros(
        (token_ids.shape[0], 1), dtype=torch.bool, device=token_ids.device
    )   # shape(batch_size, 1), a bool for every sequence
    was_training = model.training
    model.eval()
    try:
        # parallel generation in one batch
        for token_index in range(max_new_tokens):

            context_ids = token_ids[:, -context_size:]
            logits = model(context_ids)[:, -1, :]

            if top_k is not None:
                top_logits, top_indices = torch.topk(
                    logits, min(top_k, logits.shape[-1]), dim=-1
                )
                logits = torch.full_like(logits, -torch.inf).scatter(
                    -1, top_indices, top_logits
                )

            if temperature > 0:
                scaled_logits = (
                    logits - logits.max(dim=-1, keepdim=True).values
                ) / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                # greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None:
                next_token = next_token.masked_fill(finished, eos_id)
                finished = finished | (next_token == eos_id)

            token_ids = torch.cat((token_ids, next_token), dim=1)

            if progress_callback is not None:
                progress_callback(token_index + 1, max_new_tokens)

            if eos_id is not None and finished.all().item():
                break
        return token_ids
    finally:
        model.train(was_training)


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    eos_id=None,
    progress_callback=None,
):
    """Return the prompt and generated continuation as a decoded string.

    prompt - Nonempty text to continue.
    max_new_tokens - Number of tokens to append.
    temperature - Sampling temperature; 0 uses greedy decoding.
    top_k - Number of candidate tokens; None keeps the full vocabulary.
    eos_id - End token ID, or None to disable early stopping.
    progress_callback - Optional callback receiving completed and requested token counts.
    """
    token_ids = text_to_token_ids(prompt, tokenizer)
    if token_ids.shape[1] == 0:
        raise ValueError("The generation prompt must contain at least one token.")

    generated_ids = generate_token_ids(
        model,
        token_ids,
        max_new_tokens=max_new_tokens,
        context_size=model.position_embedding.num_embeddings,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
        progress_callback=progress_callback,
    )
    return token_ids_to_text(generated_ids, tokenizer)
