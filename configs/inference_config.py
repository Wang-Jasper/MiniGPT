"""Standalone inference settings, independent of training sample generation."""
# .\.venv\Scripts\python.exe -m scripts.inference

INFERENCE_CONFIG = {
    "model_source": "pretrained",  # pretrained or checkpoint.
    "pretrained_model": "gpt2",
    "checkpoint_path": None,      # In checkpoint mode, None selects the newest final checkpoint.
    "device": "auto",             # auto, cpu, cuda, or cuda:0.
    "context_size": 1024,         # Pretrained capacity; None preserves the source model's limit.
    "cache_dir": None,           # None uses the default Hugging Face cache location.
    "local_files_only": True,    # Only controls Hugging Face model file downloads.
    "generation_max_new_tokens": 128,  # Stop at EOS or this token limit, whichever comes first.
    "generation_temperature": 0.5,
    "generation_top_k": 50,     # None keeps all candidate tokens.
    "generation_eos_id": 50256,   # GPT-2 end-of-text token; None disables EOS stopping.
}
