"""Training controls, independent of model architecture."""

TRAINING_CONFIG = {
    "random_seed": 123,          # Seed for PyTorch random number generators.
    "device": "auto",            # auto, cpu, cuda, or a specific device such as cuda:0.
    "learning_rate": 0.0004,     # AdamW learning rate.
    "weight_decay": 0.1,         # AdamW weight decay.
    "batch_size": 1,             # Sequences per batch.
    "num_epochs": 10,            # Full passes over the training loader.
    "evaluation_frequency": 5,   # Optimizer steps between periodic evaluations.
    "evaluation_batches": 1,     # Batches per loader per evaluation; None uses all.
    "train_ratio": 0.90,         # Fraction of source text used for training.
    "num_workers": 0,            # Data-loading worker processes.
    "generation_prompt": "First Citizen:",  # Text to continue after each epoch.
    "generation_max_new_tokens": 50,       # New tokens per sample; 0 disables samples.
    "generation_temperature": 0.8,         # Sampling temperature; 0 uses greedy decoding.
    "generation_top_k": 50,                # Candidate tokens; None keeps all tokens.
    "generation_eos_id": None,             # End token ID; None disables early stopping.
    "output_directory": "outputs",         # Root directory for plots and training results.
    "checkpoint_directory": "checkpoints", # Root directory for model checkpoints.
}
