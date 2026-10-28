from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
import torch.nn.functional as F
import tiktoken

from configs.gpt_config import GPT_CONFIG_124M
from configs.training_config import TRAINING_CONFIG
from scripts.download_data import download_data
from ..data.preprocess import load_default_text
from ..inference.generate import generate_text
from ..model.gpt import MiniGPT
from ..tokenizer import createDataLoader
from .progress import (
    generation_progress,
    record_progress,
    show_interruption,
    show_results,
    show_sample,
    show_stage,
    show_training_setup,
    training_progress,
)
from .results import save_training_results

def forward_model(input_batch, model, device):
    """Run token IDs through the model and return logits, retaining gradients"""
    input_batch = input_batch.to(device)
    return model(input_batch)


def calculate_loss(logits, target_batch):
    """
    CROSS-ENTROPY LOSS SHAPE:
    input logits shape -> (batch_size, seq_length, vocal_size)
    softmax to logits -> (batch_size, seq_length, vocal_size)
    find the target prob -> (batch_size, seq_length)

    easier to maximize the logarithm of the probability score than the probability score itself, since multiply will transfer to addition, and small value will have larger gradient
    log(target_prob) -> (batch_size, seq_length)    and we want everything using gradient descent, so we make it positive by * (-1)

    and minimize the mean of target_prob

    mean(log(target_prob)) -> (1)
    """
    target_batch = target_batch.to(logits.device)

    # Cross entropy computes -mean(log(target_prob)) from raw logits internally.
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


@torch.no_grad()
def calculate_average_loss(data_loader, model, device, num_batches=None):
    """
    Return token-weighted loss over all batches or the first num_batches.
    """
    batch_limit = len(data_loader)
    if num_batches is not None:
        batch_limit = min(num_batches, batch_limit)
    if batch_limit <= 0:
        return float("nan")

    total_loss = 0.0
    total_tokens = 0
    for batch_index, (input_batch, target_batch) in enumerate(data_loader):
        logits = forward_model(input_batch, model, device)
        loss = calculate_loss(logits, target_batch)
        token_count = target_batch.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count
        if batch_index + 1 >= batch_limit:
            break

    return total_loss / total_tokens if total_tokens else float("nan")


@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, num_batches=None):
    """
    Return training and validation losses, restoring the original model mode
    """
    was_training = model.training
    model.eval()
    try:
        train_loss = calculate_average_loss(
            train_loader, model, device, num_batches=num_batches
        )
        validation_loss = calculate_average_loss(
            val_loader, model, device, num_batches=num_batches
        )
        return train_loss, validation_loss
    finally:
        model.train(was_training)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    evaluation_frequency=5,
    evaluation_batches=1,
    log_path=None,
    tokenizer=None,
    generation_prompt="First Citizen:",
    generation_max_new_tokens=50,
    generation_temperature=0.8,
    generation_top_k=50,
    generation_eos_id=None,
):
    """Train the model and return loss histories and processed-token counts.

    log_path - CSV progress output path, or None to print only.
    tokenizer - Tokenizer for sample generation, or None to skip samples.
    generation_prompt - Text to continue after each epoch.
    generation_max_new_tokens - New tokens per sample; 0 disables samples.
    generation_temperature - Sampling temperature; 0 uses greedy decoding.
    generation_top_k - Number of candidate tokens; None keeps all.
    generation_eos_id - End token ID, or None to disable early stopping.
    """
    train_losses = []
    validation_losses = []
    tokens_seen_history = []
    tokens_seen = 0
    global_step = 0
    total_steps = num_epochs * len(train_loader)
    start_time = perf_counter()

    for epoch in range(num_epochs):
        model.train()
        with training_progress(epoch + 1, num_epochs, len(train_loader)) as epoch_progress:
            # take each batch data
            for batch_index, (input_batch, target_batch) in enumerate(train_loader):
                # clear the gradient for each new training
                optimizer.zero_grad()
                logits = forward_model(input_batch, model, device)
                loss = calculate_loss(logits, target_batch)
                loss.backward()
                optimizer.step()

                tokens_seen += input_batch.numel()
                global_step += 1
                epoch_progress.update(
                    batch_index + 1,
                    loss=loss.item(),
                    global_step=global_step,
                    total_steps=total_steps,
                )

                is_last_batch = batch_index + 1 == len(train_loader)
                # first training || every N step || last batch
                if (
                    global_step == 1
                    or global_step % evaluation_frequency == 0
                    or is_last_batch
                ):
                    epoch_progress.set_status("Evaluating...")
                    try:
                        train_loss, validation_loss = evaluate_model(
                            model,
                            train_loader,
                            val_loader,
                            device,
                            num_batches=evaluation_batches,
                        )
                    finally:
                        epoch_progress.set_status("")
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    tokens_seen_history.append(tokens_seen)
                    progress = {
                        "epoch": epoch + 1,
                        "num_epochs": num_epochs,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "progress_percent": 100.0 * global_step / total_steps,
                        "tokens_seen": tokens_seen,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "elapsed_seconds": perf_counter() - start_time,
                    }
                    record_progress(
                        progress, log_path=log_path, progress_bar=epoch_progress
                    )

        if tokenizer is not None and generation_max_new_tokens > 0:
            with generation_progress(epoch + 1, generation_max_new_tokens) as sample_progress:
                sample = generate_text(
                    model,
                    tokenizer,
                    generation_prompt,
                    max_new_tokens=generation_max_new_tokens,
                    temperature=generation_temperature,
                    top_k=generation_top_k,
                    eos_id=generation_eos_id,
                    progress_callback=lambda completed, total: sample_progress.update(completed),
                )
            show_sample(epoch + 1, sample)

    return train_losses, validation_losses, tokens_seen_history


def main():
    gpt_config = GPT_CONFIG_124M
    settings = TRAINING_CONFIG
    torch.manual_seed(settings["random_seed"])
    device_name = settings["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # default training corpus - shakespeare
    show_stage("Preparing data...")
    download_data()
    text_data = load_default_text()

    # Initialize the model, optimizer
    show_stage(f"Initializing model on {device}...")
    model = MiniGPT(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    # Split validation data using the configured training fraction.
    train_ratio = settings["train_ratio"]
    split_idx = int(train_ratio * len(text_data))

    show_stage("Creating dataloaders...")
    train_loader = createDataLoader(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        seq_Lenth=gpt_config["context_size"],
        stride=gpt_config["context_size"] // 2,
        drop_last=True,
        shuffle=True,
        num_workers=settings["num_workers"],
    )
    val_loader = createDataLoader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        seq_Lenth=gpt_config["context_size"],
        stride=gpt_config["context_size"],
        drop_last=False,
        shuffle=False,
        num_workers=settings["num_workers"],
    )

    project_root = Path(__file__).resolve().parents[3]
    run_name = f"training_{uuid4().hex}"
    log_path = project_root / "logs" / f"{run_name}.csv"
    show_training_setup(device, len(train_loader), len(val_loader), log_path)

    tokenizer = tiktoken.get_encoding("gpt2")
    show_stage("Starting training...")
    train_losses, validation_losses, tokens_seen_history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        evaluation_frequency=settings["evaluation_frequency"],
        evaluation_batches=settings["evaluation_batches"],
        log_path=log_path,
        tokenizer=tokenizer,
        generation_prompt=settings["generation_prompt"],
        generation_max_new_tokens=settings["generation_max_new_tokens"],
        generation_temperature=settings["generation_temperature"],
        generation_top_k=settings["generation_top_k"],
        generation_eos_id=settings["generation_eos_id"],
    )

    show_stage("Saving results...")
    artifacts = save_training_results(
        model,
        optimizer,
        model_config=gpt_config,
        training_config=settings,
        history={
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "tokens_seen_history": tokens_seen_history,
        },
        output_directory=project_root / settings["output_directory"] / run_name,
        checkpoint_directory=project_root / settings["checkpoint_directory"] / run_name,
        log_path=log_path,
    )
    show_results(artifacts)
    return train_losses, validation_losses, tokens_seen_history, model


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        show_interruption()
        raise SystemExit(130) from None




