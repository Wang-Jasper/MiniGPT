import argparse
import hashlib
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
from ..model.pretrained import load_pretrained_gpt2
from ..tokenizer import createDataLoader
from .progress import (
    generation_progress,
    record_progress,
    show_interruption,
    show_results,
    show_pretrained_setup,
    show_resume_setup,
    show_sample,
    show_stage,
    show_training_setup,
    training_progress,
)
from .results import (
    read_training_checkpoint,
    restore_training_checkpoint,
    restore_training_rng,
    save_periodic_checkpoint,
    save_training_results,
)

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


def _validate_checkpoint_settings(every_n_epochs, keep_last):
    if (
        not isinstance(every_n_epochs, int)
        or isinstance(every_n_epochs, bool)
        or every_n_epochs < 0
    ):
        raise ValueError("checkpoint_every_n_epochs must be a nonnegative integer.")
    if not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 1:
        raise ValueError("checkpoint_keep_last must be a positive integer.")


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
    checkpoint_every_n_epochs=0,
    checkpoint_keep_last=2,
    checkpoint_directory=None,
    model_config=None,
    training_config=None,
    resume_state=None,
    return_state=False,
):
    """Train the model and return loss histories and processed-token counts.

    log_path - CSV progress output path, or None to print only.
    tokenizer - Tokenizer for sample generation, or None to skip samples.
    generation_prompt - Text to continue after each epoch.
    generation_max_new_tokens - New tokens per sample; 0 disables samples.
    generation_temperature - Sampling temperature; 0 uses greedy decoding.
    generation_top_k - Number of candidate tokens; None keeps all.
    generation_eos_id - End token ID, or None to disable early stopping.
    checkpoint_every_n_epochs - Save after each N completed epochs; 0 disables it.
    checkpoint_keep_last - Number of periodic snapshots retained in this run.
    checkpoint_directory - Directory dedicated to this run's checkpoints.
    model_config, training_config - Required metadata when periodic saving is enabled.
    resume_state - Validated checkpoint progress; model and optimizer must already be restored.
    return_state - Also return final epoch/step/token counters as a fourth item.
    """
    _validate_checkpoint_settings(checkpoint_every_n_epochs, checkpoint_keep_last)
    if checkpoint_every_n_epochs > 0 and (
        checkpoint_directory is None or model_config is None or training_config is None
    ):
        raise ValueError("Periodic checkpoints require checkpoint_directory, model_config, and training_config.")

    if not isinstance(num_epochs, int) or isinstance(num_epochs, bool) or num_epochs < 0:
        raise ValueError("num_epochs must be a nonnegative integer.")
    if resume_state is None:
        start_epoch = 0
        global_step = 0
        tokens_seen = 0
        train_losses, validation_losses, tokens_seen_history = [], [], []
    else:
        start_epoch = resume_state["completed_epochs"]
        global_step = resume_state["global_step"]
        tokens_seen = resume_state["tokens_seen"]
        history = resume_state["history"]
        train_losses = list(history["train_losses"])
        validation_losses = list(history["validation_losses"])
        tokens_seen_history = list(history["tokens_seen_history"])
    if start_epoch > num_epochs:
        raise ValueError("num_epochs cannot be smaller than the completed epoch count.")
    total_steps = global_step + (num_epochs - start_epoch) * len(train_loader)
    start_time = perf_counter()
    if resume_state is not None and start_epoch < num_epochs:
        # Model, optimizer, tokenizer, and data loaders have all been initialized.
        restore_training_rng(resume_state, device)

    for epoch in range(start_epoch, num_epochs):
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

        completed_epochs = epoch + 1
        if checkpoint_every_n_epochs > 0 and completed_epochs % checkpoint_every_n_epochs == 0:
            checkpoint_path = save_periodic_checkpoint(
                model, optimizer, model_config, training_config,
                history={
                    "train_losses": train_losses,
                    "validation_losses": validation_losses,
                    "tokens_seen_history": tokens_seen_history,
                },
                checkpoint_directory=checkpoint_directory,
                completed_epochs=completed_epochs,
                global_step=global_step,
                tokens_seen=tokens_seen,
                keep_last=checkpoint_keep_last,
            )
            show_stage(f"Periodic checkpoint saved: {checkpoint_path}")

        if tokenizer is not None and generation_max_new_tokens > 0:
            sampling_device = torch.device(device)
            sampling_devices = []
            if sampling_device.type == "cuda":
                sampling_devices = [
                    sampling_device.index if sampling_device.index is not None
                    else torch.cuda.current_device()
                ]
            # Sampling must not change the RNG used for data shuffling or dropout.
            with torch.random.fork_rng(devices=sampling_devices), generation_progress(
                epoch + 1, generation_max_new_tokens
            ) as sample_progress:
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

    histories = (train_losses, validation_losses, tokens_seen_history)
    if return_state:
        return (*histories, {
            "completed_epochs": num_epochs,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
        })
    return histories


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train MiniGPT from scratch, GPT-2 weights, or a checkpoint.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pretrained", help="GPT-2 model name or local Hugging Face model directory.")
    source.add_argument("--resume", help="Resume from a saved epoch-boundary training checkpoint.")
    parser.add_argument("--num-epochs", type=int, help="Cumulative target epochs, including completed epochs.")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0; overrides the saved/configured device.")
    parser.add_argument("--context-size", type=int, help="Pretrained model capacity; independent of training seq_length.")
    parser.add_argument("--cache-dir", help="Hugging Face model cache directory; requires --pretrained.")
    parser.add_argument("--local-files-only", action="store_true", help="Load pretrained weights without downloading.")
    parser.add_argument("--checkpoint-every-n-epochs", type=int, help="Periodic saving interval; 0 disables it.")
    parser.add_argument("--checkpoint-keep-last", type=int, help="Number of periodic checkpoints to retain.")
    args = parser.parse_args(argv)
    if args.num_epochs is not None and args.num_epochs < 1:
        parser.error("--num-epochs must be positive.")
    if args.context_size is not None and args.context_size < 1:
        parser.error("--context-size must be positive.")
    if args.pretrained is None and (
        args.context_size is not None or args.cache_dir is not None or args.local_files_only
    ):
        parser.error("--context-size, --cache-dir, and --local-files-only require --pretrained.")

    project_root = Path(__file__).resolve().parents[3]
    resume_state = None
    try:
        if args.resume is not None:
            resume_path = Path(args.resume).expanduser()
            if not resume_path.is_absolute():
                resume_path = project_root / resume_path
            resume_path = resume_path.resolve()
            checkpoint = read_training_checkpoint(resume_path)
            gpt_config = dict(checkpoint["model_config"])
            settings = dict(checkpoint["training_config"])
        else:
            gpt_config = dict(GPT_CONFIG_124M)
            settings = dict(TRAINING_CONFIG)

        for key, value in (
            ("num_epochs", args.num_epochs), ("device", args.device),
            ("checkpoint_every_n_epochs", args.checkpoint_every_n_epochs),
            ("checkpoint_keep_last", args.checkpoint_keep_last),
        ):
            if value is not None:
                settings[key] = value
        target_epochs = settings["num_epochs"]
        if not isinstance(target_epochs, int) or isinstance(target_epochs, bool) or target_epochs < 1:
            raise ValueError("num_epochs must be a positive integer.")
        checkpoint_every_n_epochs = settings.get("checkpoint_every_n_epochs", 0)
        checkpoint_keep_last = settings.get("checkpoint_keep_last", 2)
        _validate_checkpoint_settings(checkpoint_every_n_epochs, checkpoint_keep_last)
        output_root = project_root / settings["output_directory"]
        checkpoint_root = project_root / settings["checkpoint_directory"]

        device_name = settings.get("device", "auto")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")

        if args.resume is not None:
            show_resume_setup(
                resume_path, checkpoint["completed_epochs"], target_epochs,
                checkpoint["global_step"], checkpoint["tokens_seen"],
            )
            if checkpoint["completed_epochs"] >= target_epochs:
                return None

        # Old configurations used the model capacity as their training length.
        default_length = args.context_size if args.context_size is not None else gpt_config["context_size"]
        seq_length = settings.get("seq_length", default_length)
        if not isinstance(seq_length, int) or isinstance(seq_length, bool) or seq_length < 1:
            raise ValueError("seq_length must be a positive integer.")
        stride = settings.get("stride", max(1, seq_length // 2))
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be a positive integer.")
        settings["seq_length"], settings["stride"] = seq_length, stride
        if args.pretrained is None and seq_length > gpt_config["context_size"]:
            raise ValueError("seq_length must not exceed model context_size.")
        if args.context_size is not None and seq_length > args.context_size:
            raise ValueError("seq_length must not exceed --context-size.")

        torch.manual_seed(settings.get("random_seed", 123))
        if args.resume is not None:
            model, optimizer = restore_training_checkpoint(checkpoint, device)
            # Keep progress and RNG, releasing large source weights before training.
            resume_state = {
                key: checkpoint[key] for key in (
                    "completed_epochs", "global_step", "tokens_seen", "history",
                    "rng_state", "cuda_rng_states", "rng_device",
                ) if key in checkpoint
            }
            del checkpoint
            settings["resumed_from"] = str(resume_path)
            if not settings.get("sample_rng_isolated", False):
                show_stage("This checkpoint predates sampling RNG isolation; exact replay is not guaranteed.")
        else:
            if args.pretrained is not None:
                show_stage(f"Loading pretrained weights: {args.pretrained}")
                model, gpt_config = load_pretrained_gpt2(
                    args.pretrained, context_size=args.context_size, device=device,
                    cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                )
                if seq_length > gpt_config["context_size"]:
                    raise ValueError("seq_length must not exceed the pretrained model context_size.")
                settings["pretrained_model"] = args.pretrained
                show_pretrained_setup(args.pretrained, gpt_config["context_size"], seq_length)
            else:
                show_stage(f"Initializing model on {device}...")
                model = MiniGPT(gpt_config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
            )
        settings["sample_rng_isolated"] = True

        show_stage("Preparing data...")
        download_data()
        text_data = load_default_text()
        data_sha256 = hashlib.sha256(text_data.encode("utf-8")).hexdigest()
        if resume_state is not None:
            saved_hash = settings.get("data_sha256")
            if saved_hash is not None and saved_hash != data_sha256:
                raise ValueError("Training data differs from the checkpoint corpus; resume requires the same data.")
            if saved_hash is None:
                show_stage("This checkpoint has no corpus fingerprint; verify that the training text is unchanged.")
        settings["data_sha256"] = data_sha256

        split_idx = int(settings["train_ratio"] * len(text_data))
        show_stage("Creating dataloaders...")
        train_loader = createDataLoader(
            text_data[:split_idx], batch_size=settings["batch_size"],
            seq_Lenth=seq_length, stride=stride, drop_last=True, shuffle=True,
            num_workers=settings["num_workers"],
        )
        val_loader = createDataLoader(
            text_data[split_idx:], batch_size=settings["batch_size"],
            seq_Lenth=seq_length, stride=seq_length, drop_last=False, shuffle=False,
            num_workers=settings["num_workers"],
        )
        if not len(train_loader) or not len(val_loader):
            raise ValueError("Training and validation loaders must each contain at least one batch.")
        tokenizer = tiktoken.get_encoding("gpt2")
    except KeyError as error:
        parser.error(f"Training configuration is missing: {error.args[0]}")
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    run_name = f"training_{uuid4().hex}"
    log_path = project_root / "logs" / f"{run_name}.csv"
    checkpoint_directory = checkpoint_root / run_name
    show_training_setup(device, len(train_loader), len(val_loader), log_path)
    if checkpoint_every_n_epochs > 0:
        show_stage(
            f"Periodic checkpoints: every {checkpoint_every_n_epochs} epochs, "
            f"keep {checkpoint_keep_last} | {checkpoint_directory}"
        )

    show_stage("Starting training...")
    train_losses, validation_losses, tokens_seen_history, final_state = train_model(
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
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        checkpoint_keep_last=checkpoint_keep_last,
        checkpoint_directory=checkpoint_directory,
        model_config=gpt_config,
        training_config=settings,
        resume_state=resume_state,
        return_state=True,
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
        output_directory=output_root / run_name,
        checkpoint_directory=checkpoint_directory,
        log_path=log_path,
        **final_state,
    )
    show_results(artifacts)
    return train_losses, validation_losses, tokens_seen_history, model


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        show_interruption()
        raise SystemExit(130) from None




