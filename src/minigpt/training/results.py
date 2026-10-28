"""Save training artifacts and load model checkpoints."""

import json
import math
from pathlib import Path
import pickle
import warnings

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..model.gpt import MiniGPT


def save_checkpoint(
    model, optimizer, model_config, training_config, history, checkpoint_path,
    *, completed_epochs=None, global_step=None, tokens_seen=None,
):
    """Atomically save model, optimizer, metadata, and current PyTorch RNG state.

    Progress counters are optional for compatibility with older callers.
    This saves state only; training-loop resume is handled separately.
    """
    progress = {
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
    }
    for name, value in progress.items():
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "history": history,
        "rng_state": torch.get_rng_state(),
        "rng_device": str(next(model.parameters()).device),
        **{name: value for name, value in progress.items() if value is not None},
    }
    if torch.cuda.is_initialized():
        checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint_path


def save_periodic_checkpoint(
    model, optimizer, model_config, training_config, history, checkpoint_directory,
    completed_epochs, global_step, tokens_seen, keep_last,
):
    """Save an epoch snapshot, then prune only this run's older periodic files.

    The trainer calls this before sample generation. Saved RNG state therefore
    does not by itself guarantee exact replay when resuming after sampling.
    """
    if not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 1:
        raise ValueError("checkpoint_keep_last must be a positive integer.")
    if (
        not isinstance(completed_epochs, int)
        or isinstance(completed_epochs, bool)
        or completed_epochs < 1
    ):
        raise ValueError("completed_epochs must be a positive integer for periodic checkpoints.")
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_path = save_checkpoint(
        model, optimizer, model_config, training_config, history,
        checkpoint_directory / f"checkpoint_epoch_{completed_epochs:04d}.pth",
        completed_epochs=completed_epochs, global_step=global_step, tokens_seen=tokens_seen,
    )
    periodic_paths = [
        path for path in checkpoint_directory.glob("checkpoint_epoch_*.pth")
        if path.is_file() and path.stem.removeprefix("checkpoint_epoch_").isdigit()
    ]
    periodic_paths.sort(key=lambda path: int(path.stem.removeprefix("checkpoint_epoch_")))
    for old_path in periodic_paths[:-keep_last]:
        old_path.unlink()
    return checkpoint_path


def plot_losses(history, output_path):
    """Save training and validation loss against processed training tokens.

    history - Training loss, validation loss, and tokens_seen_history lists.
    output_path - Output image path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(
        history["tokens_seen_history"], history["train_losses"],
        label="Training loss", marker="o", markersize=3,
    )
    axis.plot(
        history["tokens_seen_history"], history["validation_losses"],
        label="Validation loss", linestyle="--", marker="o", markersize=3,
    )
    axis.set_xlabel("Training tokens processed")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    figure.clear()
    return output_path


def save_training_results(
    model,
    optimizer,
    model_config,
    training_config,
    history,
    output_directory,
    checkpoint_directory,
    log_path=None,
    *,
    completed_epochs=None,
    global_step=None,
    tokens_seen=None,
):
    """Save a final checkpoint, loss plot, and JSON training results.

    history - Loss and processed-token histories returned by training.
    output_directory - Directory for the plot and JSON results.
    checkpoint_directory - Directory for model and optimizer states.
    log_path - Associated CSV progress log, or None.
    completed_epochs, global_step, tokens_seen - Optional final training progress.
    """
    output_directory = Path(output_directory)
    checkpoint_directory = Path(checkpoint_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_checkpoint(
        model, optimizer, model_config, training_config, history,
        checkpoint_directory / "checkpoint.pth",
        completed_epochs=completed_epochs, global_step=global_step, tokens_seen=tokens_seen,
    )
    # The final file now covers this epoch; remove its redundant periodic copy
    # only after the final checkpoint has been successfully replaced.
    if completed_epochs is not None:
        periodic_path = checkpoint_directory / f"checkpoint_epoch_{completed_epochs:04d}.pth"
        periodic_path.unlink(missing_ok=True)

    plot_path = plot_losses(history, output_directory / "loss.png")
    results_path = output_directory / "training_results.json"
    # JSON uses null for unavailable or nonfinite loss measurements.
    json_history = {
        name: [
            None if isinstance(value, float) and not math.isfinite(value) else value
            for value in values
        ]
        for name, values in history.items()
    }
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(
            {
                "model_config": dict(model_config),
                "training_config": dict(training_config),
                "history": json_history,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "loss_plot_path": str(plot_path.resolve()),
                "progress_log_path": str(Path(log_path).resolve()) if log_path else None,
            },
            results_file,
            indent=2,
            allow_nan=False,
        )

    return {
        "checkpoint": checkpoint_path,
        "loss_plot": plot_path,
        "results": results_path,
    }


def load_checkpoint(checkpoint_path, device="cpu", load_optimizer=False):
    """Restore a MiniGPT model in evaluation mode from a saved checkpoint.

    checkpoint_path - Final or periodic checkpoint created by the save helpers.
    device - Device for the model and loaded tensors.
    load_optimizer - Also restore AdamW state when True.

    Return the model, optional optimizer, and checkpoint metadata/state.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MiniGPT(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    optimizer = None
    if load_optimizer:
        settings = checkpoint["training_config"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint


def _validate_training_checkpoint(checkpoint):
    """Require explicit epoch-boundary progress rather than guessing from names."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Cannot resume training: the checkpoint must contain a dictionary.")
    required = (
        "model_state_dict", "optimizer_state_dict", "model_config", "training_config",
        "history", "completed_epochs", "global_step", "tokens_seen",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "Cannot resume training: checkpoint is missing " + ", ".join(missing)
            + ". Older checkpoints may still be used for inference."
        )
    for name in ("model_state_dict", "optimizer_state_dict", "model_config", "training_config", "history"):
        if not isinstance(checkpoint[name], dict):
            raise ValueError(f"Cannot resume training: {name} must be a dictionary.")
    for name in ("completed_epochs", "global_step", "tokens_seen"):
        value = checkpoint[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Cannot resume training: {name} must be a nonnegative integer.")

    history = checkpoint["history"]
    names = ("train_losses", "validation_losses", "tokens_seen_history")
    for name in names:
        if name not in history or not isinstance(history[name], list):
            raise ValueError(f"Cannot resume training: history.{name} must be a list.")
    if len({len(history[name]) for name in names}) != 1:
        raise ValueError("Cannot resume training: loss and token histories must have matching lengths.")
    token_history = history["tokens_seen_history"]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in token_history):
        raise ValueError("Cannot resume training: token history must contain nonnegative integers.")
    if any(left > right for left, right in zip(token_history, token_history[1:])):
        raise ValueError("Cannot resume training: token history must not decrease.")
    if (token_history[-1] if token_history else 0) != checkpoint["tokens_seen"]:
        raise ValueError("Cannot resume training: the final token history entry does not match tokens_seen.")
    for name in ("learning_rate", "weight_decay"):
        if name not in checkpoint["training_config"]:
            raise ValueError(f"Cannot resume training: training_config is missing {name}.")


def read_training_checkpoint(checkpoint_path):
    """Read a resume checkpoint on CPU and validate its explicit training progress."""
    try:
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=True
        )
    except (RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Cannot read the training checkpoint: {error}") from error
    _validate_training_checkpoint(checkpoint)
    return checkpoint


def restore_training_checkpoint(checkpoint, device="cpu"):
    """Restore model and AdamW on the target device without restoring RNG yet.

    Model parameters must reach their final device before the optimizer is
    created. AdamW's state loader then places its tensors using each parameter's
    device policy, including the CPU step counters used by ordinary AdamW.
    """
    _validate_training_checkpoint(checkpoint)
    try:
        device = torch.device(device)
        model = MiniGPT(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        settings = checkpoint["training_config"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Cannot restore the training model and optimizer: {error}") from error
    model.train()
    return model, optimizer


def restore_training_rng(checkpoint, device="cpu"):
    """Restore RNG after all initialization, touching only the training CUDA device.

    Missing state or a device change permits continuation but cannot guarantee
    the same random sequence as an uninterrupted run.
    """
    device = torch.device(device)
    cpu_state = checkpoint.get("rng_state")
    if cpu_state is None:
        warnings.warn("Checkpoint has no CPU RNG state; exact replay is unavailable.", RuntimeWarning, stacklevel=2)
    else:
        if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1:
            raise ValueError("Checkpoint rng_state must be a one-dimensional uint8 tensor.")
        try:
            torch.set_rng_state(cpu_state.cpu())
        except RuntimeError as error:
            raise ValueError(f"Cannot restore CPU RNG state: {error}") from error

    saved_device = checkpoint.get("rng_device")
    if saved_device is not None:
        try:
            saved_device = torch.device(saved_device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("Checkpoint rng_device is invalid.") from error
    if device.type != "cuda":
        if saved_device is not None and saved_device.type != device.type:
            warnings.warn("Training device changed; exact replay is unavailable.", RuntimeWarning, stacklevel=2)
        return
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; cannot restore CUDA RNG state.")
    states = checkpoint.get("cuda_rng_states")
    if not isinstance(states, (list, tuple)) or not states:
        warnings.warn("Checkpoint has no CUDA RNG state; exact replay is unavailable.", RuntimeWarning, stacklevel=2)
        return
    target_index = torch.cuda.current_device() if device.index is None else device.index
    if saved_device is not None:
        if saved_device.type != "cuda":
            warnings.warn("Checkpoint was saved on a different device type; CUDA RNG cannot be restored.", RuntimeWarning, stacklevel=2)
            return
        source_index = 0 if saved_device.index is None else saved_device.index
    elif len(states) == 1:
        source_index = 0
    elif target_index < len(states):
        source_index = target_index
        warnings.warn("Checkpoint does not record its training GPU; assuming the matching CUDA index.", RuntimeWarning, stacklevel=2)
    else:
        warnings.warn("Saved CUDA RNG states cannot be mapped to this GPU; exact replay is unavailable.", RuntimeWarning, stacklevel=2)
        return
    if source_index >= len(states):
        warnings.warn("Checkpoint lacks RNG state for its training GPU; exact replay is unavailable.", RuntimeWarning, stacklevel=2)
        return
    state = states[source_index]
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("Saved CUDA RNG state must be a one-dimensional uint8 tensor.")
    if source_index != target_index or len(states) != torch.cuda.device_count():
        warnings.warn("CUDA device mapping changed; exact replay is unavailable.", RuntimeWarning, stacklevel=2)
    try:
        torch.cuda.set_rng_state(state.cpu(), device=target_index)
    except RuntimeError as error:
        raise ValueError(f"Cannot restore CUDA RNG state: {error}") from error
