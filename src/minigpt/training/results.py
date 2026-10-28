"""Save training artifacts and load model checkpoints."""

import json
import math
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..model.gpt import MiniGPT


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
):
    """Save a final checkpoint, loss plot, and JSON training results.

    history - Loss and processed-token histories returned by training.
    output_directory - Directory for the plot and JSON results.
    checkpoint_directory - Directory for model and optimizer states.
    log_path - Associated CSV progress log, or None.
    """
    output_directory = Path(output_directory)
    checkpoint_directory = Path(checkpoint_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_directory / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": dict(model_config),
            "training_config": dict(training_config),
            "history": history,
        },
        checkpoint_path,
    )

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

    checkpoint_path - Checkpoint created by save_training_results.
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