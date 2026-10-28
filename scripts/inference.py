"""Run terminal text continuation: python -m scripts.inference."""

import argparse
import math
from pathlib import Path

import tiktoken
import torch

from src.minigpt.inference.generate import generate_text
from src.minigpt.training.progress import ProgressBar, show_stage
from src.minigpt.training.results import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_checkpoint(checkpoint_path=None):
    """Find an explicit checkpoint or the newest completed training checkpoint."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.resolve()

    candidates = [
        path for path in (PROJECT_ROOT / "checkpoints").glob("training_*/checkpoint.pth")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError("No checkpoint found. Specify one with --checkpoint.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run_terminal(model, tokenizer, generation_settings):
    """Read independent prompts and print continuations until exit or interruption."""
    show_stage("MiniGPT text continuation. Type exit to quit.")
    show_stage("Each prompt starts a new continuation.")
    try:
        while True:
            prompt = input("\nPrompt> ")
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            if not prompt.strip():
                continue

            max_new_tokens = generation_settings["max_new_tokens"]
            with ProgressBar(max_new_tokens, "Generating", unit="token") as progress:
                result = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    **generation_settings,
                    progress_callback=lambda completed, total: progress.update(completed),
                )
            show_stage(f"\nContinuation:\n{result[len(prompt):]}")
    except (EOFError, KeyboardInterrupt):
        show_stage("")
    show_stage("Goodbye.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continue text with a trained MiniGPT.")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to the newest run.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--max-new-tokens", type=int, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; 0 uses greedy.")
    parser.add_argument("--top-k", type=int, help="Candidate tokens; 0 uses the full vocabulary.")
    args = parser.parse_args(argv)

    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be nonnegative.")
    if args.temperature is not None and (
        args.temperature < 0 or not math.isfinite(args.temperature)
    ):
        parser.error("--temperature must be finite and nonnegative.")
    if args.top_k is not None and args.top_k < 0:
        parser.error("--top-k must be nonnegative.")

    try:
        checkpoint_path = find_checkpoint(args.checkpoint)
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    show_stage(f"Loading checkpoint: {checkpoint_path}")
    # Read on CPU so saved AdamW tensors are never copied to GPU for inference.
    model, _, checkpoint = load_checkpoint(
        checkpoint_path, device="cpu", load_optimizer=False
    )
    settings = dict(checkpoint.get("training_config", {}))
    del checkpoint
    model.to(device)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    generation_settings = {
        "max_new_tokens": (
            args.max_new_tokens if args.max_new_tokens is not None
            else settings.get("generation_max_new_tokens", 50)
        ),
        "temperature": (
            args.temperature if args.temperature is not None
            else settings.get("generation_temperature", 0.8)
        ),
        "top_k": (
            (args.top_k or None) if args.top_k is not None
            else settings.get("generation_top_k", 50)
        ),
        "eos_id": settings.get("generation_eos_id"),
    }
    show_stage(f"Device: {device}")
    show_stage(
        f"New tokens: {generation_settings['max_new_tokens']} | "
        f"Temperature: {generation_settings['temperature']} | "
        f"Top-k: {generation_settings['top_k']}"
    )
    run_terminal(model, tokenizer, generation_settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        show_stage("\nGoodbye.")
