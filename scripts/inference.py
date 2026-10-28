"""Run terminal text continuation: python -m scripts.inference."""

import argparse
import math
from pathlib import Path

import tiktoken
import torch

from configs.inference_config import INFERENCE_CONFIG
from src.minigpt.inference.generate import generate_text
from src.minigpt.model.pretrained import load_pretrained_gpt2
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


def _resolve_loading_settings(args):
    """An explicit CLI source overrides the configured source as a whole."""
    source = INFERENCE_CONFIG.get("model_source", "checkpoint")
    if args.pretrained is not None:
        source = "pretrained"
    elif args.checkpoint is not None:
        source = "checkpoint"
    if source not in ("pretrained", "checkpoint"):
        raise ValueError("model_source must be 'pretrained' or 'checkpoint'.")
    settings = {
        "model_source": source,
        "device": args.device if args.device is not None else INFERENCE_CONFIG.get("device", "auto"),
    }
    if not isinstance(settings["device"], str) or not settings["device"].strip():
        raise ValueError("device must be a nonempty device name such as auto, cpu, or cuda:0.")
    if source == "checkpoint":
        if args.context_size is not None or args.cache_dir is not None or args.local_files_only is not None:
            raise ValueError("--context-size, --cache-dir, and --local-files-only require a pretrained source.")
        # File settings for the inactive pretrained source do not affect checkpoints.
        settings["checkpoint_path"] = (
            args.checkpoint if args.checkpoint is not None
            else INFERENCE_CONFIG.get("checkpoint_path")
        )
        return settings

    settings.update({
        "pretrained_model": (
            args.pretrained if args.pretrained is not None
            else INFERENCE_CONFIG.get("pretrained_model", "gpt2")
        ),
        "context_size": (
            args.context_size if args.context_size is not None
            else INFERENCE_CONFIG.get("context_size")
        ),
        "cache_dir": args.cache_dir if args.cache_dir is not None else INFERENCE_CONFIG.get("cache_dir"),
        "local_files_only": (
            args.local_files_only if args.local_files_only is not None
            else INFERENCE_CONFIG.get("local_files_only", False)
        ),
    })
    model_name = settings["pretrained_model"]
    if not isinstance(model_name, (str, Path)) or not str(model_name).strip():
        raise ValueError("pretrained_model must be a nonempty GPT-2 model name or local directory.")
    context_size = settings["context_size"]
    if context_size is not None and (
        not isinstance(context_size, int) or isinstance(context_size, bool) or context_size < 1
    ):
        raise ValueError("context_size must be a positive integer or None.")
    if settings["cache_dir"] is not None and not isinstance(settings["cache_dir"], (str, Path)):
        raise ValueError("cache_dir must be a path or None.")
    if not isinstance(settings["local_files_only"], bool):
        raise ValueError("local_files_only must be True or False.")
    return settings


def _resolve_generation_settings(args, saved_settings=None):
    """CLI > explicit file keys > checkpoint settings > fallback defaults.

    A present None value disables Top-k or EOS; it never means 'inherit'.
    """
    saved_settings = {} if saved_settings is None else saved_settings
    settings = {}
    for name, fallback in (("max_new_tokens", 50), ("temperature", 0.8), ("top_k", 50), ("eos_id", None)):
        key = f"generation_{name}"
        value = getattr(args, name, None)
        if name == "eos_id" and hasattr(args, name):
            settings[name] = value  # --no-eos is an explicit None.
        elif value is not None:
            settings[name] = None if name == "top_k" and value == 0 else value
        elif key in INFERENCE_CONFIG:
            settings[name] = INFERENCE_CONFIG[key]
        else:
            settings[name] = saved_settings.get(key, fallback)

    limit, temperature = settings["max_new_tokens"], settings["temperature"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("generation_max_new_tokens must be a nonnegative integer.")
    if (
        not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
        or not math.isfinite(temperature) or temperature < 0
    ):
        raise ValueError("generation_temperature must be finite and nonnegative.")
    top_k, eos_id = settings["top_k"], settings["eos_id"]
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
        raise ValueError("generation_top_k must be a positive integer or None; CLI --top-k 0 disables it.")
    if eos_id is not None and (not isinstance(eos_id, int) or isinstance(eos_id, bool) or eos_id < 0):
        raise ValueError("generation_eos_id must be a nonnegative integer or None.")
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continue text with MiniGPT or GPT-2 pretrained weights.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="Checkpoint path; overrides the configured model source.")
    source.add_argument("--pretrained", help="GPT-2 name or local directory; overrides the configured model source.")
    parser.add_argument("--context-size", type=int, help="Override the configured pretrained model context limit.")
    parser.add_argument("--cache-dir", help="Hugging Face model cache directory; requires a pretrained source.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Control pretrained downloads; overrides local_files_only in the config.")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0; defaults to the inference config.")
    parser.add_argument("--max-new-tokens", type=int, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; 0 uses greedy.")
    parser.add_argument("--top-k", type=int, help="Candidate tokens; 0 uses the full vocabulary.")
    eos = parser.add_mutually_exclusive_group()
    eos.add_argument("--eos-id", type=int, default=argparse.SUPPRESS, help="Override the end-of-text token ID.")
    eos.add_argument("--no-eos", dest="eos_id", action="store_const", const=None,
                     default=argparse.SUPPRESS, help="Disable EOS stopping for this run.")
    args = parser.parse_args(argv)

    try:
        loading_settings = _resolve_loading_settings(args)
        # Reject invalid explicit settings before loading or downloading weights.
        _resolve_generation_settings(args)
        device_name = loading_settings["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")

        if loading_settings["model_source"] == "pretrained":
            show_stage(f"Loading pretrained model: {loading_settings['pretrained_model']}")
            model, _ = load_pretrained_gpt2(
                loading_settings["pretrained_model"],
                context_size=loading_settings["context_size"],
                device="cpu",
                cache_dir=loading_settings["cache_dir"],
                local_files_only=loading_settings["local_files_only"],
            )
            settings = {"generation_eos_id": 50256}
        else:
            checkpoint_path = find_checkpoint(loading_settings["checkpoint_path"])
            show_stage(f"Loading checkpoint: {checkpoint_path}")
            # Read on CPU so saved AdamW tensors are never copied to GPU for inference.
            model, _, checkpoint = load_checkpoint(
                checkpoint_path, device="cpu", load_optimizer=False
            )
            settings = dict(checkpoint.get("training_config", {}))
            del checkpoint
        generation_settings = _resolve_generation_settings(args, settings)
        eos_id = generation_settings["eos_id"]
        if eos_id is not None and eos_id >= model.token_embedding.num_embeddings:
            raise ValueError("generation_eos_id must be smaller than the model vocabulary size.")
        model.to(device)
        model.eval()
        tokenizer = tiktoken.get_encoding("gpt2")
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    show_stage(f"Device: {device}")
    show_stage(f"Context limit: {model.position_embedding.num_embeddings}")
    show_stage(
        f"New tokens: {generation_settings['max_new_tokens']} | "
        f"Temperature: {generation_settings['temperature']} | "
        f"Top-k: {generation_settings['top_k']} | EOS: {generation_settings['eos_id']}"
    )
    run_terminal(model, tokenizer, generation_settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        show_stage("\nGoodbye.")
