# MiniGPT

A small GPT implementation in PyTorch for learning how a language model works, from next-token prediction to training and terminal text generation.

**The author wrote the core Attention computation and the main LayerNorm, GELU, and FeedForward implementations by hand.** AI assisted with some initialization code, imports, configuration wiring, code review, and targeted fixes, as well as the training and inference utilities. TransformerBlock and MiniGPT initialization, and later GPT-2 compatibility changes, include AI assistance.

The project uses PyTorch's linear layers, embedding tables, and automatic differentiation, together with tiktoken's GPT-2 tokenizer.

## Features

- Causal single-head and multi-head attention, with explicit Q/K/V projections, masking, and head splitting and merging.
- Custom LayerNorm and GELU, a feed-forward network, and Transformer blocks with residual connections.
- Sliding-window datasets for next-token prediction.
- Training from scratch and full-parameter fine-tuning from GPT-2 pretrained weights.
- Training and validation loss evaluation, live progress, CSV logs, and loss plots.
- Periodic checkpoints and training recovery at saved epoch boundaries.
- Terminal text continuation with temperature, Top-k sampling, and optional EOS stopping.

## Project structure

```text
MiniGPT/
|-- LICENSE                    # Apache License 2.0
|-- NOTICE                     # Copyright and source attribution
|-- licenses/                  # Preserved upstream license
|-- requirements.txt           # Base dependencies
|-- configs/
|   |-- gpt_config.py          # Model architectures
|   |-- training_config.py     # Training and per-epoch sample settings
|   `-- inference_config.py    # Standalone terminal inference settings
|-- scripts/
|   |-- download_data.py       # Default corpus downloader
|   `-- inference.py           # Terminal entry point
`-- src/minigpt/
    |-- tokenizer.py           # tiktoken integration, dataset, and DataLoader
    |-- data/
    |   `-- preprocess.py      # Read the default training text
    |-- model/
    |   |-- attention.py       # Single-head and multi-head causal attention
    |   |-- embedding.py       # Embedding composition for study
    |   |-- layers.py          # LayerNorm, GELU, and FeedForward
    |   |-- transformer_block.py
    |   |-- gpt.py             # MiniGPT model
    |   `-- pretrained.py      # Map GPT-2 weights into MiniGPT
    |-- training/
    |   |-- trainer.py         # Loss, evaluation, and training loop
    |   |-- progress.py        # Terminal displays and CSV records
    |   `-- results.py         # Checkpoints, plots, and result files
    `-- inference/
        `-- generate.py        # Autoregressive text generation
```

The standalone `Embedding` class and single-head `CausalAttention` are retained for study. The training model uses embedding tables directly inside `MiniGPT` and the multi-head attention implementation.

## Setup

The commands below use Windows PowerShell. Clone the repository first, then run the remaining commands from its root directory. Using the virtual environment's Python directly avoids relying on shell activation.

The local development environment uses Python 3.10.11 and PyTorch 2.9.1 with CUDA 12.8.

```powershell
git clone https://github.com/Wang-Jasper/MiniGPT.git
cd MiniGPT
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install one PyTorch build appropriate for your machine. These examples follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v291).

For a compatible NVIDIA GPU and driver, using the same CUDA build as the development environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

For CPU execution:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install the base dependencies from the version-pinned requirements file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements accept the matching PyTorch CUDA build installed above.

To load GPT-2 pretrained weights, also install Transformers:

```powershell
.\.venv\Scripts\python.exe -m pip install transformers==4.57.6
```

Transformers is optional when training from scratch or loading an existing MiniGPT checkpoint.

Check whether PyTorch can use CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

With `device="auto"`, training and inference use CUDA when available and otherwise use the CPU. Both entry points also accept `--device cpu` or `--device cuda:0`.

## Configuration

| File | Purpose |
| --- | --- |
| [configs/gpt_config.py](configs/gpt_config.py) | Model dimensions, context capacity, attention heads, and Transformer block count |
| [configs/training_config.py](configs/training_config.py) | Optimizer, data windows, evaluation, checkpoints, and per-epoch generation |
| [configs/inference_config.py](configs/inference_config.py) | Model source, cache, device, and terminal generation |

The default architecture settings are:

| Setting | From scratch | Standard GPT-2 pretrained |
| --- | --- | --- |
| Vocabulary size | 50,257 | 50,257 |
| Embedding dimension | 768 | 768 |
| Attention heads | 12 | 12 |
| Transformer blocks | 12 | 12 |
| Context capacity | 256 | 1,024 |
| Shared input/output embedding weights | No | Yes |

`GPT_CONFIG_124M` is the existing configuration name. Its from-scratch setup uses separate input and output weights, so its parameter count differs from the official GPT-2 124M model.

Current training defaults include:

| Setting | Value | Meaning |
| --- | --- | --- |
| `batch_size` | `1` | Sequences in one training batch |
| `seq_length` | `256` | Tokens in each input sequence |
| `stride` | `128` | Tokens between training window starts |
| `num_epochs` | `10` | Target number of complete training passes |
| `learning_rate` | `0.0004` | AdamW learning rate |
| `weight_decay` | `0.1` | AdamW weight decay |
| `train_ratio` | `0.90` | Fraction of source text assigned to training |
| `evaluation_frequency` | `5` | Optimizer steps between periodic evaluations |
| `evaluation_batches` | `1` | Batches per loader used in each evaluation; `None` uses all |
| `checkpoint_every_n_epochs` | `3` | Completed epochs between periodic saves; `0` disables them |
| `checkpoint_keep_last` | `2` | Periodic checkpoints retained per run |

`context_size` is the model's maximum input capacity. `seq_length` is the actual training sample length and must not exceed that capacity. Loading GPT-2 with a context capacity of 1,024 does not require training on sequences of length 1,024.

Training samples overlap according to `stride`; validation windows use `seq_length` as their stride. The source text is split into training and validation portions before tokenization and window construction. Targets are token IDs shifted one position ahead of the inputs.

Per-epoch generation uses `generation_*` settings in `training_config.py`. The current sample limit is 1,024 new tokens, with temperature `0.5`, no Top-k filtering, and no EOS stopping. Set `generation_max_new_tokens=0` to skip these samples. Terminal generation is configured separately.

## Training

### Default data

The default corpus is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), saved to:

```text
data/raw/tinyshakespeare/input.txt
```

Training downloads the file automatically if needed. A nonempty existing file is reused. To download it separately:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data
```

The first use of tiktoken's GPT-2 encoding may also download its encoding files.

### Train from scratch

```powershell
.\.venv\Scripts\python.exe -m src.minigpt.training.trainer
```

Each training step processes one batch, computes cross-entropy loss, performs backpropagation, and updates the parameters with AdamW. Each epoch traverses the training DataLoader once.

The terminal shows batch progress, loss, elapsed time, and estimated remaining time. Evaluation measurements are recorded in CSV, and a text sample is generated after each epoch unless disabled.

### Fine-tune GPT-2

```powershell
.\.venv\Scripts\python.exe -m src.minigpt.training.trainer --pretrained gpt2 --cache-dir "./checkpoints/huggingface"
```

This loads GPT-2 weights into the local MiniGPT implementation, then continues next-token training on the default corpus. It trains all model parameters using the optimizer settings in `training_config.py`.

The loader also accepts a local Hugging Face model directory. Use `--local-files-only` to require cached or local model files. Training with `--pretrained` allows model downloads by default and does not read `inference_config.py`.

`--context-size` can reduce the pretrained model's context capacity by slicing its position embeddings; it cannot increase the source model's capacity. For shorter training windows, configure `seq_length` independently.

### Resume training

Replace `<run_id>` and the checkpoint filename with an existing saved checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.minigpt.training.trainer --resume "checkpoints/training_<run_id>/checkpoint_epoch_0006.pth" --num-epochs 15
```

`--num-epochs 15` means a cumulative target of 15 epochs, including those already completed. A checkpoint saved after epoch 6 resumes with epoch 7. A final `checkpoint.pth` can also be used if it contains the required training progress.

Recovery restores model and AdamW states, saved configuration, loss history, epoch/step/token counters, and available PyTorch random-number states. It creates a new run directory and CSV while retaining the previous history and cumulative counters.

- Recovery starts at the saved epoch boundary. Ctrl+C does not create an additional checkpoint; work after the last saved checkpoint must be repeated.
- `--resume` reads the checkpoint's configuration. Supported command-line overrides are `--num-epochs`, `--device`, `--checkpoint-every-n-epochs`, and `--checkpoint-keep-last`.
- Use the same training text. Checkpoints containing a corpus fingerprint validate its contents during recovery.
- Older checkpoints without the required training counters can still be used for inference, but cannot be resumed. Missing random-number state or a change of device can prevent exact replay.
- `--pretrained` and `--resume` are mutually exclusive.

## Terminal inference

Each prompt starts an independent text continuation. Enter `exit` or `quit` to finish; empty input is ignored.

### Use a trained checkpoint

```powershell
.\.venv\Scripts\python.exe -m scripts.inference --checkpoint "checkpoints/training_<run_id>/checkpoint.pth"
```

An explicitly named periodic checkpoint is also accepted. To select the newest final checkpoint automatically, set `model_source="checkpoint"` and `checkpoint_path=None` in `inference_config.py`, then run without a source argument.

### Use GPT-2 pretrained weights

For the first model download:

```powershell
.\.venv\Scripts\python.exe -m scripts.inference --pretrained gpt2 --cache-dir "./checkpoints/huggingface" --no-local-files-only
```

To reuse the same cache without downloading model files:

```powershell
.\.venv\Scripts\python.exe -m scripts.inference --pretrained gpt2 --cache-dir "./checkpoints/huggingface" --local-files-only
```

The current inference configuration defaults to pretrained GPT-2, `local_files_only=True`, and the default Hugging Face cache (`cache_dir=None`). Override `cache_dir` as above to use another location. `local_files_only` controls Hugging Face model downloads; it does not control tiktoken's encoding cache.

### Generation settings

| Command-line option | Behavior |
| --- | --- |
| `--max-new-tokens 128` | Generate at most 128 new tokens, excluding the prompt |
| `--temperature 0.5` | Adjust sampling probabilities; `0` selects greedy decoding |
| `--top-k 50` | Sample from the 50 highest-scoring tokens; `0` disables Top-k |
| `--eos-id 50256` | Stop when the specified end token is generated |
| `--no-eos` | Disable EOS stopping |
| `--device cpu` | Override automatic device selection |

Generation settings follow this priority: command-line arguments, explicitly present inference configuration keys, saved checkpoint training settings, then fallback defaults. A configured `None` explicitly disables Top-k or EOS. The current terminal defaults are 128 new tokens, temperature `0.5`, Top-k `50`, and EOS ID `50256`.

Generation repeatedly passes the latest `context_size` tokens to the model. The complete returned text can therefore be longer than the context capacity; generation stops at the new-token limit or an enabled EOS token.

## Training artifacts

Each training run gets a unique `training_<run_id>` name:

```text
logs/
`-- training_<run_id>.csv
checkpoints/
`-- training_<run_id>/
    |-- checkpoint_epoch_0006.pth   # Example periodic checkpoint
    |-- checkpoint_epoch_0009.pth   # Latest periodic checkpoint
    `-- checkpoint.pth             # Final checkpoint after normal completion
outputs/
`-- training_<run_id>/
    |-- loss.png
    `-- training_results.json
```

With the default interval and retention, periodic checkpoints are saved after epochs 3, 6, and 9, with the latest two retained. The final checkpoint, loss plot, and JSON report are written after normal completion. A redundant periodic checkpoint from the same epoch is removed after the final checkpoint is saved successfully.

Checkpoints include optimizer state and can be substantially larger than model weights alone. The root `data/`, `checkpoints/`, `logs/`, and `outputs/` directories are excluded from Git, along with the virtual environment and IDE files.

## References

- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) - Learning reference for the model and training workflow.
- [tiktoken](https://github.com/openai/tiktoken) - GPT-2 tokenization.
- [Tiny Shakespeare / char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) - Default training corpus.
- [GPT-2](https://github.com/openai/gpt-2) - Original model implementation.

## License

MiniGPT is licensed under the [Apache License 2.0](LICENSE). Copyright and source attribution are listed in [NOTICE](NOTICE), with the reference project's license preserved in [licenses/LLMs-from-scratch-LICENSE.txt](licenses/LLMs-from-scratch-LICENSE.txt).

External libraries, pretrained model weights, and downloaded datasets retain their respective licenses.
