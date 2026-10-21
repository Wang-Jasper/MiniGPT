from pathlib import Path
from ..tokenizer import createDataLoader
from ..model.embedding import Embedding

DATA_PATH = Path(__file__).resolve().parents[3] / "data/raw/tinyshakespeare/input.txt"

def load_default_text() -> str:
    """Read the default training corpus as UTF-8 text."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            "Run python scripts/download_data.py from the project root first."
        )

    return DATA_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    try:
        text = load_default_text()
        dataloader = createDataLoader(text)
        data_iter = iter(dataloader)

        # inputs are the input sequence, targets are the prediction ground truth
        inputs, targets = next(data_iter)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    except StopIteration:
        raise SystemExit(
            "No batches available. Reduce batch_size or use drop_last=False."
        ) from None
    print(f"Input token IDs: {tuple(inputs.shape)}")
    print(f"Target token IDs: {tuple(targets.shape)}")

    # start to have embeddings



