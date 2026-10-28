from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[3] / "data/raw/tinyshakespeare/input.txt"

def load_default_text() -> str:
    """Read the default training corpus as UTF-8 text."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            "Run python scripts/download_data.py from the project root first."
        )

    return DATA_PATH.read_text(encoding="utf-8")
