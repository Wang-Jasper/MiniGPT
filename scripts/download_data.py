"""Download the default Tiny Shakespeare training corpus."""

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/tinyshakespeare/input.txt"


def download_data() -> Path:
    """Download UTF-8 text once, resolving the path from the project root."""
    if DATA_PATH.is_file() and DATA_PATH.stat().st_size > 0:
        print(f"Already downloaded: {DATA_PATH}")
        return DATA_PATH

    print(f"Downloading: {DATA_URL}")
    with urlopen(DATA_URL, timeout=60) as response:
        content = response.read()

    if not content.strip():
        raise ValueError("The downloaded corpus is empty.")
    content.decode("utf-8")  # Validate before saving the original bytes.

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_bytes(content)
    print(f"Saved {len(content):,} bytes to {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    try:
        download_data()
    except (URLError, OSError, ValueError) as error:
        raise SystemExit(f"Download failed: {error}") from error
