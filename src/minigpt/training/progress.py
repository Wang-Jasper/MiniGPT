"""
Terminal progress displays and CSV records for training.
!!! Vibe Coding File !!!
"""

import csv
from pathlib import Path
import shutil
import sys
from threading import Event, RLock, Thread
from time import perf_counter


def _duration(seconds):
    """Format elapsed or remaining seconds as a compact clock."""
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressBar:
    """Refresh one terminal line while work runs between completed steps.

    total - Expected number of batches or generated tokens.
    description - Label displayed before the bar.
    unit - Work unit used in wide terminal displays.
    stream - Output stream; None uses standard output.
    refresh_interval - Seconds between interactive refreshes.
    """

    def __init__(
        self, total, description, unit="batch", stream=None, refresh_interval=0.2
    ):
        self.total = max(0, int(total))
        self.description = description
        self.unit = unit
        self.stream = sys.stdout if stream is None else stream
        self.refresh_interval = max(0.05, refresh_interval)
        self.completed = 0
        self.loss = None
        self.global_step = None
        self.total_steps = None
        self.status = ""
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = RLock()
        self._stop = Event()
        self._thread = None
        self._start_time = None
        self._closed = False
        self._last_width = 0
        self._frame = 0

    def __enter__(self):
        with self._lock:
            self._start_time = perf_counter()
            self._render_locked()
        if self._interactive:
            self._thread = Thread(target=self._refresh, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False

    def _refresh(self):
        """Keep elapsed time and the activity spinner moving during slow work."""
        while not self._stop.wait(self.refresh_interval):
            with self._lock:
                if self._closed:
                    return
                self._render_locked()

    def _line(self):
        elapsed = perf_counter() - self._start_time if self._start_time else 0.0
        remaining = (
            elapsed * max(0, self.total - self.completed) / self.completed
            if self.completed else None
        )
        fraction = min(1.0, self.completed / self.total) if self.total else 0.0
        width = max(1, shutil.get_terminal_size(fallback=(100, 20)).columns - 1)
        spinner = " " if self._closed else "|/-\\"[self._frame % 4]
        self._frame += 1

        if width >= 110:
            counts = f" {self.completed}/{self.total} {self.unit} {fraction:.0%}"
            loss = f" | Loss {self.loss:.4f}" if self.loss is not None else ""
            overall = (
                f" | Total {self.global_step}/{self.total_steps}"
                if self.global_step is not None else ""
            )
            timing = f" | Elapsed {_duration(elapsed)} | ETA {_duration(remaining)}"
        else:
            counts = f" {self.completed}/{self.total} {fraction:.0%}"
            loss = f" L{self.loss:.3f}" if self.loss is not None else ""
            overall = (
                f" G{self.global_step}/{self.total_steps}"
                if self.global_step is not None else ""
            )
            timing = f" {_duration(elapsed)} ETA{_duration(remaining)}"
        status = f" {self.status}" if self.status else ""
        tail = f"{counts}{loss}{overall}{timing}{status} {spinner}"
        label_width = max(8, min(len(self.description), width - len(tail) - 4))
        label = self.description[:label_width]
        bar_width = max(1, min(20, width - len(label) - len(tail) - 3))
        filled = int(fraction * bar_width)
        bar = "#" * filled
        if filled < bar_width:
            bar += ">" + "-" * (bar_width - filled - 1)
        line = f"{label} [{bar}]{tail}"
        if len(line) > width:
            line = line[: max(0, width - 3)] + "..."[:width]
        return line

    def _clear_locked(self):
        if self._interactive and self._last_width:
            self.stream.write("\r" + " " * self._last_width + "\r")
            self._last_width = 0

    def _render_locked(self):
        line = self._line()
        if self._interactive:
            self.stream.write("\r" + line + " " * max(0, self._last_width - len(line)))
            self._last_width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def update(self, completed, loss=None, global_step=None, total_steps=None):
        """Report completed work without changing the expected total."""
        with self._lock:
            if self._closed:
                return
            self.completed = max(0, int(completed))
            if loss is not None:
                self.loss = float(loss)
            if global_step is not None:
                self.global_step = global_step
            if total_steps is not None:
                self.total_steps = total_steps
            self._render_locked()

    def set_status(self, message):
        """Show an active phase such as evaluation beside the progress bar."""
        with self._lock:
            if not self._closed:
                self.status = message
                self._render_locked()

    def write(self, message):
        """Print a separate message without mixing it with the active bar."""
        with self._lock:
            self._clear_locked()
            self.stream.write(str(message).rstrip("\n") + "\n")
            if self._interactive and not self._closed:
                self._render_locked()
            self.stream.flush()

    def close(self):
        """Stop refreshing and retain the actual completed count on exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._interactive:
                self._render_locked()
                self.stream.write("\n")
                self.stream.flush()


def training_progress(epoch, num_epochs, total_batches):
    """Create the progress display for one training epoch."""
    return ProgressBar(total_batches, f"Epoch {epoch}/{num_epochs}", unit="batch")


def generation_progress(epoch, max_new_tokens):
    """Create the progress display for an epoch-end text sample."""
    return ProgressBar(max_new_tokens, f"Generating sample (epoch {epoch})", unit="token")


def record_progress(progress, log_path=None, progress_bar=None):
    """Print evaluation losses and optionally append the complete CSV record."""
    message = (
        f"Evaluation | Step {progress['global_step']}/{progress['total_steps']} | "
        f"Train loss {progress['train_loss']:.4f} | "
        f"Validation loss {progress['validation_loss']:.4f} | "
        f"Tokens seen {progress['tokens_seen']}"
    )
    if progress_bar is None:
        show_stage(message)
    else:
        progress_bar.write(message)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", encoding="utf-8", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=list(progress))
            if needs_header:
                writer.writeheader()
            writer.writerow(progress)


def show_stage(message):
    """Display a phase or status message immediately."""
    print(message, flush=True)


def show_training_setup(device, train_batches, val_batches, log_path):
    """Display the selected device, loader sizes, and progress log location."""
    show_stage(f"Device: {device}")
    show_stage(f"Training batches: {train_batches}")
    show_stage(f"Validation batches: {val_batches}")
    show_stage(f"Progress log: {log_path}")


def show_sample(epoch, text):
    """Display the text generated after a training epoch."""
    show_stage(f"Sample after epoch {epoch}:\n{text}")


def show_results(artifacts):
    """Display the paths of saved training artifacts."""
    show_stage(f"Checkpoint: {artifacts['checkpoint']}")
    show_stage(f"Loss plot: {artifacts['loss_plot']}")
    show_stage(f"Training results: {artifacts['results']}")


def show_interruption():
    """Distinguish saved epoch snapshots from unsaved work and final artifacts."""
    show_stage("Training interrupted. Existing CSV records were kept.")
    show_stage("Periodic checkpoints, if any, contain progress only up to their saved epoch.")
    show_stage("This interruption does not create a new checkpoint.")
    show_stage("The final checkpoint.pth and training reports are written after normal completion.")
