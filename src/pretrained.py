"""Load a published Justin checkpoint and answer a question about a signal."""

from pathlib import Path

import torch

from .config import Config
from .model import SystemOne
from .training.evaluate import load_checkpoint


def load_pretrained(
    source: str | Path, *, filename: str | None = None, device: str | None = None
) -> tuple[SystemOne, Config]:
    """Load a checkpoint from a local file or a Hugging Face model repo.

    ``source`` is a path to a ``.pt`` file, or a repo id such as ``owner/justin-chronos`` with the
    checkpoint's ``filename`` inside it. The frozen Chronos-2 and MiniLM weights are fetched from
    their own repos on first use. Returns the model in eval mode and its training config.
    """
    path = Path(source)
    if not path.is_file():
        if filename is None:
            raise ValueError(f"{source} is not a file; pass filename= to download it from the Hub")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=str(source), filename=filename))
    return load_checkpoint(path, device or _default_device())


@torch.no_grad()
def answer(
    model: SystemOne, series: torch.Tensor, question: str, options: list[str] | None = None
) -> str | bool:
    """Answer one question about one signal.

    ``series`` is ``[channels, time]``. With ``options`` the Choice head returns the best option;
    without them the Noul (yes/no) head returns a bool.
    """
    if series.ndim != 2:
        raise ValueError("series must have shape [channels, time]")
    device = next(model.parameters()).device
    batch = series.unsqueeze(0).to(device)
    if options is None:
        return bool(model(batch, [question], head="noul").item() >= 0)
    logits = model(batch, [question], options=options)
    return options[int(logits.argmax(-1))]


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
