"""Load a published Justin model and answer a question about a signal."""

from pathlib import Path

import torch

from .config import Config
from .export import FROZEN_PREFIXES, read_export
from .model import SystemOne
from .training.evaluate import load_checkpoint
from .training.run import build_model


def load_pretrained(
    source: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    device: str | None = None,
) -> tuple[SystemOne, Config]:
    """Load a ``.safetensors`` export (or a full ``.pt`` checkpoint) from a file or a Hub repo.

    ``source`` is a local file, or a repo id such as ``owner/justin-chronos-minilm`` with the file's
    ``filename`` (and optionally the repo ``revision``). An export records the exact revisions of
    the frozen Chronos-2 and MiniLM weights it was trained on, and those are what get downloaded.
    Returns the model in eval mode and its config; ``model.temperatures`` holds the fitted
    calibration temperatures (``{}`` for a plain ``.pt`` without them).
    """
    path = Path(source)
    if not path.is_file():
        if filename is None:
            raise ValueError(f"{source} is not a file; pass filename= to download it from the Hub")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=str(source), filename=filename, revision=revision))
    device = device or _default_device()
    if path.suffix != ".safetensors":
        model, config = load_checkpoint(path, device)
        model.temperatures = {}
        return model, config

    state, meta = read_export(path)
    config = Config(**meta["config"])
    model = build_model(config, {name: info["revision"] for name, info in meta["revisions"].items()})
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if unexpected or any(k in trainable or not k.startswith(FROZEN_PREFIXES) for k in missing):
        raise ValueError(f"{path} does not match the model it describes")
    model.temperatures = meta["temperatures"]
    return model.to(torch.device(device)).eval(), config


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
