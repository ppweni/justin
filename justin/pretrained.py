"""Load a published Justin model and answer a question about a signal."""

from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

from .config import Config
from .export import FROZEN_PREFIXES, read_export
from .model import SystemOne
from .model.model import Head
from .training.calibration import probabilities
from .training.evaluate import load_checkpoint
from .training.run import build_model

ArrayLike: TypeAlias = "torch.Tensor | np.ndarray | list"


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


class Justin:
    """A trained Justin model that answers questions about signals.

    ``Justin.from_pretrained(...)`` loads a published file; ``model.answer(series, question, options)``
    picks one of the options (or answers yes/no without options). ``model.system`` is the underlying
    ``SystemOne`` module and ``model.config`` its training config.
    """

    def __init__(self, system: SystemOne, config: Config):
        self.system = system.eval()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        filename: str | None = None,
        revision: str | None = None,
        device: str | None = None,
    ) -> "Justin":
        """Load a published model from a local file or a Hugging Face repo (see ``load_pretrained``)."""
        return cls(*load_pretrained(source, filename=filename, revision=revision, device=device))

    @property
    def temperatures(self) -> dict[str, float]:
        """Calibration temperatures fitted per head (``{}`` if the file has none)."""
        return self.system.temperatures

    @property
    def device(self) -> torch.device:
        return next(self.system.parameters()).device

    @torch.no_grad()
    def answer(
        self, series: ArrayLike, question: str, options: list[str] | None = None
    ) -> str | bool:
        """Answer one question about one signal.

        ``series`` is ``[channels, time]`` (or ``[time]`` for a single channel). With ``options`` the
        best option is returned; without them the yes/no head returns a bool.
        """
        logits, head = self._logits(series, question, options)
        if options is None:
            return bool(logits.item() >= 0)
        return options[int(logits.argmax(-1))]

    @torch.no_grad()
    def probabilities(
        self, series: ArrayLike, question: str, options: list[str] | None = None
    ) -> dict[str, float] | float:
        """Calibrated probabilities: ``{option: p}`` with options, else P(yes)."""
        logits, head = self._logits(series, question, options)
        probability = probabilities(logits, head, self.temperatures.get(head, 1.0))
        if options is None:
            return float(probability.item())
        return dict(zip(options, probability[0].tolist(), strict=True))

    def _logits(
        self, series: ArrayLike, question: str, options: list[str] | None
    ) -> tuple[torch.Tensor, Head]:
        tensor = torch.as_tensor(series, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError("series must have shape [channels, time] or [time]")
        batch = tensor.unsqueeze(0).to(self.device)
        if options is None:
            return self.system(batch, [question], head="noul"), "noul"
        return self.system(batch, [question], options=options), "choice"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
