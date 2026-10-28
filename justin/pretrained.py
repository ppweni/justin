"""Load a published Justin model (and its location head) and ask it about a signal."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch

from .encoders import SERIES_ENCODERS, TextEncoder
from .features import FEATURE_VERSION, prepare, uses_features
from .files import FROZEN_PREFIXES, fingerprint, read_export, read_location
from .location import location_head_from_spec, location_logits
from .system import Head, SystemOne

ArrayLike: TypeAlias = "torch.Tensor | np.ndarray | list"
Baseline: TypeAlias = "float | tuple[int, int] | None"


def build_system(config: dict, revisions: dict[str, str] | None = None) -> SystemOne:
    """The frozen backbones at the given revisions plus untrained fusion and heads, as `config`
    (the ``config`` of an export's metadata) describes them."""
    revisions = revisions or {}
    name = config["series_encoder"]
    if name not in SERIES_ENCODERS:
        raise ValueError(f"series encoder {name} is not supported ({sorted(SERIES_ENCODERS)})")
    if not config.get("with_text", True) or config.get("text_encoder", "minilm") != TextEncoder.name:
        raise ValueError("only models with the MiniLM text encoder are supported")
    series = SERIES_ENCODERS[name].from_pretrained(revision=revisions.get("series"))
    if config.get("series_depth") is not None:
        series.truncate(config["series_depth"])
    return SystemOne(
        series,
        TextEncoder.from_pretrained(revision=revisions.get("text")),
        num_classes=config.get("num_classes"),
        score_levels=config.get("score_levels", 5),
        hidden_dim=config["hidden_dim"],
        fusion_layers=config.get("fusion_layers", 1),
        option_attention=config.get("option_attention", False),
        noul_attention=config.get("noul_attention", False),
        signal_skip=config.get("signal_skip", False),
        channel_names=config.get("channel_names", False),
        channel_features=config.get("channel_features", 0),
        channel_head=config.get("channel_head", False),
        referents=config.get("referents", False),
        kind_summary=config.get("kind_summary", 0),
        series_condition=config.get("series_condition", 0),
        series_lora_layers=config.get("series_lora_layers", 0),
        minilm_lora=config.get("minilm_lora", False),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
    )


def _resolve(source: str | Path, filename: str | None, revision: str | None) -> Path:
    """A local file, or ``filename`` downloaded from the Hub repo ``source``."""
    path = Path(source)
    if path.is_file():
        return path
    if filename is None:
        raise ValueError(f"{source} is not a file; pass filename= to download it from the Hub")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(source), filename=filename, revision=revision))


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pretrained(
    source: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    device: str | None = None,
) -> tuple[SystemOne, dict]:
    """Load a model export (``.safetensors``) from a file or a Hub repo (with ``filename``).

    The export records the exact Timestral and MiniLM revisions it was trained on; those are
    downloaded. Returns the model in eval mode and the export's metadata (``config``,
    ``revisions``, ``temperatures``); ``model.temperatures`` holds the calibration temperatures.
    """
    path = _resolve(source, filename, revision)
    state, meta = read_export(path)
    config = meta["config"]
    if (config.get("channel_features") or config.get("referents")) and meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"{path} was trained with feature version {meta.get('feature_version')}, this code computes {FEATURE_VERSION}"
        )
    model = build_system(config, {name: info["revision"] for name, info in meta["revisions"].items()})
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if unexpected or any(k in trainable or not k.startswith(FROZEN_PREFIXES) for k in missing):
        raise ValueError(f"{path} does not match the model it describes")
    model.temperatures = meta["temperatures"]
    return model.to(torch.device(device or _default_device())).eval(), meta


def load_location(
    source: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    base: str | Path | None = None,
    device: str | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load a location head export in eval mode; with ``base`` (the model export it belongs to)
    the recorded fingerprint must match. Returns the head and its spec."""
    state, meta = read_location(_resolve(source, filename, revision))
    if base is not None and fingerprint(base) != meta["base_fingerprint"]:
        raise ValueError("this location head does not belong to the loaded model (fingerprint differs)")
    head = location_head_from_spec(meta["head"])
    head.load_state_dict(state)
    return head.to(torch.device(device or _default_device())).eval(), meta["head"]


@dataclass(frozen=True)
class Question:
    """A question with its possible answers: a list of ``options``, or ``"yes_no"``, ``"score"``
    or ``"classes"`` (the model's fixed classes)."""

    question: str
    answers: list[str] | Literal["yes_no", "score", "classes"] = "yes_no"


@dataclass(frozen=True)
class Location:
    """Where something is ("Locate the onset of the spike in cpu."); needs a location head."""

    question: str


Query: TypeAlias = Question | Location


@dataclass(frozen=True)
class Answer:
    """The output of a `Question`: raw ``logits``, calibrated ``probabilities`` (one per option or
    level; for yes/no the probability of yes) and the ``answer`` (an option, a bool or a level index)."""

    logits: list[float]
    probabilities: list[float]
    answer: str | bool | int


@dataclass(frozen=True)
class Found:
    """The output of a `Location`: ``logits`` [channels * patches + 1] (the last is "not present"),
    ``null_probability`` and ``location`` (``None`` or channel, name, patch, start, end, probability)."""

    logits: list[float]
    null_probability: float
    location: dict | None = field(default=None)


class Justin:
    """A trained Justin model that answers questions about signals and locates events in them.

    ``Justin.from_pretrained(...)`` loads a published file. ``answer`` / ``probabilities`` take one
    question, ``locate`` one location question, and ``ask`` a list of `Question` / `Location`
    queries about the same series. ``model.system`` is the underlying ``SystemOne``.
    """

    def __init__(
        self,
        system: SystemOne,
        meta: dict | None = None,
        location: torch.nn.Module | None = None,
        location_spec: dict | None = None,
    ):
        self.system = system.eval()
        self.meta = meta or {}
        self.location = location.eval() if location is not None else None
        self.location_spec = location_spec

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        filename: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        location: str | Path | None = None,
        location_filename: str | None = None,
        location_revision: str | None = None,
    ) -> "Justin":
        """Load a model export from a local file or a Hugging Face repo (``filename`` in it).

        ``location`` (a file, or a repo with ``location_filename``) adds the location head trained
        on this model, for ``locate``.
        """
        path = _resolve(source, filename, revision)
        system, meta = load_pretrained(path, device=device)
        if location is None:
            return cls(system, meta)
        head, spec = load_location(
            location, filename=location_filename, revision=location_revision, base=path,
            device=str(next(system.parameters()).device),
        )  # fmt: skip
        return cls(system, meta, head, spec)

    @property
    def config(self) -> dict:
        return self.meta.get("config", {})

    @property
    def temperatures(self) -> dict[str, float]:
        """Calibration temperatures fitted per head."""
        return self.system.temperatures

    @property
    def device(self) -> torch.device:
        return next(self.system.parameters()).device

    @torch.no_grad()
    def answer(
        self,
        series: ArrayLike,
        question: str,
        options: list[str] | None = None,
        channel_names: Sequence[str] | None = None,
        *,
        step: float | None = None,
        baseline: Baseline = 0.3,
    ) -> str | bool:
        """Answer one question: the best option, or a bool without options.

        ``series`` is ``[channels, time]`` (or ``[time]``) of raw values. Models trained with
        channel names take one name per channel. ``step`` is the seconds between points;
        ``baseline`` is what counts as normal behavior for the deviation features (the first 30%
        by default, a ``(start, end)`` index range, or ``None`` if already in baseline units).
        """
        logits, _ = self._logits(series, question, options, channel_names, step, baseline)
        if options is None:
            return bool(logits.item() >= 0)
        return options[int(logits.argmax(-1))]

    @torch.no_grad()
    def probabilities(
        self,
        series: ArrayLike,
        question: str,
        options: list[str] | None = None,
        channel_names: Sequence[str] | None = None,
        *,
        step: float | None = None,
        baseline: Baseline = 0.3,
    ) -> dict[str, float] | float:
        """Calibrated probabilities: ``{option: p}`` with options, else P(yes). See ``answer``."""
        logits, head = self._logits(series, question, options, channel_names, step, baseline)
        probability = _calibrated(logits, head, self.temperatures.get(head, 1.0))
        if options is None:
            return float(probability.item())
        return dict(zip(options, probability[0].tolist(), strict=True))

    @torch.no_grad()
    def ask(
        self,
        series: ArrayLike,
        queries: Sequence[Query],
        channel_names: Sequence[str] | None = None,
        *,
        step: float | None = None,
        baseline: Baseline = 0.3,
    ) -> list[Answer | Found]:
        """Answer every query about one series, in order (the Rust crate's ``forward``)."""
        outputs: list[Answer | Found] = []
        for query in queries:
            if isinstance(query, Location):
                outputs.append(Found(**_found(self.locate(series, query.question, channel_names))))
                continue
            answers = query.answers
            options = list(answers) if isinstance(answers, list) else None
            head: Head = {"yes_no": "noul", "score": "score"}.get(answers, "choice") if options is None else "choice"  # type: ignore[arg-type]
            if options is not None and not 2 <= len(options) <= 255:
                raise ValueError("a question needs 2 to 255 options")
            logits, head = self._logits(series, query.question, options, channel_names, step, baseline, head)
            probability = _calibrated(logits, head, self.temperatures.get(head, 1.0)).reshape(-1)
            flat = logits.reshape(-1).float().cpu()
            if head == "noul":
                answer: str | bool | int = bool(flat.item() >= 0)
            elif options is not None:
                answer = options[int(flat.argmax())]
            else:
                answer = int(flat.argmax())
            outputs.append(Answer(flat.tolist(), probability.float().cpu().tolist(), answer))
        return outputs

    @torch.no_grad()
    def locate(
        self, series: ArrayLike, question: str, channel_names: Sequence[str] | None = None
    ) -> dict:
        """Point at the patch a location question asks for, or say it is not present.

        ``series`` is ``[channels, time]`` (or ``[time]``) of raw values (NaN = missing). Returns
        ``logits`` (``[channels * patches + 1]``, the last is "not present"), ``null_probability``,
        and ``location``: ``None`` or the ``channel`` index, its ``name``, the ``patch`` index, the
        sample range ``start``/``end`` (end exclusive) and its ``probability``.
        """
        if self.location is None or self.location_spec is None:
            raise ValueError("no location head loaded: pass location= to from_pretrained")
        tensor = _series(series)
        if channel_names is not None and len(channel_names) != tensor.shape[0]:
            raise ValueError("channel_names needs one name per channel")
        names = [list(channel_names)] if channel_names is not None else None
        logits = location_logits(
            self.system, self.location, tensor[None].to(self.device), None, [question], names
        )[0].float().cpu()
        probability = logits.softmax(-1)
        channels, length = tensor.shape
        patch = self.location_spec["patch_size"]
        context = self.system.series_encoder.backbone.context_length
        offset = max(0, length - context)
        kept = length - offset
        patches = -(-kept // patch)
        if (logits.shape[0] - 1) != channels * patches:
            raise ValueError("the location head needs an encoder with only patch tokens")
        pad = patches * patch - kept
        best = int(probability.argmax())
        location = None
        if best < channels * patches:
            channel, index = divmod(best, patches)
            location = {
                "channel": channel,
                "name": None if channel_names is None else channel_names[channel],
                "patch": index,
                "start": offset + max(0, index * patch - pad),
                "end": offset + min(kept, (index + 1) * patch - pad),
                "probability": float(probability[best]),
            }
        return {"logits": logits, "null_probability": float(probability[-1]), "location": location}

    def _logits(
        self,
        series: ArrayLike,
        question: str,
        options: list[str] | None,
        channel_names: Sequence[str] | None,
        step: float | None,
        baseline: Baseline,
        head: Head | None = None,
    ) -> tuple[torch.Tensor, Head]:
        tensor = _series(series)
        if channel_names is not None and len(channel_names) != tensor.shape[0]:
            raise ValueError("channel_names needs one name per channel")
        system = self.system
        extra: dict = {}
        if uses_features(system):
            if system.channel_projection is not None and channel_names is None:
                raise ValueError("this model was trained with channel names: pass channel_names")
            prepared = prepare(
                tensor.numpy(), question, options, list(channel_names) if channel_names else None,
                step=step, baseline=baseline,
            )  # fmt: skip
            tensor, question = prepared.series, prepared.question
            if system.feature_projection is not None:
                extra["channel_features"] = prepared.channel_features[None].to(self.device)
            if system.referent_question is not None:
                extra["referent_question"] = prepared.referent_question[None].to(self.device)
                extra["referent_options"] = prepared.referent_options[None].to(self.device)
        batch = tensor.unsqueeze(0).to(self.device)
        takes_names = system.channel_projection is not None or not uses_features(system)
        names = [list(channel_names)] if channel_names is not None and takes_names else None
        head = head or ("noul" if options is None else "choice")
        logits = system(batch, [question], options=options, head=head, channel_names=names, **extra)
        return logits, head


def _series(series: ArrayLike) -> torch.Tensor:
    tensor = torch.as_tensor(series, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError("series must have shape [channels, time] or [time]")
    return tensor


def _found(result: dict) -> dict:
    return {**result, "logits": result["logits"].tolist()}


def _calibrated(logits: torch.Tensor, head: Head, temperature: float = 1.0) -> torch.Tensor:
    """Logits / temperature through sigmoid (Noul: P(yes)) or softmax (Choice, Score)."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = logits.float() / temperature
    return scaled.sigmoid() if head == "noul" else scaled.softmax(dim=-1)
