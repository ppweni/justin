"""Everything a model needs computed from a series, a question and its options.

The observability models read more than the series: per-channel numbers about how far and when each
channel deviates (``deviation.py``), and numbers about the channels the question and its options
name (``referents.py``). The encoders cannot supply these, because they normalize every
channel internally. ``prepare`` computes all of them from the raw inputs, so a caller hands over a
series, a question and options and nothing else. Training reads the same functions, so a model sees
the same inputs when it is used as when it was trained.

``FEATURE_VERSION`` identifies this computation. It is stored in an export, and a model that uses
these inputs refuses to load under a different version: the features are code, not weights, and
changing them silently changes what the weights mean.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .deviation import deviation_features, normalize
from .referents import grouped_referents, referent_features

FEATURE_VERSION = "1"
_SAMPLING = "one point every"


@dataclass(frozen=True)
class Prepared:
    """The inputs of one question, as the model takes them (no batch dimension)."""

    series: torch.Tensor  # [channels, time], in baseline units when a baseline was given
    question: str
    channel_names: list[str] | None
    channel_features: torch.Tensor  # [channels, deviation.N_FEATURES]
    referent_question: torch.Tensor  # [referents.QUESTION]
    referent_options: torch.Tensor  # [len(options), referents.OPTION]


def uses_features(system) -> bool:
    """Whether a ``SystemOne`` was built to read the inputs ``prepare`` computes."""
    return system.feature_projection is not None or system.referent_question is not None


def prepare(
    series,
    question: str,
    options: list[str] | None = None,
    channel_names: list[str] | None = None,
    *,
    step: float | None = None,
    baseline: float | tuple[int, int] | None = 0.3,
) -> Prepared:
    """Compute the model inputs for one question about a ``[channels, time]`` series.

    ``baseline`` says what counts as normal behavior: the first share of the series (0.3 by
    default), a ``(start, end)`` index range, or ``None`` if the series is already in baseline
    units. ``step`` is the seconds between points; when given and the question does not already
    state it, the question gets the sentence the training questions carry. Channel names are
    needed for questions or options that name services or metrics.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("series must have shape [channels, time] or [time]")
    if not np.isfinite(values).all():
        raise ValueError("series must be finite")
    names = list(channel_names) if channel_names is not None else None
    if names is not None and len(names) != values.shape[0]:
        raise ValueError("channel_names needs one name per channel")
    z = normalize(values, baseline)
    features = deviation_features(z, 1.0 if step is None else float(step))
    if step is not None and _SAMPLING not in question:
        question = f"{question}\nThe series has one point every {step:g} s."
    labels = tuple(names) if names is not None else tuple(f"channel-{i}" for i in range(len(z)))
    choices = tuple(options or ())
    grouped = any(n.startswith("time-series ") for n in labels) or "time-series" in question.lower()
    referents = (
        grouped_referents(choices, labels, features, question)
        if grouped
        else referent_features(question, choices, labels, features)
    )
    return Prepared(
        series=torch.as_tensor(z, dtype=torch.float32),
        question=question,
        channel_names=names,
        channel_features=torch.as_tensor(features, dtype=torch.float32),
        referent_question=torch.as_tensor(referents["referent_question"], dtype=torch.float32),
        referent_options=torch.as_tensor(referents["referent_options"], dtype=torch.float32),
    )
