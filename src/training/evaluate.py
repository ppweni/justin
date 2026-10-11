"""Evaluate saved checkpoints on a held-out split: accuracy, balanced accuracy, and macro-F1 per dataset.

Balanced accuracy and class keys follow ``validation.validate`` (true answer string for open catalogs,
true class for Noul). Macro-F1 averages per-class F1 over every class that appears among the true or
predicted answers (scikit-learn's default), so predicting a class that never occurs lowers the score.

The test split is meant to be scored once per finished model, not used to choose between models.
"""

import dataclasses
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

from ..config import Config
from ..data import DATASETS
from ..model import SystemOne
from .batch import model_logits, prepare_batch
from .loader import BatchLoader
from .run import build_model


def macro_f1(pairs: Iterable[tuple[object, object]]) -> float:
    """Macro-F1 over (true, predicted) class keys; classes are those seen in either."""
    true_positive: dict[object, int] = defaultdict(int)
    false_positive: dict[object, int] = defaultdict(int)
    false_negative: dict[object, int] = defaultdict(int)
    for truth, prediction in pairs:
        if truth == prediction:
            true_positive[truth] += 1
        else:
            false_negative[truth] += 1
            false_positive[prediction] += 1
    classes = set(true_positive) | set(false_positive) | set(false_negative)
    if not classes:
        raise ValueError("no predictions to score")
    scores = []
    for c in classes:
        denominator = 2 * true_positive[c] + false_positive[c] + false_negative[c]
        scores.append(2 * true_positive[c] / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


@torch.no_grad()
def predictions(model: SystemOne, batches: Iterable[dict]) -> list[tuple[object, object]]:
    """(true class, predicted class) per sample, keyed like the validation balanced accuracy."""
    was_training = model.training
    model.eval()
    pairs: list[tuple[object, object]] = []
    try:
        for batch in batches:
            inputs, target = prepare_batch(model, batch)
            head = inputs.get("head", "choice")
            logits = model_logits(model, inputs)
            predicted = (logits >= 0).long() if head == "noul" else logits.argmax(dim=-1)
            truth = (target >= 0.5).long() if head == "noul" else target
            options = inputs.get("options") or inputs.get("option_key")
            for t, p in zip(truth.tolist(), predicted.tolist(), strict=True):
                if head == "noul":
                    pairs.append((("noul", t), ("noul", p)))
                elif head == "choice" and options is not None:
                    pairs.append((("choice", options[t]), ("choice", options[p])))
                else:
                    pairs.append(((head, t), (head, p)))
    finally:
        model.train(was_training)
    return pairs


def score(pairs: list[tuple[object, object]]) -> dict[str, float]:
    if not pairs:
        raise ValueError("no predictions to score")
    recalls: dict[object, list[int]] = defaultdict(lambda: [0, 0])
    for truth, prediction in pairs:
        recalls[truth][0] += int(truth == prediction)
        recalls[truth][1] += 1
    return {
        "samples": len(pairs),
        "accuracy": sum(t == p for t, p in pairs) / len(pairs),
        "balanced_accuracy": sum(hit / n for hit, n in recalls.values()) / len(recalls),
        "macro_f1": macro_f1(pairs),
    }


def evaluate_model(
    model: SystemOne, loaders: Mapping[str, Iterable[dict]]
) -> dict[str, dict[str, float]]:
    return {name: score(predictions(model, batches)) for name, batches in loaders.items()}


def load_checkpoint(path: str | Path, device: str | None = None) -> tuple[SystemOne, Config]:
    """Rebuild the model a checkpoint was trained as and load its weights (eval mode)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw = dict(checkpoint["config"])
    known = {field.name for field in dataclasses.fields(Config)}
    config = Config(**{key: value for key, value in raw.items() if key in known})
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(torch.device(device or config.device)).eval()
    return model, config


def evaluate_checkpoint(
    path: str | Path,
    *,
    split: str = "test",
    device: str | None = None,
    batch_size: int = 32,
    root: str | None = None,
    datasets: list[str] | None = None,
    download: bool = False,
) -> dict:
    """Score a checkpoint on `split` for its configured datasets (or a chosen subset)."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    model, config = load_checkpoint(path, device)
    names = datasets or config.datasets
    unknown = [name for name in names if name not in config.datasets]
    if unknown:
        raise ValueError(f"checkpoint was not trained on: {', '.join(unknown)}")
    loaders = {
        name: BatchLoader(
            DATASETS[name](root or config.root, split=split, download=download),
            batch_size,
            dataset_name=name,
        )
        for name in names
    }
    return {
        "checkpoint": str(path),
        "split": split,
        "series_encoder": config.series_encoder,
        "signal_skip": config.signal_skip,
        "series_lora_layers": config.series_lora_layers,
        "datasets": evaluate_model(model, loaders),
    }
