from collections.abc import Iterable, Mapping

import torch

from ..model import SystemOne
from .batch import model_logits, prepare_batch
from .loss import compute_loss


@torch.no_grad()
def validate(model: SystemOne, batches: Iterable[dict]) -> dict[str, float]:
    """Loss, micro accuracy, and balanced (macro-recall) accuracy.

    Restores the model's previous mode. Noul uses a 0.5 probability threshold
    (also for soft targets). Score accuracy compares the most likely level with
    the target level. Balanced accuracy averages per-class recall, keyed by the
    true answer string for open catalogs, so majority classes cannot dominate.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = count = 0
    per_class: dict[tuple, list[int]] = {}
    try:
        for batch in batches:
            inputs, target = prepare_batch(model, batch)
            head = inputs.get("head", "choice")
            logits = model_logits(model, inputs)
            loss = compute_loss(logits, target, head)
            prediction = logits >= 0 if head == "noul" else logits.argmax(dim=-1)
            labels = target >= 0.5 if head == "noul" else target
            count += target.numel()
            total_loss += loss.item() * target.numel()
            hit = prediction == labels
            correct += hit.sum().item()
            # Bucket each sample by its true class for per-class recall.
            options = inputs.get("options") or inputs.get("option_key")
            targets, hits = target.tolist(), hit.tolist()
            for value, was_correct in zip(targets, hits, strict=True):
                if head == "noul":
                    key = ("noul", int(value >= 0.5))
                elif head == "choice" and options is not None:
                    key = ("choice", options[int(value)])
                else:
                    key = (head, int(value))
                stats = per_class.setdefault(key, [0, 0])
                stats[0] += int(was_correct)
                stats[1] += 1
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("validation batches must not be empty")
    balanced = sum(c / n for c, n in per_class.values()) / len(per_class)
    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "balanced_accuracy": balanced,
    }


def validate_datasets(
    model: SystemOne,
    loaders: Mapping[str, Iterable[dict]],
) -> dict[str, dict[str, float]]:
    """One complete, unbalanced validation pass per dataset."""
    if not loaders:
        raise ValueError("validation datasets must not be empty")
    return {name: validate(model, batches) for name, batches in loaders.items()}
