from collections.abc import Callable, Iterable, Mapping
from typing import cast

from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from ..config import Config
from ..model import SystemOne
from .early_stopping import EarlyStopping
from .optimizer import build_optimizer
from .step import train_step
from .validation import validate, validate_datasets


def per_dataset_best(history: list[dict]) -> dict[str, dict]:
    """Each dataset's own best balanced/micro accuracy and the epoch it occurred.

    Diagnostic only: datasets peak at different epochs, so this exposes each
    dataset's ceiling independent of the single macro-selected checkpoint.
    Returns {} when validation was not recorded per dataset.
    """
    names: list[str] = []
    for row in history:
        for name in row.get("validation", {}):
            if name not in names:
                names.append(name)
    summary: dict[str, dict] = {}
    for name in names:
        points = [
            (row["validation"][name], epoch)
            for epoch, row in enumerate(history, start=1)
            if name in row.get("validation", {})
            and {"balanced_accuracy", "accuracy"} <= row["validation"][name].keys()
        ]
        if not points:  # older runs stored no per-dataset balanced accuracy
            continue
        balanced, balanced_epoch = max((m["balanced_accuracy"], e) for m, e in points)
        micro, micro_epoch = max((m["accuracy"], e) for m, e in points)
        summary[name] = {
            "balanced": balanced,
            "balanced_epoch": balanced_epoch,
            "micro": micro,
            "micro_epoch": micro_epoch,
        }
    return summary


def train_epoch(
    model: SystemOne,
    batches: Iterable[dict],
    optimizer: Optimizer,
    *,
    epoch: int = 0,
    total: int | None = None,
) -> tuple[float, int]:
    """Train over all batches; return (sample-weighted mean loss, batch count)."""
    total_loss = 0.0
    count = 0
    bar = tqdm(batches, total=total, desc=f"epoch {epoch}", unit="batch", leave=True)
    for batch in bar:
        loss = train_step(model, batch, optimizer)
        size = batch["target"].numel()
        total_loss += loss * size
        count += size
    if count == 0:
        raise ValueError("training batches must not be empty")
    return total_loss / count, bar.n


def train(
    model: SystemOne,
    train_batches: Iterable[dict],
    validation_batches: Iterable[dict] | Mapping[str, Iterable[dict]],
    config: Config,
    *,
    on_improvement: Callable[[dict[str, Tensor], list[dict]], None] | None = None,
    on_epoch: Callable[[dict, int, bool], None] | None = None,
    total_batches: int | None = None,
) -> list[dict]:
    """Train, restore the lowest-validation-loss weights, and return epoch metrics.

    Move the model to its device before calling. Supply re-iterable loaders
    using TimeSeriesDataset.collate, with one head/catalog/channel count per
    batch. The caller controls batch size, sampling, and dataset splits.
    Stop after config.patience consecutive checks without a strict improvement.
    on_improvement can persist the CPU snapshot while training is still running.
    """
    optimizer = build_optimizer(model, config)
    stopping = EarlyStopping(config.patience)
    history = []
    for epoch_num in range(1, config.epochs + 1):
        train_loss, batches_this_epoch = train_epoch(
            model, train_batches, optimizer, epoch=epoch_num, total=total_batches
        )
        if total_batches is None:
            total_batches = batches_this_epoch
        row: dict = {"train_loss": train_loss}
        if isinstance(validation_batches, Mapping):
            per_dataset = validate_datasets(
                model, cast(Mapping[str, Iterable[dict]], validation_batches)
            )
            row["validation"] = per_dataset
            # Macro averages give every dataset equal weight in the summary.
            count = len(per_dataset)
            row["validation_loss"] = sum(m["loss"] for m in per_dataset.values()) / count
            row["validation_accuracy"] = sum(m["accuracy"] for m in per_dataset.values()) / count
            row["validation_balanced_accuracy"] = (
                sum(m["balanced_accuracy"] for m in per_dataset.values()) / count
            )
        else:
            metrics = validate(model, validation_batches)
            row.update(
                validation_loss=metrics["loss"],
                validation_accuracy=metrics["accuracy"],
                validation_balanced_accuracy=metrics["balanced_accuracy"],
            )
        history.append(row)
        # Select the checkpoint by balanced accuracy (higher is better), not loss,
        # so majority-class fitting cannot win the run. EarlyStopping minimizes.
        is_best = stopping.update(model, -row["validation_balanced_accuracy"])
        if is_best and on_improvement is not None:
            assert stopping.best_state is not None
            on_improvement(stopping.best_state, history)
        if on_epoch is not None:
            on_epoch(row, len(history), is_best)
        if stopping.should_stop:
            break
    stopping.restore(model)
    return history
