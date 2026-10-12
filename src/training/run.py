from collections.abc import Mapping
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import torch

from ..cache import CachedBatchLoader, CachedDataset, FeatureCache
from ..config import Config
from ..data import DATASETS
from ..model import SystemOne
from ..persistence import Store
from .calibration import fit_temperatures
from .checkpoint import save_checkpoint
from .loader import BatchLoader
from .mixing import BalancedBatches
from .train import per_dataset_best, train


def build_model(config: Config, revisions: Mapping[str, str] | None = None) -> SystemOne:
    """Pretrained frozen encoders plus untrained fusion and heads, as configured.

    `revisions` pins the backbones ("series", "text") to Hub commits; default is the pinned ones.
    """
    return SystemOne.from_pretrained(
        num_classes=config.num_classes,
        score_levels=config.score_levels,
        hidden_dim=config.hidden_dim,
        fusion_layers=config.fusion_layers,
        dropout=config.dropout,
        with_text=config.with_text,
        option_attention=config.option_attention,
        signal_skip=config.signal_skip,
        series_encoder=config.series_encoder,
        revisions=revisions,
        series_lora_layers=config.series_lora_layers,
        minilm_lora=config.minilm_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    )


def run(config: Config) -> Path:
    """Save each best epoch, restore it, then calibrate and finalize the checkpoint.

    The test split is untouched. Existing output paths are rejected before
    training. Dataset downloads require download=True in the configuration.
    """
    if config.split != "train":
        raise ValueError("training requires split: train; validation is loaded separately")
    if not config.with_text and any(name in ("tsqa", "ecg") for name in config.datasets):
        raise ValueError("TSQA and ECG require with_text: true for runtime questions")
    if len(config.datasets) > 1 and (not config.with_text or config.num_classes is not None):
        raise ValueError("multiple datasets require with_text: true and num_classes: null")
    output = Path(config.output)
    if output.exists():
        raise FileExistsError(f"checkpoint already exists: {output}")
    device = torch.device(config.device)
    # Validate every dataset before loading weights, so bad configs fail cheaply.
    splits: dict[str, tuple] = {}
    for name in config.datasets:
        dataset_class = DATASETS[name]
        train_data = dataset_class(config.root, split="train", download=config.download)
        validation_data = dataset_class(config.root, split="validation", download=config.download)
        if len(train_data) == 0 or len(validation_data) == 0:
            raise ValueError(f"train and validation datasets must both be nonempty: {name}")
        first = train_data[0]
        if not config.with_text and config.num_classes != len(first["options"]):
            raise ValueError(f"num_classes must be {len(first['options'])} for {name}")
        splits[name] = (train_data, validation_data)
        print(f"  {name}: {len(train_data)} train / {len(validation_data)} validation", flush=True)
    model = build_model(config).to(device)
    # LoRA moves the cache boundary: cache the frozen prefix (signal) and skip text.
    prefix_start = model.lora_boundary if config.series_lora_layers > 0 else None
    cache_text = not config.minilm_lora
    train_loaders, validation_loaders = {}, {}
    for name, (train_data, validation_data) in splits.items():
        if config.cache:
            train_cache = FeatureCache.load_or_build(
                train_data, model, cache_dir=config.cache, name=name, split="train",
                prefix_start=prefix_start, cache_text=cache_text,
            )
            validation_cache = FeatureCache.load_or_build(
                validation_data, model, cache_dir=config.cache, name=name, split="validation",
                prefix_start=prefix_start, cache_text=cache_text,
            )
            train_loaders[name] = CachedBatchLoader(
                CachedDataset(
                    train_data, train_cache,
                    weighted=name in config.class_weighted, live_text=config.minilm_lora,
                ),
                train_cache,
                config.batch_size,
                shuffle=True,
                dataset_name=name,
            )
            validation_loaders[name] = CachedBatchLoader(
                CachedDataset(validation_data, validation_cache, live_text=config.minilm_lora),
                validation_cache,
                config.batch_size,
                dataset_name=name,
            )
        else:
            train_loaders[name] = BatchLoader(
                train_data,
                config.batch_size,
                shuffle=True,
                dataset_name=name,
            )
            validation_loaders[name] = BatchLoader(
                validation_data,
                config.batch_size,
                dataset_name=name,
            )
    print(
        f"Loaded {len(train_loaders)} datasets "
        f"(batch_size={config.batch_size}, epochs={config.epochs}, patience={config.patience})",
        flush=True,
    )
    total_batches = None
    store = Store(config.db) if config.db else None
    run_id = store.create_run(asdict(config)) if store is not None else None
    latest_cp: Path | None = None
    saved = False

    def save_best(state: dict, history: list[dict]) -> None:
        nonlocal saved, latest_cp
        if store is not None:
            cp = Store.checkpoint_name_for(run_id, len(history), output)  # type: ignore[arg-type]
            save_checkpoint(
                cp,
                {
                    "model": state,
                    "config": asdict(config),
                    "history": history,
                    "best_epoch": len(history),
                    "temperatures": {},
                },
                overwrite=False,
            )
            latest_cp = cp
        else:
            save_checkpoint(
                output,
                {
                    "model": state,
                    "config": asdict(config),
                    "history": history,
                    "best_epoch": len(history),
                    "temperatures": {},
                },
                overwrite=saved,
            )
        saved = True

    def on_epoch(row: dict, epoch: int, is_best: bool) -> None:
        tag = " *" if is_best else ""
        print(
            f"Epoch {epoch}: train={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.3f} "
            f"val_bacc={row['validation_balanced_accuracy']:.3f}{tag}",
            flush=True,
        )
        if store is not None:
            store.record_epoch(
                run_id,  # type: ignore[arg-type]
                epoch,
                row,
                is_best=is_best,
                checkpoint_name=str(latest_cp) if is_best else None,
            )

    print("Training started.", flush=True)
    history = train(
        model,
        BalancedBatches(train_loaders),
        validation_loaders,
        config,
        on_improvement=save_best,
        on_epoch=on_epoch,
        total_batches=total_batches,
    )
    summary = per_dataset_best(history)
    if summary:
        print("Per-dataset best epochs (diagnostic):", flush=True)
        for name, m in summary.items():
            print(
                f"  {name}: balanced={m['balanced']:.3f}@e{m['balanced_epoch']} "
                f"micro={m['micro']:.3f}@e{m['micro_epoch']}",
                flush=True,
            )
    temperatures = fit_temperatures(model, chain(*validation_loaders.values()))
    load_from = latest_cp if (store is not None and latest_cp is not None) else output
    checkpoint = torch.load(load_from, map_location="cpu", weights_only=True)
    checkpoint.update(history=history, temperatures=temperatures, per_dataset_best=summary)
    save_checkpoint(output, checkpoint, overwrite=output.exists())
    if store is not None:
        store.finish_run(run_id, str(output))  # type: ignore[arg-type]
    return output


def build_cache(config: Config) -> None:
    """Precompute and persist frozen-encoder features for every dataset, no training.

    Builds the same on-disk cache that run() consumes when config.cache is set,
    so a later training run loads it instead of re-encoding.
    """
    if not config.cache:
        raise ValueError("build_cache requires config.cache to be set")
    device = torch.device(config.device)
    model = build_model(config).to(device)
    prefix_start = model.lora_boundary if config.series_lora_layers > 0 else None
    cache_text = not config.minilm_lora
    for name in config.datasets:
        dataset_class = DATASETS[name]
        for split in ("train", "validation"):
            data = dataset_class(config.root, split=split, download=config.download)
            if len(data) == 0:
                raise ValueError(f"{split} dataset must be nonempty: {name}")
            FeatureCache.load_or_build(
                data, model, cache_dir=config.cache, name=name, split=split,
                prefix_start=prefix_start, cache_text=cache_text,
            )
