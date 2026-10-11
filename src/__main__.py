from pathlib import Path

import click
import yaml

from .config import Config


def _load_config(path: Path) -> Config:
    try:
        return Config.from_yaml(path)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        raise click.ClickException(str(error)) from error


@click.group()
def main() -> None:
    """Download data or train and calibrate a Justin model."""


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def download(config: Path) -> None:
    """Cache train and validation data for every dataset in CONFIG (YAML)."""
    settings = _load_config(config)

    from .data import DATASETS

    for name in settings.datasets:
        dataset_class = DATASETS[name]
        for split in ("train", "validation"):
            dataset_class(settings.root, split=split, download=True)
            click.echo(f"Cached {name} {split} in {settings.root}")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def cache(config: Path) -> None:
    """Precompute and persist encoder features for CONFIG (YAML), without training."""
    settings = _load_config(config)
    if settings.cache is None:
        raise click.ClickException("set 'cache' in the config to a directory path")

    from .training.run import build_cache

    build_cache(settings)
    click.echo(f"Cache ready: {settings.cache}")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def train(config: Path) -> None:
    """Train, calibrate, and save a checkpoint using CONFIG (YAML)."""
    settings = _load_config(config)

    from .training.run import run

    checkpoint = run(settings)
    click.echo(f"Saved checkpoint: {checkpoint}")


@main.command()
@click.argument("checkpoints", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--split", type=click.Choice(["validation", "test"]), default="test", show_default=True)
@click.option("--dataset", "datasets", multiple=True, help="Only these datasets (default: all trained on).")
@click.option("--device", default=None, help="Override the device stored in the checkpoint config.")
@click.option("--batch-size", default=32, show_default=True)
@click.option("--root", default=None, help="Override the data root stored in the checkpoint config.")
@click.option("--download", is_flag=True, help="Fetch missing split files (may use the network).")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Write results as JSON.")
def evaluate(checkpoints, split, datasets, device, batch_size, root, download, output) -> None:
    """Score saved CHECKPOINTS on a held-out split (accuracy, balanced accuracy, macro-F1).

    Score the test split once per finished model, not to choose between models.
    """
    import json

    from .training.evaluate import evaluate_checkpoint

    results = []
    for checkpoint in checkpoints:
        try:
            result = evaluate_checkpoint(
                checkpoint, split=split, device=device, batch_size=batch_size, root=root,
                datasets=list(datasets) or None, download=download,
            )
        except (OSError, ValueError, KeyError) as error:
            raise click.ClickException(f"{checkpoint}: {error}") from error
        results.append(result)
        click.echo(
            f"\n{checkpoint}  [{split}] encoder={result['series_encoder']} "
            f"skip={result['signal_skip']} lora={result['series_lora_layers']}"
        )
        click.echo(f"  {'dataset':8s} {'n':>7s} {'bal-acc':>8s} {'macro-F1':>9s} {'acc':>7s}")
        for name, m in result["datasets"].items():
            click.echo(
                f"  {name:8s} {m['samples']:7d} {m['balanced_accuracy'] * 100:8.1f} "
                f"{m['macro_f1'] * 100:9.1f} {m['accuracy'] * 100:7.1f}"
            )
    if output is not None:
        output.write_text(json.dumps(results, indent=2))
        click.echo(f"\nWrote {output}")


if __name__ == "__main__":
    main()
