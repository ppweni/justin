"""Multi-dataset tests: one shared model, no dataset= key, variable channel counts."""

from collections import Counter
from dataclasses import asdict
from importlib import import_module
from unittest.mock import patch

import pytest
import torch
import yaml
from click.testing import CliRunner

from src import Config
from src.__main__ import main
from src.data import DATASETS
from src.model import SystemOne
from src.training.calibration import fit_temperatures
from src.training.loader import BatchLoader
from src.training.mixing import BalancedBatches
from src.training.optimizer import build_optimizer
from src.training.run import run
from src.training.step import train_step
from src.training.train import train
from src.training.validation import validate_datasets

CHANNELS = {"har": 3, "sleep": 1, "tsqa": 1, "ecg": 12}


@pytest.mark.parametrize("sizes", [{"a": 2, "b": 5}, {"a": 1, "b": 3, "c": 2}, {"a": 4}])
def test_balanced_epochs_repeat_small_loaders_and_cover_largest(sizes):
    loaders = {
        name: [{"dataset": name, "id": i} for i in range(size)] for name, size in sizes.items()
    }
    mixed = BalancedBatches(loaders)
    for _ in range(2):
        batches = list(mixed)
        assert Counter(b["dataset"] for b in batches) == dict.fromkeys(sizes, max(sizes.values()))
        for name, size in sizes.items():
            actual = [b["id"] for b in batches if b["dataset"] == name]
            assert actual == [i % size for i in range(max(sizes.values()))]
        for start in range(0, len(batches), len(sizes)):
            assert {b["dataset"] for b in batches[start : start + len(sizes)]} == set(sizes)


def test_mixer_restarts_loaders_instead_of_caching_batches():
    class FreshLoader:
        calls = 0

        def __iter__(self):
            self.calls += 1
            yield {"dataset": "small", "pass": self.calls}

    small = FreshLoader()
    batches = list(BalancedBatches({"small": small, "big": [{"dataset": "big"}] * 3}))
    assert [b["pass"] for b in batches if b["dataset"] == "small"] == [1, 2, 3]


def test_mixer_rejects_missing_empty_or_one_shot_loaders():
    with pytest.raises(ValueError, match="at least one"):
        BalancedBatches({})
    with pytest.raises(ValueError, match="empty"):
        list(BalancedBatches({"empty": [], "other": [{}]}))
    with pytest.raises(ValueError, match="re-iterable"):
        list(BalancedBatches({"once": iter([{}]), "other": [{}, {}]}))


def test_single_model_handles_variable_channel_counts(make_chronos2_encoder, text_encoder):
    """One model instance handles 1-channel and 3-channel series without dataset= routing."""
    model = SystemOne(make_chronos2_encoder(), text_encoder, num_classes=2, hidden_dim=16)
    for n_channels in (1, 3):
        logits = model(torch.randn(2, n_channels, 8), ["is this rising?"])
        assert logits.shape == (2, 2)
        assert logits.isfinite().all()


def test_single_model_trains_on_mixed_channel_batches(make_chronos2_encoder, text_encoder):
    """Optimizer updates shared fusion/head params from batches with different channel counts."""
    model = SystemOne(make_chronos2_encoder(), text_encoder, num_classes=2, hidden_dim=16)
    optimizer = build_optimizer(model, Config(weight_decay=0.1))
    before_fusion = {
        name: p.detach().clone()
        for name, p in model.fusion.named_parameters()
    }
    for n_channels in (1, 3, 1, 3):
        batch = {
            "series": torch.randn(2, n_channels, 8),
            "questions": ["is this rising?"],
            "options": ["rising", "falling"],
            "target": torch.tensor([0, 1]),
            "head": "choice",
        }
        train_step(model, batch, optimizer)
    # Fusion should have been updated by mixed-channel batches
    for name, p in model.fusion.named_parameters():
        assert not torch.equal(p, before_fusion[name])


def test_validation_summary_is_macro_averaged(model, monkeypatch):
    batch = {"series": torch.randn(2, 1, 8), "target": torch.tensor([0, 1])}
    metrics = {
        "small": {"loss": 1.0, "accuracy": 1.0, "balanced_accuracy": 0.9},
        "large": {"loss": 3.0, "accuracy": 0.0, "balanced_accuracy": 0.1},
    }
    monkeypatch.setattr(
        "src.training.validation.validate",
        lambda model, batches: metrics["small" if len(batches) == 1 else "large"],
    )
    history = train(model, [batch], {"small": [batch], "large": [batch] * 3}, Config(epochs=1))
    assert history[0]["validation"] == metrics
    assert history[0]["validation_loss"] == 2.0
    assert history[0]["validation_accuracy"] == 0.5
    assert history[0]["validation_balanced_accuracy"] == 0.5
    with pytest.raises(ValueError, match="validation datasets"):
        validate_datasets(model, {})


def test_download_command_visits_all_datasets(tmp_path, monkeypatch):
    path = tmp_path / "all.yaml"
    path.write_text(yaml.safe_dump({"dataset": list(CHANNELS), "root": str(tmp_path)}))
    calls = []
    for name in CHANNELS:

        def download(root, *, split, download, dataset=name):
            calls.append((dataset, split, download))

        monkeypatch.setitem(DATASETS, name, download)
    with patch.object(
        SystemOne, "from_pretrained", side_effect=AssertionError("No model download")
    ):
        result = CliRunner().invoke(main, ["download", str(path)])
    assert result.exit_code == 0, result.output
    assert calls == [(name, split, True) for name in CHANNELS for split in ("train", "validation")]


@pytest.mark.parametrize("changes", [{"with_text": False}, {"num_classes": 8}])
def test_joint_configuration_is_checked_before_data_or_weights(changes, monkeypatch):
    config = Config(dataset=["har", "sleep"], **changes)
    monkeypatch.setitem(
        DATASETS, "har", lambda *args, **kwargs: pytest.fail("Data loaded too early")
    )
    with pytest.raises(ValueError, match="multiple datasets"):
        run(config)


def test_all_four_datasets_train_calibrate_and_reload_together(
    har_root,
    sleep_root,
    tsqa_root,
    ecg_root,
    tmp_path,
    make_chronos2_encoder,
    text_encoder,
    monkeypatch,
):
    assert har_root == sleep_root == tsqa_root == ecg_root
    config = Config(
        dataset=list(CHANNELS),
        root=str(har_root),
        epochs=2,
        batch_size=3,
        hidden_dim=16,
        output=str(tmp_path / "joint.pt"),
    )
    model = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16)
    steps, splits = [], []
    step = import_module("src.training.train").train_step

    def record_step(model, batch, optimizer):
        steps.append(batch["dataset"])
        assert batch["series"].shape[1] == CHANNELS[batch["dataset"]]
        return step(model, batch, optimizer)

    monkeypatch.setattr(import_module("src.training.train"), "train_step", record_step)
    for name, dataset_class in list(DATASETS.items()):

        def load(root, *, split, download, cls=dataset_class, dataset=name):
            splits.append((dataset, split))
            return cls(root, split=split, download=download)

        monkeypatch.setitem(DATASETS, name, load)

    with (
        patch.object(SystemOne, "from_pretrained", return_value=model) as factory,
        patch("src.training.run.fit_temperatures", wraps=fit_temperatures) as calibrate,
    ):
        path = run(config)
    # from_pretrained is called without channels kwarg
    assert "channels" not in factory.call_args.kwargs
    assert len(set(Counter(steps).values())) == 1
    assert set(steps) == set(CHANNELS)
    assert splits == [(name, split) for name in CHANNELS for split in ("train", "validation")]
    # calibrate is called once over all validation loaders combined
    assert calibrate.call_count == 1
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["config"] == asdict(config)
    assert "channels" not in checkpoint
    # temperatures is {head: float} not {dataset: {head: float}}
    assert all(isinstance(v, float) for v in checkpoint["temperatures"].values())
    assert all(0.01 <= t <= 100 for t in checkpoint["temperatures"].values())
    assert len(checkpoint["history"]) == config.epochs
    for row in checkpoint["history"]:
        assert set(row["validation"]) == set(CHANNELS)
    # Reload without channels
    restored = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16)
    restored.load_state_dict(checkpoint["model"])
    for name, count in CHANNELS.items():
        inputs = {
            "series": torch.randn(2, count, 8),
            "questions": ["rising"],
            "head": "noul",
        }
        torch.testing.assert_close(restored(**inputs), model(**inputs))
    # Validation/calibration loaders do not oversample the smaller datasets.
    for name in CHANNELS:
        dataset = DATASETS[name](str(har_root), split="validation", download=False)
        batches = BatchLoader(dataset, config.batch_size, dataset_name=name)
        assert sum(batch["target"].numel() for batch in batches) == len(dataset)
