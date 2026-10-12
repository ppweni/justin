import runpy
import sys
from dataclasses import asdict

import pytest
import torch
import yaml
from click.testing import CliRunner

from src import Config
from src.__main__ import _load_config, main
from src.data import DATASETS
from src.model import SystemOne
from src.training.run import run


@pytest.fixture
def run_config(har_root, tmp_path):
    return Config(
        dataset="har",
        root=str(har_root),
        num_classes=8,
        with_text=False,
        hidden_dim=16,
        epochs=1,
        batch_size=3,
        output=str(tmp_path / "runs" / "model.pt"),
    )


def test_package_exports_config():
    from src.config import Config as ConfigClass

    assert Config is ConfigClass


def test_module_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["src", "--help"])
    monkeypatch.delitem(sys.modules, "src.__main__")
    with pytest.raises(SystemExit) as error:
        runpy.run_module("src", run_name="__main__")
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "download" in output and "train" in output and "cache" in output


@pytest.mark.parametrize("command", ["train", "download", "cache"])
def test_config_path_required(command):
    result = CliRunner().invoke(main, [command])
    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_main_passes_yaml_config(run_config, tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(run_config)))
    calls = []

    def fake_run(config):
        calls.append(config)
        return config.output

    monkeypatch.setattr("src.training.run.run", fake_run)
    result = CliRunner().invoke(main, ["train", str(path)])
    assert result.exit_code == 0, result.output
    assert calls == [run_config]
    assert f"Saved checkpoint: {run_config.output}" in result.output


def test_cache_command_passes_yaml_config(tmp_path, monkeypatch, har_root):
    config = Config(
        dataset="har",
        root=str(har_root),
        hidden_dim=16,
        cache=str(tmp_path / "features"),
        output=str(tmp_path / "m.pt"),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    calls = []

    monkeypatch.setattr("src.training.run.build_cache", calls.append)
    result = CliRunner().invoke(main, ["cache", str(path)])
    assert result.exit_code == 0, result.output
    assert calls == [config]
    assert f"Cache ready: {config.cache}" in result.output


def test_cache_command_requires_cache_setting(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dataset": "har", "root": str(tmp_path)}))
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: pytest.fail("no weights"))
    result = CliRunner().invoke(main, ["cache", str(path)])
    assert result.exit_code == 1
    assert "set 'cache'" in result.output


def test_cache_command_builds_without_training(
    tmp_path, monkeypatch, make_chronos2_encoder, text_encoder, har_root
):
    model = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    monkeypatch.setattr("src.training.run.run", lambda *a, **k: pytest.fail("cache must not train"))
    cache_dir = tmp_path / "features"
    config = Config(
        dataset="har",
        root=str(har_root),
        hidden_dim=16,
        cache=str(cache_dir),
        output=str(tmp_path / "m.pt"),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    result = CliRunner().invoke(main, ["cache", str(path)])
    assert result.exit_code == 0, result.output
    assert (cache_dir / "har-train-series.pt").exists()
    assert (cache_dir / "har-validation-series.pt").exists()
    assert not (tmp_path / "m.pt").exists()


@pytest.mark.parametrize("dataset", ["har", "sleep", "tsqa", "ecg"])
def test_download_command_uses_dataset_only(tmp_path, monkeypatch, dataset):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dataset": dataset, "root": str(tmp_path), "download": False}))
    calls = []

    def download_dataset(root, *, split, download):
        calls.append((root, split, download))

    def unexpected_training(*args, **kwargs):
        pytest.fail("Download must not train or load model weights")

    monkeypatch.setitem(DATASETS, dataset, download_dataset)
    monkeypatch.setattr(SystemOne, "from_pretrained", unexpected_training)
    monkeypatch.setattr("src.training.run.run", unexpected_training)
    result = CliRunner().invoke(main, ["download", str(path)])
    assert result.exit_code == 0, result.output
    assert calls == [(str(tmp_path), "train", True), (str(tmp_path), "validation", True)]
    assert f"Cached {dataset} validation" in result.output


def test_download_reuses_cached_data(run_config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(run_config)))
    for _ in range(2):
        result = CliRunner().invoke(main, ["download", str(path)])
        assert result.exit_code == 0, result.output
    assert not (tmp_path / "runs" / "model.pt").exists()


@pytest.mark.parametrize("command", ["download", "train", "cache"])
@pytest.mark.parametrize("content", ["channels: [", "unknown_field: true", "epochs: 0"])
def test_invalid_cli_config_is_readable(tmp_path, command, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    result = CliRunner().invoke(main, [command, str(path)])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_missing_config_is_rejected(tmp_path):
    result = CliRunner().invoke(main, ["download", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_config_read_error_becomes_click_error(tmp_path):
    import click

    with pytest.raises(click.ClickException):
        _load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "dataset,root_fixture,num_classes,with_text",
    [
        ("har", "har_root", 8, False),
        ("sleep", "sleep_root", 5, False),
        ("tsqa", "tsqa_root", None, True),
        ("ecg", "ecg_root", None, True),
    ],
)
def test_training_run_saves_complete_checkpoint(
    request,
    tmp_path,
    monkeypatch,
    make_chronos2_encoder,
    text_encoder,
    dataset,
    root_fixture,
    num_classes,
    with_text,
):
    config = Config(
        dataset=dataset,
        root=str(request.getfixturevalue(root_fixture)),
        num_classes=num_classes,
        with_text=with_text,
        hidden_dim=16,
        epochs=2,
        batch_size=4,
        output=str(tmp_path / "runs" / f"{dataset}.pt"),
    )
    model = SystemOne(
        make_chronos2_encoder(),
        text_encoder if with_text else None,
        num_classes=num_classes,
        hidden_dim=16,
    )
    factory_calls = []

    def load_model(**kwargs):
        factory_calls.append(kwargs)
        return model

    monkeypatch.setattr(SystemOne, "from_pretrained", load_model)
    splits = []
    dataset_class = DATASETS[dataset]

    def load_dataset(root, *, split, download):
        splits.append((split, download))
        return dataset_class(root, split=split, download=download)

    monkeypatch.setitem(DATASETS, dataset, load_dataset)
    path = run(config)
    checkpoint = torch.load(path, weights_only=True)
    assert splits == [("train", False), ("validation", False)]
    assert factory_calls == [
        {
            "num_classes": num_classes,
            "with_text": with_text,
            "option_attention": config.option_attention,
            "signal_skip": config.signal_skip,
            "score_levels": config.score_levels,
            "hidden_dim": 16,
            "fusion_layers": config.fusion_layers,
            "dropout": config.dropout,
            "series_encoder": config.series_encoder,
            "revisions": None,  # the encoder classes' pinned revisions
            "series_lora_layers": config.series_lora_layers,
            "minilm_lora": config.minilm_lora,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
        }
    ]
    assert checkpoint["config"] == asdict(config)
    assert len(checkpoint["history"]) == 2
    assert "channels" not in checkpoint
    # temperatures is {head: float}, not per-dataset
    assert all(isinstance(v, float) for v in checkpoint["temperatures"].values())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"split": "test"}, "split: train"),
        ({"dataset": "tsqa"}, "with_text: true"),
        ({"dataset": "ecg"}, "with_text: true"),
        ({"num_classes": 3}, "num_classes must be 8"),
    ],
)
def test_invalid_run_config_fails_before_loading_weights(run_config, monkeypatch, change, message):
    def unexpected_load(**kwargs):
        pytest.fail("Must validate configuration before loading model weights")

    monkeypatch.setattr(SystemOne, "from_pretrained", unexpected_load)
    config = Config(**(asdict(run_config) | change))
    with pytest.raises(ValueError, match=message):
        run(config)


def test_existing_checkpoint_is_preserved(run_config, tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"existing checkpoint")
    run_config.output = str(path)
    with pytest.raises(FileExistsError):
        run(run_config)
    assert path.read_bytes() == b"existing checkpoint"


@pytest.mark.parametrize("empty_split", ["train", "validation"])
def test_empty_dataset_split(run_config, monkeypatch, empty_split):
    dataset_class = DATASETS["har"]

    def load_dataset(root, *, split, download):
        dataset = dataset_class(root, split=split)
        if split == empty_split:
            dataset.rows = dataset.rows.select([])
        return dataset

    monkeypatch.setitem(DATASETS, "har", load_dataset)
    with pytest.raises(ValueError, match="both be nonempty"):
        run(run_config)
