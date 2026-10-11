from dataclasses import asdict

import pytest
import yaml

from src.config import Config


def test_defaults():
    assert asdict(Config()) == dict(
        root="data",
        split="train",
        download=False,
        dataset="tsqa",
        num_classes=None,
        score_levels=5,
        hidden_dim=256,
        fusion_layers=1,
        dropout=0.0,
        with_text=True,
        option_attention=False,
        signal_skip=False,
        series_encoder="chronos",
        series_lora_layers=0,
        minilm_lora=False,
        lora_rank=8,
        lora_alpha=16,
        epochs=3,
        patience=2,
        learning_rate=1e-3,
        lora_learning_rate=1e-4,
        weight_decay=0.01,
        batch_size=1,
        device="cpu",
        output="model.pt",
        db=None,
        cache=None,
        class_weighted=[],
    )


def test_yaml_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    values = dict(
        root="signals",
        split="validation",
        download=True,
        dataset="har",
        num_classes=8,
        score_levels=4,
        hidden_dim=32,
        fusion_layers=3,
        dropout=0.2,
        with_text=False,
        option_attention=False,
        signal_skip=False,
        series_encoder="chronos",
        series_lora_layers=4,
        minilm_lora=False,
        lora_rank=16,
        lora_alpha=32,
        epochs=2,
        patience=4,
        learning_rate=0.005,
        lora_learning_rate=0.0005,
        weight_decay=0,
        batch_size=4,
        device="cuda",
        output="runs/har.pt",
        db=None,
        cache=None,
        class_weighted=["har"],
    )
    path.write_text(yaml.safe_dump(values))
    assert asdict(Config.from_yaml(str(path))) == values


@pytest.mark.parametrize("contents", ["", "# Defaults\n", "{}", "null"])
def test_empty_yaml_uses_defaults(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    assert Config.from_yaml(path) == Config()


@pytest.mark.parametrize("contents", ["[]", "42", "false", "hello"])
def test_yaml_requires_mapping(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError, match="mapping"):
        Config.from_yaml(path)


def test_unknown_fields_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("unknown_setting: 0.001")
    with pytest.raises(TypeError, match="unknown_setting"):
        Config.from_yaml(path)


@pytest.mark.parametrize(
    "contents", ["dataset: [", "!!python/object/apply:builtins.str ['unsafe']"]
)
def test_invalid_or_executable_yaml_rejected(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(yaml.YAMLError):
        Config.from_yaml(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "values",
    [
        {"hidden_dim": -1},
        {"score_levels": 1},
        {"num_classes": 1},
        {"num_classes": 256},
        {"num_classes": 3.5},
        {"with_text": "false"},
        {"download": 1},
        {"root": 123},
        {"split": "invalid"},
        {"epochs": 0},
        {"epochs": True},
        {"patience": 0},
        {"patience": -1},
        {"patience": True},
        {"patience": "2"},
        {"learning_rate": 0},
        {"learning_rate": "0.01"},
        {"learning_rate": float("nan")},
        {"weight_decay": -0.1},
        {"weight_decay": float("inf")},
        {"weight_decay": False},
        {"dataset": "unknown"},
        {"dataset": []},
        {"dataset": ["har", "har"]},
        {"dataset": ["har", "unknown"]},
        {"dataset": ["har", []]},
        {"dataset": 42},
        {"batch_size": 0},
        {"batch_size": True},
        {"device": ""},
        {"device": 1},
        {"output": " "},
        {"output": False},
        {"cache": " "},
        {"cache": 5},
        {"db": " "},
        {"db": 5},
        {"class_weighted": "yes"},
        {"class_weighted": 1},
        {"class_weighted": ["unknown"]},
        {"class_weighted": ["sleep"]},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"dropout": "0.1"},
        {"fusion_layers": 0},
        {"fusion_layers": 1.5},
        {"series_encoder": "bogus"},
        {"series_lora_layers": -1},
        {"series_lora_layers": True},
        {"minilm_lora": "yes"},
        {"option_attention": "yes"},
        {"signal_skip": 1},
        {"signal_skip": True, "with_text": False},
        {"option_attention": True, "with_text": False},
        {"lora_rank": 0},
        {"lora_alpha": 0},
        {"lora_learning_rate": 0},
        {"lora_learning_rate": -1},
    ],
)
def test_invalid_config_values(tmp_path, values):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises((TypeError, ValueError)):
        Config.from_yaml(path)


def test_multiple_datasets_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: [har, sleep, tsqa, ecg]\n")
    config = Config.from_yaml(path)
    assert config.datasets == ["har", "sleep", "tsqa", "ecg"]
    assert Config(dataset="har").datasets == ["har"]
