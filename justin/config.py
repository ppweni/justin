from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Config:
    """Dataset, model, and training arguments; load overrides from YAML."""

    root: str = "data"
    split: Literal["train", "validation", "test"] = "train"
    download: bool = False
    dataset: str | list[str] = "tsqa"

    num_classes: int | None = None
    score_levels: int = 5
    hidden_dim: int = 256
    fusion_layers: int = 1
    dropout: float = 0.0
    with_text: bool = True
    option_attention: bool = False
    signal_skip: bool = False
    series_encoder: str = "chronos"
    series_lora_layers: int = 0
    minilm_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    epochs: int = 3
    patience: int = 2
    learning_rate: float = 1e-3
    lora_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1
    device: str = "cpu"
    output: str = "model.pt"
    db: str | None = None
    cache: str | None = None
    class_weighted: list[str] = field(default_factory=list)
    seed: int | None = None  # seeds torch before the model is built, so runs can be repeated

    def __post_init__(self) -> None:
        if not isinstance(self.root, str):
            raise TypeError("root must be a string")
        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {self.split}")
        if not isinstance(self.dataset, (str, list)):
            raise ValueError("dataset must be a name or a nonempty list of unique names")
        if not self.datasets or any(
            name not in ("har", "sleep", "tsqa", "ecg") for name in self.datasets
        ):
            raise ValueError(f"unknown dataset: {self.dataset}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("dataset names must be unique")
        for name in ("device", "output"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("download", "with_text", "minilm_lora", "option_attention", "signal_skip"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if (self.option_attention or self.signal_skip) and not self.with_text:
            raise ValueError("option_attention and signal_skip require with_text: true")
        if self.series_encoder not in ("chronos",):
            raise ValueError(f"unknown series_encoder: {self.series_encoder}")
        # Upper bound depends on the encoder's block count; SystemOne enforces it.
        if type(self.series_lora_layers) is not int or self.series_lora_layers < 0:
            raise ValueError("series_lora_layers must be a nonnegative integer")
        for name, minimum in (
            ("hidden_dim", 1),
            ("score_levels", 2),
            ("fusion_layers", 1),
            ("lora_rank", 1),
            ("lora_alpha", 1),
            ("epochs", 1),
            ("patience", 1),
            ("batch_size", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.dropout) not in (int, float) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        if self.num_classes is not None:
            if type(self.num_classes) is not int or not 2 <= self.num_classes <= 255:
                raise ValueError("num_classes must be an integer between 2 and 255, or null")
        for name in ("learning_rate", "lora_learning_rate", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.learning_rate == 0 or self.lora_learning_rate == 0:
            raise ValueError("learning_rate and lora_learning_rate must be positive")
        if self.db is not None and (not isinstance(self.db, str) or not self.db.strip()):
            raise ValueError("db must be a nonempty string or null")
        if self.cache is not None and (not isinstance(self.cache, str) or not self.cache.strip()):
            raise ValueError("cache must be a nonempty string or null")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or null")
        if not isinstance(self.class_weighted, list) or any(
            name not in self.datasets for name in self.class_weighted
        ):
            raise ValueError("class_weighted must be a list of configured dataset names")

    @property
    def datasets(self) -> list[str]:
        """Normalize the single-dataset shorthand without changing saved YAML values."""
        return [self.dataset] if isinstance(self.dataset, str) else self.dataset

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open(encoding="utf-8") as file:
            values = yaml.safe_load(file)
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError("config YAML must contain a mapping")
        return cls(**values)
