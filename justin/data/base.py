import hashlib
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TypedDict
from urllib.request import urlopen
from zipfile import ZipFile

import torch
from datasets import Dataset as ArrowDataset
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

Split = Literal["train", "validation", "test"]


class Sample(TypedDict):
    series: Tensor  # float32 [channels, time], original scale
    question: str
    options: tuple[str, ...]  # empty for Noul
    target: int | float  # class index for Choice, 0/1 for Noul
    head: Literal["choice", "noul"]
    group: str  # subject/signal identity when available


class TimeSeriesDataset(Dataset[Sample], ABC):
    """Map-style dataset. Downloads are explicit; iteration never uses the network."""

    name: str

    def __init__(
        self, root: str | Path = "data", split: Split = "train", *, download: bool = False
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split: {split}")
        self.root = Path(root) / self.name
        self.split = split
        if download:
            self.download()
        self.rows = self._load()

    @abstractmethod
    def download(self) -> None:
        """Download missing source files for this split into root."""

    @abstractmethod
    def _load(self) -> ArrowDataset: ...

    @abstractmethod
    def _sample(self, row: dict) -> Sample: ...

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        sample = self._sample(self.rows[index])
        series = torch.as_tensor(sample["series"], dtype=torch.float32)
        if series.ndim != 2 or min(series.shape) == 0 or not series.isfinite().all():
            raise ValueError(f"invalid [channels, time] series at index {index}")
        sample["series"] = series
        return sample

    def __iter__(self) -> Iterator[Sample]:
        return (self[i] for i in range(len(self)))

    def _csv(self, filename: str, columns: list[str]) -> ArrowDataset:
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; construct with download=True")
        return load_dataset(
            "csv",
            data_files=str(path),
            split="train",
            usecols=columns,
            keep_default_na=False,
            cache_dir=str(self.root / ".cache"),
        )

    @staticmethod
    def collate(samples: list[Sample]) -> dict:
        """Left-pad a batch with one head, option catalog, and channel count.

        Pop `target` from the result, then pass the remaining kwargs to SystemOne.
        For fixed catalogs without MiniLM, also omit `questions` and `options`.
        """
        if not samples:
            raise ValueError("cannot collate an empty batch")
        first = samples[0]
        channels = first["series"].shape[0]
        if any(
            (s["head"], s["options"], s["series"].shape[0])
            != (first["head"], first["options"], channels)
            for s in samples
        ):
            raise ValueError("batch items must share head, options, and channel count")
        length = max(s["series"].shape[-1] for s in samples)
        series = torch.zeros(len(samples), channels, length)
        mask = torch.ones_like(series, dtype=torch.bool)
        for i, sample in enumerate(samples):
            n = sample["series"].shape[-1]
            series[i, :, -n:] = sample["series"]
            mask[i, :, -n:] = False
        return {
            "series": series,
            "padding_mask": mask,
            "questions": [s["question"] for s in samples],
            "options": list(first["options"]) or None,
            "head": first["head"],
            "target": torch.tensor(
                [s["target"] for s in samples],
                dtype=torch.float32 if first["head"] == "noul" else torch.long,
            ),
        }


def group_split(key: str) -> Split:
    """Stable approximate 80/10/10 split; identical groups never cross splits."""
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def download_file(url: str, path: Path) -> None:
    """Cache a complete download atomically; interrupted files are never reused."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
                if expected is not None and output.tell() != int(expected):
                    raise OSError(f"incomplete download: {url}")
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def download_csv_zip(url: str, root: Path, filename: str) -> None:
    """Extract just the requested CSV, regardless of the archive's folder prefix."""
    path = root / filename
    if path.is_file():
        return
    archive = root / "source.zip"
    download_file(url, archive)
    with ZipFile(archive) as source:
        matches = [
            n
            for n in source.namelist()
            if Path(n).name == filename and not n.startswith("__MACOSX/")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {filename} in {archive}")
        with (
            source.open(matches[0]) as input_file,
            tempfile.NamedTemporaryFile(dir=root, delete=False) as output,
        ):
            temporary = Path(output.name)
            try:
                shutil.copyfileobj(input_file, output)
                output.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
