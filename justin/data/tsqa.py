import hashlib
import json
import re
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk

from .base import Sample, TimeSeriesDataset, group_split


class TSQADataset(TimeSeriesDataset):
    """ChengsenWang/TSQA. Split by signal identity; preserve supplied MCQ options.

    The published dataset has one source split and three-way questions.
    """

    name = "tsqa"

    def download(self):
        path = self.root / "source"
        if not path.is_dir():
            self.root.mkdir(parents=True, exist_ok=True)
            rows = load_dataset(
                "ChengsenWang/TSQA", split="train", cache_dir=str(self.root / ".cache")
            )
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                staging = Path(temporary) / "source"
                rows.save_to_disk(str(staging))
                staging.replace(path)

    def _load(self):
        path = self.root / "source"
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {path}; construct with download=True")
        rows = load_from_disk(str(path))
        return rows.filter(lambda x: group_split(x) == self.split, input_columns=["Series"])

    def _sample(self, row) -> Sample:
        text = row["Question"]
        matches = list(re.finditer(r"^\(([a-z])\)\s*(.+)$", text, re.MULTILINE))
        if len(matches) < 2:
            raise ValueError("TSQA question has no recognizable option list")
        letters = [m[1] for m in matches]
        options = tuple(m[2].strip() for m in matches)
        target = letters.index(row["Answer"].strip().lower().strip("()"))
        return {
            "series": torch.tensor([json.loads(row["Series"])], dtype=torch.float32),
            "question": text[: matches[0].start()].strip(),
            "options": options,
            "target": target,
            "head": "choice",
            "group": hashlib.sha256(row["Series"].encode()).hexdigest(),
        }
