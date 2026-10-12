import random
import re
from ast import literal_eval

import torch

from .base import Sample, TimeSeriesDataset, download_file


class SleepDataset(TimeSeriesDataset):
    """OpenTSLM Sleep-CoT. Split by subject, not by neighboring EEG epochs.

    The release stores 1,500 samples per 30-second epoch (50 Hz).
    Merge N4 into N3. The release has six subjects: four train, one validation,
    one test (seed 42). These differ from OpenTSLM's random epoch splits.
    """

    name = "sleep"
    labels = ("Wake", "N1", "N2", "N3", "REM")
    label_ids = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 3, "REM": 4}
    url = "https://polybox.ethz.ch/index.php/s/ZryWSdCFJZ9JR3R/download"

    @staticmethod
    def subject(recording: str) -> str:
        # SC4ssNE0 / ST7ssNJ0: ss identifies the subject; N is the night.
        if not re.fullmatch(r"s[ct]\d{4}[a-z]\d", recording.lower()):
            raise ValueError(f"unrecognized Sleep-EDF recording ID: {recording}")
        return recording[:5].lower()

    def download(self):
        download_file(self.url, self.root / "sleep_cot.csv")

    def _load(self):
        rows = self._csv("sleep_cot.csv", ["id", "time_series", "label"])
        subjects = sorted({self.subject(x) for x in rows["id"]})
        if len(subjects) < 3:
            raise ValueError("Sleep needs at least three subjects for disjoint splits")
        random.Random(42).shuffle(subjects)
        n = max(1, len(subjects) // 10)
        selected = {
            "train": subjects[2 * n :],
            "validation": subjects[:n],
            "test": subjects[n : 2 * n],
        }[self.split]
        return rows.filter(lambda x: self.subject(x) in selected, input_columns=["id"])

    def _sample(self, row) -> Sample:
        return {
            "series": torch.tensor(literal_eval(row["time_series"]), dtype=torch.float32),
            "question": "Which sleep stage is shown by this 30-second EEG epoch?",
            "options": self.labels,
            "target": self.label_ids[row["label"]],
            "head": "choice",
            "group": self.subject(row["id"]),
        }
