from ast import literal_eval

import torch

from .base import Sample, TimeSeriesDataset, download_csv_zip


class HARDataset(TimeSeriesDataset):
    """OpenTSLM HAR-CoT: original splits, XYZ accelerometer, eight activities.

    Source: OpenTSLM's har_cot_loader.py. The CSVs contain no subject IDs.
    """

    name = "har"
    labels = (
        "biking",
        "lying",
        "running",
        "sitting",
        "standing",
        "walking",
        "walking_down",
        "walking_up",
    )
    url = "https://polybox.ethz.ch/index.php/s/kD74GnMYxn3HBEM/download"

    @property
    def filename(self):
        split = "val" if self.split == "validation" else self.split
        return f"har_cot_{split}_cot.csv"

    def download(self):
        download_csv_zip(self.url, self.root, self.filename)

    def _load(self):
        rows = self._csv(self.filename, ["x_axis", "y_axis", "z_axis", "label"])
        # The release contains repeated CSV header rows.
        return rows.filter(lambda x: x != "x_axis", input_columns=["x_axis"])

    def _sample(self, row) -> Sample:
        return {
            "series": torch.tensor(
                [literal_eval(row[f"{axis}_axis"]) for axis in "xyz"], dtype=torch.float32
            ),
            "question": "Which activity is shown by this 2.56-second XYZ accelerometer window?",
            "options": self.labels,
            "target": self.labels.index(row["label"]),
            "head": "choice",
            "group": "",
        }
