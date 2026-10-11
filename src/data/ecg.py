import csv
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor

import torch
import wfdb

from .base import Sample, TimeSeriesDataset, download_csv_zip, download_file


class ECGDataset(TimeSeriesDataset):
    """OpenTSLM ECG-QA-CoT with PTB-XL 1.0.3's native 100 Hz recordings.

    Uses published CoT splits and template answer catalogs. Binary templates
    use Noul; other single-answer templates use Choice (including 'not sure').
    Multi-record / multi-answer rows are rejected, never reduced to one label.
    """

    name = "ecg"
    leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    url = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"
    answers_url = "https://raw.githubusercontent.com/Jwoo5/ecg-qa/master/ecgqa/ptbxl/answers_for_each_template.csv"
    records_url = "https://physionet.org/files/ptb-xl/1.0.3/"

    @property
    def filename(self):
        split = "val" if self.split == "validation" else self.split
        return f"ecg_qa_cot_{split}.csv"

    @staticmethod
    def record_id(value: str) -> int:
        ids = literal_eval(value)
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError("ECG-CoT requires one recording per question")
        return int(ids[0])

    def _metadata(self):
        path = self.root / "ptbxl" / "ptbxl_database.csv"
        with path.open(newline="") as file:
            return {
                int(r["ecg_id"]): (r["filename_lr"], r["patient_id"]) for r in csv.DictReader(file)
            }

    def download(self):
        download_csv_zip(self.url, self.root, self.filename)
        download_file(self.answers_url, self.root / "answers.csv")
        download_file(
            self.records_url + "ptbxl_database.csv", self.root / "ptbxl" / "ptbxl_database.csv"
        )
        records = self._metadata()
        rows = self._csv(self.filename, ["ecg_id"])
        names = sorted({records[self.record_id(x)][0] for x in rows["ecg_id"]})

        def fetch(name):
            for extension in (".hea", ".dat"):
                download_file(
                    self.records_url + name + extension, self.root / "ptbxl" / (name + extension)
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(fetch, names))

    def _load(self):
        rows = self._csv(self.filename, ["question", "answer", "ecg_id", "template_id"])
        self.records = self._metadata()
        with (self.root / "answers.csv").open(newline="") as file:
            self.answers = {
                int(r["template_id"]): tuple(literal_eval(r["classes"]))
                for r in csv.DictReader(file)
            }
        # Pre-load all waveforms into RAM so _sample never touches disk.
        needed = sorted({self.records[self.record_id(x)][0] for x in rows["ecg_id"]})

        def _read(name):
            signal, fields = wfdb.rdsamp(str(self.root / "ptbxl" / name))
            names = [lead.upper() for lead in fields["sig_name"]]
            if fields["fs"] != 100 or sorted(names) != sorted(
                lead.upper() for lead in self.leads
            ):
                raise ValueError("expected a 100 Hz, 12-lead PTB-XL recording")
            order = [names.index(lead.upper()) for lead in self.leads]
            return torch.tensor(signal[:, order].T, dtype=torch.float32)

        with ThreadPoolExecutor(max_workers=4) as pool:
            tensors = list(pool.map(_read, needed))
        self._signals: dict[str, torch.Tensor] = dict(zip(needed, tensors))
        return rows

    def _sample(self, row) -> Sample:
        options = self.answers[int(row["template_id"])]
        answer = row["answer"].strip()
        if answer not in options:
            raise ValueError(
                f"unsupported ECG answer for template {row['template_id']}: {answer!r}"
            )
        name, patient = self.records[self.record_id(row["ecg_id"])]
        binary = set(options) == {"no", "yes"}
        return {
            "series": self._signals[name],
            "question": row["question"],
            "options": () if binary else options,
            "target": float(answer == "yes") if binary else options.index(answer),
            "head": "noul" if binary else "choice",
            "group": patient,
        }
