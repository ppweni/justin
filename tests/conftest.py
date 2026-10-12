import csv
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import wfdb
from datasets import Dataset as ArrowDataset
from datasets import disable_progress_bars
from transformers import BertConfig, BertModel, BertTokenizer

from justin.data import ECGDataset, HARDataset, SleepDataset
from justin.model import Chronos2Encoder, SystemOne, TextEncoder
from justin.model.chronos2 import Chronos2


@pytest.fixture(autouse=True)
def deterministic_offline_tests(monkeypatch):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    disable_progress_bars()
    connect = socket.socket.connect

    def offline_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"Tests must not access the network: {address}")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", offline_connect)


def _tiny_chronos2_backbone():
    return Chronos2(
        d_model=16,
        d_ff=24,
        d_kv=8,
        num_heads=2,
        num_layers=2,
        patch_size=4,
        context_length=64,
        time_encoding_scale=64,
    )


@pytest.fixture
def make_chronos2_encoder():
    def make():
        return Chronos2Encoder(_tiny_chronos2_backbone())

    return make


@pytest.fixture
def chronos2_encoder(make_chronos2_encoder):
    return make_chronos2_encoder()


@pytest.fixture
def text_encoder(tmp_path):
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "is",
        "this",
        "rising",
        "falling",
        "flat",
        "yes",
        "no",
        "?",
    ]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tokenizer = BertTokenizer(vocab=str(path))
    backbone = BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    )
    return TextEncoder(backbone, tokenizer)


@pytest.fixture
def model(make_chronos2_encoder, text_encoder):
    return SystemOne(
        make_chronos2_encoder(), text_encoder, num_classes=3, score_levels=4, hidden_dim=16
    )


@pytest.fixture
def write_csv():
    def write(path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    return write


@pytest.fixture
def har_rows():
    return [
        dict(
            x_axis="[1, 2, 3]",
            y_axis="[4, 5, 6]",
            z_axis="[7, 8, 9]",
            label=label,
            rationale="Never pass this answer to the model.",
        )
        for label in HARDataset.labels
    ]


@pytest.fixture
def har_root(tmp_path, write_csv, har_rows):
    for split in ("train", "val", "test"):
        write_csv(tmp_path / "har" / f"har_cot_{split}_cot.csv", har_rows)
    return tmp_path


@pytest.fixture
def sleep_rows():
    return [
        dict(id=f"sc4{subject:02d}{night}e0", time_series="[[1, 2, 3]]", label=label)
        for subject in range(6)
        for night in (1, 2)
        for label in SleepDataset.label_ids
    ]


@pytest.fixture
def sleep_root(tmp_path, write_csv, sleep_rows):
    write_csv(tmp_path / "sleep" / "sleep_cot.csv", sleep_rows)
    return tmp_path


@pytest.fixture
def tsqa_rows():
    return [
        dict(
            Series=json.dumps([float(i), 0.0, 1.0]),
            Question="Classify the trend.\n(a) rising\n(b) falling\n(c) flat\nOnly answer (a), (b), or (c).",
            Answer="(b)",
            Label="falling",
            Task="Trend",
        )
        for i in range(30)
    ]


@pytest.fixture
def tsqa_root(tmp_path, tsqa_rows):
    ArrowDataset.from_list(tsqa_rows).save_to_disk(tmp_path / "tsqa" / "source")
    return tmp_path


@pytest.fixture
def ecg_rows():
    return [
        dict(question="Is this abnormal?", answer=answer, ecg_id="[1]", template_id=template)
        for answer, template in (("yes", 2), ("no", 2), ("not sure", 1), ("none", 3))
    ]


@pytest.fixture
def ecg_root(tmp_path, write_csv, ecg_rows):
    root = tmp_path / "ecg"
    for split in ("train", "val", "test"):
        write_csv(root / f"ecg_qa_cot_{split}.csv", ecg_rows)
    write_csv(
        root / "answers.csv",
        [
            {"template_id": 1, "classes": "['no', 'not sure', 'yes']"},
            {"template_id": 2, "classes": "['no', 'yes']"},
            {"template_id": 3, "classes": "['normal', 'abnormal', 'none']"},
        ],
    )
    write_csv(
        root / "ptbxl" / "ptbxl_database.csv",
        [
            {"ecg_id": 1, "filename_lr": "records100/00000/00001_lr", "patient_id": "patient-1"},
        ],
    )
    directory = root / "ptbxl" / "records100" / "00000"
    directory.mkdir(parents=True)
    # Reverse the stored leads and uppercase AVR/AVL/AVF, as PTB-XL does.
    signal = np.tile(np.arange(12, dtype=float), (24, 1))
    signal += np.arange(24)[:, None] / 100
    wfdb.wrsamp(
        "00001_lr",
        fs=100,
        units=["mV"] * 12,
        sig_name=[lead.upper() for lead in reversed(ECGDataset.leads)],
        p_signal=signal[:, ::-1],
        fmt=["16"] * 12,
        write_dir=str(directory),
    )
    return tmp_path
