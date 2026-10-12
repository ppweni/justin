from unittest.mock import patch
from zipfile import ZipFile

import pytest
import torch
import wfdb

from justin.data import ECGDataset
from justin.model import SystemOne


@pytest.mark.parametrize(
    "split,suffix", [("train", "train"), ("validation", "val"), ("test", "test")]
)
def test_ecg_native_splits_and_target_types(ecg_root, split, suffix):
    dataset = ECGDataset(ecg_root, split)
    assert dataset.filename == f"ecg_qa_cot_{suffix}.csv"
    assert dataset[0]["head"] == "noul" and dataset[0]["target"] == 1.0
    assert dataset[1]["head"] == "noul" and dataset[1]["target"] == 0.0
    assert dataset[0]["options"] == ()
    uncertain = dataset[2]
    assert uncertain["head"] == "choice"
    assert isinstance(uncertain["target"], int)
    assert uncertain["options"][uncertain["target"]] == "not sure"
    none = dataset[3]
    assert isinstance(none["target"], int)
    assert none["options"][none["target"]] == "none"


def test_ecg_preserves_units_and_restores_lead_order(ecg_root):
    dataset = ECGDataset(ecg_root)
    sample = dataset[0]
    assert sample["series"].shape == (12, 24)
    torch.testing.assert_close(
        sample["series"][:, 0], torch.arange(12, dtype=torch.float32), atol=1e-3, rtol=0
    )
    assert sample["group"] == "patient-1"
    assert sample["question"] == "Is this abnormal?"


def test_ecg_download_fetches_only_referenced_records(tmp_path, ecg_root, monkeypatch):
    source = ecg_root / "ecg"
    archive = tmp_path / "qa.zip"
    with ZipFile(archive, "w") as file:
        file.write(source / "ecg_qa_cot_train.csv", "nested/ecg_qa_cot_train.csv")
    monkeypatch.setattr(ECGDataset, "url", archive.as_uri())
    monkeypatch.setattr(ECGDataset, "answers_url", (source / "answers.csv").as_uri())
    monkeypatch.setattr(ECGDataset, "records_url", (source / "ptbxl").as_uri() + "/")
    dataset = ECGDataset(tmp_path / "downloaded", download=True)
    files = list((dataset.root / "ptbxl" / "records100").rglob("*.*"))
    assert {p.suffix for p in files} == {".hea", ".dat"} and len(files) == 2
    assert len(dataset) == 4
    torch.testing.assert_close(dataset[0]["series"], ECGDataset(ecg_root)[0]["series"])
    dataset.download()


@pytest.mark.parametrize("value", ["[]", "[1, 2]", "1", "'1'"])
def test_ecg_rejects_multi_record_or_non_list_ids(value):
    with pytest.raises(ValueError, match="one recording"):
        ECGDataset.record_id(value)


def test_ecg_single_record_id():
    assert ECGDataset.record_id("[13625]") == 13625


@pytest.mark.parametrize("answer", ["unknown diagnosis", "yes, no", "['yes', 'no']"])
def test_ecg_rejects_unrepresentable_answers(ecg_root, answer):
    dataset = ECGDataset(ecg_root)
    with pytest.raises(ValueError, match="unsupported ECG answer"):
        dataset._sample(dict(dataset.rows[0], answer=answer))


@pytest.mark.parametrize("change", [{"fs": 500}, {"sig_name": ["I"] * 12}])
def test_ecg_rejects_wrong_sampling_rate_or_leads(ecg_root, change):
    # Waveforms are validated when the dataset preloads them at construction.
    path = ecg_root / "ecg" / "ptbxl" / "records100" / "00000" / "00001_lr"
    signal, metadata = wfdb.rdsamp(str(path))
    with patch("justin.data.ecg.wfdb.rdsamp", return_value=(signal, {**metadata, **change})):
        with pytest.raises(ValueError, match="100 Hz, 12-lead"):
            ECGDataset(ecg_root)


def test_ecg_missing_waveform_fails_without_download(ecg_root):
    # Preloading reads waveforms from disk only; a missing file must fail loudly
    # rather than silently download (the offline fixture guards against network).
    (ecg_root / "ecg" / "ptbxl" / "records100" / "00000" / "00001_lr.dat").unlink()
    with pytest.raises(FileNotFoundError):
        ECGDataset(ecg_root)


def test_ecg_batch_to_binary_model_backward(ecg_root, make_chronos2_encoder, text_encoder):
    dataset = ECGDataset(ecg_root)
    batch = dataset.collate([dataset[0], dataset[1]])
    targets = batch.pop("target")
    model = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16)
    output = model(**batch)
    torch.nn.functional.binary_cross_entropy_with_logits(output, targets).backward()
    assert output.shape == (2,)
    assert model.noul.weight.grad is not None
    assert model.noul.weight.grad.abs().sum() > 0
