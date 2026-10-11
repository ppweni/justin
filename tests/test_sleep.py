import pytest
import torch

from src.data import SleepDataset


def test_sleep_subjects_and_nights_never_cross_splits(sleep_root):
    datasets = [SleepDataset(sleep_root, split) for split in ("train", "validation", "test")]
    groups = [{s["group"] for s in dataset} for dataset in datasets]
    assert [len(g) for g in groups] == [4, 1, 1]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sum(map(len, datasets)) == 72
    for dataset, expected in zip(datasets, groups, strict=True):
        assert {s["group"] for s in SleepDataset(sleep_root, dataset.split)} == expected


def test_sleep_labels_and_original_scale(sleep_root):
    dataset = SleepDataset(sleep_root)
    assert dataset.labels == ("Wake", "N1", "N2", "N3", "REM")
    samples = list(dataset)
    assert {s["target"] for s in samples} == set(range(5))
    n4 = next(i for i, row in enumerate(dataset.rows) if row["label"] == "N4")
    assert dataset[n4]["target"] == 3
    torch.testing.assert_close(samples[0]["series"], torch.tensor([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize(
    "recording,subject", [("SC4102E0", "sc410"), ("sc4101e0", "sc410"), ("st7022j0", "st702")]
)
def test_sleep_subject_parsing(recording, subject):
    assert SleepDataset.subject(recording) == subject


@pytest.mark.parametrize("recording", ["", "patient-1", "sc4102e0-extra", "xx4102e0"])
def test_invalid_subject_ids(recording):
    with pytest.raises(ValueError, match="recording ID"):
        SleepDataset.subject(recording)


def test_sleep_requires_three_subjects(tmp_path, write_csv, sleep_rows):
    write_csv(tmp_path / "sleep" / "sleep_cot.csv", sleep_rows[:12])
    with pytest.raises(ValueError, match="three subjects"):
        SleepDataset(tmp_path)


def test_sleep_download(tmp_path, monkeypatch, write_csv, sleep_rows):
    source = write_csv(tmp_path / "source.csv", sleep_rows)
    monkeypatch.setattr(SleepDataset, "url", source.as_uri())
    dataset = SleepDataset(tmp_path / "data", download=True)
    assert len(dataset) == 48
    source.unlink()
    dataset.download()


def test_subject_splits_do_not_depend_on_row_order(tmp_path, write_csv, sleep_rows):
    for name, rows in (("original", sleep_rows), ("reversed", sleep_rows[::-1])):
        write_csv(tmp_path / name / "sleep" / "sleep_cot.csv", rows)
    for split in ("train", "validation", "test"):
        a = SleepDataset(tmp_path / "original", split)
        b = SleepDataset(tmp_path / "reversed", split)
        assert {s["group"] for s in a} == {s["group"] for s in b}
