from zipfile import ZipFile

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import HARDataset
from src.model import SystemOne


@pytest.mark.parametrize(
    "split,suffix", [("train", "train"), ("validation", "val"), ("test", "test")]
)
def test_har_splits_labels_and_channel_order(har_root, split, suffix):
    dataset = HARDataset(har_root, split)
    assert dataset.filename == f"har_cot_{suffix}_cot.csv"
    assert [s["target"] for s in dataset] == list(range(8))
    sample = dataset[0]
    torch.testing.assert_close(
        sample["series"], torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    )
    assert sample["options"] == dataset.labels
    assert "rationale" not in dataset.rows.column_names
    assert "rationale" not in sample


def test_har_download_and_repeated_headers(tmp_path, monkeypatch, write_csv, har_rows):
    header = {key: key for key in har_rows[0]}
    path = write_csv(tmp_path / "source.csv", [har_rows[0], header, har_rows[1]])
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as file:
        file.write(path, "nested/har_cot_train_cot.csv")
    monkeypatch.setattr(HARDataset, "url", archive.as_uri())
    dataset = HARDataset(tmp_path / "data", download=True)
    assert len(dataset) == 2
    assert [s["target"] for s in dataset] == [0, 1]


def test_har_unknown_labels_are_not_silently_remapped(tmp_path, write_csv, har_rows):
    write_csv(tmp_path / "har" / "har_cot_train_cot.csv", [dict(har_rows[0], label="unknown")])
    with pytest.raises(ValueError):
        HARDataset(tmp_path)[0]


def test_dataloader_to_fixed_catalog_model_backward(har_root, make_chronos2_encoder):
    dataset = HARDataset(har_root)
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=dataset.collate)))
    targets = batch.pop("target")
    batch.pop("questions")
    batch.pop("options")
    model = SystemOne(make_chronos2_encoder(), num_classes=8, hidden_dim=16)
    output = model(**batch)
    torch.nn.functional.cross_entropy(output, targets).backward()
    assert output.shape == (4, 8)
    assert model.classes is not None and model.classes.weight.grad is not None
    assert model.classes.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.series_encoder.backbone.parameters())
