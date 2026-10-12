from collections import Counter
from io import BytesIO
from typing import cast
from unittest.mock import patch
from zipfile import BadZipFile, ZipFile

import pytest
import torch

from justin.data import HARDataset, Sample, TimeSeriesDataset
from justin.data.base import download_csv_zip, download_file, group_split


def test_base_class_is_abstract(tmp_path):
    with pytest.raises(TypeError, match="abstract"):
        TimeSeriesDataset(tmp_path)  # ty: ignore[call-non-callable] -- tests the runtime guard


def test_indexing_and_iteration(har_root):
    dataset = HARDataset(har_root)
    assert len(dataset) == 8
    assert dataset[-1]["target"] == 7
    assert [s["target"] for s in dataset] == list(range(8))
    for index in (8, -9):
        with pytest.raises(IndexError):
            dataset[index]


def test_split_must_exist(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        HARDataset(tmp_path, split="invalid")  # ty: ignore[invalid-argument-type] -- invalid input


def test_missing_data_explains_download_option(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=True"):
        HARDataset(tmp_path)


@pytest.mark.parametrize(
    "series",
    [
        torch.ones(3),
        torch.ones(1, 1, 3),
        torch.zeros(0, 3),
        torch.zeros(1, 0),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
    ],
)
def test_invalid_series_rejected(har_root, series):
    dataset = HARDataset(har_root)
    sample = dict(dataset[0], series=series)
    with patch.object(dataset, "_sample", return_value=sample):
        with pytest.raises(ValueError, match="invalid.*series"):
            dataset[0]


def test_collate_preserves_order_and_left_pads(har_root):
    dataset = HARDataset(har_root)
    a, b = dataset[0], dataset[1]
    b["series"] = b["series"][:, :1]
    batch = dataset.collate([a, b])
    assert batch["series"].shape == (2, 3, 3)
    torch.testing.assert_close(batch["series"][0], a["series"])
    torch.testing.assert_close(batch["series"][1, :, -1:], b["series"])
    assert batch["series"][1, :, :2].eq(0).all()
    assert batch["padding_mask"][1, :, :2].all()
    assert not batch["padding_mask"][0].any()
    assert not batch["padding_mask"][1, :, -1:].any()
    assert batch["questions"] == [a["question"], b["question"]]
    assert batch["options"] == list(dataset.labels)
    assert batch["target"].tolist() == [0, 1]
    assert batch["target"].dtype == torch.long
    assert batch["series"].dtype == torch.float32


def test_binary_collate_uses_float_targets_and_no_options(har_root):
    sample = HARDataset(har_root)[0]
    sample.update(head="noul", options=(), target=1.0)
    batch = TimeSeriesDataset.collate([sample])
    assert batch["options"] is None
    assert batch["target"].dtype == torch.float32
    assert batch["target"].tolist() == [1.0]


@pytest.mark.parametrize(
    "field,value", [("head", "noul"), ("options", ("different",)), ("series", torch.ones(2, 3))]
)
def test_incompatible_batch_rejected(har_root, field, value):
    sample = HARDataset(har_root)[0]
    other = cast(Sample, {**sample, field: value})
    with pytest.raises(ValueError, match="share head, options, and channel count"):
        TimeSeriesDataset.collate([sample, other])


def test_empty_batch_rejected():
    with pytest.raises(ValueError, match="empty batch"):
        TimeSeriesDataset.collate([])


def test_group_split_is_stable_and_approximately_80_10_10():
    groups = [f"subject-{i}" for i in range(1000)]
    assignments = {group: group_split(group) for group in groups}
    assert assignments == {group: group_split(group) for group in reversed(groups)}
    counts = Counter(assignments.values())
    assert 750 < counts["train"] < 850
    assert 60 < counts["validation"] < 140
    assert 60 < counts["test"] < 140


def test_download_creates_parents_and_reuses_complete_file(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")
    path = tmp_path / "cache" / "nested" / "download.csv"
    download_file(source.as_uri(), path)
    assert path.read_bytes() == source.read_bytes()
    source.unlink()
    download_file(source.as_uri(), path)
    assert list(path.parent.iterdir()) == [path]


class Response(BytesIO):
    def __init__(self, content, length=None):
        super().__init__(content)
        self.headers = {} if length is None else {"Content-Length": str(length)}


@pytest.mark.parametrize("length", [None, 4])
def test_download_handles_responses_with_and_without_length(tmp_path, length):
    with patch("justin.data.base.urlopen", return_value=Response(b"data", length)):
        download_file("https://example.test/data", tmp_path / "data")
    assert (tmp_path / "data").read_bytes() == b"data"


def test_truncated_download_is_not_cached_and_can_retry(tmp_path):
    path = tmp_path / "data"
    with patch("justin.data.base.urlopen", return_value=Response(b"part", 100)):
        with pytest.raises(OSError, match="incomplete"):
            download_file("https://example.test/data", path)
    assert list(tmp_path.iterdir()) == []
    with patch("justin.data.base.urlopen", return_value=Response(b"complete", 8)):
        download_file("https://example.test/data", path)
    assert path.read_bytes() == b"complete"


def test_failed_download_cleans_up(tmp_path):
    with patch("justin.data.base.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(TimeoutError):
            download_file("https://example.test/data", tmp_path / "data")
    assert list(tmp_path.iterdir()) == []


def test_zip_extracts_only_requested_csv_and_reuses_cache(tmp_path):
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as file:
        file.writestr("nested/train.csv", "value\n1\n")
        file.writestr("__MACOSX/train.csv", "junk")
        file.writestr("../outside.txt", "must not be extracted")
    root = tmp_path / "cache"
    download_csv_zip(archive.as_uri(), root, "train.csv")
    assert (root / "train.csv").read_text() == "value\n1\n"
    assert not (tmp_path / "outside.txt").exists()
    archive.unlink()
    download_csv_zip(archive.as_uri(), root, "train.csv")


@pytest.mark.parametrize("names", [[], ["a/train.csv", "b/train.csv"]])
def test_zip_requires_exactly_one_matching_csv(tmp_path, names):
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as file:
        for name in names:
            file.writestr(name, "value\n1\n")
    with pytest.raises(ValueError, match="expected one"):
        download_csv_zip(archive.as_uri(), tmp_path / "cache", "train.csv")
    assert not (tmp_path / "cache" / "train.csv").exists()


def test_invalid_zip_is_rejected(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_text("not a ZIP")
    with pytest.raises(BadZipFile):
        download_csv_zip(archive.as_uri(), tmp_path / "cache", "train.csv")


def test_failed_extraction_leaves_no_partial_csv(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    with ZipFile(root / "source.zip", "w") as file:
        file.writestr("train.csv", "value\n1\n")
    with patch("justin.data.base.shutil.copyfileobj", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            download_csv_zip("unused", root, "train.csv")
    assert {p.name for p in root.iterdir()} == {"source.zip"}
