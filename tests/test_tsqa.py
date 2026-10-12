from unittest.mock import patch

import pytest
import torch
from datasets import Dataset as ArrowDataset

from justin.data import TSQADataset


def test_tsqa_choices_are_separate_from_question(tsqa_root):
    dataset = TSQADataset(tsqa_root)
    sample = dataset[0]
    assert sample["question"] == "Classify the trend."
    assert sample["options"] == ("rising", "falling", "flat")
    assert sample["target"] == 1 and sample["head"] == "choice"
    assert sample["series"].shape == (1, 3)
    assert sample["series"].dtype == torch.float32


def test_same_signal_different_questions_stay_in_one_split(tmp_path, tsqa_rows):
    duplicates = [
        dict(row, Question=row["Question"].replace("trend", "signal")) for row in tsqa_rows
    ]
    ArrowDataset.from_list(tsqa_rows + duplicates).save_to_disk(tmp_path / "tsqa" / "source")
    datasets = [TSQADataset(tmp_path, split) for split in ("train", "validation", "test")]
    groups = [{s["group"] for s in dataset} for dataset in datasets]
    assert all(groups)
    assert sum(map(len, datasets)) == 60
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    for dataset, group in zip(datasets, groups, strict=True):
        assert len(dataset) == 2 * len(group)


def test_tsqa_download_is_cached(tmp_path, tsqa_rows):
    with patch(
        "justin.data.tsqa.load_dataset", return_value=ArrowDataset.from_list(tsqa_rows)
    ) as load:
        dataset = TSQADataset(tmp_path, download=True)
        dataset.download()
        reloaded = TSQADataset(tmp_path, download=True)
    assert load.call_count == 1
    assert load.call_args.args == ("ChengsenWang/TSQA",)
    assert load.call_args.kwargs["split"] == "train"
    assert len(reloaded) == len(dataset) > 0


def test_failed_tsqa_save_does_not_poison_cache(tmp_path):
    with patch("justin.data.tsqa.load_dataset") as load:
        load.return_value.save_to_disk.side_effect = OSError("disk full")
        with pytest.raises(OSError, match="disk full"):
            TSQADataset(tmp_path, download=True)
    assert not (tmp_path / "tsqa" / "source").exists()
    assert list((tmp_path / "tsqa").iterdir()) == []


def test_tsqa_missing_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=True"):
        TSQADataset(tmp_path)


@pytest.mark.parametrize("update", [{"Question": "No options"}, {"Answer": "(z)"}])
def test_tsqa_malformed_question_or_answer(tsqa_root, update):
    dataset = TSQADataset(tsqa_root)
    with pytest.raises(ValueError):
        dataset._sample({**dataset.rows[0], **update})


def test_tsqa_answer_letter_case_and_option_order(tsqa_root):
    dataset = TSQADataset(tsqa_root)
    row = dict(
        dataset.rows[0],
        Answer=" (B) ",
        Question="Trend?\n(c) flat\n(a) rising\n(b) falling\nOnly answer.",
    )
    sample = dataset._sample(row)
    assert sample["options"] == ("flat", "rising", "falling")
    assert sample["target"] == 2
