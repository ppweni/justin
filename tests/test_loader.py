from collections import Counter

import torch
from torch.utils.data import RandomSampler

from justin.data import ECGDataset, HARDataset
from justin.training.loader import BatchLoader


def test_fixed_catalog_batches_include_final_partial_batch(har_root):
    dataset = HARDataset(har_root)
    loader = BatchLoader(dataset, batch_size=3)
    for _ in range(2):
        batches = list(loader)
        assert [batch["target"].numel() for batch in batches] == [3, 3, 2]
        torch.testing.assert_close(torch.cat([b["target"] for b in batches]), torch.arange(8))


def test_runtime_catalogs_and_heads_remain_separate(ecg_root):
    dataset = ECGDataset(ecg_root)
    batches = list(BatchLoader(dataset, batch_size=4))
    assert [batch["target"].numel() for batch in batches] == [2, 1, 1]
    actual = Counter(
        (batch["head"], tuple(batch["options"] or ()), float(target))
        for batch in batches
        for target in batch["target"]
    )
    expected = Counter(
        (sample["head"], sample["options"], float(sample["target"])) for sample in dataset
    )
    assert actual == expected


def test_shuffled_loader_visits_every_sample(har_root):
    loader = BatchLoader(HARDataset(har_root), batch_size=3, shuffle=True)
    assert isinstance(loader.loader.sampler, RandomSampler)
    for _ in range(2):
        labels = torch.cat([batch["target"] for batch in loader])
        torch.testing.assert_close(labels.sort().values, torch.arange(8))
