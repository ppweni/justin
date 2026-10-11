from collections.abc import Iterator

from torch.utils.data import DataLoader

from ..data import Sample, TimeSeriesDataset


class BatchLoader:
    """Re-iterable batches, grouped by compatible heads, options, and channels.

    batch_size is an upper bound: each chunk is split into compatible groups.
    Every sample is used once per pass, without preloading the whole dataset.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        dataset_name: str | None = None,
    ):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=list)
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[dict]:
        for samples in self.loader:
            groups: dict[tuple, list[Sample]] = {}
            for sample in samples:
                key = (sample["head"], sample["options"], sample["series"].shape[0])
                groups.setdefault(key, []).append(sample)
            for group in groups.values():
                batch = TimeSeriesDataset.collate(group)
                if self.dataset_name is not None:
                    batch["dataset"] = self.dataset_name
                yield batch
