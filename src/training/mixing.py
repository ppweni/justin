from collections.abc import Iterable, Iterator, Mapping

import torch


class BalancedBatches:
    """Equal optimizer steps per dataset; cycle small loaders until all finish once.

    Each round draws one compatible batch per dataset, in random dataset order.
    Restarting a shuffled loader reshuffles its data. No batches are cached.
    Loaders must be nonempty and re-iterable. Validation must not use this mixer.
    """

    def __init__(self, loaders: Mapping[str, Iterable[dict]]):
        if not loaders:
            raise ValueError("at least one dataset loader is required")
        self.loaders = dict(loaders)

    def __iter__(self) -> Iterator[dict]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        seen: set[str] = set()
        exhausted: set[str] = set()
        while True:
            batches = {}
            for name, iterator in iterators.items():
                batch = next(iterator, None)
                if batch is None:
                    if name not in seen:
                        raise ValueError(f"dataset loader must not be empty: {name}")
                    exhausted.add(name)
                else:
                    seen.add(name)
                    batches[name] = batch
            if len(exhausted) == len(iterators):
                return
            for name in iterators:
                if name not in batches:
                    iterators[name] = iter(self.loaders[name])
                    batch = next(iterators[name], None)
                    if batch is None:
                        raise ValueError(f"dataset loader must be nonempty and re-iterable: {name}")
                    batches[name] = batch
            names = list(batches)
            for index in torch.randperm(len(names)).tolist():
                yield batches[names[index]]
