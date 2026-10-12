from collections import Counter
from collections.abc import Iterator

import torch
from torch.utils.data import Dataset

from ..data import TimeSeriesDataset
from .features import FeatureCache, series_key


def _class_key(head, target, option_key) -> tuple:
    """True-class identity, matching validation's balanced-accuracy buckets."""
    if head == "noul":
        return ("noul", int(target >= 0.5))
    if head == "choice" and option_key is not None:
        return ("choice", option_key[int(target)])
    return (head, int(target))


def effective_number_weights(counts: dict, beta: float = 0.999) -> dict:
    """Class-balanced weights (Cui et al. 2019), normalized to sample-mean 1.

    Rare classes get more weight; common classes less. beta -> 1 sharpens the
    reweighting. Bounded, so extreme-tail classes cannot explode the loss.
    """
    weights = {c: (1.0 - beta) / (1.0 - beta**n) for c, n in counts.items()}
    total = sum(counts.values())
    scale = total / sum(n * weights[c] for c, n in counts.items())
    return {c: w * scale for c, w in weights.items()}


class CachedDataset(Dataset):
    """Serves precomputed encoder outputs in place of raw series and text.

    Row order stays positional (base.rows[i]), so DataLoader shuffling only
    permutes access, matching the raw pipeline.
    """

    def __init__(
        self,
        base: TimeSeriesDataset,
        cache: FeatureCache,
        *,
        weighted: bool = False,
        live_text: bool = False,
    ):
        self.cache = cache
        self.prefix = cache.prefix_start is not None
        self.live_text = live_text  # True: emit question/option strings for a LoRA MiniLM
        self.index: list[tuple] = []
        for i in range(len(base)):
            sample = base[i]
            option_key = tuple(sample["options"]) if sample["options"] else None
            self.index.append(
                (
                    series_key(sample["series"]),
                    sample["question"] or None,
                    option_key,
                    sample["target"],
                    sample["head"],
                )
            )
        self.weights: list[float] | None = None
        if weighted:
            counts = Counter(_class_key(r[4], r[3], r[2]) for r in self.index)
            table = effective_number_weights(counts)
            self.weights = [table[_class_key(r[4], r[3], r[2])] for r in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        key, question, option_key, target, head = self.index[i]
        sample: dict = {"option_key": option_key, "target": target, "head": head}
        sample["signal_prefix" if self.prefix else "signal"] = self.cache.series[key]
        if self.live_text:
            sample["question_str"] = question
        elif self.cache.questions:
            sample["question"] = self.cache.questions.get(question) if question else None
        if self.weights is not None:
            sample["weight"] = self.weights[i]
        return sample


def cached_collate(samples: list[dict], cache: FeatureCache) -> dict:
    """Pad cached tokens to batch max; masks make padding inert.

    Signal is either a final H_ts (pad tokens) or a frozen prefix [C, N, d]
    (pad patches). Text is either cached token states or live question/option
    strings (LoRA MiniLM), depending on what the dataset emitted.
    """
    head = samples[0]["head"]
    option_key = samples[0]["option_key"]
    batch = len(samples)
    out: dict = {"head": head}

    if "signal_prefix" in samples[0]:  # frozen prefix -> live LoRA suffix
        prefixes = [s["signal_prefix"] for s in samples]  # each [C, N, d]
        channels, dim = prefixes[0].shape[0], prefixes[0].shape[2]
        width = max(p.shape[1] for p in prefixes)
        signal_prefix = torch.zeros(batch, channels, width, dim, dtype=prefixes[0].dtype)
        prefix_valid = torch.zeros(batch, channels, width, dtype=torch.bool)
        for i, p in enumerate(prefixes):
            signal_prefix[i, :, : p.shape[1]] = p
            prefix_valid[i, :, : p.shape[1]] = True
        out["signal_prefix"] = signal_prefix
        out["prefix_valid"] = prefix_valid
    else:
        signals = [s["signal"] for s in samples]
        width = max(x.shape[0] for x in signals)
        dim = signals[0].shape[1]
        signal = torch.zeros(batch, width, dim, dtype=signals[0].dtype)
        signal_mask = torch.zeros(batch, width, dtype=torch.bool)
        for i, x in enumerate(signals):
            signal[i, : x.shape[0]] = x
            signal_mask[i, : x.shape[0]] = True
        out["signal"] = signal
        out["signal_mask"] = signal_mask

    if "question_str" in samples[0]:  # live MiniLM (LoRA): pass raw strings
        out["questions"] = [s.get("question_str") or "" for s in samples]
        out["options"] = list(option_key) if option_key else None
    else:
        questions = [s.get("question") for s in samples]
        if any(q is not None for q in questions):
            length = max(q.shape[0] for q in questions if q is not None)
            text_dim = next(q for q in questions if q is not None).shape[1]
            q_tokens = torch.zeros(batch, length, text_dim, dtype=torch.float16)
            q_mask = torch.zeros(batch, length, dtype=torch.bool)
            for i, q in enumerate(questions):
                if q is not None:
                    q_tokens[i, : q.shape[0]] = q
                    q_mask[i, : q.shape[0]] = True
                else:
                    q_mask[i, 0] = True  # keep the masked mean well-defined
            out["q_tokens"] = q_tokens
            out["q_mask"] = q_mask
        if option_key is not None and option_key in cache.options:
            out["option_embeddings"] = cache.options[option_key]

    out["option_key"] = option_key  # lets validation recover the true class label
    dtype = torch.float32 if head == "noul" else torch.long
    out["target"] = torch.tensor([s["target"] for s in samples], dtype=dtype)
    if "weight" in samples[0]:
        out["weight"] = torch.tensor([s["weight"] for s in samples], dtype=torch.float32)
    return out


class CachedBatchLoader:
    """Re-iterable cached batches, packed full within each (head, option set) group.

    Bucketing by (head, option_key) before chunking keeps every batch at the
    target size instead of fragmenting a shuffled chunk into many tiny batches.
    """

    def __init__(
        self,
        dataset: CachedDataset,
        cache: FeatureCache,
        batch_size: int,
        *,
        shuffle: bool = False,
        dataset_name: str | None = None,
    ):
        self.dataset = dataset
        self.cache = cache
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_name = dataset_name

    def _shuffle(self, items: list) -> list:
        return [items[i] for i in torch.randperm(len(items)).tolist()]

    def __iter__(self) -> Iterator[dict]:
        groups: dict[tuple, list[int]] = {}
        for index, row in enumerate(self.dataset.index):
            groups.setdefault((row[4], row[2]), []).append(index)  # (head, option_key)
        batches: list[list[int]] = []
        for indices in groups.values():
            if self.shuffle:
                indices = self._shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        if self.shuffle:
            batches = self._shuffle(batches)
        for indices in batches:
            batch = cached_collate([self.dataset[i] for i in indices], self.cache)
            if self.dataset_name is not None:
                batch["dataset"] = self.dataset_name
            yield batch
