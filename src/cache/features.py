import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from ..data import TimeSeriesDataset
from ..model import SystemOne

CACHE_VERSION = 2


def series_key(series: Tensor) -> str:
    """Stable content hash of a [C, T] series, keyed on shape and float32 bytes."""
    digest = hashlib.sha256()
    digest.update(repr(tuple(series.shape)).encode())
    digest.update(series.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _encoder_ids(model: SystemOne) -> dict[str, str]:
    ids = {"series": f"{model.series_encoder.name}-d{model.series_encoder.hidden_size}"}
    if model.text_encoder is not None:
        ids["text"] = f"d{model.text_encoder.hidden_size}"
    return ids


class FeatureCache:
    """Precomputed frozen-encoder outputs for one dataset split, held in RAM.

    series maps a series content hash to its series-encoder token states [M, d] (fp16;
    the token mask is all-valid for a natively-encoded sample). questions maps a
    question string to its MiniLM token states [L, d_txt] (fp16). options maps an
    options tuple to pooled MiniLM embeddings [n, d_txt] (fp16). The trainable
    fusion and heads consume these; no backbone runs during training.
    """

    def __init__(
        self,
        series: dict[str, Tensor],
        questions: dict[str, Tensor],
        options: dict[tuple[str, ...], Tensor],
        prefix_start: int | None = None,
    ):
        self.series = series
        self.questions = questions
        self.options = options
        # None: series holds final H_ts [M, d]. int: series holds a frozen prefix
        # activation [C, N, d] at that block boundary (top blocks run live with LoRA).
        self.prefix_start = prefix_start

    @classmethod
    def load_or_build(
        cls,
        dataset: TimeSeriesDataset,
        model: SystemOne,
        *,
        cache_dir: str | Path,
        name: str,
        split: str,
        batch_size: int = 64,
        prefix_start: int | None = None,
        cache_text: bool = True,
    ) -> "FeatureCache":
        """Load a matching on-disk cache, else build it once and persist it.

        prefix_start=None caches the final H_ts; an int caches the frozen prefix at
        that series-encoder block. cache_text=False skips text (MiniLM runs live under LoRA).
        """
        directory = Path(cache_dir)
        tag = "" if prefix_start is None else f"-p{prefix_start}"
        meta_path = directory / f"{name}-{split}{tag}-meta.json"
        want = {
            "version": CACHE_VERSION,
            "encoders": _encoder_ids(model),
            "size": len(dataset),
            "prefix_start": prefix_start,
            "cache_text": cache_text,
        }
        if meta_path.exists() and json.loads(meta_path.read_text()) == want:
            return cls._load(directory, name, split, tag, prefix_start)
        print(f"  caching {name}/{split}{tag} ({len(dataset)} samples)...", flush=True)
        cache = cls._build(
            dataset, model, batch_size=batch_size, prefix_start=prefix_start, cache_text=cache_text
        )
        cache._save(directory, name, split, tag, want)
        return cache

    @classmethod
    def _build(
        cls,
        dataset: TimeSeriesDataset,
        model: SystemOne,
        *,
        batch_size: int,
        prefix_start: int | None = None,
        cache_text: bool = True,
    ) -> "FeatureCache":
        device = next(model.parameters()).device
        want_text = cache_text and model.text_encoder is not None
        by_shape: dict[tuple[int, ...], dict[str, Tensor]] = {}
        question_set: dict[str, None] = {}
        option_set: dict[tuple[str, ...], None] = {}
        for index in tqdm(range(len(dataset)), desc="  scanning", unit="sample", leave=False):
            sample = dataset[index]
            key = series_key(sample["series"])
            by_shape.setdefault(tuple(sample["series"].shape), {}).setdefault(
                key, sample["series"]
            )
            if want_text:
                if sample["question"]:
                    question_set.setdefault(sample["question"], None)
                if sample["options"]:
                    option_set.setdefault(tuple(sample["options"]), None)

        series: dict[str, Tensor] = {}
        unique_series = sum(len(items) for items in by_shape.values())
        with tqdm(total=unique_series, desc=f"  {model.series_encoder.name}", unit="series", leave=False) as bar:
            for items in by_shape.values():
                keys = list(items)
                for start in range(0, len(keys), batch_size):
                    chunk = keys[start : start + batch_size]
                    stacked = torch.stack([items[k] for k in chunk]).to(device)
                    # Equal length within a shape group: no padding, mask is all-valid.
                    if prefix_start is None:
                        states, _ = model.series_encoder(stacked)  # [B, M, d]
                    else:
                        states, _ = model.series_encoder.encode_prefix(
                            stacked, None, prefix_start
                        )  # [B, C, N, d]
                    for offset, key in enumerate(chunk):
                        series[key] = states[offset].detach().to(torch.float16).cpu()
                    bar.update(len(chunk))

        questions: dict[str, Tensor] = {}
        options: dict[tuple[str, ...], Tensor] = {}
        if want_text:
            texts = list(question_set)
            for start in tqdm(
                range(0, len(texts), batch_size), desc="  questions", unit="batch", leave=False
            ):
                chunk = texts[start : start + batch_size]
                hidden, mask = model.text_encoder.token_states(chunk)
                for offset, text in enumerate(chunk):
                    valid = hidden[offset][mask[offset]]
                    questions[text] = valid.detach().to(torch.float16).cpu()
            for opts in tqdm(option_set, desc="  options", unit="set", leave=False):
                embedding = model.text_encoder(list(opts))
                options[opts] = embedding.detach().to(torch.float16).cpu()
        return cls(series, questions, options, prefix_start)

    def _save(self, directory: Path, name: str, split: str, tag: str, meta: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{name}-{split}{tag}"
        torch.save(self.series, directory / f"{stem}-series.pt")
        torch.save(self.questions, directory / f"{stem}-questions.pt")
        torch.save(self.options, directory / f"{stem}-options.pt")
        (directory / f"{stem}-meta.json").write_text(json.dumps(meta))

    @classmethod
    def _load(
        cls, directory: Path, name: str, split: str, tag: str, prefix_start: int | None
    ) -> "FeatureCache":
        stem = f"{name}-{split}{tag}"
        series = torch.load(directory / f"{stem}-series.pt", weights_only=False)
        questions = torch.load(directory / f"{stem}-questions.pt", weights_only=False)
        options = torch.load(directory / f"{stem}-options.pt", weights_only=False)
        return cls(series, questions, options, prefix_start)
