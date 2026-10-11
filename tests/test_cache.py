import json

import pytest
import torch

from src.cache import (
    CachedBatchLoader,
    CachedDataset,
    FeatureCache,
    cached_collate,
    series_key,
)
from src.cache.dataset import _class_key, effective_number_weights
from src.config import Config
from src.data import ECGDataset, HARDataset
from src.model import SystemOne
from src.training.batch import model_logits, prepare_batch
from src.training.loss import compute_loss
from src.training.run import build_cache, run
from src.training.validation import validate


def _model(make_chronos2_encoder, text_encoder, **kwargs):
    return SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16, **kwargs)


def test_series_key_stable_and_shape_sensitive():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert series_key(a) == series_key(a.clone())
    changed = a.clone()
    changed[0, 0] += 1
    assert series_key(a) != series_key(changed)
    assert series_key(a) != series_key(a.reshape(3, 2))


def test_answer_cached_matches_forward_choice(make_chronos2_encoder, text_encoder):
    model = _model(make_chronos2_encoder, text_encoder, num_classes=3).eval()
    series = torch.randn(1, 2, 8)
    question, options = "is this rising ?", ["rising", "falling", "flat"]
    live = model(series, [question], options=options, head="choice")

    features = model.series_encoder(series)
    q_tokens, q_mask = model.text_encoder.token_states([question])
    option_embeddings = model.text_encoder(options)
    cached = model.answer_cached(features, q_tokens, q_mask, option_embeddings, head="choice")
    assert torch.allclose(live, cached, atol=1e-5)


def test_answer_cached_matches_forward_noul_and_catalog(make_chronos2_encoder, text_encoder):
    model = _model(make_chronos2_encoder, text_encoder, num_classes=4).eval()
    series = torch.randn(2, 1, 8)
    question = "is this rising ?"
    features = model.series_encoder(series)
    q_tokens, q_mask = model.text_encoder.token_states([question] * 2)

    for head in ("noul", "choice"):
        live = model(series, [question] * 2, head=head)
        cached = model.answer_cached(features, q_tokens, q_mask, head=head)
        assert torch.allclose(live, cached, atol=1e-5)


def test_cached_collate_pads_and_masks():
    cache = FeatureCache(series={}, questions={}, options={("a", "b"): torch.randn(2, 4)})
    samples = [
        {
            "signal": torch.randn(3, 5),
            "question": torch.randn(2, 4),
            "option_key": ("a", "b"),
            "target": 0,
            "head": "choice",
        },
        {
            "signal": torch.randn(5, 5),
            "question": torch.randn(4, 4),
            "option_key": ("a", "b"),
            "target": 1,
            "head": "choice",
        },
    ]
    batch = cached_collate(samples, cache)
    assert batch["signal"].shape == (2, 5, 5)
    assert batch["signal_mask"][0].tolist() == [True, True, True, False, False]
    assert batch["q_tokens"].shape == (2, 4, 4)
    assert batch["q_mask"][1].tolist() == [True, True, True, True]
    assert batch["option_embeddings"].shape == (2, 4)
    assert batch["target"].dtype == torch.long and batch["target"].tolist() == [0, 1]


def test_cached_collate_mixed_questions_keeps_mask_defined():
    cache = FeatureCache(series={}, questions={}, options={})
    samples = [
        {"signal": torch.randn(2, 4), "question": torch.randn(3, 4), "option_key": None, "target": 0, "head": "choice"},
        {"signal": torch.randn(2, 4), "question": None, "option_key": None, "target": 1, "head": "choice"},
    ]
    batch = cached_collate(samples, cache)
    assert batch["q_mask"][0].tolist() == [True, True, True]
    # The question-less row still exposes one valid slot so the masked mean is finite.
    assert batch["q_mask"][1].tolist() == [True, False, False]


def test_answer_cached_zero_query_matches_forward(make_chronos2_encoder, text_encoder):
    model = _model(make_chronos2_encoder, text_encoder, num_classes=3).eval()
    series = torch.randn(2, 1, 8)
    features = model.series_encoder(series)
    live = model(series, head="choice")
    cached = model.answer_cached(features, head="choice")
    assert torch.allclose(live, cached, atol=1e-5)


def test_answer_cached_rejects_bad_head(make_chronos2_encoder, text_encoder):
    model = _model(make_chronos2_encoder, text_encoder, num_classes=3)
    features = model.series_encoder(torch.randn(1, 1, 8))
    with pytest.raises(ValueError, match="unknown head"):
        model.answer_cached(features, head="mystery")


def test_answer_cached_rejects_options_for_non_choice(make_chronos2_encoder, text_encoder):
    model = _model(make_chronos2_encoder, text_encoder, num_classes=3)
    features = model.series_encoder(torch.randn(1, 1, 8))
    options = model.text_encoder(["yes", "no"])
    with pytest.raises(ValueError, match="only supported by the Choice head"):
        model.answer_cached(features, option_embeddings=options, head="noul")


def test_answer_cached_requires_text_encoder(make_chronos2_encoder):
    model = SystemOne(make_chronos2_encoder(), None, num_classes=3, hidden_dim=16)
    features = model.series_encoder(torch.randn(1, 1, 8))
    with pytest.raises(ValueError, match="require a text encoder"):
        model.answer_cached(features, q_tokens=torch.randn(1, 2, 8), head="choice")


def test_answer_cached_choice_requires_options_or_classes(make_chronos2_encoder, text_encoder):
    model = _model(make_chronos2_encoder, text_encoder).eval()  # num_classes=None
    features = model.series_encoder(torch.randn(1, 1, 8))
    q_tokens, q_mask = model.text_encoder.token_states(["is this rising ?"])
    with pytest.raises(ValueError, match="Choice requires options or num_classes"):
        model.answer_cached(features, q_tokens, q_mask, head="choice")


def test_answer_cached_no_text_pooling_matches_forward(make_chronos2_encoder):
    model = SystemOne(make_chronos2_encoder(), None, num_classes=3, hidden_dim=16).eval()
    series = torch.randn(2, 1, 8)
    features = model.series_encoder(series)
    live = model(series, head="choice")
    cached = model.answer_cached(features, head="choice")
    assert torch.allclose(live, cached, atol=1e-5)


class _FixedModel:
    """Minimal stand-in whose answer_cached returns preset logits, for metric tests."""

    training = False
    text_encoder = object()

    def __init__(self, logits):
        self._logits = logits

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def series_features(self, inputs):
        return inputs["signal"], inputs["signal_mask"]

    def answer_cached(self, features, q_tokens=None, q_mask=None, option_embeddings=None, *, head):
        return self._logits


def test_validate_balanced_accuracy_penalizes_majority_bias():
    # 4 of class "a", 1 of class "b"; the model always predicts class 0 ("a").
    logits = torch.tensor([[1.0, 0.0, 0.0]] * 5)
    batch = {
        "signal": torch.zeros(5, 1, 4),
        "signal_mask": torch.ones(5, 1, dtype=torch.bool),
        "option_key": ("a", "b", "c"),
        "head": "choice",
        "target": torch.tensor([0, 0, 0, 0, 1]),
    }
    result = validate(_FixedModel(logits), [batch])
    assert result["accuracy"] == pytest.approx(0.8)  # 4/5 correct
    # recall("a")=1.0, recall("b")=0.0, "c" absent -> balanced = 0.5
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_validate_noul_balanced_accuracy():
    # 3 negatives, 1 positive; model always predicts negative (logit < 0).
    logits = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    batch = {
        "signal": torch.zeros(4, 1, 4),
        "signal_mask": torch.ones(4, 1, dtype=torch.bool),
        "head": "noul",
        "target": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    result = validate(_FixedModel(logits), [batch])
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_cached_collate_includes_option_key():
    cache = FeatureCache(series={}, questions={}, options={("a", "b"): torch.randn(2, 4)})
    samples = [
        {"signal": torch.randn(2, 4), "question": None, "option_key": ("a", "b"), "target": 1, "head": "choice"}
    ]
    batch = cached_collate(samples, cache)
    assert batch["option_key"] == ("a", "b")


def test_cached_loader_packs_full_batches(har_root, make_chronos2_encoder, text_encoder):
    dataset = HARDataset(har_root)  # one head, one option set -> a single group
    model = _model(make_chronos2_encoder, text_encoder, num_classes=8).eval()
    cache = FeatureCache._build(dataset, model, batch_size=8)
    cached = CachedDataset(dataset, cache)
    assert len(cached) == len(dataset)
    loader = CachedBatchLoader(cached, cache, batch_size=3, dataset_name="har")
    sizes = []
    seen = 0
    for batch in loader:
        size = batch["target"].numel()
        sizes.append(size)
        seen += size
        assert batch["head"] == "choice" and batch["option_key"] is not None
    assert seen == len(dataset)  # every sample used exactly once
    # Full batches except one remainder: no fragmentation into tiny batches.
    assert sizes.count(3) == len(dataset) // 3
    assert sum(sizes) == len(dataset)


def test_class_key_covers_all_heads():
    assert _class_key("noul", 1.0, None) == ("noul", 1)
    assert _class_key("noul", 0.0, None) == ("noul", 0)
    assert _class_key("choice", 2, ("a", "b", "c")) == ("choice", "c")
    # Fixed catalog (no options) and score fall back to the integer target.
    assert _class_key("choice", 3, None) == ("choice", 3)
    assert _class_key("score", 4, None) == ("score", 4)


def test_effective_number_weights_favor_rare_classes():
    counts = {"common": 1000, "rare": 10}
    weights = effective_number_weights(counts)
    assert weights["rare"] > weights["common"]
    # Normalized to sample-mean 1.
    total = sum(counts.values())
    mean = sum(counts[c] * weights[c] for c in counts) / total
    assert mean == pytest.approx(1.0)


def test_cached_dataset_weights_minority_more(ecg_root, make_chronos2_encoder, text_encoder):
    dataset = ECGDataset(ecg_root)  # rows include a majority "no" and minority answers
    model = _model(make_chronos2_encoder, text_encoder)
    cache = FeatureCache._build(dataset, model, batch_size=8)
    plain = CachedDataset(dataset, cache)
    assert plain.weights is None
    weighted = CachedDataset(dataset, cache, weighted=True)
    assert weighted.weights is not None and len(weighted.weights) == len(dataset)
    assert all(w > 0 for w in weighted.weights)


def test_cached_collate_stacks_weights():
    cache = FeatureCache(series={}, questions={}, options={})
    samples = [
        {"signal": torch.randn(2, 4), "question": None, "option_key": None, "target": 0, "head": "noul", "weight": 2.0},
        {"signal": torch.randn(2, 4), "question": None, "option_key": None, "target": 1, "head": "noul", "weight": 0.5},
    ]
    batch = cached_collate(samples, cache)
    assert batch["weight"].tolist() == [2.0, 0.5]


def test_compute_loss_weights_change_result():
    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    target = torch.tensor([0, 0])  # second sample is wrong
    plain = compute_loss(logits, target, "choice")
    # Up-weighting the wrong sample raises the loss above the unweighted mean.
    weighted = compute_loss(logits, target, "choice", torch.tensor([1.0, 3.0]))
    assert weighted > plain
    # Equal weights reproduce the plain mean.
    equal = compute_loss(logits, target, "choice", torch.tensor([1.0, 1.0]))
    assert equal == pytest.approx(plain.item())


def test_cached_collate_noul_target_is_float():
    cache = FeatureCache(series={}, questions={}, options={})
    samples = [
        {"signal": torch.randn(2, 4), "question": None, "option_key": None, "target": 1.0, "head": "noul"}
    ]
    batch = cached_collate(samples, cache)
    assert batch["target"].dtype == torch.float32
    assert "option_embeddings" not in batch


def test_build_dedups_shared_series(ecg_root, make_chronos2_encoder, text_encoder):
    dataset = ECGDataset(ecg_root)
    model = _model(make_chronos2_encoder, text_encoder)
    cache = FeatureCache._build(dataset, model, batch_size=8)
    # All four rows reference recording [1]; questions are identical.
    assert len(cache.series) == 1
    assert len(cache.questions) == 1
    # Binary template contributes no options; two distinct choice catalogs remain.
    assert len(cache.options) == 2


def test_load_or_build_persists_and_rebuilds(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch
):
    dataset = HARDataset(har_root)
    model = _model(make_chronos2_encoder, text_encoder, num_classes=8)
    cache_dir = tmp_path / "cache"

    calls = {"n": 0}
    original = FeatureCache._build

    def spy(dataset, model, **kwargs):
        calls["n"] += 1
        return original(dataset, model, **kwargs)

    monkeypatch.setattr(FeatureCache, "_build", staticmethod(spy))

    FeatureCache.load_or_build(dataset, model, cache_dir=cache_dir, name="har", split="train")
    FeatureCache.load_or_build(dataset, model, cache_dir=cache_dir, name="har", split="train")
    assert calls["n"] == 1  # second call reloads from disk
    assert (cache_dir / "har-train-series.pt").exists()

    (cache_dir / "har-train-meta.json").write_text(json.dumps({"version": 999}))
    FeatureCache.load_or_build(dataset, model, cache_dir=cache_dir, name="har", split="train")
    assert calls["n"] == 2  # stale meta triggers a rebuild


def test_cached_loader_feeds_training(har_root, make_chronos2_encoder, text_encoder):
    dataset = HARDataset(har_root)
    model = _model(make_chronos2_encoder, text_encoder, num_classes=8).eval()
    cache = FeatureCache._build(dataset, model, batch_size=8)
    loader = CachedBatchLoader(
        CachedDataset(dataset, cache), cache, batch_size=8, dataset_name="har"
    )
    batch = next(iter(loader))
    assert batch["dataset"] == "har"
    inputs, target = prepare_batch(model, batch)
    logits = model_logits(model, inputs)
    assert logits.shape[0] == target.numel()
    assert torch.isfinite(logits).all()


def test_run_with_cache_persists_and_reloads(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch
):
    model = _model(make_chronos2_encoder, text_encoder)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    cache_dir = tmp_path / "cache"

    calls = {"n": 0}
    original = FeatureCache._build

    def spy(dataset, model, **kwargs):
        calls["n"] += 1
        return original(dataset, model, **kwargs)

    monkeypatch.setattr(FeatureCache, "_build", staticmethod(spy))

    def config(output):
        return Config(
            dataset="har",
            root=str(har_root),
            epochs=1,
            patience=1,
            hidden_dim=16,
            batch_size=8,
            cache=str(cache_dir),
            output=str(output),
        )

    path = run(config(tmp_path / "first.pt"))
    assert path.exists()
    assert (cache_dir / "har-train-series.pt").exists()
    checkpoint = torch.load(path, weights_only=True)
    assert "temperatures" in checkpoint and "history" in checkpoint
    built_after_first = calls["n"]

    run(config(tmp_path / "second.pt"))
    assert calls["n"] == built_after_first  # reused the persisted cache, no rebuild


def test_build_cache_writes_all_splits(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch
):
    model = _model(make_chronos2_encoder, text_encoder)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    cache_dir = tmp_path / "features"
    config = Config(
        dataset="har", root=str(har_root), hidden_dim=16, cache=str(cache_dir), output=str(tmp_path / "m.pt")
    )
    build_cache(config)
    for split in ("train", "validation"):
        assert (cache_dir / f"har-{split}-series.pt").exists()
        assert (cache_dir / f"har-{split}-meta.json").exists()


def test_build_cache_requires_cache_config(har_root):
    config = Config(dataset="har", root=str(har_root), hidden_dim=16)
    with pytest.raises(ValueError, match="requires config.cache"):
        build_cache(config)


def test_build_cache_rejects_empty_split(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch
):
    from src.data import DATASETS

    model = _model(make_chronos2_encoder, text_encoder)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    dataset_class = DATASETS["har"]

    def load_dataset(root, *, split, download):
        dataset = dataset_class(root, split=split)
        if split == "validation":
            dataset.rows = dataset.rows.select([])
        return dataset

    monkeypatch.setitem(DATASETS, "har", load_dataset)
    config = Config(
        dataset="har", root=str(har_root), hidden_dim=16, cache=str(tmp_path / "features")
    )
    with pytest.raises(ValueError, match="must be nonempty"):
        build_cache(config)


def test_run_with_class_weighting(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch
):
    model = _model(make_chronos2_encoder, text_encoder)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    config = Config(
        dataset="har",
        root=str(har_root),
        epochs=1,
        patience=1,
        hidden_dim=16,
        batch_size=8,
        cache=str(tmp_path / "cache"),
        class_weighted=["har"],
        output=str(tmp_path / "weighted.pt"),
    )
    checkpoint = torch.load(run(config), weights_only=True)
    assert checkpoint["config"]["class_weighted"] == ["har"]
    assert "validation_balanced_accuracy" in checkpoint["history"][-1]


@pytest.mark.parametrize(
    "series_lora_layers,minilm_lora",
    [(1, False), (0, True), (1, True)],  # A: series-only, B: MiniLM-only, C: both
)
def test_run_with_lora_conditions(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch,
    series_lora_layers, minilm_lora,
):
    model = SystemOne(
        make_chronos2_encoder(), text_encoder, hidden_dim=16,
        series_lora_layers=series_lora_layers, minilm_lora=minilm_lora,
        lora_rank=4, lora_alpha=8,
    )
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    config = Config(
        dataset="har",
        root=str(har_root),
        epochs=1,
        patience=1,
        hidden_dim=16,
        batch_size=8,
        cache=str(tmp_path / "cache"),
        series_lora_layers=series_lora_layers,
        minilm_lora=minilm_lora,
        lora_rank=4,
        lora_alpha=8,
        output=str(tmp_path / f"lora_{series_lora_layers}_{minilm_lora}.pt"),
    )
    checkpoint = torch.load(run(config), weights_only=True)
    assert "per_dataset_best" in checkpoint
    assert "har" in checkpoint["per_dataset_best"]


def test_run_cache_matches_uncached_shapes(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch
):
    model = _model(make_chronos2_encoder, text_encoder)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    config = Config(
        dataset="har",
        root=str(har_root),
        epochs=1,
        patience=1,
        hidden_dim=16,
        batch_size=8,
        cache=str(tmp_path / "cache"),
        output=str(tmp_path / "cached.pt"),
    )
    checkpoint = torch.load(run(config), weights_only=True)
    assert checkpoint["config"]["cache"] == str(tmp_path / "cache")
    assert set(checkpoint["temperatures"]) <= {"choice", "noul", "score"}
    # Per-dataset best-epoch diagnostic is embedded in the checkpoint.
    assert "har" in checkpoint["per_dataset_best"]
    assert set(checkpoint["per_dataset_best"]["har"]) == {
        "balanced", "balanced_epoch", "micro", "micro_epoch"
    }


@pytest.mark.parametrize("series_lora_layers", [0, 1])
def test_run_with_chronos2_encoder(
    har_root, make_chronos2_encoder, text_encoder, tmp_path, monkeypatch, series_lora_layers
):
    model = SystemOne(
        make_chronos2_encoder(), text_encoder, hidden_dim=16,
        series_lora_layers=series_lora_layers, lora_rank=4, lora_alpha=8,
    )
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    config = Config(
        dataset="har",
        root=str(har_root),
        epochs=1,
        patience=1,
        hidden_dim=16,
        batch_size=8,
        cache=str(tmp_path / "cache"),
        series_encoder="chronos",
        series_lora_layers=series_lora_layers,
        lora_rank=4,
        lora_alpha=8,
        output=str(tmp_path / f"chronos_{series_lora_layers}.pt"),
    )
    checkpoint = torch.load(run(config), weights_only=True)
    assert "per_dataset_best" in checkpoint
    meta = [p.read_text() for p in (tmp_path / "cache").rglob("*.json")]
    assert meta and all("chronos" in text for text in meta)
