import copy
import math
from importlib import import_module

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.config import Config
from src.data import HARDataset
from src.model import SystemOne
from src.training import train, validate
from src.training.batch import prepare_batch
from src.training.loss import compute_loss, negative_log_likelihood
from src.training.optimizer import build_optimizer
from src.training.step import train_step
from src.training.train import per_dataset_best, train_epoch


@pytest.fixture
def batch():
    return {
        "series": torch.randn(3, 1, 8),
        "padding_mask": torch.zeros(3, 1, 8, dtype=torch.bool),
        "questions": ["is this rising?"] * 3,
        "options": None,
        "head": "choice",
        "target": torch.tensor([0, 1, 2]),
    }


def test_prepare_batch_preserves_inputs(model, batch):
    original = batch.copy()
    inputs, target = prepare_batch(model, batch)
    assert "target" not in inputs
    assert torch.equal(target, original["target"])
    assert inputs["series"].device == next(model.parameters()).device
    assert batch.keys() == original.keys()
    assert all(batch[key] is value for key, value in original.items())
    assert inputs["questions"] == original["questions"]


def test_prepare_fixed_catalog_without_text(make_chronos2_encoder, batch):
    model = SystemOne(make_chronos2_encoder(), num_classes=3)
    batch["options"] = ["rising", "falling", "flat"]
    inputs, _ = prepare_batch(model, batch)
    assert "questions" not in inputs and "options" not in inputs
    assert batch["options"] == ["rising", "falling", "flat"]
    assert model(**inputs).shape == (3, 3)
    minimal = {key: value for key, value in batch.items() if key not in ("questions", "options")}
    assert prepare_batch(model, minimal)[0].keys() == inputs.keys()


@pytest.mark.parametrize(
    "target", [[0], torch.tensor([]), torch.tensor([[0]]), torch.tensor([float("nan")])]
)
def test_prepare_batch_rejects_invalid_targets(model, batch, target):
    batch["target"] = target
    with pytest.raises(ValueError, match="target"):
        prepare_batch(model, batch)


@pytest.mark.parametrize("head", ["choice", "score", "noul"])
def test_loss_value_and_gradient(head):
    logits = torch.zeros((2,) if head == "noul" else (2, 3), requires_grad=True)
    target = torch.tensor([0, 1])
    loss = negative_log_likelihood(logits, target, head)
    assert loss.item() == pytest.approx(math.log(2 if head == "noul" else 3))
    loss.backward()
    assert logits.grad is not None
    if head == "noul":
        torch.testing.assert_close(logits.grad, torch.tensor([0.25, -0.25]))
    else:
        torch.testing.assert_close(
            logits.grad, (torch.full((2, 3), 1 / 3) - F.one_hot(target, 3)) / 2
        )


@pytest.mark.parametrize("target", [-0.1, 1.1])
def test_binary_target_range(target):
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_loss(torch.zeros(1), torch.tensor([target]), "noul")


def test_unknown_loss_head():
    with pytest.raises(ValueError, match="unknown head"):
        compute_loss(torch.zeros(1), torch.zeros(1), "invalid")  # ty: ignore[invalid-argument-type]


def test_nonfinite_loss():
    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_loss(torch.tensor([float("nan")]), torch.zeros(1), "noul")


def test_optimizer_contains_only_trainable_parameters(model):
    config = Config(learning_rate=0.02, weight_decay=0.03)
    optimizer = build_optimizer(model, config)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.02
    assert group["weight_decay"] == 0.03
    assert {id(p) for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }


@pytest.mark.parametrize("head", ["choice", "score", "noul"])
def test_step_updates_heads_and_preserves_encoders(model, batch, head):
    batch["head"] = head
    if head == "noul":
        batch["target"] = torch.tensor([0.0, 1.0, 1.0])
    if head == "choice":
        batch["options"] = ["rising", "falling", "flat"]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = build_optimizer(model, Config())
    model.eval()
    inputs, target = prepare_batch(model, batch)
    expected_loss = compute_loss(model(**inputs), target, head).item()
    # Stale gradients must be cleared, including those of unused heads.
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, float("nan"))
    assert train_step(model, batch, optimizer) == pytest.approx(expected_loss)
    assert model.training
    assert model.series_encoder.training
    assert not model.series_encoder.backbone.training
    assert model.text_encoder is not None and not model.text_encoder.training
    from src.model.fusion import CrossAttentionFusion
    assert isinstance(model.fusion, CrossAttentionFusion)
    assert not torch.equal(
        before["fusion.query_projection.weight"], model.fusion.query_projection.weight
    )
    for name, p in model.named_parameters():
        assert p.isfinite().all()
        if not p.requires_grad:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert p.grad is None
    assert "target" in batch


def test_failed_step_does_not_update_weights(model, batch):
    before = copy.deepcopy(model.state_dict())
    batch["head"] = "noul"
    batch["target"] = torch.tensor([0.0, 1.0, 2.0])
    optimizer = build_optimizer(model, Config())
    with pytest.raises(ValueError):
        train_step(model, batch, optimizer)
    assert not optimizer.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("head", ["choice", "score", "noul"])
@pytest.mark.parametrize("training", [True, False])
def test_validation_weights_samples_and_preserves_model(model, batch, head, training):
    batch["head"] = head
    if head == "noul":
        batch["target"] = torch.tensor([0.0, 1.0, 1.0])
    model.train(training)
    before = copy.deepcopy(model.state_dict())
    inputs, target = prepare_batch(model, batch)
    with torch.no_grad():
        logits = model(**inputs)
        expected_loss = compute_loss(logits, target, head).item()
        expected_accuracy = (
            (
                ((logits >= 0) == (target >= 0.5))
                if head == "noul"
                else (logits.argmax(-1) == target)
            )
            .float()
            .mean()
            .item()
        )
    # An uneven final batch must not count as much as the first batch.
    batches = [
        {
            key: value[part] if isinstance(value, (torch.Tensor, list)) else value
            for key, value in batch.items()
        }
        for part in (slice(0, 2), slice(2, 3))
    ]
    result = validate(model, batches)
    assert result["loss"] == pytest.approx(expected_loss)
    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert model.training == training
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_validation_restores_mode_on_error(model, batch):
    model.train()
    batch["target"] = torch.tensor([])
    with pytest.raises(ValueError):
        validate(model, [batch])
    assert model.training


def test_empty_loaders_fail(model):
    with pytest.raises(ValueError, match="validation batches"):
        validate(model, [])
    with pytest.raises(ValueError, match="training batches"):
        train_epoch(model, [], build_optimizer(model, Config()))


def test_epoch_loss_weights_samples(model, monkeypatch):
    losses = iter([1.0, 4.0])
    monkeypatch.setattr(
        import_module("src.training.train"), "train_step", lambda *args: next(losses)
    )
    result = train_epoch(
        model,
        [{"target": torch.zeros(3)}, {"target": torch.zeros(1)}],
        build_optimizer(model, Config()),
    )
    assert result[0] == 1.75


def test_epoch_returns_batch_count(model, monkeypatch):
    monkeypatch.setattr(import_module("src.training.train"), "train_step", lambda *args: 0.0)
    batches = [{"target": torch.zeros(1)} for _ in range(7)]
    _, batch_count = train_epoch(model, batches, build_optimizer(model, Config()))
    assert batch_count == 7


def test_epoch_uses_total_for_tqdm(model, monkeypatch):
    captured = {}
    real_tqdm = import_module("src.training.train").tqdm

    def spy_tqdm(it, **kwargs):
        captured.update(kwargs)
        return real_tqdm(it, **kwargs)

    monkeypatch.setattr(import_module("src.training.train"), "train_step", lambda *args: 0.0)
    monkeypatch.setattr(import_module("src.training.train"), "tqdm", spy_tqdm)
    train_epoch(
        model,
        [{"target": torch.zeros(1)}],
        build_optimizer(model, Config()),
        epoch=3,
        total=42,
    )
    assert captured["total"] == 42
    assert captured["desc"] == "epoch 3"


def test_per_dataset_best_picks_each_datasets_own_peak():
    history = [
        {"validation": {
            "har": {"balanced_accuracy": 0.5, "accuracy": 0.6},
            "sleep": {"balanced_accuracy": 0.8, "accuracy": 0.9},
        }},
        {"validation": {
            "har": {"balanced_accuracy": 0.7, "accuracy": 0.65},
            "sleep": {"balanced_accuracy": 0.6, "accuracy": 0.7},
        }},
        {"validation": {
            "har": {"balanced_accuracy": 0.6, "accuracy": 0.8},
            "sleep": {"balanced_accuracy": 0.5, "accuracy": 0.6},
        }},
    ]
    summary = per_dataset_best(history)
    assert summary["har"] == {
        "balanced": 0.7, "balanced_epoch": 2, "micro": 0.8, "micro_epoch": 3
    }
    assert summary["sleep"] == {
        "balanced": 0.8, "balanced_epoch": 1, "micro": 0.9, "micro_epoch": 1
    }


def test_per_dataset_best_empty_without_per_dataset_validation():
    assert per_dataset_best([{"validation_loss": 1.0, "validation_accuracy": 0.5}]) == {}


def test_per_dataset_best_skips_datasets_missing_balanced_accuracy():
    # Older runs stored per-dataset validation without balanced_accuracy.
    history = [{"validation": {"har": {"loss": 0.5, "accuracy": 0.6}}}]
    assert per_dataset_best(history) == {}


def test_training_with_dataset_collator(make_chronos2_encoder, har_root):
    dataset = HARDataset(har_root)
    loader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate)
    model = SystemOne(make_chronos2_encoder(), num_classes=len(dataset.labels), hidden_dim=16)
    history = train(model, loader, loader, Config(epochs=2))
    assert len(history) == 2
    assert all(row["train_loss"] > 0 for row in history)
    # The restored checkpoint is the highest balanced-accuracy epoch.
    assert max(row["validation_balanced_accuracy"] for row in history) == pytest.approx(
        validate(model, loader)["balanced_accuracy"]
    )


def test_training_learns_and_retains_optimizer_state(model, batch):
    # Compare the orchestrator with uninterrupted manual steps to catch optimizer resets.
    reference = copy.deepcopy(model)
    config = Config(epochs=15, patience=15, learning_rate=0.02)
    optimizer = build_optimizer(reference, config)
    history = train(model, [batch], [batch], config)
    best_epoch = (
        max(range(len(history)), key=lambda i: history[i]["validation_balanced_accuracy"]) + 1
    )
    for _ in range(best_epoch):
        train_step(reference, batch, optimizer)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
