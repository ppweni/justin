import pytest
import torch

from justin.config import Config
from justin.data import HARDataset
from justin.model import SystemOne
from justin.training.evaluate import (
    evaluate_model,
    load_checkpoint,
    macro_f1,
    predictions,
    score,
)
from justin.training.loader import BatchLoader
from justin.training.validation import validate


def test_macro_f1_matches_hand_computation():
    # a: 2 true, 1 hit; b: 1 true, 1 hit; plus one wrong "c" prediction for an "a".
    pairs = [("a", "a"), ("a", "c"), ("b", "b")]
    # F1(a) = 2*1/(2+0+1) = 2/3, F1(b) = 1, F1(c) = 0 (predicted, never true)
    assert macro_f1(pairs) == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert macro_f1([("a", "a"), ("b", "b")]) == 1.0
    with pytest.raises(ValueError):
        macro_f1([])


def test_score_reports_accuracy_balanced_accuracy_and_f1():
    pairs = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b")]
    result = score(pairs)
    assert result["samples"] == 4
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert 0 < result["macro_f1"] <= 1


def test_evaluation_matches_the_validation_metrics(har_root, chronos2_encoder, text_encoder):
    model = SystemOne(chronos2_encoder, text_encoder, hidden_dim=16)
    loader = BatchLoader(HARDataset(har_root, split="validation"), 4, dataset_name="har")
    expected = validate(model, loader)
    result = evaluate_model(model, {"har": loader})["har"]
    assert result["accuracy"] == pytest.approx(expected["accuracy"])
    assert result["balanced_accuracy"] == pytest.approx(expected["balanced_accuracy"])
    assert result["samples"] == len(HARDataset(har_root, split="validation"))
    assert len(predictions(model, loader)) == result["samples"]


def test_load_checkpoint_rebuilds_the_trained_model(
    tmp_path, monkeypatch, make_chronos2_encoder, text_encoder
):
    config = Config(series_encoder="chronos", signal_skip=True, hidden_dim=16, device="cpu")
    trained = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16, signal_skip=True)
    from dataclasses import asdict

    path = tmp_path / "model.pt"
    torch.save({"model": trained.state_dict(), "config": asdict(config)}, path)
    monkeypatch.setattr(
        SystemOne,
        "from_pretrained",
        classmethod(
            lambda cls, **kwargs: cls(
                make_chronos2_encoder(), text_encoder, hidden_dim=kwargs["hidden_dim"],
                signal_skip=kwargs["signal_skip"],
            )
        ),
    )
    model, loaded = load_checkpoint(path, device="cpu")
    assert loaded.series_encoder == "chronos" and loaded.signal_skip
    assert not model.training
    for key, value in trained.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
