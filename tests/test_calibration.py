import copy
import math

import pytest
import torch

from justin.training import fit_temperatures, probabilities
from justin.training.loss import negative_log_likelihood


def calibration_batch(logits, target, head):
    """For tests with an identity forward: the series tensor supplies known logits."""
    return {"series": logits, "target": target, "head": head}


@pytest.mark.parametrize("head", ["choice", "score", "noul"])
@pytest.mark.parametrize("scale", [0.2, 4.0])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="requires an MPS device"
            ),
        ),
    ],
)
def test_fitting_recovers_known_temperature(model, monkeypatch, head, scale, device):
    model.to(device)
    monkeypatch.setattr(model, "forward", lambda series, **kwargs: series)
    if head == "noul":
        logits = torch.full((20,), scale)
        target = torch.tensor([1.0] * 15 + [0.0] * 5)
    else:
        logits = torch.tensor([[scale, 0.0, 0.0]]).repeat(20, 1)
        target = torch.tensor([0] * 12 + [1] * 4 + [2] * 4)
    before = negative_log_likelihood(logits, target, head).item()
    # Uneven batches with very different label frequencies expose weighting errors.
    batches = [
        calibration_batch(logits[:3], target[:3], head),
        calibration_batch(logits[3:], target[3:], head),
    ]
    temperature = fit_temperatures(model, batches)[head]
    assert temperature == pytest.approx(scale / math.log(3), rel=0.005)
    after = negative_log_likelihood(logits / temperature, target, head).item()
    assert after < before
    combined = fit_temperatures(model, [calibration_batch(logits, target, head)])[head]
    assert temperature == pytest.approx(combined, rel=1e-6)


def test_fitting_handles_mixed_heads_and_option_counts(model, monkeypatch):
    monkeypatch.setattr(model, "forward", lambda series, **kwargs: series)
    batches = [
        calibration_batch(
            torch.tensor([[4.0, 0.0]]).repeat(4, 1), torch.tensor([0, 0, 0, 1]), "choice"
        ),
        calibration_batch(
            torch.tensor([[3.0, 0.0, 0.0]]).repeat(3, 1), torch.tensor([0, 0, 2]), "choice"
        ),
        calibration_batch(torch.tensor([3.0, 3.0, 3.0]), torch.tensor([0.0, 1.0, 1.0]), "noul"),
        calibration_batch(
            torch.tensor([[3.0, 0.0, 0.0]]).repeat(3, 1), torch.tensor([0, 0, 1]), "score"
        ),
    ]
    fitted = fit_temperatures(model, batches)
    assert set(fitted) == {"choice", "noul", "score"}
    for head, temperature in fitted.items():
        matching = [batch for batch in batches if batch["head"] == head]
        before = sum(
            negative_log_likelihood(b["series"], b["target"], head) * b["target"].numel()
            for b in matching
        )
        after = sum(
            negative_log_likelihood(b["series"] / temperature, b["target"], head)
            * b["target"].numel()
            for b in matching
        )
        assert after < before


@pytest.mark.parametrize("training", [True, False])
def test_fitting_preserves_model_and_existing_gradients(model, training):
    model.train(training)
    before = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(torch.is_grad_enabled()))
    batch = {"series": torch.randn(3, 2, 16), "target": torch.tensor([0, 1, 2])}
    # Calibration still needs gradients for its scalar when called under no_grad.
    with torch.no_grad():
        temperatures = fit_temperatures(model, [batch])
    hook.remove()
    assert calls == [False]
    assert set(temperatures) == {"choice"}
    assert model.training == training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            torch.testing.assert_close(p.grad, torch.ones_like(p))
        else:
            assert p.grad is None


@pytest.mark.parametrize("kind", ["uniform", "perfect", "wrong"])
def test_degenerate_data_produces_bounded_temperature(model, monkeypatch, kind):
    monkeypatch.setattr(model, "forward", lambda series, **kwargs: series)
    logits = torch.zeros(3) if kind == "uniform" else torch.full((3,), 3.0)
    target = torch.ones(3) if kind == "perfect" else torch.zeros(3)
    temperature = fit_temperatures(model, [calibration_batch(logits, target, "noul")])["noul"]
    assert 0.01 - 1e-10 <= temperature <= 100 + 1e-10
    assert negative_log_likelihood(logits / temperature, target, "noul") <= (
        negative_log_likelihood(logits, target, "noul")
    )
    if kind == "uniform":
        assert temperature == 1.0


def test_empty_calibration_data(model):
    with pytest.raises(ValueError, match="must not be empty"):
        fit_temperatures(model, [])


def test_fitting_restores_mode_on_forward_error(model):
    model.train()
    with pytest.raises(ValueError):
        fit_temperatures(model, [{"series": torch.zeros(3, 1, 8), "target": torch.tensor([])}])
    assert model.training


def test_nonfinite_calibration_logits(model, monkeypatch):
    monkeypatch.setattr(model, "forward", lambda series, **kwargs: series)
    with pytest.raises(ValueError, match="logits must be finite"):
        fit_temperatures(
            model, [calibration_batch(torch.tensor([float("inf")]), torch.ones(1), "noul")]
        )


@pytest.mark.parametrize("head", ["choice", "score", "noul"])
def test_calibrated_probabilities_and_decisions(head):
    logits = torch.tensor([-2.0, 0.0, 3.0]) if head == "noul" else torch.tensor([[2.0, -1.0, 0.0]])
    raw = probabilities(logits, head)
    scaled = probabilities(logits, head, 2.0)
    if head == "noul":
        torch.testing.assert_close(scaled, (logits / 2).sigmoid())
        assert torch.equal(raw >= 0.5, scaled >= 0.5)
    else:
        torch.testing.assert_close(scaled, (logits / 2).softmax(-1))
        torch.testing.assert_close(scaled.sum(-1), torch.ones(1))
        assert torch.equal(raw.argmax(-1), scaled.argmax(-1))
        assert scaled.max() < raw.max()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sharpening_half_precision_logits_stays_finite(dtype):
    result = probabilities(torch.tensor([[1000.0, -1000.0]], dtype=dtype), "choice", 0.01)
    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        probabilities(torch.ones(1), "noul", temperature)


def test_invalid_probability_head():
    with pytest.raises(ValueError, match="unknown head"):
        probabilities(torch.ones(1), "bad")  # ty: ignore[invalid-argument-type]


def test_temperature_checkpoint_roundtrip(model, tmp_path):
    batches = [{"series": torch.randn(3, 2, 16), "target": torch.tensor([0, 1, 2])}]
    temperatures = fit_temperatures(model, batches)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "temperatures": temperatures}, path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["temperatures"] == temperatures
    logits = torch.randn(3, 3)
    torch.testing.assert_close(
        probabilities(logits, "choice", temperatures["choice"]),
        probabilities(logits, "choice", checkpoint["temperatures"]["choice"]),
    )
