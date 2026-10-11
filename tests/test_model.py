from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from src.model import SystemOne
from src.model.fusion import CrossAttentionFusion


@pytest.mark.parametrize("with_text", [True, False])
def test_pretrained_factory(make_chronos2_encoder, text_encoder, with_text):
    encoder = make_chronos2_encoder()
    with (
        patch(
            "src.model.encoders.Chronos2Encoder.from_pretrained", return_value=encoder
        ) as series_load,
        patch(
            "src.model.model.TextEncoder.from_pretrained", return_value=text_encoder
        ) as text_load,
    ):
        model = SystemOne.from_pretrained(with_text=with_text, num_classes=3)
    series_load.assert_called_once_with()
    assert text_load.call_count == int(with_text)
    assert (model.text_encoder is not None) == with_text
    assert model(torch.randn(2, 1, 8)).shape == (2, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0},
        {"score_levels": 1},
        {"num_classes": 1},
        {"num_classes": 256},
    ],
)
def test_invalid_dimensions(chronos2_encoder, kwargs):
    with pytest.raises(ValueError):
        SystemOne(chronos2_encoder, **kwargs)


@pytest.mark.parametrize("head,shape", [("choice", (2, 3)), ("noul", (2,)), ("score", (2, 4))])
def test_heads_and_gradient_isolation(model, head, shape):
    model.train()
    logits = model(torch.randn(2, 1, 8), ["is this rising ?"], head=head)
    assert logits.shape == shape and logits.isfinite().all()
    target = torch.tensor([0.0, 1.0]) if head == "noul" else torch.tensor([0, 1])
    loss = (
        F.binary_cross_entropy_with_logits(logits, target)
        if head == "noul"
        else F.cross_entropy(logits, target)
    )
    loss.backward()
    assert isinstance(model.fusion, CrossAttentionFusion)
    assert model.fusion.query_projection.weight.grad.abs().sum() > 0
    trained = model.classes if head == "choice" else getattr(model, head)
    assert trained.weight.grad.abs().sum() > 0
    # backbone stays frozen
    for p in model.series_encoder.backbone.parameters():
        assert p.grad is None
    assert model.text_encoder is not None
    for p in model.text_encoder.parameters():
        assert p.grad is None
    assert all(not m.training for m in model.series_encoder.backbone.modules())
    assert all(not m.training for m in model.text_encoder.modules())


def test_runtime_options_are_permutation_equivariant(model):
    series = torch.randn(2, 1, 8)
    questions = ["is this rising ?", "is this falling ?"]
    options = ["rising", "falling", "flat"]
    logits = model(series, questions, options=options)
    permuted = model(series, questions, options=[options[i] for i in (2, 0, 1)])
    torch.testing.assert_close(permuted, logits[:, [2, 0, 1]])
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.option_projection.weight.grad.abs().sum() > 0
    assert model.classes.weight.grad is None


def test_reused_features_and_shared_question_match_forward(model):
    series = torch.randn(2, 1, 8)
    features = model.series_encoder(series)
    expected = model(series, ["is this rising ?"])
    with patch.object(
        model.series_encoder, "forward", side_effect=AssertionError("Re-encoded signal")
    ):
        actual = model.answer(features, ["is this rising ?"] * 2)
    torch.testing.assert_close(actual, expected)


def test_question_changes_how_signal_is_interpreted(model):
    features = model.series_encoder(torch.randn(2, 1, 8))
    rising = model.answer(features, ["rising"], head="noul")
    falling = model.answer(features, ["falling"], head="noul")
    assert not torch.isclose((rising - falling)[0], (rising - falling)[1], atol=1e-7)


def test_answers_are_independent_of_other_batch_items(model):
    series = torch.randn(2, 1, 8)
    questions = ["rising", "falling"]
    actual = model(series, questions, head="noul")
    expected = torch.cat(
        [model(series[i : i + 1], [q], head="noul") for i, q in enumerate(questions)]
    )
    torch.testing.assert_close(actual, expected)


def test_checkpoint_round_trip(model, tmp_path):
    series = torch.randn(2, 1, 8)
    expected = model(series)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        model.classes.weight.zero_()
    model.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model(series), expected)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"head": "unknown"}, "unknown head"),
        ({"head": "noul", "options": ["yes", "no"]}, "only supported"),
        ({"questions": []}, "one question"),
        ({"questions": ["yes"] * 3}, "one question"),
        ({"options": []}, "2 to 255"),
        ({"options": ["yes"]}, "2 to 255"),
        ({"options": ["yes"] * 256}, "2 to 255"),
    ],
)
def test_invalid_requests(model, kwargs, match):
    with pytest.raises(ValueError, match=match):
        model(torch.randn(2, 1, 8), **kwargs)


@pytest.mark.parametrize("series", [torch.zeros(1, 8), torch.zeros(1, 1, 1, 8)])
def test_wrong_rank(model, series):
    with pytest.raises(ValueError, match="shape"):
        model(series)


def test_choice_requires_options_or_catalog(chronos2_encoder, text_encoder):
    m = SystemOne(chronos2_encoder, text_encoder)
    with pytest.raises(ValueError, match="options or num_classes"):
        m(torch.randn(1, 1, 8))


@pytest.mark.parametrize("kwargs", [{"questions": ["rising"]}, {"options": ["yes", "no"]}])
def test_text_requires_encoder(chronos2_encoder, kwargs):
    m = SystemOne(chronos2_encoder, num_classes=3)
    with pytest.raises(ValueError, match="text encoder"):
        m(torch.randn(1, 1, 8), **kwargs)


def test_no_dataset_key_required(model):
    # forward must work without any dataset= kwarg
    logits = model(torch.randn(2, 1, 8), ["rising"])
    assert logits.isfinite().all()


def test_option_attention_starts_equal_to_dot_product_logits(chronos2_encoder, text_encoder):
    model = SystemOne(chronos2_encoder, text_encoder, hidden_dim=16, option_attention=True).eval()
    series = torch.randn(2, 2, 12)
    options = ["rising", "falling", "flat"]
    with torch.no_grad():
        enabled = model(series, ["is this rising ?"], options=options)
        attention, model.option_attention = model.option_attention, None
        baseline = model(series, ["is this rising ?"], options=options)
        model.option_attention = attention
    assert enabled.shape == (2, 3)
    assert torch.equal(enabled, baseline)


def test_option_attention_learns_a_correction_and_only_applies_to_options(
    chronos2_encoder, text_encoder
):
    model = SystemOne(
        chronos2_encoder, text_encoder, num_classes=3, hidden_dim=16, option_attention=True
    )
    series = torch.randn(2, 2, 12)
    assert model.option_attention is not None
    torch.nn.init.normal_(model.option_attention.scorer.weight)
    with_options = model(series, ["is this rising ?"], options=["rising", "falling"])
    with_options.sum().backward()
    grad = model.option_attention.kv_projection.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    # A fixed class catalog (no runtime options) and the other heads skip the extra path.
    assert model(series, ["is this rising ?"]).shape == (2, 3)
    assert model(series, ["is this rising ?"], head="noul").shape == (2,)


def test_option_attention_requires_text_encoder(chronos2_encoder):
    with pytest.raises(ValueError, match="option_attention"):
        SystemOne(chronos2_encoder, num_classes=3, option_attention=True)


def test_signal_skip_flows_through_system_one(chronos2_encoder, text_encoder):
    model = SystemOne(chronos2_encoder, text_encoder, num_classes=3, hidden_dim=16, signal_skip=True)
    assert model.fusion.signal_skip is not None
    logits = model(torch.randn(2, 2, 12), ["is this rising ?"])
    logits.sum().backward()
    assert model.fusion.signal_skip[1].weight.grad is not None


def test_signal_skip_requires_text_encoder(chronos2_encoder):
    with pytest.raises(ValueError, match="signal_skip"):
        SystemOne(chronos2_encoder, num_classes=3, signal_skip=True)
