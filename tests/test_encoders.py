from unittest.mock import patch

import pytest
import torch

from src.model import Chronos2Encoder, TextEncoder


def test_chronos2_factory(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    with patch(
        "src.model.encoders.Chronos2.from_pretrained", return_value=encoder.backbone
    ) as load:
        result = Chronos2Encoder.from_pretrained()
        Chronos2Encoder.from_pretrained(revision="abc")
    assert [c.args for c in load.call_args_list] == [(Chronos2Encoder.revision,), ("abc",)]
    assert result.hidden_size == 16


def test_text_factory(text_encoder):
    with (
        patch(
            "src.model.encoders.AutoModel.from_pretrained", return_value=text_encoder.backbone
        ) as load,
        patch(
            "src.model.encoders.AutoTokenizer.from_pretrained", return_value=text_encoder.tokenizer
        ) as tokenize,
    ):
        encoder = TextEncoder.from_pretrained()
    pin = TextEncoder.revision
    load.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", revision=pin)
    tokenize.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", revision=pin)
    assert encoder.hidden_size == 16


def test_frozen_backbone_stays_in_eval(chronos2_encoder, text_encoder):
    assert chronos2_encoder.train() is chronos2_encoder
    assert chronos2_encoder.training
    assert all(not module.training for module in chronos2_encoder.backbone.modules())
    assert all(not p.requires_grad for p in chronos2_encoder.backbone.parameters())
    text_encoder.train()
    assert all(not module.training for module in text_encoder.modules())
    assert all(not p.requires_grad for p in text_encoder.parameters())


def test_series_output_shape(chronos2_encoder):
    # patch_length=4, so 8 timesteps -> N=2 patches (+ [REG] + future token = S=4); C=3 -> M=12
    series = torch.randn(2, 3, 8)
    H_ts, mask = chronos2_encoder(series)
    assert H_ts.shape == (2, 12, 16)
    assert mask.shape == (2, 12)
    assert mask.dtype == torch.bool
    assert mask.all()  # no padding -> all valid
    assert H_ts.isfinite().all()


def test_series_rejects_invalid_shapes(chronos2_encoder):
    for shape in [(2, 8), (1, 1, 1, 8), (0, 1, 8), (1, 0, 8), (1, 1, 0)]:
        with pytest.raises(ValueError, match="shape"):
            chronos2_encoder(torch.zeros(shape))


def test_padding_mask_marks_tokens_invalid(chronos2_encoder):
    # 8 timesteps, patch_length=4 -> N=2 patches per channel, plus [REG] and a future token (S=4)
    # Left-pad first 4 timesteps of channel 0: first patch is all-padding -> invalid
    series = torch.randn(1, 2, 8)
    mask = torch.zeros(1, 2, 8, dtype=torch.bool)
    mask[0, 0, :4] = True  # first patch of channel 0 is padded

    H_ts, token_mask = chronos2_encoder(series, mask)
    # M = C*S = 2*4 = 8 tokens per channel: [patch0, patch1, REG, future]
    assert token_mask.shape == (1, 8)
    assert not token_mask[0, 0]  # ch0 patch0: all padded -> invalid
    assert token_mask[0, 1:4].all()  # ch0 patch1, REG, future: valid
    assert token_mask[0, 4:].all()  # ch1: all valid


def test_padding_mask_shape_mismatch_rejected(chronos2_encoder):
    with pytest.raises(ValueError, match="padding_mask"):
        chronos2_encoder(torch.ones(1, 1, 8), torch.zeros(1, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="padding_mask"):
        chronos2_encoder(torch.ones(1, 1, 8), torch.zeros(1, 1, 8))  # not bool


def test_no_gradients_through_backbone(chronos2_encoder, text_encoder):
    # Run a full forward through SystemOne so there IS a grad path (via fusion),
    # then check that the frozen backbone never accumulated gradients.
    from src.model import SystemOne

    model = SystemOne(chronos2_encoder, text_encoder, num_classes=2, hidden_dim=16)
    model.train()
    logits = model(torch.randn(1, 1, 8), ["rising"])
    logits.sum().backward()
    for p in chronos2_encoder.backbone.parameters():
        assert p.grad is None


def test_text_embeddings_ignore_padding_and_are_normalized(text_encoder):
    text_encoder.train()
    alone = text_encoder(["rising"])
    batched = text_encoder(["rising", "is this falling ?"])
    torch.testing.assert_close(alone[0], batched[0])
    torch.testing.assert_close(batched.norm(dim=-1), torch.ones(2))
    assert not batched.requires_grad
    assert not torch.allclose(batched[0], batched[1])


def test_token_states_shape_and_mask(text_encoder):
    hidden, mask = text_encoder.token_states(["rising", "is this falling ?"])
    assert hidden.ndim == 3
    assert hidden.shape[:2] == mask.shape
    assert mask.dtype == torch.bool
    # longer sequence has more valid tokens
    assert mask[1].sum() > mask[0].sum()


def test_token_states_is_not_normalized(text_encoder):
    hidden, _ = text_encoder.token_states(["rising"])
    # raw hidden states are not L2-normalized like forward()
    norms = hidden[0].norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms))


def test_text_truncates_to_256_tokens(text_encoder):
    short = " ".join(["rising"] * 254)
    long = short + " " + " ".join(["falling"] * 80)
    torch.testing.assert_close(text_encoder([short]), text_encoder([long]))
