import pytest
import torch
from torch import nn

from src.model import SystemOne
from src.model.lora import LoRALinear, inject_lora


def test_lora_linear_is_identity_at_init_and_trainable():
    base = nn.Linear(4, 3)
    lora = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(5, 4)
    torch.testing.assert_close(lora(x), base(x))  # B zero-init => delta is 0
    assert not lora.base.weight.requires_grad
    assert lora.a.weight.requires_grad and lora.b.weight.requires_grad


def test_lora_linear_rejects_bad_hyperparameters():
    with pytest.raises(ValueError, match="positive"):
        LoRALinear(nn.Linear(4, 3), rank=0, alpha=4)
    with pytest.raises(ValueError, match="positive"):
        LoRALinear(nn.Linear(4, 3), rank=2, alpha=0)


def test_inject_lora_targets_only_matching_and_freezes_base():
    root = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))  # children "0", "2"
    count = inject_lora(root, lambda name: name == "0", rank=2, alpha=4)
    assert count == 1
    assert isinstance(root[0], LoRALinear) and isinstance(root[2], nn.Linear)
    assert not root[0].base.weight.requires_grad


def test_prefix_suffix_equals_full_encode(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(2, 3, 20)
    h_full, mask_full = encoder.forward(series)
    for boundary in (0, encoder.num_blocks):
        prefix, valid = encoder.encode_prefix(series, None, boundary)
        h_suffix, mask_suffix = encoder.encode_suffix(prefix, valid, boundary)
        torch.testing.assert_close(h_full, h_suffix, atol=1e-5, rtol=0)
        assert torch.equal(mask_full, mask_suffix)


def test_series_lora_only_adapters_train(make_chronos2_encoder, text_encoder):
    model = SystemOne(
        make_chronos2_encoder(), text_encoder,
        num_classes=3, hidden_dim=16, series_lora_layers=1, lora_rank=4, lora_alpha=8,
    ).train()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # The tiny backbone has 2 blocks; one adapted layer means block 1 carries the adapters.
    assert any("encoder.block.1." in n and ".b.weight" in n for n in trainable)
    assert not any("encoder.block.0." in n for n in trainable)
    # The frozen base of an adapted block stays frozen.
    assert not any(".q.base.weight" in n for n in trainable)

    prefix, valid = model.series_encoder.encode_prefix(torch.randn(2, 3, 20), None, model.lora_boundary)
    features = model.series_encoder.encode_suffix(prefix, valid, model.lora_boundary)
    model.answer_cached(features, head="choice").sum().backward()
    lora_b = [
        p for n, p in model.named_parameters()
        if "encoder.block" in n and ".b.weight" in n
    ]
    assert lora_b and any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_b)
    base = model.series_encoder.backbone.encoder.block[1].layer[0].self_attention.q.base
    assert base.weight.grad is None  # frozen base carries no gradient


def test_minilm_lora_sets_trainable_and_flows_grad(make_chronos2_encoder, text_encoder):
    model = SystemOne(
        make_chronos2_encoder(), text_encoder,
        num_classes=3, hidden_dim=16, minilm_lora=True, lora_rank=4, lora_alpha=8,
    ).train()
    assert model.text_encoder.trainable is True
    features = model.series_encoder(torch.randn(2, 1, 8))  # frozen signal
    logits = model.answer(
        features, ["is this rising ?"] * 2, options=["rising", "falling", "flat"], head="choice"
    )
    logits.sum().backward()
    text_b = [
        p for n, p in model.named_parameters()
        if "text_encoder" in n and ".b.weight" in n
    ]
    assert text_b and any(p.grad is not None and p.grad.abs().sum() > 0 for p in text_b)
