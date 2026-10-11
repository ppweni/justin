import pytest
import torch

from src.model.fusion import CrossAttentionFusion, OptionAttention


def _fusion(**kwargs):
    return CrossAttentionFusion(signal_dim=8, text_dim=6, hidden_dim=16, heads=4, **kwargs)


def test_multi_layer_fusion_stacks_blocks_and_flows_gradients():
    fusion = _fusion(layers=3, dropout=0.1)
    assert len(fusion.blocks) == 3
    signal = torch.randn(2, 5, 8, requires_grad=True)
    signal_mask = torch.ones(2, 5, dtype=torch.bool)
    question = torch.randn(2, 4, 6)
    question_mask = torch.ones(2, 4, dtype=torch.bool)
    z = fusion(signal, signal_mask, question, question_mask)
    assert z.shape == (2, 16)
    z.sum().backward()
    # A deep block's parameters receive gradients.
    assert fusion.blocks[-1].ffn[0].weight.grad is not None


def test_fusion_rejects_bad_construction():
    with pytest.raises(ValueError, match="positive"):
        _fusion(layers=0)
    with pytest.raises(ValueError, match="divisible"):
        CrossAttentionFusion(signal_dim=8, text_dim=6, hidden_dim=15, heads=4)


def test_fusion_rejects_bad_inputs():
    fusion = _fusion()
    good_signal = torch.randn(2, 5, 8)
    good_mask = torch.ones(2, 5, dtype=torch.bool)
    question = torch.randn(2, 4, 6)
    question_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="\\[B, tokens, dim\\]"):
        fusion(torch.randn(2, 8), good_mask, question, question_mask)
    with pytest.raises(ValueError, match="masks must be"):
        fusion(good_signal, torch.ones(2, 4, dtype=torch.bool), question, question_mask)
    empty = torch.zeros(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one valid"):
        fusion(good_signal, empty, question, question_mask)


def _option_attention():
    module = OptionAttention(signal_dim=8, hidden_dim=16, heads=4)
    torch.nn.init.normal_(module.scorer.weight)  # leave the zero init so outputs are nonzero
    torch.nn.init.normal_(module.scorer.bias)
    return module.eval()


def test_option_attention_starts_at_zero_evidence():
    module = OptionAttention(signal_dim=8, hidden_dim=16, heads=4)
    out = module(torch.randn(3, 16), torch.randn(5, 16), torch.randn(3, 7, 8), torch.ones(3, 7).bool())
    assert out.shape == (3, 5)
    assert torch.equal(out, torch.zeros(3, 5))


def test_option_attention_permutes_with_options_and_keeps_options_independent():
    module = _option_attention()
    z, keys = torch.randn(2, 16), torch.randn(4, 16)
    signal, mask = torch.randn(2, 6, 8), torch.ones(2, 6).bool()
    out = module(z, keys, signal, mask)
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(module(z, keys[perm], signal, mask), out[:, perm], atol=1e-6)
    changed = keys.clone()
    changed[1] = torch.randn(16)
    other = module(z, changed, signal, mask)
    assert not torch.allclose(other[:, 1], out[:, 1])
    assert torch.allclose(other[:, [0, 2, 3]], out[:, [0, 2, 3]], atol=1e-6)


def test_option_attention_ignores_padded_signal_and_samples_do_not_mix():
    module = _option_attention()
    z, keys = torch.randn(2, 16), torch.randn(3, 16)
    signal = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    out = module(z, keys, signal, mask)
    noisy = signal.clone()
    noisy[0, 4:] = 1e3
    assert torch.allclose(module(z, keys, noisy, mask), out, atol=1e-5)
    noisy[1] = torch.randn(6, 8)
    assert torch.allclose(module(z, keys, noisy, mask)[0], out[0], atol=1e-5)


def test_option_attention_gradients_reach_attention_and_scorer():
    module = _option_attention().train()
    z = torch.randn(2, 16, requires_grad=True)
    out = module(z, torch.randn(3, 16), torch.randn(2, 6, 8), torch.ones(2, 6).bool())
    out.sum().backward()
    for parameter in (module.scorer.weight, module.kv_projection.weight, module.attention.in_proj_weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_option_attention_rejects_bad_dims():
    with pytest.raises(ValueError):
        OptionAttention(signal_dim=8, hidden_dim=15, heads=4)


def _inputs():
    signal = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False
    return signal, mask, torch.randn(2, 4, 6), torch.ones(2, 4, dtype=torch.bool)


def test_signal_skip_adds_a_pooled_signal_path_and_off_has_no_extra_parameters():
    torch.manual_seed(0)
    plain = _fusion().eval()
    skipped = _fusion(signal_skip=True).eval()
    assert plain.signal_skip is None and skipped.signal_skip is not None
    assert sum(p.numel() for p in skipped.parameters()) > sum(p.numel() for p in plain.parameters())
    signal, mask, question, question_mask = _inputs()
    assert skipped(signal, mask, question, question_mask).shape == (2, 16)


def test_signal_skip_ignores_padded_tokens_and_trains_its_projection():
    fusion = _fusion(signal_skip=True).eval()
    signal, mask, question, question_mask = _inputs()
    noisy = signal.clone()
    noisy[0, 4:] = 1e3
    assert torch.allclose(
        fusion(signal, mask, question, question_mask),
        fusion(noisy, mask, question, question_mask),
        atol=1e-5,
    )
    fusion.train()
    fusion(signal, mask, question, question_mask).sum().backward()
    weight = fusion.signal_skip[1].weight
    assert weight.grad is not None and weight.grad.abs().sum() > 0
