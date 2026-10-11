import pytest
import torch

from src.model import Chronos2Encoder, SeriesEncoder, SystemOne
from src.model.chronos2 import Chronos2


def test_encoder_implements_the_abc(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    assert isinstance(encoder, SeriesEncoder)
    assert encoder.num_blocks >= 1
    assert Chronos2Encoder.name == "chronos"
    with pytest.raises(TypeError):
        SeriesEncoder(torch.nn.Linear(1, 1))  # ty: ignore[call-non-callable]


def test_forward_shapes_and_mask(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(2, 3, 10)  # padded to 12 -> 3 patches + REG + future = 5 tokens
    states, mask = encoder(series)
    assert states.shape == (2, 15, encoder.hidden_size)
    assert mask.shape == (2, 15) and mask.dtype == torch.bool
    # The left-pad patch is partially observed, so every token is valid.
    assert mask.all()
    assert not states.requires_grad


def test_left_padding_is_masked_out(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(1, 2, 12)
    padding = torch.zeros(1, 2, 12, dtype=torch.bool)
    padding[..., :4] = True
    _, mask = encoder(series, padding)
    assert mask.view(1, 2, 5)[0, :, 0].tolist() == [False, False]
    assert mask.view(1, 2, 5)[0, :, 1:].all()


def test_padding_value_does_not_matter(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(1, 2, 12)
    padding = torch.zeros(1, 2, 12, dtype=torch.bool)
    padding[..., :4] = True
    other = series.clone()
    other[..., :4] = 1e6
    a, mask = encoder(series, padding)
    b, _ = encoder(other, padding)
    assert torch.allclose(a[mask], b[mask], atol=1e-5)


def test_fully_padded_channel_is_invisible_to_group_attention(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(1, 3, 8)
    padding = torch.zeros(1, 3, 8, dtype=torch.bool)
    padding[:, 2] = True
    changed = series.clone()
    changed[:, 2] = 1e3
    a, mask = encoder(series, padding)
    b, _ = encoder(changed, padding)
    assert not mask.view(1, 3, 4)[0, 2].any()
    assert torch.allclose(a[mask], b[mask], atol=1e-5)


def test_samples_do_not_leak_across_the_batch(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(2, 2, 8)
    changed = series.clone()
    changed[1] = torch.randn(2, 8)
    a, _ = encoder(series)
    b, _ = encoder(changed)
    assert torch.allclose(a[0], b[0], atol=1e-5)
    assert not torch.allclose(a[1], b[1], atol=1e-5)


def test_channels_mix_through_group_attention(make_chronos2_encoder):
    encoder = make_chronos2_encoder()
    series = torch.randn(1, 2, 8)
    changed = series.clone()
    changed[0, 1] = torch.randn(8)
    a, _ = encoder(series)
    b, _ = encoder(changed)
    first_channel = slice(0, a.shape[1] // 2)
    assert not torch.allclose(a[0, first_channel], b[0, first_channel], atol=1e-5)


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_suffix_of_prefix_equals_forward(make_chronos2_encoder, boundary):
    encoder = make_chronos2_encoder()
    series = torch.randn(2, 3, 12)
    expected, expected_mask = encoder(series)
    prefix, valid = encoder.encode_prefix(series, None, boundary)
    actual, mask = encoder.encode_suffix(prefix, valid, boundary)
    assert torch.equal(mask, expected_mask)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_suffix_accepts_right_padded_prefixes(make_chronos2_encoder):
    """Cache collation right-pads shorter series; results must match the unpadded run."""
    encoder = make_chronos2_encoder()
    series = torch.randn(1, 2, 8)
    expected, _ = encoder(series)
    prefix, valid = encoder.encode_prefix(series, None, 1)
    S = prefix.shape[2]
    wide = torch.zeros(1, 2, S + 3, prefix.shape[3])
    wide_valid = torch.zeros(1, 2, S + 3, dtype=torch.bool)
    wide[:, :, :S], wide_valid[:, :, :S] = prefix, valid
    actual, mask = encoder.encode_suffix(wide, wide_valid, 1)
    actual = actual.view(1, 2, S + 3, -1)[:, :, :S].reshape(1, 2 * S, -1)
    assert torch.allclose(actual, expected, atol=1e-5)
    assert mask.view(1, 2, S + 3)[:, :, S:].sum() == 0


def test_context_is_truncated_to_the_most_recent_steps(make_chronos2_encoder):
    encoder = make_chronos2_encoder()  # context_length 64
    series = torch.randn(1, 1, 100)
    a, _ = encoder(series)
    b, _ = encoder(series[..., -64:])
    assert torch.allclose(a, b, atol=1e-6)


def test_lora_is_identity_at_init_and_trains_only_adapters(
    make_chronos2_encoder, text_encoder
):
    encoder = make_chronos2_encoder()
    series = torch.randn(1, 2, 8)
    before, _ = encoder(series)
    model = SystemOne(
        encoder, text_encoder, num_classes=3, hidden_dim=16, series_lora_layers=1,
        lora_rank=4, lora_alpha=8,
    )
    assert model.lora_boundary == 1
    after, _ = encoder(series)
    assert torch.allclose(before, after, atol=1e-6)
    names = [n for n, p in encoder.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("backbone.encoder.block.1.") for n in names)
    assert all(n.endswith((".a.weight", ".b.weight")) for n in names)


def test_lora_layer_count_is_bounded_by_the_encoder(make_chronos2_encoder, text_encoder):
    with pytest.raises(ValueError, match="series_lora_layers"):
        SystemOne(make_chronos2_encoder(), text_encoder, series_lora_layers=3)


def test_system_one_runs_end_to_end_with_chronos2(make_chronos2_encoder, text_encoder):
    model = SystemOne(make_chronos2_encoder(), text_encoder, num_classes=3, hidden_dim=16)
    logits = model(torch.randn(2, 2, 12), ["is this rising ?"], head="choice")
    assert logits.shape == (2, 3)


def test_state_dict_keys_follow_upstream_layout():
    keys = set(Chronos2(num_layers=1, d_model=16, d_ff=24, d_kv=8, num_heads=2).state_dict())
    for expected in (
        "shared.weight",
        "input_patch_embedding.hidden_layer.weight",
        "input_patch_embedding.residual_layer.bias",
        "encoder.block.0.layer.0.self_attention.q.weight",
        "encoder.block.0.layer.0.layer_norm.weight",
        "encoder.block.0.layer.1.self_attention.o.weight",
        "encoder.block.0.layer.2.mlp.wi.weight",
        "encoder.block.0.layer.2.layer_norm.weight",
        "encoder.final_layer_norm.weight",
    ):
        assert expected in keys
