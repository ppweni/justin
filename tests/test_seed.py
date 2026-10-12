import pytest
import torch

from justin.config import Config
from justin.model import SystemOne
from justin.training.run import run


def test_seed_must_be_a_nonnegative_integer_or_null():
    assert Config().seed is None and Config(seed=0).seed == 0
    for bad in (-1, 1.5, "1", True):
        with pytest.raises(ValueError, match="seed"):
            Config(seed=bad)


def train(tmp_path, monkeypatch, har_root, make_chronos2_encoder, text_encoder, name, seed):
    # Frozen backbones come from a fixed seed, as pretrained weights do; the run seed must
    # then decide everything that is trained and shuffled.
    def load_model(**kwargs):
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(1234)
        encoder = make_chronos2_encoder()
        torch.random.set_rng_state(generator_state)
        return SystemOne(encoder, None, num_classes=8, hidden_dim=16)

    monkeypatch.setattr(SystemOne, "from_pretrained", load_model)
    config = Config(
        dataset="har", root=str(har_root), num_classes=8, with_text=False, hidden_dim=16,
        epochs=1, batch_size=4, output=str(tmp_path / f"{name}.pt"), seed=seed,
    )  # fmt: skip
    return torch.load(run(config), weights_only=True)["model"]


def test_same_seed_repeats_a_run_and_another_seed_changes_it(
    tmp_path, monkeypatch, har_root, make_chronos2_encoder, text_encoder
):
    args = (tmp_path, monkeypatch, har_root, make_chronos2_encoder, text_encoder)
    first = train(*args, "a", seed=7)
    again = train(*args, "b", seed=7)
    other = train(*args, "c", seed=8)
    trained = [k for k in first if k.startswith(("fusion.", "classes."))]
    assert trained
    assert all(torch.equal(first[k], again[k]) for k in trained)
    assert any(not torch.equal(first[k], other[k]) for k in trained)
