from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import justin
from justin.config import Config
from justin.model import SystemOne
from justin.pretrained import Justin, load_pretrained


@pytest.fixture
def small_model(make_chronos2_encoder, text_encoder):
    return SystemOne(make_chronos2_encoder(), text_encoder, num_classes=3, hidden_dim=16).eval()


def test_answer_picks_a_supplied_option_or_a_boolean(small_model):
    justin = Justin(small_model, Config(hidden_dim=16))
    series = torch.randn(2, 12)
    assert justin.answer(series, "is this rising ?", ["rising", "falling"]) in ("rising", "falling")
    assert isinstance(justin.answer(series, "is this rising ?"), bool)
    with pytest.raises(ValueError, match="shape"):
        justin.answer(torch.randn(1, 2, 12), "is this rising ?")


def test_series_may_be_numpy_lists_or_a_single_channel(small_model):
    justin = Justin(small_model, Config(hidden_dim=16))
    values = torch.randn(12)
    options = ["rising", "falling"]
    expected = justin.probabilities(values, "is this rising ?", options)
    for series in (values.numpy(), values.tolist(), values.unsqueeze(0)):
        actual = justin.probabilities(series, "is this rising ?", options)
        assert actual == pytest.approx(expected)


def test_probabilities_are_calibrated_with_the_stored_temperature(small_model):
    justin = Justin(small_model, Config(hidden_dim=16))
    series, options = torch.randn(2, 12), ["rising", "falling", "flat"]
    plain = justin.probabilities(series, "is this rising ?", options)
    assert list(plain) == options and sum(plain.values()) == pytest.approx(1)
    assert justin.answer(series, "is this rising ?", options) == max(plain, key=plain.__getitem__)
    small_model.temperatures = {"choice": 5.0, "noul": 5.0}
    soft = justin.probabilities(series, "is this rising ?", options)
    assert max(soft.values()) < max(plain.values())
    assert 0 < justin.probabilities(series, "is this rising ?") < 1


def test_justin_is_available_from_the_package_and_loads_from_a_path(tmp_path, small_model):
    assert justin.Justin is Justin
    path = tmp_path / "model.pt"
    config = Config(series_encoder="chronos", hidden_dim=16, device="cpu")
    torch.save({"model": small_model.state_dict(), "config": asdict(config)}, path)
    with patch(
        "justin.training.evaluate.build_model",
        return_value=SystemOne(
            small_model.series_encoder, small_model.text_encoder, num_classes=3, hidden_dim=16
        ),
    ):
        loaded = Justin.from_pretrained(path, device="cpu")
    assert loaded.config.series_encoder == "chronos" and loaded.temperatures == {}
    assert loaded.device.type == "cpu"


def test_load_pretrained_from_a_local_file(tmp_path, small_model):
    path = tmp_path / "model.pt"
    config = Config(series_encoder="chronos", hidden_dim=16, device="cpu")
    torch.save({"model": small_model.state_dict(), "config": asdict(config)}, path)
    with patch(
        "justin.training.evaluate.build_model",
        return_value=SystemOne(
            small_model.series_encoder, small_model.text_encoder, num_classes=3, hidden_dim=16
        ),
    ):
        model, loaded = load_pretrained(path, device="cpu")
    assert loaded.series_encoder == "chronos" and not model.training


def test_load_pretrained_downloads_from_the_hub_when_given_a_repo(tmp_path):
    with pytest.raises(ValueError, match="filename"):
        load_pretrained("owner/justin-chronos")
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(tmp_path / "m.pt")) as download,
        patch("justin.pretrained.load_checkpoint", return_value=(SimpleNamespace(), "config")) as load,
    ):
        result = load_pretrained("owner/justin-chronos", filename="m.pt", device="cpu")
    download.assert_called_once_with(
        repo_id="owner/justin-chronos", filename="m.pt", revision=None
    )
    load.assert_called_once()
    assert result[1] == "config" and result[0].temperatures == {}
