"""Loading and asking with tiny random models (no network): the export format, every head,
the feature path, the list-of-queries API and the location head."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from justin import Answer, Found, Justin, Location, Question
from justin.files import FROZEN_PREFIXES, fingerprint, read_export
from justin.location import LocationAttentionHead, LocationHead, location_logits
from justin.system import SystemOne

CONFIG = dict(
    num_classes=None, score_levels=4, hidden_dim=16, fusion_layers=1, with_text=True,
    option_attention=False, signal_skip=True, channel_names=True, channel_features=9,
    channel_head=False, referents=True, series_encoder="timestral5m", text_encoder="minilm",
    series_depth=None, series_condition=2, series_lora_layers=2, minilm_lora=False, lora_rank=4,
    lora_alpha=8,
)  # fmt: skip
NAMES = ["cpu usage", "memory usage"]


@pytest.fixture
def exported(tmp_path, series_encoder, text_encoder):
    """(build, model, export path): an export with random trained weights, written like `justin export`."""

    def build(*_args, **_kwargs):
        keys = ("num_classes", "score_levels", "hidden_dim", "fusion_layers", "option_attention",
                "signal_skip", "channel_names", "channel_features", "channel_head", "referents",
                "series_condition", "series_lora_layers", "minilm_lora", "lora_rank", "lora_alpha")  # fmt: skip
        return SystemOne(copy.deepcopy(series_encoder), copy.deepcopy(text_encoder), **{k: CONFIG[k] for k in keys})

    torch.manual_seed(1)
    model = build().eval()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in trainable:
                parameter.add_(torch.randn_like(parameter) * 0.1)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k in trainable or not k.startswith(FROZEN_PREFIXES)}
    revisions = {"series": {"repo": "test/tiny-timestral", "revision": "0" * 40},
                 "text": {"repo": "test/tiny-bert", "revision": "1" * 40}}  # fmt: skip
    metadata = {"justin_format": "1", "config": json.dumps(CONFIG), "revisions": json.dumps(revisions),
                "temperatures": json.dumps({"choice": 1.5, "noul": 0.5}), "feature_version": json.dumps("1")}  # fmt: skip
    path = tmp_path / "model.safetensors"
    save_file(state, path, metadata=metadata)
    return build, model, path


def write_location(path, head, spec, base):
    metadata = {"justin_location_format": "1", "head": json.dumps(spec), "base_fingerprint": json.dumps(fingerprint(base))}
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, path, metadata=metadata)
    return path


def series():
    torch.manual_seed(2)
    values = torch.randn(2, 30).cumsum(-1)
    values[1, 20:] += 25
    return values


def test_an_export_loads_with_its_recorded_revisions_and_answers_like_the_model(exported):
    build, model, path = exported
    with patch("justin.pretrained.build_system", side_effect=build) as factory:
        justin = Justin.from_pretrained(path, device="cpu")
    assert factory.call_args.args[1] == {"series": "0" * 40, "text": "1" * 40}
    assert justin.temperatures == {"choice": 1.5, "noul": 0.5} and justin.config["hidden_dim"] == 16
    model.temperatures = {"choice": 1.5, "noul": 0.5}
    reference = Justin(model)
    options = ["cpu", "memory"]
    for a, b in [
        (justin.probabilities(series(), "is this rising ?", options, NAMES, step=2.0),
         reference.probabilities(series(), "is this rising ?", options, NAMES, step=2.0)),
        (justin.probabilities(series(), "is this rising ?", None, NAMES), reference.probabilities(series(), "is this rising ?", None, NAMES)),
    ]:  # fmt: skip
        assert a == pytest.approx(b)
    assert justin.answer(series(), "is this rising ?", options, NAMES) in options
    assert isinstance(justin.answer(series(), "is this rising ?", None, NAMES), bool)


def test_ask_answers_a_list_of_queries_like_the_single_calls(exported):
    build, _, path = exported
    with patch("justin.pretrained.build_system", side_effect=build):
        justin = Justin.from_pretrained(path, device="cpu")
    options = ["cpu", "memory", "flat"]
    outputs = justin.ask(
        series(), [Question("is this rising ?", options), Question("is this rising ?"), Question("is it ?", "score")], NAMES
    )
    assert [type(o) for o in outputs] == [Answer, Answer, Answer]
    choice, yes_no, score = outputs
    assert choice.answer == justin.answer(series(), "is this rising ?", options, NAMES)
    assert choice.probabilities == pytest.approx(list(justin.probabilities(series(), "is this rising ?", options, NAMES).values()))
    assert yes_no.probabilities[0] == pytest.approx(justin.probabilities(series(), "is this rising ?", None, NAMES))
    assert len(score.logits) == 4 and sum(score.probabilities) == pytest.approx(1) and isinstance(score.answer, int)
    with pytest.raises(ValueError, match="2 to 255"):
        justin.ask(series(), [Question("is this rising ?", ["only"])], NAMES)
    with pytest.raises(ValueError, match="no location head"):
        justin.ask(series(), [Location("where is the spike ?")], NAMES)
    with pytest.raises(ValueError, match="channel names"):
        justin.answer(series(), "is this rising ?")


@pytest.mark.parametrize("kind", ["attn", "pointer"])
def test_a_location_head_points_at_a_patch_and_refuses_another_model(tmp_path, exported, kind):
    build, _, path = exported
    torch.manual_seed(3)
    if kind == "attn":
        head = LocationAttentionHead(16, 16, 8, layers=1, heads=2, kernel=3, max_tokens=64)
        spec = {"type": "attn", "signal_dim": 16, "text_dim": 16, "hidden_dim": 8, "patch_size": 4,
                "layers": 1, "heads": 2, "kernel": 3, "max_tokens": 64}  # fmt: skip
    else:
        head = LocationHead(16, 16, 8)
        spec = {"type": "pointer", "signal_dim": 16, "text_dim": 16, "hidden_dim": 8, "patch_size": 4}
    location = write_location(tmp_path / "location.safetensors", head, spec, path)
    with patch("justin.pretrained.build_system", side_effect=build):
        justin = Justin.from_pretrained(path, device="cpu", location=location)
    values = series()[:, :26]  # 7 patches of 4 with 2 points of left padding
    result = justin.locate(values, "where is the spike ?", NAMES)
    expected = location_logits(justin.system, justin.location, values[None], None, ["where is the spike ?"], [NAMES])[0]
    torch.testing.assert_close(result["logits"], expected.detach())
    best = int(expected.argmax())
    if best == 14:
        assert result["location"] is None
    else:
        channel, index = divmod(best, 7)
        assert result["location"]["channel"] == channel and result["location"]["name"] == NAMES[channel]
        assert (result["location"]["start"], result["location"]["end"]) == (max(0, index * 4 - 2), min(26, (index + 1) * 4 - 2))
    found = justin.ask(values, [Location("where is the spike ?")], NAMES)[0]
    assert isinstance(found, Found) and found.location == result["location"]

    state, meta = read_export(path)
    other = tmp_path / "other.safetensors"
    save_file({k: v + 1 if k.startswith("fusion.") else v for k, v in state.items()}, other,
              metadata={"justin_format": "1", **{k: json.dumps(v) for k, v in meta.items()}})  # fmt: skip
    with patch("justin.pretrained.build_system", side_effect=build), pytest.raises(ValueError, match="belong"):
        Justin.from_pretrained(other, device="cpu", location=location)


def test_files_that_do_not_fit_are_refused(tmp_path, exported):
    build, _, path = exported
    state, meta = read_export(path)
    del state[next(k for k in state if k.startswith("fusion."))]
    broken = tmp_path / "broken.safetensors"
    save_file(state, broken, metadata={"justin_format": "1", **{k: json.dumps(v) for k, v in meta.items()}})
    with patch("justin.pretrained.build_system", side_effect=build), pytest.raises(ValueError, match="does not match"):
        Justin.from_pretrained(broken, device="cpu")
    save_file({"w": torch.zeros(1)}, tmp_path / "foreign.safetensors")
    with pytest.raises(ValueError, match="not a Justin export"):
        Justin.from_pretrained(tmp_path / "foreign.safetensors", device="cpu")
    with pytest.raises(ValueError, match="filename"):
        Justin.from_pretrained("owner/justin-timestral-minilm")


def test_only_timestral_chronos_and_minilm_models_are_built():
    from justin.pretrained import build_system

    with pytest.raises(ValueError, match="not supported"):
        build_system({**CONFIG, "series_encoder": "toto22m"})
    with pytest.raises(ValueError, match="MiniLM"):
        build_system({**CONFIG, "text_encoder": "embeddinggemma2"})


def test_a_hub_source_downloads_the_named_file(tmp_path, exported):
    build, _, path = exported
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download,
        patch("justin.pretrained.build_system", side_effect=build),
    ):
        Justin.from_pretrained("owner/justin", filename="model.safetensors", revision="v0.2.0", device="cpu")
    download.assert_called_once_with(repo_id="owner/justin", filename="model.safetensors", revision="v0.2.0")
