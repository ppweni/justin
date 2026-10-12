import copy
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from click.testing import CliRunner
from safetensors.torch import save_file

from src.__main__ import main
from src.config import Config
from src.export import backbone_revisions, export_checkpoint, read_export
from src.model import Chronos2Encoder, SystemOne, TextEncoder
from src.pretrained import answer, load_pretrained

CONFIG = Config(
    dataset="har", hidden_dim=16, signal_skip=True, series_lora_layers=1, lora_rank=4,
    lora_alpha=8, device="cuda", output="runs/secret/model.pt", db="runs/training.db",
)  # fmt: skip


@pytest.fixture
def trained(make_chronos2_encoder, text_encoder):
    """(build, model, checkpoint path factory): `build()` makes a fresh model on the same frozen weights."""
    series, text = make_chronos2_encoder(), text_encoder

    def build(*_args, **_kwargs):
        return SystemOne(
            copy.deepcopy(series), copy.deepcopy(text), hidden_dim=16, signal_skip=True,
            series_lora_layers=1, lora_rank=4, lora_alpha=8,
        )  # fmt: skip

    model = build()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.1)
    return build, model.eval()


def write_checkpoint(path, model, **extra):
    torch.save(
        {"model": model.state_dict(), "config": asdict(CONFIG), "history": [{"loss": 1.0}],
         "temperatures": {"choice": 1.25, "noul": 0.9}, **extra},
        path,
    )  # fmt: skip


def test_export_keeps_only_what_was_trained(tmp_path, trained):
    build, model = trained
    write_checkpoint(tmp_path / "m.pt", model)
    with patch("src.export.build_model", build):
        out = export_checkpoint(tmp_path / "m.pt", tmp_path / "m.safetensors")
    state, meta = read_export(out)
    full = model.state_dict()
    assert set(state) < set(full)
    frozen = [k for k in state if k.startswith(("series_encoder.", "text_encoder."))]
    assert frozen and all(k.endswith((".a.weight", ".b.weight")) for k in frozen)  # LoRA only
    assert not any(".base." in k or "text_encoder" in k for k in state)
    assert out.stat().st_size < (tmp_path / "m.pt").stat().st_size
    assert meta["temperatures"] == {"choice": 1.25, "noul": 0.9}
    assert meta["revisions"] == {
        "series": {"repo": Chronos2Encoder.repo, "revision": Chronos2Encoder.revision},
        "text": {"repo": TextEncoder.repo, "revision": TextEncoder.revision},
    }
    assert meta["revisions"] == backbone_revisions(CONFIG)
    # Run artifacts (paths, devices, training history) do not leak into the export.
    assert "output" not in meta["config"] and "device" not in meta["config"]
    assert "runs/secret" not in out.with_suffix(".json").read_text()
    assert "history" not in out.with_suffix(".json").read_text()


def test_export_rejects_frozen_weights_that_differ_from_the_pinned_backbones(tmp_path, trained):
    build, model = trained
    key = next(k for k in model.state_dict() if k.startswith("text_encoder."))
    model.state_dict()[key].add_(1.0)
    write_checkpoint(tmp_path / "m.pt", model)
    with patch("src.export.build_model", build), pytest.raises(ValueError, match="other revisions"):
        export_checkpoint(tmp_path / "m.pt", tmp_path / "m.safetensors")


def test_export_needs_a_safetensors_name(tmp_path, trained):
    _, model = trained
    write_checkpoint(tmp_path / "m.pt", model)
    with pytest.raises(ValueError, match="safetensors"):
        export_checkpoint(tmp_path / "m.pt", tmp_path / "m.bin")


def test_loading_an_export_reproduces_the_trained_model_with_the_recorded_revisions(
    tmp_path, trained
):
    build, model = trained
    write_checkpoint(tmp_path / "m.pt", model)
    with patch("src.export.build_model", build):
        out = export_checkpoint(tmp_path / "m.pt", tmp_path / "m.safetensors")
    # Another release would record other revisions; the export, not the code pin, must win.
    state, meta = read_export(out)
    meta["revisions"]["series"]["revision"] = "old-revision"
    with patch("src.pretrained.build_model", side_effect=build) as factory:
        save_file(
            state, tmp_path / "other.safetensors",
            metadata={"justin_format": "1", **{k: __import__("json").dumps(v) for k, v in meta.items()}},
        )  # fmt: skip
        loaded, config = load_pretrained(tmp_path / "other.safetensors", device="cpu")
    assert factory.call_args.args[1] == {"series": "old-revision", "text": TextEncoder.revision}
    assert config.signal_skip and config.series_lora_layers == 1 and not loaded.training
    assert loaded.temperatures == {"choice": 1.25, "noul": 0.9}
    series = torch.randn(2, 12)
    options = ["rising", "falling"]
    with patch("src.pretrained.build_model", build):
        loaded, _ = load_pretrained(out, device="cpu")
    torch.testing.assert_close(
        loaded(series[None], ["is it ?"], options=options),
        model(series[None], ["is it ?"], options=options),
    )
    assert answer(loaded, series, "is it ?", options) == answer(model, series, "is it ?", options)


def test_loading_rejects_a_file_missing_trained_tensors(tmp_path, trained):
    build, model = trained
    write_checkpoint(tmp_path / "m.pt", model)
    with patch("src.export.build_model", build):
        out = export_checkpoint(tmp_path / "m.pt", tmp_path / "m.safetensors")
    state, meta = read_export(out)
    del state[next(k for k in state if k.startswith("fusion."))]
    json = __import__("json")
    save_file(
        state, tmp_path / "bad.safetensors",
        metadata={"justin_format": "1", **{k: json.dumps(v) for k, v in meta.items()}},
    )  # fmt: skip
    with patch("src.pretrained.build_model", build), pytest.raises(ValueError, match="does not match"):
        load_pretrained(tmp_path / "bad.safetensors", device="cpu")


def test_read_export_rejects_foreign_files(tmp_path):
    save_file({"w": torch.zeros(1)}, tmp_path / "x.safetensors")
    with pytest.raises(ValueError, match="not a Justin export"):
        read_export(tmp_path / "x.safetensors")


def test_export_command(tmp_path, trained):
    build, model = trained
    write_checkpoint(tmp_path / "m.pt", model)
    with patch("src.export.build_model", build):
        result = CliRunner().invoke(
            main, ["export", str(tmp_path / "m.pt"), str(tmp_path / "m.safetensors")]
        )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "m.safetensors").exists() and (tmp_path / "m.json").exists()
    bad = CliRunner().invoke(main, ["export", str(tmp_path / "m.pt"), str(tmp_path / "m.bin")])
    assert bad.exit_code != 0 and "safetensors" in bad.output
