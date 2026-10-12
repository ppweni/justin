"""Export the trained part of a checkpoint as a small safetensors file.

A training checkpoint also holds the frozen Chronos-2 and MiniLM weights. The export keeps only what
this project trained (fusion, heads, ``signal_skip``, LoRA adapters), plus the model-defining config,
the fitted temperatures, and the exact Hub revisions of the frozen backbones, so loading rebuilds the
same model. Every dropped tensor is checked against the pinned backbone, so a wrong pin fails here
instead of silently changing results.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import Config
from .model import SystemOne
from .model.encoders import SERIES_ENCODERS, TextEncoder
from .training.run import build_model

FORMAT_VERSION = "1"
FROZEN_PREFIXES = ("series_encoder.", "text_encoder.")
# The fields that define the model; paths, devices and optimizer settings are run artifacts.
MODEL_FIELDS = (
    "dataset", "num_classes", "score_levels", "hidden_dim", "fusion_layers", "dropout",
    "with_text", "option_attention", "signal_skip", "series_encoder", "series_lora_layers",
    "minilm_lora", "lora_rank", "lora_alpha",
)  # fmt: skip


def backbone_revisions(config: Config) -> dict[str, dict[str, str]]:
    """Repo and pinned revision of every frozen backbone the config uses."""
    encoder = SERIES_ENCODERS[config.series_encoder]
    revisions = {"series": {"repo": encoder.repo, "revision": encoder.revision}}
    if config.with_text:
        revisions["text"] = {"repo": TextEncoder.repo, "revision": TextEncoder.revision}
    return revisions


def trained_keys(model: SystemOne, state: dict[str, torch.Tensor]) -> list[str]:
    """State-dict keys that are not just a copy of a frozen pretrained backbone."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return [k for k in state if k in trainable or not k.startswith(FROZEN_PREFIXES)]


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> Path:
    """Write ``output`` (.safetensors, plus a ``.json`` with the same metadata) from a ``.pt`` checkpoint."""
    output = Path(output)
    if output.suffix != ".safetensors":
        raise ValueError("output must end in .safetensors")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = saved["config"]
    config = Config(**{key: raw[key] for key in MODEL_FIELDS if key in raw})
    revisions = backbone_revisions(config)
    pinned = build_model(config, {name: info["revision"] for name, info in revisions.items()})

    state = saved["model"]
    keep = set(trained_keys(pinned, state))
    reference = pinned.state_dict()
    changed = [
        key for key in state if key not in keep and not torch.equal(state[key], reference[key])
    ]
    if changed:
        raise ValueError(
            f"{len(changed)} frozen tensors differ from the pinned backbones "
            f"(first: {changed[0]}); the checkpoint was trained on other revisions"
        )

    metadata = {
        "justin_format": FORMAT_VERSION,
        "config": json.dumps({key: asdict(config)[key] for key in MODEL_FIELDS}),
        "revisions": json.dumps(revisions),
        "temperatures": json.dumps(saved.get("temperatures", {})),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: state[key].contiguous() for key in sorted(keep)}, output, metadata=metadata)
    output.with_suffix(".json").write_text(
        json.dumps({k: json.loads(v) if k != "justin_format" else v for k, v in metadata.items()}, indent=2)
    )
    return output


def read_export(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """The tensors and the decoded metadata (config, revisions, temperatures) of an export."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        raw = file.metadata() or {}
        if raw.get("justin_format") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a Justin export (format {raw.get('justin_format')})")
        state = {key: file.get_tensor(key) for key in file.keys()}  # noqa: SIM118
    return state, {k: json.loads(v) for k, v in raw.items() if k != "justin_format"}
