"""Reading the files `justin export` and `justin export-location` write (training repository)."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

FORMAT_VERSION = "1"
LOCATION_FORMAT_VERSION = "1"
FROZEN_PREFIXES = ("series_encoder.", "text_encoder.")


def read_export(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """The tensors and the decoded metadata (config, revisions, temperatures) of a model export."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        raw = file.metadata() or {}
        if raw.get("justin_format") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a Justin export (format {raw.get('justin_format')})")
        state = {key: file.get_tensor(key) for key in file.keys()}  # noqa: SIM118
    return state, {k: json.loads(v) for k, v in raw.items() if k != "justin_format"}


def read_location(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """The tensors and the decoded metadata (head spec, base fingerprint) of a location export."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        raw = file.metadata() or {}
        if raw.get("justin_location_format") != LOCATION_FORMAT_VERSION:
            raise ValueError(f"{path} is not a Justin location export")
        state = {key: file.get_tensor(key) for key in file.keys()}  # noqa: SIM118
    return state, {k: json.loads(v) for k, v in raw.items() if k != "justin_location_format"}


def fingerprint(path: str | Path) -> str:
    """sha256 over an export's tensors (sorted names, each name, a zero byte, its raw bytes); a
    location head records the fingerprint of the model export it was trained on."""
    digest = hashlib.sha256()
    with safe_open(str(path), framework="pt", device="cpu") as file:
        for key in sorted(file.keys()):  # noqa: SIM118
            digest.update(key.encode() + b"\0")
            digest.update(file.get_tensor(key).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
