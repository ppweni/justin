import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def save_checkpoint(path: Path, checkpoint: dict, *, overwrite: bool = False) -> None:
    """Publish a complete checkpoint atomically; first save refuses existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(checkpoint, temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard-link publication is atomic and fails if another file appeared.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
