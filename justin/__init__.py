"""Justin: time-series question answering on a frozen Chronos-2 encoder."""

from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .pretrained import Justin

__all__ = ["Config", "Justin"]


def __getattr__(name: str):
    # Loaded on first use so `import justin` and the CLI do not import torch/transformers.
    if name == "Justin":
        from .pretrained import Justin

        return Justin
    raise AttributeError(f"module 'justin' has no attribute {name!r}")
