"""Minimal LoRA adapters for selectively unfreezing frozen backbones.

A LoRALinear wraps a frozen nn.Linear with a trainable low-rank delta
(y = base(x) + (alpha/rank) * B(dropout(A(x)))). B is zero-initialized so the
adapter is the identity at start and training begins from the frozen behavior.
inject_lora swaps matching nn.Linear modules in place, so the surrounding module
keeps working unchanged while only the low-rank A/B parameters are trainable.
"""

from collections.abc import Callable

from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if rank < 1 or alpha < 1:
            raise ValueError("rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)  # start as identity: delta = 0

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scaling * self.b(self.dropout(self.a(x.to(self.a.weight.dtype))))


def inject_lora(
    root: nn.Module,
    predicate: Callable[[str], bool],
    *,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
) -> int:
    """Replace every nn.Linear whose qualified name passes `predicate` with a LoRALinear.

    Returns the number of layers adapted. Frozen bases stay frozen; only the
    injected low-rank parameters become trainable.
    """
    targets = [
        name
        for name, module in root.named_modules()
        if isinstance(module, nn.Linear) and predicate(name)
    ]
    for name in targets:
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        attr = name.rsplit(".", 1)[-1]
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    return len(targets)
