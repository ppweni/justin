from collections.abc import Iterable
from math import isfinite, log

import torch
from torch import Tensor

from ..model import SystemOne
from ..model.model import Head
from .batch import model_logits, prepare_batch
from .loss import negative_log_likelihood


@torch.enable_grad()
def _fit_temperature(batches: list[tuple[Tensor, Tensor]], head: Head) -> float:
    """Fit a scalar on cached CPU logits, weighted by samples, not batches."""
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    count = sum(target.numel() for _, target in batches)

    def nll(temperature: Tensor | float) -> Tensor:
        return (
            torch.stack(
                [
                    negative_log_likelihood(logits / temperature, target, head) * target.numel()
                    for logits, target in batches
                ]
            ).sum()
            / count
        )

    best_temperature = 1.0
    best_loss = nll(1.0).item()
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        nonlocal best_temperature, best_loss
        optimizer.zero_grad()
        # Bounds prevent runaway scales on perfectly separable or uninformative data.
        temperature = log_temperature.clamp(log(0.01), log(100.0)).exp().clamp(0.01, 100.0)
        loss = nll(temperature)
        loss.backward()
        if loss.item() < best_loss:
            best_temperature, best_loss = temperature.item(), loss.item()
        return loss

    optimizer.step(closure)
    return best_temperature


def fit_temperatures(model: SystemOne, batches: Iterable[dict]) -> dict[Head, float]:
    """Fit one temperature per observed head on held-out validation batches.

    Cache logits once in CPU float64; only the scalar temperatures are optimized.
    Model weights, gradients, and mode are preserved. Choice batches may have
    different option counts. Call separately for each deployment dataset/domain.

    Temperatures stay in [0.01, 100] and never worsen fitted NLL versus T=1.
    Save the returned mapping with the checkpoint; use probabilities at inference.
    Evaluate calibration on a separate split from the one used for fitting.
    """
    cached: dict[Head, list[tuple[Tensor, Tensor]]] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                inputs, target = prepare_batch(model, batch)
                head = inputs.get("head", "choice")
                # MPS has no float64 support; move first, then widen on CPU.
                logits = model_logits(model, inputs).detach().cpu().double()
                if not logits.isfinite().all():
                    raise ValueError("calibration logits must be finite")
                cached.setdefault(head, []).append((logits, target.detach().cpu()))
    finally:
        model.train(was_training)
    if not cached:
        raise ValueError("calibration batches must not be empty")
    return {head: _fit_temperature(values, head) for head, values in cached.items()}


def probabilities(logits: Tensor, head: Head, temperature: float = 1.0) -> Tensor:
    """Apply a fitted temperature to raw logits; Noul returns P(yes)."""
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Scale in at least float32 so sharpening cannot overflow half-precision logits.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    scaled = logits / temperature
    if head == "noul":
        return scaled.sigmoid()
    if head in ("choice", "score"):
        return scaled.softmax(dim=-1)
    raise ValueError(f"unknown head: {head}")
