import torch
from torch import Tensor
from torch.nn import functional as F

from ..model.model import Head


def _weighted_mean(per_sample: Tensor, weights: Tensor | None) -> Tensor:
    """Sample-weighted mean, or plain mean when no weights are given."""
    if weights is None:
        return per_sample.mean()
    weights = weights.to(per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def negative_log_likelihood(
    logits: Tensor, target: Tensor, head: Head, weights: Tensor | None = None
) -> Tensor:
    """Mean CE for Choice/Score indices; mean BCE for Noul probabilities.

    weights are optional per-sample loss weights (e.g. inverse class frequency).
    """
    if head == "noul":
        if ((target < 0) | (target > 1)).any():
            raise ValueError("Noul targets must be between 0 and 1")
        per_sample = F.binary_cross_entropy_with_logits(
            logits, target.to(logits.dtype), reduction="none"
        )
    elif head in ("choice", "score"):
        per_sample = F.cross_entropy(logits, target, reduction="none")
    else:
        raise ValueError(f"unknown head: {head}")
    loss = _weighted_mean(per_sample, weights)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    return loss


def ranked_probability_score(
    logits: Tensor, target: Tensor, weights: Tensor | None = None
) -> Tensor:
    """Mean squared CDF error over K-1 thresholds for ordered levels 0..K-1.

    Normalized to [0, 1]: a confident miss costs more the farther it is
    from the target. The final CDF entry is always one and is excluded.
    weights are optional per-sample loss weights.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("Score logits must have nonempty shape [batch, levels >= 2]")
    if target.shape != logits.shape[:1] or target.dtype != torch.long:
        raise ValueError("Score targets must be one integer level per sample")
    observed = F.one_hot(target, num_classes=logits.shape[-1]).to(logits.dtype)
    predicted_cdf = logits.softmax(dim=-1).cumsum(dim=-1)[:, :-1]
    observed_cdf = observed.cumsum(dim=-1)[:, :-1]
    per_sample = (predicted_cdf - observed_cdf).square().mean(dim=-1)
    return _weighted_mean(per_sample, weights)


def compute_loss(
    logits: Tensor, target: Tensor, head: Head, weights: Tensor | None = None
) -> Tensor:
    """Choice: CE. Noul: BCE. Score: CE + normalized ranked probability score.

    weights optionally reweights each sample (used for class-balanced training).
    """
    loss = negative_log_likelihood(logits, target, head, weights)
    if head == "score":
        loss = loss + ranked_probability_score(logits, target, weights)
    return loss
