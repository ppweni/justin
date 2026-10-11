import torch
from torch import Tensor

from ..model import SystemOne


def prepare_batch(model: SystemOne, batch: dict) -> tuple[dict, Tensor]:
    """Move tensors to the model's device without modifying the collated batch."""
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
    target = inputs.pop("target")
    if not isinstance(target, Tensor) or target.ndim != 1 or target.numel() == 0:
        raise ValueError("target must be a nonempty tensor with one value per sample")
    if not torch.isfinite(target).all():
        raise ValueError("target must be finite")
    inputs.pop("dataset", None)
    if model.text_encoder is None:
        inputs.pop("questions", None)
        inputs.pop("options", None)
    return inputs, target


def model_logits(model: SystemOne, inputs: dict) -> Tensor:
    """Run the model, dispatching cached batches to the encoder-light paths.

    Signal comes from a cached final H_ts or a cached prefix + live LoRA suffix;
    text comes from cached token states or, when MiniLM is LoRA-adapted, live strings.
    """
    if "signal" in inputs or "signal_prefix" in inputs:
        features = model.series_features(inputs)
        if "questions" in inputs or "options" in inputs:
            return model.answer(
                features,
                inputs.get("questions"),
                options=inputs.get("options"),
                head=inputs["head"],
            )
        return model.answer_cached(
            features,
            inputs.get("q_tokens"),
            inputs.get("q_mask"),
            inputs.get("option_embeddings"),
            head=inputs["head"],
        )
    return model(**inputs)
