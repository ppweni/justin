from torch.optim import Optimizer

from ..model import SystemOne
from .batch import model_logits, prepare_batch
from .loss import compute_loss


def train_step(model: SystemOne, batch: dict, optimizer: Optimizer) -> float:
    """One update; return the mean loss before the update as a detached scalar."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, target = prepare_batch(model, batch)
    logits = model_logits(model, inputs)
    loss = compute_loss(logits, target, inputs.get("head", "choice"), inputs.get("weight"))
    loss.backward()
    optimizer.step()
    return loss.item()
