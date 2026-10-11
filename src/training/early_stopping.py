from math import isfinite

from torch import Tensor, nn


class EarlyStopping:
    """Minimize validation loss; ties count as checks without improvement."""

    def __init__(self, patience: int):
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.best_state: dict[str, Tensor] | None = None

    def update(self, model: nn.Module, loss: float) -> bool:
        """Snapshot each strict improvement on CPU; return whether it improved."""
        if not isfinite(loss):
            raise ValueError("validation loss must be finite")
        if loss < self.best_loss:
            self.best_loss = loss
            self.bad_epochs = 0
            self.best_state = {
                name: value.detach().to(device="cpu", copy=True)
                for name, value in model.state_dict().items()
            }
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("no best model has been recorded")
        model.load_state_dict(self.best_state)
