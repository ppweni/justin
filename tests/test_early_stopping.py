from importlib import import_module
from pathlib import Path

import pytest
import torch
from torch import nn

from src import Config
from src.model import SystemOne
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.run import run
from src.training.train import train


@pytest.mark.parametrize(
    "losses,patience,improvements,best",
    [
        ([3, 2, 2, 2], 2, [True, True, False, False], 2),
        ([3, 4, 2, 2, 3], 2, [True, False, True, False, False], 2),
        ([3, 3], 1, [True, False], 3),
    ],
)
def test_patience_counts_consecutive_non_improvements(losses, patience, improvements, best):
    stopping = EarlyStopping(patience)
    model = nn.Linear(1, 1)
    for i, (loss, improved) in enumerate(zip(losses, improvements, strict=True)):
        assert stopping.update(model, loss) == improved
        assert stopping.should_stop == (i == len(losses) - 1)
    assert stopping.best_loss == best


@pytest.mark.parametrize("patience", [0, -1, True, 1.5])
def test_invalid_patience(patience):
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_validation_cannot_replace_best(loss):
    stopping = EarlyStopping(2)
    model = nn.Linear(1, 1)
    stopping.update(model, 1.0)
    before = stopping.best_state
    with pytest.raises(ValueError, match="finite"):
        stopping.update(model, loss)
    assert stopping.best_state is before
    assert stopping.best_loss == 1.0 and stopping.bad_epochs == 0


def test_snapshot_is_independent_and_restores_parameters_and_buffers():
    model = nn.Linear(1, 1)
    model.register_buffer("counter", torch.tensor(7))
    original = {name: value.clone() for name, value in model.state_dict().items()}
    stopping = EarlyStopping(2)
    with pytest.raises(RuntimeError, match="no best model"):
        stopping.restore(model)
    stopping.update(model, 1.0)
    assert stopping.best_state is not None
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for value in stopping.best_state.values()
    )
    with torch.no_grad():
        model.weight.fill_(50)
        model.bias.fill_(-50)
        model.get_buffer("counter").fill_(9)
    stopping.update(model, 2.0)
    stopping.restore(model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_checkpoint_publication_is_exclusive_and_replaceable(tmp_path):
    path = tmp_path / "nested" / "best.pt"
    save_checkpoint(path, {"epoch": 1, "model": torch.ones(2)})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, {"epoch": 2})
    assert torch.load(path, weights_only=True)["epoch"] == 1
    save_checkpoint(path, {"epoch": 3}, overwrite=True)
    assert torch.load(path, weights_only=True) == {"epoch": 3}
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
def test_failed_checkpoint_write_does_not_publish_partial_data(tmp_path, monkeypatch, existing):
    path = tmp_path / "best.pt"
    if existing:
        save_checkpoint(path, {"epoch": 1})

    def broken_save(checkpoint, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("src.training.checkpoint.torch.save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, {"epoch": 2}, overwrite=existing)
    if existing:
        assert torch.load(path, weights_only=True) == {"epoch": 1}
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


@pytest.mark.parametrize(
    "losses,epochs,patience,expected_epochs,best_epoch",
    [
        ([3, 2, 2, 4, 1], 5, 2, 4, 2),
        ([3, 3, 1], 3, 1, 2, 1),
        ([3, 2, 1], 3, 2, 3, 3),
        ([1, 2, 3], 2, 5, 2, 1),
    ],
)
def test_train_stops_and_restores_best_even_at_epoch_limit(
    model,
    monkeypatch,
    losses,
    epochs,
    patience,
    expected_epochs,
    best_epoch,
):
    module = import_module("src.training.train")
    updates = []
    saved = []

    def epoch(model, batches, optimizer, **kwargs):
        updates.append(1)
        with torch.no_grad():
            model.get_parameter("fusion.query_projection.weight").fill_(len(updates))
        return 0.5, 1

    def validate(model, batches):
        # Selection is by balanced accuracy; make it rank inversely to loss so the
        # parametrized best epoch (lowest loss) is also the highest-balanced one.
        loss = losses[len(updates) - 1]
        return {"loss": loss, "accuracy": 0.5, "balanced_accuracy": -loss}

    def save(state, history):
        saved.append((len(history), state["fusion.query_projection.weight"].clone()))

    monkeypatch.setattr(module, "train_epoch", epoch)
    monkeypatch.setattr(module, "validate", validate)
    history = train(model, [], [], Config(epochs=epochs, patience=patience), on_improvement=save)
    assert len(history) == len(updates) == expected_epochs
    assert history[-1]["validation_loss"] == losses[expected_epochs - 1]
    assert torch.all(model.get_parameter("fusion.query_projection.weight") == best_epoch)
    assert saved[-1][0] == best_epoch
    for epoch_number, weights in saved:
        assert torch.all(weights == epoch_number)


def test_run_saves_during_training_selects_macro_loss_and_calibrates_best(
    har_root,
    sleep_root,
    tmp_path,
    make_chronos2_encoder,
    text_encoder,
    monkeypatch,
):
    model = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16)
    config = Config(
        dataset=["har", "sleep"],
        root=str(har_root),
        epochs=8,
        patience=2,
        output=str(tmp_path / "best.pt"),
    )
    path = Path(config.output)
    epochs = []
    calibrated = []
    # HAR alone prefers the last epoch, but the macro loss is lowest at epoch 2.
    losses = [(0.4, 1.6), (0.6, 1.0), (0.2, 1.8), (0.1, 2.1)]

    def epoch(model, batches, optimizer, **kwargs):
        if epochs:
            best = torch.load(path, weights_only=True)
            assert best["best_epoch"] == min(len(epochs), 2)
            assert best["temperatures"] == {}
            assert torch.all(
                best["model"]["fusion.query_projection.weight"] == best["best_epoch"]
            )
        epochs.append(1)
        with torch.no_grad():
            model.get_parameter("fusion.query_projection.weight").fill_(len(epochs))
        return 0.5, 1

    def validate(model, loaders):
        # Balanced accuracy drives selection; set it inverse to loss so the
        # best macro-balanced epoch coincides with the lowest macro-loss epoch (2).
        return {
            name: {"loss": loss, "accuracy": 0.5, "balanced_accuracy": -loss}
            for name, loss in zip(loaders, losses[len(epochs) - 1], strict=True)
        }

    def calibrate(model, batches):
        assert torch.all(model.get_parameter("fusion.query_projection.weight") == 2)
        assert torch.load(path, weights_only=True)["best_epoch"] == 2
        calibrated.append(True)
        return {"choice": 1.5}

    module = import_module("src.training.train")
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    monkeypatch.setattr(module, "train_epoch", epoch)
    monkeypatch.setattr(module, "validate_datasets", validate)
    monkeypatch.setattr("src.training.run.fit_temperatures", calibrate)
    assert run(config) == path
    best = torch.load(path, weights_only=True)
    assert len(epochs) == len(best["history"]) == 4
    assert best["best_epoch"] == 2
    assert best["history"][1]["validation_loss"] == pytest.approx(0.8)
    assert calibrated == [True]  # calibrate called once over all validation data
    assert best["temperatures"] == {"choice": 1.5}
    assert torch.all(best["model"]["fusion.query_projection.weight"] == 2)


@pytest.mark.parametrize("failure", ["training", "calibration"])
def test_best_checkpoint_survives_later_failure(
    har_root,
    tmp_path,
    make_chronos2_encoder,
    text_encoder,
    monkeypatch,
    failure,
):
    config = Config(dataset="har", root=str(har_root), epochs=2, output=str(tmp_path / "best.pt"))
    model = SystemOne(make_chronos2_encoder(), text_encoder, hidden_dim=16)
    module = import_module("src.training.train")
    real_epoch = module.train_epoch
    calls = 0

    def epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and failure == "training":
            raise RuntimeError("training failed")
        return real_epoch(*args, **kwargs)

    def calibrate(*args):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kwargs: model)
    monkeypatch.setattr(module, "train_epoch", epoch)
    monkeypatch.setattr("src.training.run.fit_temperatures", calibrate)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        run(config)
    checkpoint = torch.load(config.output, weights_only=True)
    assert checkpoint["best_epoch"] >= 1 and checkpoint["temperatures"] == {}
    assert torch.isfinite(checkpoint["model"]["fusion.query_projection.weight"]).all()
