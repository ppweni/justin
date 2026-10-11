"""Tests for src/persistence.Store."""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from src.config import Config
from src.persistence import Store


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "training.db"


@pytest.fixture
def store(db_path):
    return Store(db_path)


def _tables(db_path):
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def test_creates_db_and_tables(db_path):
    Store(db_path)
    assert db_path.exists()
    assert {"runs", "epochs"} <= _tables(db_path)


def test_creates_parent_directories(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "training.db"
    Store(deep)
    assert deep.exists()


def test_create_run_returns_positive_integer(store):
    config = asdict(Config())
    run_id = store.create_run(config)
    assert isinstance(run_id, int) and run_id > 0


def test_create_run_inserts_row(store, db_path):
    config = asdict(Config(dataset="tsqa"))
    run_id = store.create_run(config)
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT config_json, finished_at FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    assert json.loads(row[0])["dataset"] == "tsqa"
    assert row[1] is None  # not finished yet


def test_multiple_runs_have_distinct_ids(store):
    config = asdict(Config())
    ids = [store.create_run(config) for _ in range(3)]
    assert len(set(ids)) == 3


def test_record_epoch_inserts_row(store, db_path):
    run_id = store.create_run(asdict(Config()))
    metrics = {"train_loss": 1.0, "validation_loss": 0.8, "validation_accuracy": 0.6}
    store.record_epoch(run_id, 1, metrics, is_best=True, checkpoint_name="runs/joint-run0001-ep001.pt")
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT train_loss, validation_loss, validation_accuracy, is_best, checkpoint_name "
            "FROM epochs WHERE run_id = ? AND epoch_number = 1",
            (run_id,),
        ).fetchone()
    assert row == (1.0, 0.8, 0.6, 1, "runs/joint-run0001-ep001.pt")


def test_record_epoch_non_best_has_no_checkpoint(store, db_path):
    run_id = store.create_run(asdict(Config()))
    metrics = {"train_loss": 1.0, "validation_loss": 0.9, "validation_accuracy": 0.5}
    store.record_epoch(run_id, 1, metrics, is_best=False)
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT is_best, checkpoint_name FROM epochs WHERE run_id = ?", (run_id,)
        ).fetchone()
    assert row == (0, None)


def test_record_epoch_stores_full_metrics_json(store, db_path):
    run_id = store.create_run(asdict(Config()))
    metrics = {
        "train_loss": 1.0,
        "validation_loss": 0.8,
        "validation_accuracy": 0.6,
        "validation": {"har": {"loss": 0.7, "accuracy": 0.65}},
    }
    store.record_epoch(run_id, 1, metrics, is_best=False)
    with sqlite3.connect(db_path) as con:
        raw = con.execute("SELECT metrics_json FROM epochs WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert json.loads(raw) == metrics


def test_duplicate_epoch_raises(store):
    run_id = store.create_run(asdict(Config()))
    metrics = {"train_loss": 1.0, "validation_loss": 0.8, "validation_accuracy": 0.6}
    store.record_epoch(run_id, 1, metrics, is_best=False)
    with pytest.raises(sqlite3.IntegrityError):
        store.record_epoch(run_id, 1, metrics, is_best=False)


def test_finish_run_sets_fields(store, db_path):
    run_id = store.create_run(asdict(Config()))
    store.finish_run(run_id, "runs/joint.pt")
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT finished_at, checkpoint_path FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
    assert row[0] is not None  # finished_at set
    assert row[1] == "runs/joint.pt"


def test_checkpoint_name_for_derives_correct_path():
    base = Path("runs/joint.pt")
    result = Store.checkpoint_name_for(1, 5, base)
    assert result == Path("runs/joint-run0001-ep005.pt")


def test_checkpoint_name_for_pads_ids():
    base = Path("runs/joint.pt")
    assert Store.checkpoint_name_for(12, 3, base) == Path("runs/joint-run0012-ep003.pt")
    assert Store.checkpoint_name_for(1000, 100, base) == Path("runs/joint-run1000-ep100.pt")


def test_checkpoint_name_for_preserves_directory_and_suffix():
    base = Path("some/deep/dir/model.pt")
    result = Store.checkpoint_name_for(2, 7, base)
    assert result.parent == Path("some/deep/dir")
    assert result.suffix == ".pt"


# --- Integration: run() with db active ---

def test_run_records_epochs_and_unique_checkpoints(
    tmp_path, har_root, make_chronos2_encoder, monkeypatch
):
    """run() with config.db set writes DB rows and unique checkpoint files."""
    import torch

    from src.config import Config
    from src.data import DATASETS, HARDataset
    from src.model import SystemOne
    from src.training.run import run

    db_path = tmp_path / "training.db"
    output = tmp_path / "model.pt"

    config = Config(
        dataset="har",
        root=str(har_root),
        num_classes=8,
        with_text=False,
        hidden_dim=16,
        epochs=2,
        patience=2,
        batch_size=3,
        output=str(output),
        db=str(db_path),
    )

    model = SystemOne(make_chronos2_encoder(), None, num_classes=8, hidden_dim=16)
    monkeypatch.setattr(SystemOne, "from_pretrained", lambda **kw: model)

    def load_dataset(root, *, split, download):
        return HARDataset(root, split=split, download=download)

    monkeypatch.setitem(DATASETS, "har", load_dataset)
    run(config)

    # DB assertions
    assert db_path.exists()
    with sqlite3.connect(db_path) as con:
        runs = con.execute("SELECT id, finished_at, checkpoint_path FROM runs").fetchall()
        epochs = con.execute(
            "SELECT epoch_number, is_best, checkpoint_name FROM epochs ORDER BY epoch_number"
        ).fetchall()

    assert len(runs) == 1
    run_row = runs[0]
    assert run_row[1] is not None  # finished_at set
    assert run_row[2] == str(output)  # final checkpoint path

    assert len(epochs) == 2
    best_epochs = [e for e in epochs if e[1] == 1]
    assert len(best_epochs) >= 1
    for e in best_epochs:
        cp = Path(e[2])
        assert cp.exists()
        assert cp != output  # unique name, not the final output

    # Final checkpoint exists with temperatures
    assert output.exists()
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert "temperatures" in checkpoint
    assert "history" in checkpoint
    assert len(checkpoint["history"]) == 2
