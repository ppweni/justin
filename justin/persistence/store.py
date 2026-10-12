"""SQLite-backed store for training run and epoch history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    finished_at     TEXT,
    checkpoint_path TEXT
);

CREATE TABLE IF NOT EXISTS epochs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL REFERENCES runs(id),
    epoch_number        INTEGER NOT NULL,
    recorded_at         TEXT NOT NULL,
    train_loss          REAL NOT NULL,
    validation_loss     REAL NOT NULL,
    validation_accuracy REAL NOT NULL,
    metrics_json        TEXT NOT NULL,
    is_best             INTEGER NOT NULL DEFAULT 0,
    checkpoint_name     TEXT,
    UNIQUE(run_id, epoch_number)
);
"""


class Store:
    """Persist run and epoch records to a SQLite database.

    The database file and its parent directories are created automatically.
    All writes use WAL mode for safe concurrent reads.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as con:
            con.executescript(_DDL)
            con.execute("PRAGMA journal_mode=WAL")

    def create_run(self, config: dict) -> int:
        """Insert a new run row and return its id."""
        with sqlite3.connect(self._path) as con:
            cur = con.execute(
                "INSERT INTO runs (started_at, config_json) VALUES (?, ?)",
                (_now(), json.dumps(config)),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def record_epoch(
        self,
        run_id: int,
        epoch_number: int,
        metrics: dict,
        *,
        is_best: bool,
        checkpoint_name: str | None = None,
    ) -> None:
        """Insert one epoch row.

        metrics must contain train_loss, validation_loss, validation_accuracy.
        The full metrics dict is stored as JSON for per-dataset breakdowns.
        """
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                INSERT INTO epochs
                    (run_id, epoch_number, recorded_at,
                     train_loss, validation_loss, validation_accuracy,
                     metrics_json, is_best, checkpoint_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    epoch_number,
                    _now(),
                    metrics["train_loss"],
                    metrics["validation_loss"],
                    metrics["validation_accuracy"],
                    json.dumps(metrics),
                    int(is_best),
                    checkpoint_name,
                ),
            )

    def finish_run(self, run_id: int, checkpoint_path: str) -> None:
        """Mark the run as finished and record the final checkpoint path."""
        with sqlite3.connect(self._path) as con:
            con.execute(
                "UPDATE runs SET finished_at = ?, checkpoint_path = ? WHERE id = ?",
                (_now(), checkpoint_path, run_id),
            )

    @staticmethod
    def checkpoint_name_for(run_id: int, epoch: int, base_path: Path) -> Path:
        """Derive a unique checkpoint path from base_path, run_id, and epoch.

        Example: base_path=runs/joint.pt, run_id=1, epoch=5
                 -> runs/joint-run0001-ep005.pt
        """
        stem = f"{base_path.stem}-run{run_id:04d}-ep{epoch:03d}"
        return base_path.with_name(stem + base_path.suffix)
