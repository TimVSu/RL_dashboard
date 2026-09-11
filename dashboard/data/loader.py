"""Read-only access to recorded runs on disk.

This is the only module that touches the filesystem. Everything above it
(pages/components/callbacks) works with pandas DataFrames and the typed
models in dashboard/data/models.py - never with raw file paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from dashboard.data.cache import cached
from dashboard.data.models import RunMetadata
from dashboard.data.schema import (
    OPERATOR_METRIC_COLUMNS,
    SCHEMA_VERSION,
    STATION_COLUMNS,
    TRAINING_METRIC_COLUMNS,
    VEHICLE_COLUMNS,
    SchemaError,
    validate_columns,
)


def list_runs(runs_dir: Path) -> list[str]:
    """Run ids found under runs_dir, i.e. subdirectories with a metadata.json."""

    if not runs_dir.exists():
        return []
    return sorted(
        p.name for p in runs_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()
    )


@cached(maxsize=16)
def load_metadata(runs_dir: Path, run_id: str) -> RunMetadata:
    path = runs_dir / run_id / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No metadata.json found for run '{run_id}' at {path}")
    with path.open() as f:
        data = json.load(f)
    metadata = RunMetadata.from_dict(data)
    if metadata.schema_version != SCHEMA_VERSION:
        raise SchemaError(
            f"Run '{run_id}' was recorded with schema_version="
            f"{metadata.schema_version!r}, but this dashboard expects "
            f"{SCHEMA_VERSION!r}. Regenerate the run or update "
            "dashboard/data/schema.py."
        )
    return metadata


def _read_partitioned(path: Path, epoch: Optional[int], run_id: str, table: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Run '{run_id}' is missing expected table '{table}' at {path}")
    filters = [("epoch", "=", epoch)] if epoch is not None else None
    return pd.read_parquet(path, filters=filters)


@cached(maxsize=64)
def load_vehicle_states(runs_dir: Path, run_id: str, epoch: int) -> pd.DataFrame:
    df = _read_partitioned(
        runs_dir / run_id / "vehicle_states.parquet", epoch, run_id, "vehicle_states"
    )
    validate_columns(df, VEHICLE_COLUMNS, table="vehicle_states", run_id=run_id)
    return df


@cached(maxsize=64)
def load_station_states(runs_dir: Path, run_id: str, epoch: int) -> pd.DataFrame:
    df = _read_partitioned(
        runs_dir / run_id / "station_states.parquet", epoch, run_id, "station_states"
    )
    validate_columns(df, STATION_COLUMNS, table="station_states", run_id=run_id)
    return df


@cached(maxsize=64)
def load_operator_metrics(
    runs_dir: Path, run_id: str, epoch: Optional[int] = None
) -> pd.DataFrame:
    df = _read_partitioned(
        runs_dir / run_id / "operator_metrics.parquet", epoch, run_id, "operator_metrics"
    )
    validate_columns(df, OPERATOR_METRIC_COLUMNS, table="operator_metrics", run_id=run_id)
    return df


@cached(maxsize=8)
def load_training_metrics(runs_dir: Path, run_id: str) -> pd.DataFrame:
    path = runs_dir / run_id / "training_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run '{run_id}' is missing training_metrics.parquet at {path}")
    df = pd.read_parquet(path)
    validate_columns(df, TRAINING_METRIC_COLUMNS, table="training_metrics", run_id=run_id)
    return df


def at_timestep(df: pd.DataFrame, rollout_id: int, timestep: int) -> pd.DataFrame:
    """Rows for one rollout at one timestep. Used for map/fleet/station views."""

    return df[(df["rollout_id"] == rollout_id) & (df["timestep"] == timestep)]


def for_rollout(df: pd.DataFrame, rollout_id: int) -> pd.DataFrame:
    """All timesteps for one rollout. Used for selected-vehicle/station history."""

    return df[df["rollout_id"] == rollout_id]
