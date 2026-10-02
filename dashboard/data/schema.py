"""On-disk schema contract between the recorder (urban-mobility-simulation)
and this dashboard.

The two projects are separate repos with no shared code, so this file - plus
the matching `schema_version` recorded in each run's metadata.json - is the
entire integration contract. If the recorder's output changes shape, bump
SCHEMA_VERSION here and there together; a mismatch fails loudly instead of
silently misreading columns (requirements.md section 54: "fail clearly if
required data is missing").
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = "0.1.0"

# Column -> pandas dtype (informational; validated by presence, dtype
# mismatches surface naturally when the UI does numeric/boolean ops on them).
VEHICLE_COLUMNS: dict[str, str] = {
    "run_id": "object",
    "epoch": "int64",
    "rollout_id": "int64",
    "timestep": "int64",
    "operator_id": "int64",
    "vehicle_id": "int64",
    "soc": "float64",
    "last_known_position_h3": "object",
    "destination_h3": "object",
    "position_h3": "object",
    "position_known": "bool",
    "has_reservation": "bool",
    "reservation_start": "float64",  # nullable int -> float w/ NaN
    "reservation_end": "float64",
    "has_commitment": "bool",
    "committed_station": "float64",  # nullable int -> float w/ NaN
    "allocation": "int64",
    "is_dummy_assignment": "bool",
    "bid_price": "float64",
    "is_bid": "bool",
}

STATION_COLUMNS: dict[str, str] = {
    "run_id": "object",
    "epoch": "int64",
    "rollout_id": "int64",
    "timestep": "int64",
    "station_id": "int64",
    "h3": "object",
    "capacity": "int64",
    "occupancy": "int64",
    "price": "float64",
    "num_bids": "int64",
    "num_winning_bids": "int64",
}

OPERATOR_METRIC_COLUMNS: dict[str, str] = {
    "run_id": "object",
    "epoch": "int64",
    "rollout_id": "int64",
    "timestep": "int64",
    "operator_id": "int64",
    "reward": "float64",
    "value": "float64",
}

TRAINING_METRIC_COLUMNS: dict[str, str] = {
    "run_id": "object",
    "epoch": "int64",
    "operator_id": "int64",
    "total_reward": "float64",
    "mean_reward": "float64",
    "mean_value": "float64",
}

# reward_components.parquet is optional: it was added after operator_metrics,
# so runs recorded before then (e.g. demo_run, synthetic_week_run) simply
# don't have this table. loader.load_reward_components() returns an empty
# frame with these columns rather than raising when it's missing - callers
# should treat "empty" and "not recorded for this run" the same way, and
# show that state in the UI rather than erroring.
REWARD_COMPONENT_COLUMNS: dict[str, str] = {
    "run_id": "object",
    "epoch": "int64",
    "rollout_id": "int64",
    "timestep": "int64",
    "operator_id": "int64",
    "reward": "float64",
    "earnings": "float64",
    "cost": "float64",
    "winnings_factor": "float64",
    "penalty_factor": "float64",
    "low_soc_penalty": "float64",
    "cost_penalty": "float64",
}


class SchemaError(RuntimeError):
    """A recorded run does not match the schema this dashboard expects."""


def validate_columns(
    df: pd.DataFrame, expected: dict[str, str], *, table: str, run_id: str
) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Run '{run_id}': table '{table}' is missing required column(s) "
            f"{missing}. Expected columns: {sorted(expected)}. Check that "
            f"the recorder producing this run matches dashboard "
            f"schema_version={SCHEMA_VERSION!r}."
        )
