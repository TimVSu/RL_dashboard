"""Typed, in-repo data model for recorded runs.

These types describe what this dashboard reads and expects, independently
of whatever internal representation the recorder in urban-mobility-simulation
uses. Keeping this model self-contained (rather than importing that repo's
classes) is what keeps the two projects decoupled - the only shared contract
is the on-disk schema in dashboard/data/schema.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd

PositionStatus = Literal["known", "estimated"]


def _opt_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return int(value)


@dataclass(frozen=True)
class OperatorInfo:
    operator_id: int
    name: str


@dataclass(frozen=True)
class RunMetadata:
    """Parsed contents of runs/<run_id>/metadata.json.

    This is the single source of truth for counts/indices that must never
    be hard-coded in the dashboard (requirements.md section 58).
    """

    run_id: str
    schema_version: str
    created_at: str
    description: str
    num_epochs: int
    num_parallel_envs: int
    steps_per_rollout: int
    num_vehicles: int
    num_stations: int
    num_operators: int
    dummy_station_index: int
    h3_resolution: int
    operators: list[OperatorInfo]
    config: dict
    seed: Optional[int] = None
    recording_level: str = "full"
    is_evaluation: bool = False

    @property
    def epoch_ids(self) -> list[int]:
        return list(range(self.num_epochs))

    @property
    def rollout_ids(self) -> list[int]:
        return list(range(self.num_parallel_envs))

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        required = [
            "run_id",
            "schema_version",
            "created_at",
            "description",
            "num_epochs",
            "num_parallel_envs",
            "steps_per_rollout",
            "num_vehicles",
            "num_stations",
            "num_operators",
            "dummy_station_index",
            "h3_resolution",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"metadata.json for run '{data.get('run_id', '?')}' is missing "
                f"required field(s): {missing}"
            )
        operators = [OperatorInfo(**op) for op in data.get("operators", [])]
        return cls(
            run_id=data["run_id"],
            schema_version=data["schema_version"],
            created_at=data["created_at"],
            description=data["description"],
            num_epochs=data["num_epochs"],
            num_parallel_envs=data["num_parallel_envs"],
            steps_per_rollout=data["steps_per_rollout"],
            num_vehicles=data["num_vehicles"],
            num_stations=data["num_stations"],
            num_operators=data["num_operators"],
            dummy_station_index=data["dummy_station_index"],
            h3_resolution=data["h3_resolution"],
            operators=operators,
            config=data.get("config", {}),
            seed=data.get("seed"),
            recording_level=data.get("recording_level", "full"),
            is_evaluation=data.get("is_evaluation", False),
        )


@dataclass(frozen=True)
class VehicleState:
    """One vehicle, at one (epoch, rollout, timestep)."""

    run_id: str
    epoch: int
    rollout_id: int
    timestep: int
    operator_id: int
    vehicle_id: int
    soc: float
    last_known_position_h3: str
    destination_h3: str
    position_h3: str
    position_known: bool
    has_reservation: bool
    reservation_start: Optional[int]
    reservation_end: Optional[int]
    has_commitment: bool
    committed_station: Optional[int]
    allocation: int
    is_dummy_assignment: bool
    bid_price: float
    is_bid: bool

    @property
    def position_status(self) -> PositionStatus:
        return "known" if self.position_known else "estimated"

    @classmethod
    def from_row(cls, row: pd.Series) -> "VehicleState":
        return cls(
            run_id=row["run_id"],
            epoch=int(row["epoch"]),
            rollout_id=int(row["rollout_id"]),
            timestep=int(row["timestep"]),
            operator_id=int(row["operator_id"]),
            vehicle_id=int(row["vehicle_id"]),
            soc=float(row["soc"]),
            last_known_position_h3=row["last_known_position_h3"],
            destination_h3=row["destination_h3"],
            position_h3=row["position_h3"],
            position_known=bool(row["position_known"]),
            has_reservation=bool(row["has_reservation"]),
            reservation_start=_opt_int(row["reservation_start"]),
            reservation_end=_opt_int(row["reservation_end"]),
            has_commitment=bool(row["has_commitment"]),
            committed_station=_opt_int(row["committed_station"]),
            allocation=int(row["allocation"]),
            is_dummy_assignment=bool(row["is_dummy_assignment"]),
            bid_price=float(row["bid_price"]),
            is_bid=bool(row["is_bid"]),
        )


@dataclass(frozen=True)
class StationState:
    """One charging station, at one (epoch, rollout, timestep)."""

    run_id: str
    epoch: int
    rollout_id: int
    timestep: int
    station_id: int
    h3: str
    capacity: int
    occupancy: int
    price: float
    num_bids: int
    num_winning_bids: int

    @classmethod
    def from_row(cls, row: pd.Series) -> "StationState":
        return cls(
            run_id=row["run_id"],
            epoch=int(row["epoch"]),
            rollout_id=int(row["rollout_id"]),
            timestep=int(row["timestep"]),
            station_id=int(row["station_id"]),
            h3=row["h3"],
            capacity=int(row["capacity"]),
            occupancy=int(row["occupancy"]),
            price=float(row["price"]),
            num_bids=int(row["num_bids"]),
            num_winning_bids=int(row["num_winning_bids"]),
        )


@dataclass(frozen=True)
class OperatorMetric:
    """Reward/value for one operator, at one (epoch, rollout, timestep)."""

    run_id: str
    epoch: int
    rollout_id: int
    timestep: int
    operator_id: int
    reward: float
    value: float


def is_real_assignment(allocation: int, dummy_station_index: int) -> bool:
    """True if `allocation` is a real charging-station assignment.

    This is the one place that should ever compare an allocation value
    against the dummy index (requirements.md sections 7, 42). Never infer
    "no bid" from bid_price == 0 alone.
    """

    return allocation != dummy_station_index


def format_allocation(allocation: int, dummy_station_index: int) -> str:
    if is_real_assignment(allocation, dummy_station_index):
        return f"Station {allocation}"
    return "No assignment"
