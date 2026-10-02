"""Generate a small synthetic recorded run for local development and tests.

There is no recorder in this repo - it lives in urban-mobility-simulation.
This script produces data that matches exactly the on-disk schema documented
in dashboard/data/schema.py and README.md, so the dashboard can be built and
exercised end-to-end before any real recording exists.

The generated behavior is a plausible-looking heuristic (low SOC -> more
likely to bid; nearer/cheaper stations preferred; occasional multi-step
charging commitments; a mild "improves with epoch" trend), not a real
policy - it exists only to populate every field the UI needs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from dashboard.data.schema import SCHEMA_VERSION

CENTER_LAT, CENTER_LNG = 37.7749, -122.4194

# Mirror urban-mobility-simulation's GymWrapper._calculate_reward shape
# closely enough that reward_components.parquet and operator_metrics.parquet
# stay consistent with each other in this fixture, the way they are in a
# real run (operator_metrics.reward is the same value _calculate_reward
# returned, just reshaped from the PPO buffers).
PENALTY_DECAY_RATE = 5.0
EARNINGS_PER_CHARGE = 0.2


@dataclass
class VehicleTravelState:
    soc: float
    last_known_h3: str
    dest_h3: str
    travel_start: int
    travel_duration: int
    commitment_station: int | None = None
    commitment_until: int | None = None
    commitment_start: int | None = None


def _station_layout(
    rng: np.random.Generator, num_stations: int, h3_resolution: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    center = h3.latlng_to_cell(CENTER_LAT, CENTER_LNG, h3_resolution)
    pool = sorted(h3.grid_disk(center, 12))
    if len(pool) < num_stations:
        raise ValueError(
            f"H3 resolution {h3_resolution} disk too small for {num_stations} stations "
            f"({len(pool)} candidate cells); use a finer resolution."
        )
    station_cells = list(rng.choice(pool, size=num_stations, replace=False))
    capacities = rng.integers(2, 6, size=num_stations)
    base_prices = rng.uniform(0.15, 0.35, size=num_stations)
    return station_cells, capacities, base_prices


def _area_pool(h3_resolution: int) -> list[str]:
    center = h3.latlng_to_cell(CENTER_LAT, CENTER_LNG, h3_resolution)
    return sorted(h3.grid_disk(center, 12))


def _interpolated_position(state: VehicleTravelState, timestep: int) -> str:
    elapsed = timestep - state.travel_start
    frac = min(max(elapsed / state.travel_duration, 0.0), 1.0)
    path = h3.grid_path_cells(state.last_known_h3, state.dest_h3)
    if len(path) <= 1:
        return state.dest_h3
    idx = min(int(round(frac * (len(path) - 1))), len(path) - 1)
    return path[idx]


def _simulate_rollout(
    rng: np.random.Generator,
    *,
    epoch: int,
    rollout_id: int,
    steps_per_rollout: int,
    num_vehicles: int,
    num_stations: int,
    num_operators: int,
    dummy_station_index: int,
    area_pool: list[str],
    station_cells: list[str],
    capacities: np.ndarray,
    base_prices: np.ndarray,
    learning_progress: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    vehicle_rows: list[dict] = []
    station_rows: list[dict] = []
    reward_component_rows: list[dict] = []
    operator_reward_acc = np.zeros(num_operators)
    operator_value_acc = np.zeros(num_operators)

    states: list[VehicleTravelState] = []
    for v in range(num_vehicles):
        start_h3 = rng.choice(area_pool)
        states.append(
            VehicleTravelState(
                soc=float(rng.uniform(0.3, 0.9)),
                last_known_h3=start_h3,
                dest_h3=start_h3,
                travel_start=0,
                travel_duration=0,
            )
        )

    for t in range(steps_per_rollout):
        station_occupancy = np.zeros(num_stations, dtype=int)
        station_num_bids = np.zeros(num_stations, dtype=int)

        vehicle_alloc: list[int] = []
        vehicle_price: list[float] = []

        for v, state in enumerate(states):
            operator_id = v % num_operators

            arrived = t >= state.travel_start + state.travel_duration
            if arrived:
                state.last_known_h3 = state.dest_h3
                position_known = True
                position_h3 = state.dest_h3
                if rng.random() < 0.25:
                    new_dest = rng.choice(area_pool)
                    duration = max(1, min(6, h3.grid_distance(state.last_known_h3, new_dest)))
                    state.dest_h3 = new_dest
                    state.travel_start = t
                    state.travel_duration = int(duration)
            else:
                position_known = False
                position_h3 = _interpolated_position(state, t)

            charging = (
                state.commitment_station is not None
                and state.commitment_until is not None
                and t <= state.commitment_until
            )
            drain = rng.uniform(0.005, 0.02)
            if charging:
                state.soc = float(np.clip(state.soc + rng.uniform(0.03, 0.07), 0.0, 1.0))
            else:
                state.soc = float(np.clip(state.soc - drain, 0.0, 1.0))

            has_commitment = (
                state.commitment_station is not None
                and state.commitment_until is not None
                and t <= state.commitment_until
            )
            if not has_commitment:
                state.commitment_station = None
                state.commitment_until = None
                state.commitment_start = None

            # Urgency-driven bidding: lower SOC -> more likely to bid.
            # `learning_progress` (0..1 across epochs) makes the policy
            # slightly less wasteful over time, for a visible training trend.
            p_bid = float(np.clip((1.0 - state.soc) * (0.9 + 0.2 * learning_progress), 0.05, 0.95))
            if has_commitment:
                # A vehicle mid-commitment keeps bidding for its own station.
                allocation = state.commitment_station
                price = float(base_prices[allocation] * rng.uniform(0.9, 1.3))
            elif rng.random() < p_bid:
                veh_cell = position_h3 if position_known else state.last_known_h3
                distances = np.array(
                    [h3.grid_distance(veh_cell, sc) for sc in station_cells], dtype=float
                )
                weights = 1.0 / (1.0 + distances + base_prices * 5.0)
                weights = weights / weights.sum()
                allocation = int(rng.choice(num_stations, p=weights))
                price = float(base_prices[allocation] * rng.uniform(0.8, 1.4))
                if rng.random() < 0.6:
                    duration = int(rng.integers(2, 5))
                    state.commitment_station = allocation
                    state.commitment_start = t
                    state.commitment_until = t + duration
            else:
                allocation = dummy_station_index
                price = 0.0

            is_dummy = allocation == dummy_station_index
            is_bid = not is_dummy
            if not is_dummy:
                station_num_bids[allocation] += 1
                if has_commitment:
                    station_occupancy[allocation] += 1

            vehicle_alloc.append(allocation)
            vehicle_price.append(price)

            vehicle_rows.append(
                {
                    "epoch": epoch,
                    "rollout_id": rollout_id,
                    "timestep": t,
                    "operator_id": operator_id,
                    "vehicle_id": v,
                    "soc": round(state.soc, 4),
                    "last_known_position_h3": state.last_known_h3,
                    "destination_h3": state.dest_h3,
                    "position_h3": position_h3,
                    "position_known": bool(position_known),
                    "has_reservation": bool(has_commitment),
                    "reservation_start": state.commitment_start,
                    "reservation_end": state.commitment_until,
                    "has_commitment": bool(has_commitment),
                    "committed_station": state.commitment_station,
                    "allocation": int(allocation),
                    "is_dummy_assignment": bool(is_dummy),
                    "bid_price": round(price, 4),
                    "is_bid": bool(is_bid),
                }
            )

        for s in range(num_stations):
            occupancy = int(min(station_occupancy[s], capacities[s]))
            num_bids = int(station_num_bids[s])
            num_winning = int(min(num_bids, max(capacities[s] - occupancy, 0)))
            price = float(base_prices[s] * (1.0 + 0.15 * occupancy))
            station_rows.append(
                {
                    "epoch": epoch,
                    "rollout_id": rollout_id,
                    "timestep": t,
                    "station_id": s,
                    "h3": station_cells[s],
                    "capacity": int(capacities[s]),
                    "occupancy": occupancy,
                    "price": round(price, 4),
                    "num_bids": num_bids,
                    "num_winning_bids": num_winning,
                }
            )

        if t == 0:
            operator_rows: list[dict] = []
        for op in range(num_operators):
            op_vehicle_ids = [v for v in range(num_vehicles) if v % num_operators == op]
            fleet_size = len(op_vehicle_ids)
            socs = np.array([states[v].soc for v in op_vehicle_ids])

            # Earnings: bid value of this operator's vehicles that landed a
            # real (non-dummy) allocation this step - a fixture-only proxy
            # for "won an auction slot", since the fixture doesn't run an
            # actual per-station auction. Cost is a fraction of earnings,
            # standing in for operating/charging cost.
            earnings = sum(
                vehicle_price[v] for v in op_vehicle_ids if vehicle_alloc[v] != dummy_station_index
            )
            cost = earnings * float(rng.uniform(0.3, 0.6))
            winnings_factor = earnings / fleet_size if fleet_size else 0.0
            penalty_factor = float(np.mean(np.exp(-PENALTY_DECAY_RATE * socs))) if fleet_size else 0.0
            low_soc_penalty = EARNINGS_PER_CHARGE * penalty_factor
            cost_penalty = cost / fleet_size if fleet_size else 0.0

            # Exactly winnings_factor - low_soc_penalty - cost_penalty, no
            # extra noise term - this must match _calculate_reward's
            # relationship exactly, the same way it does in a real run.
            reward = winnings_factor - low_soc_penalty - cost_penalty
            # The "improves with epoch" trend lives in earnings (vehicles
            # bid more successfully as learning_progress rises via p_bid
            # above); value is the critic's own (noisy) estimate of reward.
            value = reward + float(rng.normal(0, 0.15))
            operator_reward_acc[op] = reward
            operator_value_acc[op] = value

            reward_component_rows.append(
                {
                    "epoch": epoch,
                    "rollout_id": rollout_id,
                    "timestep": t,
                    "operator_id": op,
                    "reward": round(reward, 4),
                    "earnings": round(earnings, 4),
                    "cost": round(cost, 4),
                    "winnings_factor": round(winnings_factor, 4),
                    "penalty_factor": round(penalty_factor, 4),
                    "low_soc_penalty": round(low_soc_penalty, 4),
                    "cost_penalty": round(cost_penalty, 4),
                }
            )
            operator_rows.append(
                {
                    "epoch": epoch,
                    "rollout_id": rollout_id,
                    "timestep": t,
                    "operator_id": op,
                    "reward": round(float(operator_reward_acc[op]), 4),
                    "value": round(float(operator_value_acc[op]), 4),
                }
            )

    return vehicle_rows, station_rows, operator_rows, reward_component_rows


def generate_fixture(
    out_dir: Path,
    *,
    run_id: str = "run_fixture",
    num_epochs: int = 3,
    num_parallel_envs: int = 3,
    steps_per_rollout: int = 24,
    num_vehicles: int = 12,
    num_stations: int = 8,
    num_operators: int = 2,
    h3_resolution: int = 9,
    seed: int = 42,
) -> Path:
    run_dir = out_dir / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    area_pool = _area_pool(h3_resolution)
    station_cells, capacities, base_prices = _station_layout(rng, num_stations, h3_resolution)
    dummy_station_index = num_stations

    all_vehicle_rows: list[dict] = []
    all_station_rows: list[dict] = []
    all_operator_rows: list[dict] = []
    all_reward_component_rows: list[dict] = []

    for epoch in range(num_epochs):
        learning_progress = epoch / max(num_epochs - 1, 1)
        for rollout_id in range(num_parallel_envs):
            epoch_rng = np.random.default_rng((seed, epoch, rollout_id))
            v_rows, s_rows, o_rows, rc_rows = _simulate_rollout(
                epoch_rng,
                epoch=epoch,
                rollout_id=rollout_id,
                steps_per_rollout=steps_per_rollout,
                num_vehicles=num_vehicles,
                num_stations=num_stations,
                num_operators=num_operators,
                dummy_station_index=dummy_station_index,
                area_pool=area_pool,
                station_cells=station_cells,
                capacities=capacities,
                base_prices=base_prices,
                learning_progress=learning_progress,
            )
            all_vehicle_rows.extend(v_rows)
            all_station_rows.extend(s_rows)
            all_operator_rows.extend(o_rows)
            all_reward_component_rows.extend(rc_rows)

    vehicle_df = pd.DataFrame(all_vehicle_rows)
    vehicle_df.insert(0, "run_id", run_id)
    station_df = pd.DataFrame(all_station_rows)
    station_df.insert(0, "run_id", run_id)
    operator_df = pd.DataFrame(all_operator_rows)
    operator_df.insert(0, "run_id", run_id)
    reward_components_df = pd.DataFrame(all_reward_component_rows)
    reward_components_df.insert(0, "run_id", run_id)

    vehicle_df.to_parquet(run_dir / "vehicle_states.parquet", partition_cols=["epoch"])
    station_df.to_parquet(run_dir / "station_states.parquet", partition_cols=["epoch"])
    operator_df.to_parquet(run_dir / "operator_metrics.parquet", partition_cols=["epoch"])
    reward_components_df.to_parquet(run_dir / "reward_components.parquet", partition_cols=["epoch"])

    training_df = (
        operator_df.groupby(["run_id", "epoch", "operator_id"])
        .agg(
            total_reward=("reward", "sum"),
            mean_reward=("reward", "mean"),
            mean_value=("value", "mean"),
        )
        .reset_index()
    )
    training_df.to_parquet(run_dir / "training_metrics.parquet")

    metadata = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Synthetic fixture run for local dashboard development (not real training data).",
        "num_epochs": num_epochs,
        "num_parallel_envs": num_parallel_envs,
        "steps_per_rollout": steps_per_rollout,
        "num_vehicles": num_vehicles,
        "num_stations": num_stations,
        "num_operators": num_operators,
        "dummy_station_index": dummy_station_index,
        "h3_resolution": h3_resolution,
        "operators": [
            {"operator_id": op, "name": f"Operator {chr(65 + op)}"} for op in range(num_operators)
        ],
        "config": {"source": "scripts/make_fixture.py", "seed": seed},
        "seed": seed,
        "recording_level": "full",
        "is_evaluation": False,
    }
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
        help="Directory to write the run into (default: ./runs)",
    )
    parser.add_argument("--run-id", default="run_fixture")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--stations", type=int, default=8)
    parser.add_argument("--operators", type=int, default=2)
    parser.add_argument("--h3-resolution", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = generate_fixture(
        args.out,
        run_id=args.run_id,
        num_epochs=args.epochs,
        num_parallel_envs=args.rollouts,
        steps_per_rollout=args.steps,
        num_vehicles=args.vehicles,
        num_stations=args.stations,
        num_operators=args.operators,
        h3_resolution=args.h3_resolution,
        seed=args.seed,
    )
    print(f"Wrote fixture run to {run_dir}")


if __name__ == "__main__":
    main()
