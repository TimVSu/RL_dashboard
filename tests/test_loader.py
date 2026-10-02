from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.data import loader
from dashboard.data.cache import clear_all
from dashboard.data.schema import SchemaError
from scripts.make_fixture import generate_fixture


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    clear_all()
    return generate_fixture(
        tmp_path,
        run_id="test_run",
        num_epochs=2,
        num_parallel_envs=2,
        steps_per_rollout=6,
        num_vehicles=4,
        num_stations=3,
        num_operators=2,
        h3_resolution=9,
        seed=1,
    ).parent


def test_list_runs(run_dir: Path) -> None:
    assert loader.list_runs(run_dir) == ["test_run"]


def test_list_runs_missing_dir(tmp_path: Path) -> None:
    assert loader.list_runs(tmp_path / "does_not_exist") == []


def test_load_metadata(run_dir: Path) -> None:
    meta = loader.load_metadata(run_dir, "test_run")
    assert meta.num_vehicles == 4
    assert meta.num_stations == 3
    assert meta.dummy_station_index == 3
    assert meta.epoch_ids == [0, 1]
    assert meta.rollout_ids == [0, 1]
    assert len(meta.operators) == 2


def test_load_metadata_missing_run(run_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load_metadata(run_dir, "no_such_run")


def test_load_vehicle_states_partition_pruning(run_dir: Path) -> None:
    df = loader.load_vehicle_states(run_dir, "test_run", 0)
    assert set(df["epoch"].astype(int)) == {0}
    assert set(df["rollout_id"]) == {0, 1}
    assert set(df["timestep"]) == set(range(6))
    assert len(df) == 2 * 6 * 4  # rollouts * timesteps * vehicles


def test_dummy_station_semantics(run_dir: Path) -> None:
    df = loader.load_vehicle_states(run_dir, "test_run", 0)
    dummy_index = 3
    real = df[~df["is_dummy_assignment"]]
    dummy = df[df["is_dummy_assignment"]]
    assert (real["allocation"] != dummy_index).all()
    assert (dummy["allocation"] == dummy_index).all()
    assert (dummy["bid_price"] == 0.0).all()
    assert (real["is_bid"]).all()
    assert (~dummy["is_bid"]).all()


def test_at_timestep_and_for_rollout(run_dir: Path) -> None:
    df = loader.load_vehicle_states(run_dir, "test_run", 0)
    at_t = loader.at_timestep(df, rollout_id=0, timestep=2)
    assert len(at_t) == 4  # num_vehicles
    assert set(at_t["rollout_id"]) == {0}
    assert set(at_t["timestep"]) == {2}

    rollout_df = loader.for_rollout(df, rollout_id=1)
    assert set(rollout_df["rollout_id"]) == {1}
    assert len(rollout_df) == 6 * 4


def test_load_station_states(run_dir: Path) -> None:
    df = loader.load_station_states(run_dir, "test_run", 1)
    assert len(df) == 2 * 6 * 3  # rollouts * timesteps * stations
    assert (df["occupancy"] <= df["capacity"]).all()


def test_load_operator_metrics_and_training_metrics(run_dir: Path) -> None:
    om = loader.load_operator_metrics(run_dir, "test_run", 0)
    assert set(om["operator_id"]) == {0, 1}
    tm = loader.load_training_metrics(run_dir, "test_run")
    assert set(tm["epoch"].astype(int)) == {0, 1}
    assert set(tm["operator_id"]) == {0, 1}


def test_load_reward_components(run_dir: Path) -> None:
    df = loader.load_reward_components(run_dir, "test_run", 0)
    assert len(df) == 2 * 6 * 2  # rollouts * timesteps * operators
    assert set(df["operator_id"]) == {0, 1}
    # reward = winnings_factor - low_soc_penalty - cost_penalty, same
    # relationship urban-mobility-simulation's GymWrapper._calculate_reward
    # defines - the fixture mirrors it deliberately. Compared with a small
    # tolerance since each column was independently rounded to 4dp on disk.
    recomputed = df["winnings_factor"] - df["low_soc_penalty"] - df["cost_penalty"]
    assert (recomputed - df["reward"]).abs().max() < 1e-3


def test_load_reward_components_missing_table_is_empty_not_an_error(run_dir: Path) -> None:
    import shutil

    shutil.rmtree(run_dir / "test_run" / "reward_components.parquet")
    clear_all()
    df = loader.load_reward_components(run_dir, "test_run", 0)
    assert df.empty
    assert set(df.columns) == {
        "run_id",
        "epoch",
        "rollout_id",
        "timestep",
        "operator_id",
        "reward",
        "earnings",
        "cost",
        "winnings_factor",
        "penalty_factor",
        "low_soc_penalty",
        "cost_penalty",
    }
    # at_timestep/for_rollout must keep working on the empty-but-columned frame.
    assert loader.at_timestep(df, rollout_id=0, timestep=0).empty
    assert loader.for_rollout(df, rollout_id=0).empty


def test_schema_version_mismatch_raises(run_dir: Path) -> None:
    import json

    meta_path = run_dir / "test_run" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["schema_version"] = "999.0.0"
    meta_path.write_text(json.dumps(data))
    clear_all()
    with pytest.raises(SchemaError):
        loader.load_metadata(run_dir, "test_run")
