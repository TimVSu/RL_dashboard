# RL Auction Dashboard

Local, browser-based replay and debugging dashboard for the multi-operator EV
charging-station auction agent. This is a **standalone** project: it never
runs the SimPy simulation and never imports code from `urban-mobility-simulation`.
It only reads recorded run data from disk.

See `requirements.md` for the full product requirements this implementation
follows.

## Running

```bash
uv sync
uv run python dashboard/app.py
```

Then open <http://localhost:8050>.

By default the app looks for recorded runs in `./runs`. Override with:

```bash
THESIS_DASHBOARD_RUNS_DIR=/path/to/urban-mobility-simulation/runs uv run python dashboard/app.py
```

## Generating a synthetic fixture run

There is no recorder in this repo (it lives in `urban-mobility-simulation`).
To develop and test the dashboard without a real training run, generate a
small synthetic fixture:

```bash
uv run python scripts/make_fixture.py
```

This writes a fixture run to `runs/run_fixture` using the exact on-disk
schema documented below, so the app behaves as it would against real data.

## Data contract

The dashboard treats `runs/<run_id>/` as read-only. A run directory must
contain:

```text
runs/<run_id>/
    metadata.json
    training_metrics.parquet
    vehicle_states.parquet/      (partitioned by epoch)
    station_states.parquet/      (partitioned by epoch)
    operator_metrics.parquet/    (partitioned by epoch)
```

`metadata.json` carries a `schema_version` field. `dashboard/data/schema.py`
declares the version and column set this app expects and validates every
loaded table against it, failing with a clear error rather than guessing at
missing/renamed columns. If the recorder in `urban-mobility-simulation`
changes its output, bump `schema_version` on both sides deliberately.

Nothing about vehicle/station/operator counts, rollout length, the
no-assignment station index, or the H3 resolution is hard-coded anywhere in
this app — all of it is read from `metadata.json` per run.
