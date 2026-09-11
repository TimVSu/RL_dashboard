# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, browser-based **replay/debugging dashboard** for a multi-operator RL auction agent (EV-to-charging-station assignment via SimPy + PPO, trained in a separate project). It is not a live simulation UI: it only reads recorded run data from disk and replays it. Full product requirements are in `requirements.md` — read it before making product-level decisions; this file is implementation orientation, not a substitute.

**This repo is intentionally standalone**, decoupled from the sibling project `../urban-mobility-simulation` (which owns the SimPy sim, the RL training loop, and — eventually — the diagnostic recorder). The only integration contract between the two is the on-disk schema described below. Never import code from `urban-mobility-simulation`, and never add a REST API, database, or network boundary between the two — that was an explicit, deliberate choice by the project owner, not an oversight.

## Commands

```bash
uv sync                              # install deps into .venv
uv run python dashboard/app.py       # run the app -> http://localhost:8050
uv run python scripts/make_fixture.py  # (re)generate the synthetic dev fixture at runs/run_fixture
uv run pytest -q                     # run tests
uv run pytest tests/test_loader.py::test_dummy_station_semantics  # run a single test
uvx ruff check --select F dashboard scripts tests   # lint (unused imports/vars, undefined names)
```

Override where the app looks for recorded runs (defaults to `./runs`):

```bash
THESIS_DASHBOARD_RUNS_DIR=/path/to/urban-mobility-simulation/runs uv run python dashboard/app.py
```

There is no real recorder yet — `scripts/make_fixture.py` is the only source of run data until one exists in `urban-mobility-simulation`. Regenerate the fixture after changing anything in `dashboard/data/schema.py`.

## Architecture

**Data flow:** `runs/<run_id>/` (Parquet + JSON on disk) → `dashboard/data/loader.py` (cached reads, schema-validated) → `dashboard/components/*.py` (pure figure/layout builders, no I/O) → `dashboard/callbacks/*.py` (wiring) → `dashboard/pages/*.py` (Dash Pages layouts). Nothing above `loader.py` touches the filesystem, and nothing in `components/` reads Dash callback state — that separation is deliberate (mirrors requirements.md section 54) and should be preserved when extending the app.

**On-disk data contract** (`dashboard/data/schema.py` is the single source of truth):

```text
runs/<run_id>/
    metadata.json                # RunMetadata: counts, dummy_station_index, h3_resolution, operators, schema_version
    training_metrics.parquet     # epoch-level: run_id, epoch, operator_id, total_reward, mean_reward, mean_value
    vehicle_states.parquet/      # partitioned by epoch=N/ for predicate-pushdown reads
    station_states.parquet/      # partitioned by epoch=N/
    operator_metrics.parquet/    # partitioned by epoch=N/ — reward/value per (rollout, timestep, operator)
```

`metadata.json`'s `schema_version` is checked against `schema.SCHEMA_VERSION` on every load; a mismatch raises `SchemaError` rather than silently misreading columns. Bump both together deliberately if the shape changes. **Nothing about vehicle/station/operator counts, rollout length, the no-assignment station index, or H3 resolution is ever hard-coded** — it all comes from `metadata.json` per run (`RunMetadata` in `dashboard/data/models.py`).

**The dummy-station rule** (requirements.md sections 7, 42 — the one semantic rule everything else depends on): a vehicle's `allocation` equal to `dummy_station_index` means "no assignment"; anything else is a real station bid. `bid_price == 0` is *not* sufficient to infer "no bid" — only the allocation index is authoritative. This logic lives in exactly one place, `dashboard.data.models.is_real_assignment` / `format_allocation`; always go through those rather than re-deriving the comparison inline.

**Caching:** `dashboard/data/cache.py` wraps loader functions in `functools.lru_cache`, keyed on `(runs_dir, run_id, epoch)`. Runs are immutable once written, so this is always safe and is what keeps epoch/rollout/timestep switching fast without re-reading Parquet per callback. `loader.at_timestep()` / `loader.for_rollout()` then do cheap pandas filtering on the cached per-epoch frame.

**Dash structure:** multi-page app via `dash.register_page` (`use_pages=True` in `dashboard/app.py`). `pages/replay.py` is the one primary MVP page (the H3 map + fleet table + vehicle/station detail panels, per requirements.md section 15's layout); `pages/training.py` is a small separate reward/value view; `pages/strategy.py` and `pages/compare.py` are registered but intentionally unbuilt placeholders (post-MVP per the phased order in requirements.md section 50 — do not build these out before Phase 1-3 concerns are solid). Page `layout()` functions are called per navigation and are where dynamic dropdown options (available runs, epochs, rollouts) get computed server-side; most cross-widget state flows through `dcc.Store` components (`run-metadata-store`, `selected-vehicle-store`, `selected-station-store`, `playback-playing-store`) rather than page-global Python state.

**Map rendering caveat:** uses Plotly's `go.Scattermap` (MapLibre-based, no Mapbox token needed — `layout.map.style = "carto-positron"`), *not* the older `Scattermapbox` (removed in this Plotly version). Its marker only reliably renders per-point `color`/`size` arrays for `symbol="circle"` — every marker trace in `map_view.py` uses circles and differentiates vehicles/stations/selection by size, opacity and color rather than by symbol shape. Click-to-select relies on `customdata` being a `[kind, id]` pair (`"vehicle"` or `"station"`) on every clickable trace; if you add a new clickable trace, follow that convention or `callbacks/map.py`'s `on_map_click` won't route the click correctly.

**Multiple Dash callbacks write the same Output** (e.g. `timestep-slider.value` from both nav buttons/playback and from epoch/rollout-change reset) — these use `allow_duplicate=True` with `dash.ctx.triggered_id` to disambiguate, consolidated into as few callbacks as possible to avoid write conflicts. Follow this pattern rather than introducing a new writer to a Store/prop that already has one.

**Testing:** `tests/` covers the data layer only (`loader.py`, `schema.py`) against `scripts/make_fixture.py`'s `generate_fixture()`, using `tmp_path` fixtures — there is no UI/callback test coverage yet. When testing loader behavior, call `dashboard.data.cache.clear_all()` first if a previous test wrote a run at the same path/run_id, since reads are cached process-wide.
