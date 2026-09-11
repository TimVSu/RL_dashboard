"""Fleet table: filter options, filtering, and row-click selection (sections 21-23)."""

from __future__ import annotations

import dash
from dash import Input, Output, State

from dashboard import config
from dashboard.components.fleet_table import apply_filters, to_table_records
from dashboard.data import loader


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("fleet-filter-operator", "options"),
        Output("fleet-filter-station", "options"),
        Input("run-metadata-store", "data"),
    )
    def update_filter_options(metadata):
        # Must always include "Any", matching build_fleet_filters' initial
        # options and the dropdowns' default value="any": dcc.Dropdown
        # silently clears its value to None when the options it's given no
        # longer contain the currently-selected value, which previously
        # turned "any" into None here and crashed apply_filters' int(None).
        if not metadata:
            return [{"label": "Any", "value": "any"}], [{"label": "Any", "value": "any"}]
        operator_options = [{"label": "Any", "value": "any"}] + [
            {"label": op["name"], "value": op["operator_id"]} for op in metadata["operators"]
        ]
        station_options = [{"label": "Any", "value": "any"}] + [
            {"label": f"Station {s}", "value": s} for s in range(metadata["num_stations"])
        ]
        return operator_options, station_options

    @app.callback(
        Output("fleet-table", "data"),
        Input("run-dropdown", "value"),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        Input("timestep-slider", "value"),
        Input("fleet-filter-soc", "value"),
        Input("fleet-filter-bid", "value"),
        Input("fleet-filter-assignment", "value"),
        Input("fleet-filter-operator", "value"),
        Input("fleet-filter-station", "value"),
        Input("fleet-filter-flags", "value"),
        State("run-metadata-store", "data"),
    )
    def update_fleet_table(
        run_id, epoch, rollout, timestep, soc, bid, assignment, operator, station, flags, metadata
    ):
        if not run_id or epoch is None or rollout is None or timestep is None or not metadata:
            return []
        df = loader.load_vehicle_states(config.RUNS_DIR, run_id, epoch)
        current = loader.at_timestep(df, rollout, timestep)
        filtered = apply_filters(
            current,
            {
                "soc": soc,
                "bid": bid,
                "assignment": assignment,
                "operator": operator,
                "station": station,
                "flags": flags or [],
            },
        )
        return to_table_records(filtered, metadata["dummy_station_index"])

    @app.callback(
        Output("selected-vehicle-store", "data", allow_duplicate=True),
        Input("fleet-table", "active_cell"),
        State("fleet-table", "data"),
        prevent_initial_call=True,
    )
    def on_fleet_row_click(active_cell, rows):
        if not active_cell or not rows:
            return dash.no_update
        row = rows[active_cell["row"]]
        # [operator_id, vehicle_id]: vehicle_id alone is only unique within
        # one operator's fleet (see dashboard/components/map_view.py).
        return [int(row["operator_id"]), int(row["vehicle_id"])]
