"""Selected-station detail and history (requirements.md sections 27-28)."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, html

from dashboard import config
from dashboard.components.station_panel import build_station_detail, build_station_history_figure
from dashboard.components.station_table import to_table_records as station_table_records
from dashboard.data import loader
from dashboard.data.models import StationState


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("station-table", "data"),
        Input("run-dropdown", "value"),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        Input("timestep-slider", "value"),
    )
    def update_station_table(run_id, epoch, rollout, timestep):
        if not run_id or epoch is None or rollout is None or timestep is None:
            return []
        sdf = loader.load_station_states(config.RUNS_DIR, run_id, epoch)
        current = loader.at_timestep(sdf, rollout, timestep)
        return station_table_records(current)

    @app.callback(
        Output("selected-station-store", "data", allow_duplicate=True),
        Input("station-table", "active_cell"),
        State("station-table", "data"),
        prevent_initial_call=True,
    )
    def on_station_row_click(active_cell, rows):
        if not active_cell or not rows:
            return dash.no_update
        row = rows[active_cell["row"]]
        return int(row["station_id"])

    @app.callback(
        Output("station-detail-panel", "children"),
        Output("station-history-graph", "figure"),
        Input("selected-station-store", "data"),
        Input("run-dropdown", "value"),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        Input("timestep-slider", "value"),
    )
    def update_station_views(station_id, run_id, epoch, rollout, timestep):
        empty_fig = go.Figure()
        if station_id is None or not run_id or epoch is None or rollout is None:
            return html.P("Select a station", className="placeholder-text"), empty_fig

        sdf = loader.load_station_states(config.RUNS_DIR, run_id, epoch)
        current = loader.at_timestep(sdf, rollout, timestep)
        match = current[current["station_id"] == station_id]
        if match.empty:
            return (
                html.P(f"Station {station_id} not found at this timestep", className="placeholder-text"),
                empty_fig,
            )
        station = StationState.from_row(match.iloc[0])

        vdf = loader.load_vehicle_states(config.RUNS_DIR, run_id, epoch)
        vehicles_now = loader.at_timestep(vdf, rollout, timestep)
        bidding = vehicles_now[
            (vehicles_now["allocation"] == station_id) & (vehicles_now["is_bid"])
        ]
        detail = build_station_detail(station, bidding)

        history = loader.for_rollout(sdf, rollout)
        station_history = history[history["station_id"] == station_id]
        history_fig = build_station_history_figure(station_history)

        return detail, history_fig
