"""Selected-vehicle detail, decision inspection, and history (sections 24-26)."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, html

from dashboard import config
from dashboard.components.vehicle_panel import (
    build_decision_inspection,
    build_vehicle_detail,
    build_vehicle_history_figure,
)
from dashboard.data import loader
from dashboard.data.models import VehicleState


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("vehicle-detail-panel", "children"),
        Output("decision-panel", "children"),
        Output("vehicle-history-graph", "figure"),
        Input("selected-vehicle-store", "data"),
        Input("run-dropdown", "value"),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        Input("timestep-slider", "value"),
        State("run-metadata-store", "data"),
    )
    def update_vehicle_views(selected_vehicle, run_id, epoch, rollout, timestep, metadata):
        empty_fig = go.Figure()
        if not selected_vehicle or not run_id or epoch is None or rollout is None or not metadata:
            return html.P("Select a vehicle", className="placeholder-text"), None, empty_fig

        # selected-vehicle-store holds [operator_id, vehicle_id]: vehicle_id
        # alone is only unique within one operator's fleet, so both are
        # required to identify one real vehicle (see map_view.py).
        operator_id, vehicle_id = selected_vehicle

        vdf = loader.load_vehicle_states(config.RUNS_DIR, run_id, epoch)
        current = loader.at_timestep(vdf, rollout, timestep)
        match = current[(current["operator_id"] == operator_id) & (current["vehicle_id"] == vehicle_id)]
        if match.empty:
            return (
                html.P(
                    f"Vehicle {vehicle_id} (operator {operator_id}) not found at this timestep",
                    className="placeholder-text",
                ),
                None,
                empty_fig,
            )

        vehicle = VehicleState.from_row(match.iloc[0])
        dummy_station_index = metadata["dummy_station_index"]
        detail = build_vehicle_detail(vehicle, dummy_station_index)

        sdf = loader.load_station_states(config.RUNS_DIR, run_id, epoch)
        stations_now = loader.at_timestep(sdf, rollout, timestep)
        decision = build_decision_inspection(vehicle, stations_now, dummy_station_index, timestep)

        history = loader.for_rollout(vdf, rollout)
        vehicle_history = history[(history["operator_id"] == operator_id) & (history["vehicle_id"] == vehicle_id)]
        history_fig = build_vehicle_history_figure(vehicle_history, dummy_station_index)

        return detail, decision, history_fig
