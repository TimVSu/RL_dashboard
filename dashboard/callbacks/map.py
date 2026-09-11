"""Map figure + click-to-select (requirements.md sections 17-20, 23)."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State

from dashboard import config
from dashboard.components.map_view import build_map_figure
from dashboard.data import loader


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("map-graph", "figure"),
        Input("run-dropdown", "value"),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        Input("timestep-slider", "value"),
        Input("map-layers", "value"),
        Input("selected-vehicle-store", "data"),
        Input("selected-station-store", "data"),
    )
    def update_map(run_id, epoch, rollout, timestep, layers, selected_vehicle, selected_station):
        if not run_id or epoch is None or rollout is None or timestep is None:
            return go.Figure()
        layers = layers or []
        vdf = loader.load_vehicle_states(config.RUNS_DIR, run_id, epoch)
        sdf = loader.load_station_states(config.RUNS_DIR, run_id, epoch)
        vehicles = loader.at_timestep(vdf, rollout, timestep)
        stations = loader.at_timestep(sdf, rollout, timestep)
        return build_map_figure(
            vehicles,
            stations,
            selected_vehicle_id=selected_vehicle,
            selected_station_id=selected_station,
            show_vehicles="vehicles" in layers,
            show_stations="stations" in layers,
            show_destinations="destinations" in layers,
            color_stations_by_price="station_prices" in layers,
            show_bids="bids" in layers,
            show_h3_cells="h3_cells" in layers,
        )

    @app.callback(
        Output("selected-vehicle-store", "data", allow_duplicate=True),
        Output("selected-station-store", "data", allow_duplicate=True),
        Input("map-graph", "clickData"),
        State("selected-vehicle-store", "data"),
        State("selected-station-store", "data"),
        prevent_initial_call=True,
    )
    def on_map_click(click_data, current_vehicle, current_station):
        if not click_data or not click_data.get("points"):
            return current_vehicle, current_station
        point = click_data["points"][0]
        customdata = point.get("customdata")
        if not customdata or len(customdata) != 2:
            return current_vehicle, current_station
        kind, entity_id = customdata
        if kind == "vehicle":
            return int(entity_id), current_station
        if kind == "station":
            return current_vehicle, int(entity_id)
        return current_vehicle, current_station
