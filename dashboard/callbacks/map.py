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
        State("map-view-store", "data"),
    )
    def update_map(run_id, epoch, rollout, timestep, layers, selected_vehicle, selected_station, saved_view):
        if not run_id or epoch is None or rollout is None or timestep is None:
            return go.Figure()
        layers = layers or []
        vdf = loader.load_vehicle_states(config.RUNS_DIR, run_id, epoch)
        sdf = loader.load_station_states(config.RUNS_DIR, run_id, epoch)
        vehicles = loader.at_timestep(vdf, rollout, timestep)
        stations = loader.at_timestep(sdf, rollout, timestep)
        # selected-vehicle-store holds [operator_id, vehicle_id] (vehicle_id
        # alone is only unique within one operator's fleet - see map_view.py).
        selected_key = tuple(selected_vehicle) if selected_vehicle else None
        return build_map_figure(
            vehicles,
            stations,
            selected_vehicle=selected_key,
            selected_station_id=selected_station,
            show_vehicles="vehicles" in layers,
            show_stations="stations" in layers,
            show_destinations="destinations" in layers,
            color_stations_by_price="station_prices" in layers,
            show_bids="bids" in layers,
            show_h3_cells="h3_cells" in layers,
            saved_view=saved_view,
        )

    @app.callback(
        Output("map-view-store", "data", allow_duplicate=True),
        Input("map-graph", "relayoutData"),
        State("map-view-store", "data"),
        prevent_initial_call=True,
    )
    def on_map_relayout(relayout_data, current_view):
        # relayoutData carries keys like "map.center"/"map.zoom" only for the
        # properties the user actually changed by dragging/scrolling - merge
        # them into whatever we already had rather than replacing wholesale.
        if not relayout_data:
            return dash.no_update
        view = dict(current_view or {})
        changed = False
        for key, value in relayout_data.items():
            if key.endswith(".center"):
                view["center"] = value
                changed = True
            elif key.endswith(".zoom"):
                view["zoom"] = value
                changed = True
        return view if changed else dash.no_update

    @app.callback(
        Output("map-view-store", "data", allow_duplicate=True),
        Input("run-dropdown", "value"),
        prevent_initial_call=True,
    )
    def reset_map_view_on_run_change(_run_id):
        # A different run may cover a different city/area entirely - don't
        # carry over a pan/zoom that might now be pointed at nothing.
        return None

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
        if not customdata:
            return current_vehicle, current_station
        kind = customdata[0]
        if kind == "vehicle" and len(customdata) == 3:
            _, operator_id, vehicle_id = customdata
            return [int(operator_id), int(vehicle_id)], current_station
        if kind == "station" and len(customdata) == 2:
            _, station_id = customdata
            return current_vehicle, int(station_id)
        return current_vehicle, current_station
