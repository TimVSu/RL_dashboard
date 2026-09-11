"""Selected-vehicle detail, decision inspection, and history (sections 24-26).

This is the heart of the "why did this happen?" workflow (section 45): show
what the agent saw, what it chose, and what happened before/after - without
claiming to explain the policy's internal reasoning (section 26).
"""

from __future__ import annotations


import h3
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, html
from plotly.subplots import make_subplots

from dashboard.data.models import VehicleState, format_allocation, is_real_assignment


def _field(label: str, value: str) -> html.Div:
    return html.Div(
        className="detail-field",
        children=[html.Span(label, className="detail-label"), html.Span(value, className="detail-value")],
    )


def build_vehicle_detail(vehicle: VehicleState, dummy_station_index: int) -> html.Div:
    reservation = (
        f"{vehicle.reservation_start} - {vehicle.reservation_end}"
        if vehicle.has_reservation
        else "None"
    )
    commitment = (
        f"Station {vehicle.committed_station}" if vehicle.has_commitment else "None"
    )
    action = format_allocation(vehicle.allocation, dummy_station_index)
    bid = f"{vehicle.bid_price:.2f}" if vehicle.is_bid else "None"

    return html.Div(
        className="detail-panel",
        children=[
            html.H4(f"Vehicle {vehicle.vehicle_id} (operator {vehicle.operator_id})"),
            _field("SOC", f"{vehicle.soc:.0%}"),
            _field("Position", vehicle.position_h3),
            _field("Position certainty", "Known" if vehicle.position_known else "Estimated"),
            _field("Destination", vehicle.destination_h3),
            _field("Reservation", reservation),
            _field("Charging commitment", commitment),
            _field("Current action", action),
            _field("Bid", bid),
        ],
    )


def build_decision_inspection(
    vehicle: VehicleState, stations: pd.DataFrame, dummy_station_index: int, timestep: int
) -> html.Div:
    candidates = stations.copy()
    candidates["distance"] = candidates["h3"].apply(
        lambda cell: h3.grid_distance(vehicle.position_h3, cell)
    )
    candidates["is_selected"] = candidates["station_id"] == vehicle.allocation
    candidates = candidates.sort_values("distance").head(6)

    table = dash_table.DataTable(
        columns=[
            {"name": "Station", "id": "station_id"},
            {"name": "Distance (cells)", "id": "distance"},
            {"name": "Price", "id": "price"},
            {"name": "Occupancy", "id": "occupancy_display"},
        ],
        data=[
            {
                "station_id": int(row["station_id"]),
                "distance": int(row["distance"]),
                "price": f"{row['price']:.2f}",
                "occupancy_display": f"{int(row['occupancy'])}/{int(row['capacity'])}",
            }
            for _, row in candidates.iterrows()
        ],
        style_cell={"padding": "4px 8px", "fontSize": 13, "textAlign": "left"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[
            {
                "if": {"filter_query": f"{{station_id}} = {vehicle.allocation}"},
                "backgroundColor": "#e8f0fe",
                "fontWeight": "bold",
            }
        ],
    )

    action_summary = (
        format_allocation(vehicle.allocation, dummy_station_index)
        + (f" at {vehicle.bid_price:.2f}" if is_real_assignment(vehicle.allocation, dummy_station_index) else "")
    )

    return html.Div(
        className="decision-panel",
        children=[
            html.H4(f"Decision at timestep {timestep}"),
            html.P(
                "State the agent observed and the action it took. This does "
                "not explain the network's internal reasoning, only the "
                "observed state and resulting action."
            ),
            _field("Selected action", action_summary),
            html.H5("Nearby candidate stations"),
            table,
        ],
    )


def build_vehicle_history_figure(history: pd.DataFrame, dummy_station_index: int) -> go.Figure:
    history = history.sort_values("timestep")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("SOC over time", "Bid price over time", "Commitment"),
    )
    fig.add_trace(
        go.Scatter(x=history["timestep"], y=history["soc"], mode="lines", name="SOC"),
        row=1,
        col=1,
    )
    bid_y = history["bid_price"].where(history["is_bid"])
    fig.add_trace(
        go.Scatter(
            x=history["timestep"],
            y=bid_y,
            mode="markers+lines",
            name="Bid price",
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history["timestep"],
            y=history["has_commitment"].astype(int),
            mode="lines",
            line_shape="hv",
            name="Has commitment",
            fill="tozeroy",
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 1], ticktext=["No", "Yes"], row=3, col=1)
    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    fig.update_layout(height=420, margin={"l": 40, "r": 10, "t": 30, "b": 30}, showlegend=False)
    return fig
