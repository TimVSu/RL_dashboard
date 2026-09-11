"""Selected-station detail and history (requirements.md sections 27-28)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, html
from plotly.subplots import make_subplots

from dashboard.data.models import StationState


def _field(label: str, value: str) -> html.Div:
    return html.Div(
        className="detail-field",
        children=[html.Span(label, className="detail-label"), html.Span(value, className="detail-value")],
    )


def build_station_detail(station: StationState, bidding_vehicles: pd.DataFrame) -> html.Div:
    bidder_table = dash_table.DataTable(
        columns=[
            {"name": "Vehicle", "id": "vehicle_id"},
            {"name": "Operator", "id": "operator_id"},
            {"name": "SOC", "id": "soc_display"},
            {"name": "Bid", "id": "bid_display"},
        ],
        data=[
            {
                "vehicle_id": int(row["vehicle_id"]),
                "operator_id": int(row["operator_id"]),
                "soc_display": f"{row['soc']:.0%}",
                "bid_display": f"{row['bid_price']:.2f}",
            }
            for _, row in bidding_vehicles.iterrows()
        ],
        style_cell={"padding": "4px 8px", "fontSize": 13, "textAlign": "left"},
        style_header={"fontWeight": "bold"},
        page_size=8,
    )

    return html.Div(
        className="detail-panel",
        children=[
            html.H4(f"Station {station.station_id}"),
            _field("Location", station.h3),
            _field("Capacity", str(station.capacity)),
            _field("Occupancy", f"{station.occupancy}/{station.capacity}"),
            _field("Price", f"{station.price:.2f}"),
            _field("Bids", f"{station.num_bids} ({station.num_winning_bids} winning)"),
            html.H5(f"Vehicles bidding here ({len(bidding_vehicles)})"),
            bidder_table,
        ],
    )


def build_station_history_figure(history: pd.DataFrame) -> go.Figure:
    history = history.sort_values("timestep")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.34, 0.33, 0.33],
        vertical_spacing=0.06,
        subplot_titles=("Occupancy", "Price", "Bids"),
    )
    fig.add_trace(
        go.Scatter(x=history["timestep"], y=history["occupancy"], mode="lines", line_shape="hv"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=history["timestep"], y=history["price"], mode="lines"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=history["timestep"], y=history["num_bids"], name="Bids"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=history["timestep"], y=history["num_winning_bids"], name="Winning bids"),
        row=3,
        col=1,
    )
    fig.update_layout(
        height=420,
        margin={"l": 40, "r": 10, "t": 30, "b": 30},
        showlegend=False,
        barmode="overlay",
    )
    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    return fig
