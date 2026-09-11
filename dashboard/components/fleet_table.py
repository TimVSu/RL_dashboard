"""Sortable/filterable fleet table (requirements.md sections 21-22).

150 vehicles as permanent cards is unreadable; a table the user can sort,
filter, and click into is the requirement. This module builds the filter
controls, applies them to a vehicle-states DataFrame, and formats the
result into the columns dash_table.DataTable expects.
"""

from __future__ import annotations


import pandas as pd
from dash import dash_table, dcc, html

from dashboard.data.models import format_allocation

SOC_THRESHOLDS = {
    "any": None,
    "lt10": 0.10,
    "lt20": 0.20,
    "lt30": 0.30,
}

FLEET_TABLE_COLUMNS = [
    {"name": "Vehicle", "id": "vehicle_id"},
    {"name": "Operator", "id": "operator_id"},
    {"name": "SOC", "id": "soc_display"},
    {"name": "Position", "id": "position_display"},
    {"name": "Destination", "id": "destination_h3"},
    {"name": "Reservation", "id": "reservation_display"},
    {"name": "Bid", "id": "bid_display"},
    {"name": "Selected station", "id": "selected_station_display"},
    {"name": "Bid price", "id": "bid_price_display"},
    {"name": "Commitment", "id": "commitment_display"},
    {"name": "Status", "id": "status_display"},
]


def build_fleet_filters(operator_options: list[dict], station_options: list[dict]) -> html.Div:
    return html.Div(
        className="fleet-filters",
        children=[
            html.Div(
                [
                    html.Label("SOC"),
                    dcc.Dropdown(
                        id="fleet-filter-soc",
                        options=[
                            {"label": "Any", "value": "any"},
                            {"label": "< 10%", "value": "lt10"},
                            {"label": "< 20%", "value": "lt20"},
                            {"label": "< 30%", "value": "lt30"},
                        ],
                        value="any",
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Bid"),
                    dcc.Dropdown(
                        id="fleet-filter-bid",
                        options=[
                            {"label": "Any", "value": "any"},
                            {"label": "Placed bid", "value": "bid"},
                            {"label": "No bid", "value": "no_bid"},
                        ],
                        value="any",
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Assignment"),
                    dcc.Dropdown(
                        id="fleet-filter-assignment",
                        options=[
                            {"label": "Any", "value": "any"},
                            {"label": "Real station", "value": "real"},
                            {"label": "No assignment (dummy)", "value": "dummy"},
                        ],
                        value="any",
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Operator"),
                    dcc.Dropdown(
                        id="fleet-filter-operator",
                        options=[{"label": "Any", "value": "any"}] + operator_options,
                        value="any",
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Selected station"),
                    dcc.Dropdown(
                        id="fleet-filter-station",
                        options=[{"label": "Any", "value": "any"}] + station_options,
                        value="any",
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="fleet-filter-flags",
                        options=[
                            {"label": " Has reservation", "value": "has_reservation"},
                            {"label": " Has commitment", "value": "has_commitment"},
                        ],
                        value=[],
                    ),
                ]
            ),
        ],
    )


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    # `or "any"` guards against a dropdown value of None (e.g. a dcc.Dropdown
    # whose options were replaced without "any" among them, which makes Dash
    # clear .value to None client-side) - treat that the same as "no filter"
    # rather than letting int(None) below raise.
    soc_threshold = SOC_THRESHOLDS.get(filters.get("soc") or "any")
    if soc_threshold is not None:
        mask &= df["soc"] < soc_threshold

    bid = filters.get("bid") or "any"
    if bid == "bid":
        mask &= df["is_bid"]
    elif bid == "no_bid":
        mask &= ~df["is_bid"]

    assignment = filters.get("assignment") or "any"
    if assignment == "real":
        mask &= ~df["is_dummy_assignment"]
    elif assignment == "dummy":
        mask &= df["is_dummy_assignment"]

    operator = filters.get("operator") or "any"
    if operator != "any":
        mask &= df["operator_id"] == int(operator)

    station = filters.get("station") or "any"
    if station != "any":
        mask &= df["allocation"] == int(station)

    flags = filters.get("flags", [])
    if "has_reservation" in flags:
        mask &= df["has_reservation"]
    if "has_commitment" in flags:
        mask &= df["has_commitment"]

    return df[mask]


def to_table_records(df: pd.DataFrame, dummy_station_index: int) -> list[dict]:
    if df.empty:
        return []
    out = pd.DataFrame(
        {
            "vehicle_id": df["vehicle_id"],
            "operator_id": df["operator_id"],
            "soc_display": (df["soc"] * 100).round(0).astype(int).astype(str) + "%",
            "position_display": df.apply(
                lambda r: r["position_h3"] if r["position_known"] else f"~{r['position_h3']}", axis=1
            ),
            "destination_h3": df["destination_h3"],
            "reservation_display": df["has_reservation"].map({True: "Yes", False: "No"}),
            "bid_display": df["is_bid"].map({True: "Yes", False: "No"}),
            "selected_station_display": df["allocation"].apply(
                lambda a: format_allocation(int(a), dummy_station_index)
            ),
            "bid_price_display": df.apply(
                lambda r: f"{r['bid_price']:.2f}" if r["is_bid"] else "-", axis=1
            ),
            "commitment_display": df["has_commitment"].map({True: "Yes", False: "No"}),
            "status_display": df["position_known"].map({True: "Known", False: "Estimated"}),
        }
    )
    return out.to_dict("records")


def build_fleet_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="fleet-table",
        columns=FLEET_TABLE_COLUMNS,
        data=[],
        row_selectable=False,
        sort_action="native",
        filter_action="none",
        page_size=20,
        style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
        style_cell={"padding": "4px 8px", "fontSize": 13, "textAlign": "left"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[
            {
                "if": {"filter_query": "{selected_station_display} = 'No assignment'"},
                "color": "#888888",
            },
        ],
    )
