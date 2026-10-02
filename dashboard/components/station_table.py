"""Sortable station list, replacing the map as the primary way to pick a
station to inspect (the map was dropped - clutter, and there are no live
vehicle locations to justify it; see station_panel.py for the resulting
detail/history views).
"""

from __future__ import annotations

import pandas as pd
from dash import dash_table

STATION_TABLE_COLUMNS = [
    {"name": "Station", "id": "station_id"},
    {"name": "Location", "id": "h3"},
    {"name": "Occupancy", "id": "occupancy_display"},
    {"name": "Price", "id": "price_display"},
    {"name": "Bids", "id": "num_bids"},
    {"name": "Winning bids", "id": "num_winning_bids"},
]


def to_table_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = pd.DataFrame(
        {
            "station_id": df["station_id"],
            "h3": df["h3"],
            "occupancy_display": df["occupancy"].astype(str) + "/" + df["capacity"].astype(str),
            "price_display": df["price"].round(2),
            "num_bids": df["num_bids"],
            "num_winning_bids": df["num_winning_bids"],
            # Not listed in STATION_TABLE_COLUMNS, so not rendered as a
            # visible column - used only by style_data_conditional below.
            # A string (not a bool) to match dash_table's string-based
            # filter_query comparisons used elsewhere in this codebase.
            "_full": (df["occupancy"] >= df["capacity"]).map({True: "yes", False: "no"}),
        }
    )
    return out.to_dict("records")


def build_station_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="station-table",
        columns=STATION_TABLE_COLUMNS,
        data=[],
        sort_action="native",
        filter_action="none",
        page_size=20,
        style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
        style_cell={"padding": "4px 8px", "fontSize": 13, "textAlign": "left"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[
            {
                "if": {"filter_query": "{_full} = 'yes'"},
                "backgroundColor": "#fdecea",
            },
        ],
    )
