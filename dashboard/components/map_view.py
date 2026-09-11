"""H3 city map - the central visualization (requirements.md section 17-20).

Encodes, deliberately, only a few independent visual channels so the map
stays readable at ~150 vehicles / ~70 stations. Plotly's Scattermap marker
only renders per-point `color`/`size` arrays reliably for the "circle"
symbol, so every marker trace here uses "circle" and differentiates by
color/size/opacity instead of by symbol shape:

- vehicle color: SOC (low -> warm/red, high -> cool/green). SOC already
  doubles as the "urgency" signal requirements.md section 18 asks for, so
  there is no separate urgency toggle.
- vehicle opacity: full when position is known, faded when the vehicle is
  travelling and its position is only estimated (section 13) - split into
  two traces so hover text can say so explicitly too.
- vehicle size: slightly larger when the vehicle has an active bid.
- station size: occupancy / capacity ratio.
- station color: price, only when that layer is enabled (kept separate from
  vehicle color so "busy" and "expensive" are never conflated - section 44).
- selection: a dedicated highlight trace per selected vehicle/station, never
  reusing the semantic color channel.

All layers are optional booleans passed in by the callback; this module only
builds figures, it never reads callback state or touches the filesystem.
"""

from __future__ import annotations

from typing import Optional

import h3
import pandas as pd
import plotly.graph_objects as go

MAP_STYLE = "carto-positron"

_SOC_COLORSCALE = "RdYlGn"  # low SOC (0) -> red, high SOC (1) -> green
_VEHICLE_BASE_SIZE = 9
_VEHICLE_BID_SIZE = 14


def _latlng(cell: str) -> tuple[float, float]:
    lat, lng = h3.cell_to_latlng(cell)
    return lat, lng


def _center(cells: list[str]) -> tuple[float, float]:
    if not cells:
        return 37.7749, -122.4194
    lats, lngs = zip(*(_latlng(c) for c in cells))
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def _h3_cells_trace(cells: list[str]) -> go.Scattermap:
    lats: list[Optional[float]] = []
    lngs: list[Optional[float]] = []
    for cell in cells:
        boundary = h3.cell_to_boundary(cell)
        ring = list(boundary) + [boundary[0]]
        for lat, lng in ring:
            lats.append(lat)
            lngs.append(lng)
        lats.append(None)
        lngs.append(None)
    return go.Scattermap(
        lat=lats,
        lon=lngs,
        mode="lines",
        fill="toself",
        fillcolor="rgba(100,120,140,0.08)",
        line={"width": 1, "color": "rgba(100,120,140,0.35)"},
        hoverinfo="skip",
        name="H3 cells",
        showlegend=False,
    )


def _vehicle_hover_text(row: pd.Series) -> str:
    status = "known" if row["position_known"] else "estimated"
    action = f"Station {row['allocation']}" if row["is_bid"] else "No assignment"
    text = (
        f"Vehicle {row['vehicle_id']} (operator {row['operator_id']})<br>"
        f"SOC: {row['soc']:.0%}<br>"
        f"Position: {status}<br>"
        f"Action: {action}"
    )
    if row["is_bid"]:
        text += f"<br>Bid: {row['bid_price']:.2f}"
    return text


def _vehicles_trace(vehicles: pd.DataFrame, *, known: bool, showscale: bool) -> Optional[go.Scattermap]:
    subset = vehicles[vehicles["position_known"] == known]
    if subset.empty:
        return None
    lats, lngs = zip(*(_latlng(c) for c in subset["position_h3"]))
    sizes = [_VEHICLE_BID_SIZE if b else _VEHICLE_BASE_SIZE for b in subset["is_bid"]]
    texts = [_vehicle_hover_text(row) for _, row in subset.iterrows()]
    return go.Scattermap(
        lat=list(lats),
        lon=list(lngs),
        mode="markers",
        marker={
            "size": sizes,
            "symbol": "circle",
            "color": list(subset["soc"]),
            "colorscale": _SOC_COLORSCALE,
            "cmin": 0,
            "cmax": 1,
            "opacity": 0.95 if known else 0.45,
            "showscale": showscale,
            "colorbar": {"title": "SOC", "x": 1.0, "len": 0.5, "y": 0.75},
        },
        customdata=[["vehicle", int(v)] for v in subset["vehicle_id"]],
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        name="Vehicles" if known else "Vehicles (estimated pos.)",
        showlegend=False,
    )


def _destination_trace(vehicles: pd.DataFrame) -> go.Scattermap:
    lats, lngs = [], []
    for _, row in vehicles.iterrows():
        if row["last_known_position_h3"] == row["destination_h3"]:
            continue
        lat0, lng0 = _latlng(row["last_known_position_h3"])
        lat1, lng1 = _latlng(row["destination_h3"])
        lats.extend([lat0, lat1, None])
        lngs.extend([lng0, lng1, None])
    return go.Scattermap(
        lat=lats,
        lon=lngs,
        mode="lines",
        line={"width": 1, "color": "rgba(80,80,80,0.35)"},
        hoverinfo="skip",
        name="Destinations",
        showlegend=False,
    )


def _selected_vehicle_traces(
    vehicles: pd.DataFrame, selected_vehicle_id: int
) -> list[go.Scattermap]:
    match = vehicles[vehicles["vehicle_id"] == selected_vehicle_id]
    if match.empty:
        return []
    row = match.iloc[0]
    lat0, lng0 = _latlng(row["position_h3"])
    lat1, lng1 = _latlng(row["destination_h3"])
    traces = [
        go.Scattermap(
            lat=[lat0],
            lon=[lng0],
            mode="markers",
            marker={"size": 22, "symbol": "circle", "color": "rgba(30,60,214,0.25)"},
            hoverinfo="skip",
            showlegend=False,
            name="Selected vehicle",
        )
    ]
    if row["last_known_position_h3"] != row["destination_h3"] or row["position_h3"] != row["destination_h3"]:
        traces.append(
            go.Scattermap(
                lat=[lat0, lat1],
                lon=[lng0, lng1],
                mode="lines",
                line={"width": 3, "color": "#1e3fd6"},
                hoverinfo="skip",
                showlegend=False,
                name="Selected vehicle path",
            )
        )
        traces.append(
            go.Scattermap(
                lat=[lat1],
                lon=[lng1],
                mode="markers",
                marker={"size": 13, "symbol": "circle", "color": "#1e3fd6"},
                hoverinfo="text",
                text=["Destination"],
                showlegend=False,
                name="Selected vehicle destination",
            )
        )
    return traces


def _stations_trace(stations: pd.DataFrame, color_by_price: bool) -> go.Scattermap:
    lats, lngs = zip(*(_latlng(c) for c in stations["h3"]))
    ratios = [(o / c) if c else 0.0 for o, c in zip(stations["occupancy"], stations["capacity"])]
    sizes = [16 + 14 * min(r, 1.0) for r in ratios]
    texts = [
        f"Station {row['station_id']}<br>"
        f"Occupancy: {row['occupancy']}/{row['capacity']}<br>"
        f"Price: {row['price']:.2f}<br>"
        f"Bids: {row['num_bids']} ({row['num_winning_bids']} winning)"
        for _, row in stations.iterrows()
    ]
    marker: dict = {"size": sizes, "symbol": "circle", "opacity": 0.9}
    if color_by_price:
        marker.update(
            color=list(stations["price"]),
            colorscale="Oranges",
            showscale=True,
            colorbar={"title": "Price", "x": 1.0, "len": 0.5, "y": 0.25},
        )
    else:
        marker["color"] = "#444444"
    return go.Scattermap(
        lat=list(lats),
        lon=list(lngs),
        mode="markers",
        marker=marker,
        customdata=[["station", int(s)] for s in stations["station_id"]],
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        name="Stations",
        showlegend=False,
    )


def _selected_station_trace(stations: pd.DataFrame, selected_station_id: int) -> Optional[go.Scattermap]:
    match = stations[stations["station_id"] == selected_station_id]
    if match.empty:
        return None
    row = match.iloc[0]
    lat, lng = _latlng(row["h3"])
    return go.Scattermap(
        lat=[lat],
        lon=[lng],
        mode="markers",
        marker={"size": 34, "symbol": "circle", "color": "rgba(220,30,30,0.22)"},
        hoverinfo="skip",
        showlegend=False,
        name="Selected station",
    )


def _bids_trace(vehicles: pd.DataFrame, stations: pd.DataFrame) -> go.Scattermap:
    station_pos = {row["station_id"]: row["h3"] for _, row in stations.iterrows()}
    lats, lngs = [], []
    for _, row in vehicles.iterrows():
        if not row["is_bid"]:
            continue
        station_h3 = station_pos.get(row["allocation"])
        if station_h3 is None:
            continue
        lat0, lng0 = _latlng(row["position_h3"])
        lat1, lng1 = _latlng(station_h3)
        lats.extend([lat0, lat1, None])
        lngs.extend([lng0, lng1, None])
    return go.Scattermap(
        lat=lats,
        lon=lngs,
        mode="lines",
        line={"width": 1, "color": "rgba(30,120,220,0.5)"},
        hoverinfo="skip",
        name="Bids",
        showlegend=False,
    )


def build_map_figure(
    vehicles: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    selected_vehicle_id: Optional[int] = None,
    selected_station_id: Optional[int] = None,
    show_vehicles: bool = True,
    show_stations: bool = True,
    show_destinations: bool = False,
    color_stations_by_price: bool = False,
    show_bids: bool = False,
    show_h3_cells: bool = False,
) -> go.Figure:
    fig = go.Figure()

    all_cells = list(stations["h3"]) + list(vehicles["position_h3"])
    if show_h3_cells and all_cells:
        fig.add_trace(_h3_cells_trace(sorted(set(all_cells))))

    if show_destinations and len(vehicles):
        fig.add_trace(_destination_trace(vehicles))

    if show_bids and len(vehicles) and len(stations):
        fig.add_trace(_bids_trace(vehicles, stations))

    if show_stations and len(stations):
        fig.add_trace(_stations_trace(stations, color_stations_by_price))
        if selected_station_id is not None:
            trace = _selected_station_trace(stations, selected_station_id)
            if trace is not None:
                fig.add_trace(trace)

    if show_vehicles and len(vehicles):
        known_trace = _vehicles_trace(vehicles, known=True, showscale=True)
        estimated_trace = _vehicles_trace(vehicles, known=False, showscale=False)
        if known_trace is not None:
            fig.add_trace(known_trace)
        if estimated_trace is not None:
            fig.add_trace(estimated_trace)
        if selected_vehicle_id is not None:
            for trace in _selected_vehicle_traces(vehicles, selected_vehicle_id):
                fig.add_trace(trace)

    center_lat, center_lng = _center(all_cells)
    fig.update_layout(
        map={
            "style": MAP_STYLE,
            "center": {"lat": center_lat, "lon": center_lng},
            "zoom": 12.5,
            # layout.map.uirevision documents itself as defaulting to
            # layout.uirevision, but setting it explicitly here too rules
            # out any inheritance-order surprise - both must survive every
            # figure rebuild (new layer toggled, new timestep, etc.) so the
            # user's pan/zoom is never silently reset by an unrelated change.
            "uirevision": "map",
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        uirevision="map",  # preserve user pan/zoom across timestep updates
        showlegend=False,
        height=560,
    )
    return fig
