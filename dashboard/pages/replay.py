"""Episode Replay / Debugging page - the MVP's one primary page (section 15).

No map: with no live vehicle locations to justify it, it was mostly clutter
and told the user little about *why* something was going wrong. In its
place: an always-visible per-operator summary (money, bids, reward and its
components) for the current step, a station list standing in for the map's
old role of picking something to inspect, and the pre-existing vehicle view.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from dashboard import config
from dashboard.components.controls import build_epoch_rollout_selectors, build_timestep_controls
from dashboard.components.fleet_table import build_fleet_filters, build_fleet_table
from dashboard.components.station_table import build_station_table
from dashboard.data import loader

dash.register_page(__name__, path="/", name="Replay")


def layout() -> html.Div:
    run_ids = loader.list_runs(config.RUNS_DIR)
    run_options = [{"label": r, "value": r} for r in run_ids]
    default_run = run_ids[0] if run_ids else None

    return html.Div(
        className="page replay-page",
        children=[
            # Selection/navigation state, shared across every callback on this page.
            dcc.Store(id="run-metadata-store", data=None),
            dcc.Store(id="selected-vehicle-store", data=None),
            dcc.Store(id="selected-station-store", data=None),
            dcc.Store(id="playback-playing-store", data=False),
            html.Div(
                className="control-bar",
                children=[
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Run"),
                            dcc.Dropdown(
                                id="run-dropdown",
                                options=run_options,
                                value=default_run,
                                clearable=False,
                            ),
                        ],
                    )
                    if run_options
                    else html.Div(
                        className="control",
                        children=[
                            html.Label("Run"),
                            html.P(
                                f"No runs found in {config.RUNS_DIR}. Generate one with "
                                "scripts/make_fixture.py.",
                                className="placeholder-text",
                            ),
                            dcc.Dropdown(id="run-dropdown", options=[], value=None),
                        ],
                    ),
                    build_epoch_rollout_selectors(),
                    html.Div(className="control timestep-control-wide", children=[build_timestep_controls()]),
                ],
            ),
            html.Div(id="operator-cards", className="operator-row"),
            html.Div(
                className="main-grid",
                children=[
                    html.Div(
                        className="fleet-column",
                        children=[
                            html.H4("Fleet"),
                            build_fleet_filters(operator_options=[], station_options=[]),
                            build_fleet_table(),
                        ],
                    ),
                    html.Div(
                        className="station-column",
                        children=[
                            html.H4("Charging stations"),
                            build_station_table(),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="detail-grid",
                children=[
                    html.Div(
                        className="vehicle-detail-column",
                        children=[
                            html.Div(id="vehicle-detail-panel", children=[_placeholder("Select a vehicle")]),
                            html.Div(id="decision-panel"),
                            dcc.Graph(id="vehicle-history-graph"),
                        ],
                    ),
                    html.Div(
                        className="station-detail-column",
                        children=[
                            html.Div(id="station-detail-panel", children=[_placeholder("Select a station")]),
                            dcc.Graph(id="station-history-graph"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="operator-history-section",
                children=[
                    html.H4("Reward breakdown by operator (this rollout)"),
                    dcc.Graph(id="operator-reward-breakdown-graph"),
                ],
            ),
        ],
    )


def _placeholder(text: str) -> html.P:
    return html.P(text, className="placeholder-text")
