"""Reward/training view (requirements.md section 29)."""

from __future__ import annotations

import dash
from dash import dcc, html

from dashboard import config
from dashboard.data import loader

dash.register_page(__name__, path="/training", name="Training")


def layout() -> html.Div:
    run_ids = loader.list_runs(config.RUNS_DIR)
    run_options = [{"label": r, "value": r} for r in run_ids]
    default_run = run_ids[0] if run_ids else None

    return html.Div(
        className="page training-page",
        children=[
            html.Div(
                className="control-bar",
                children=[
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Run"),
                            dcc.Dropdown(
                                id="training-run-dropdown",
                                options=run_options,
                                value=default_run,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Rollout (for timestep view)"),
                            dcc.Dropdown(id="training-rollout-dropdown", options=[], value=None, clearable=False),
                        ],
                    ),
                ],
            ),
            dcc.Graph(id="reward-by-epoch-graph"),
            dcc.Graph(id="reward-value-by-timestep-graph"),
        ],
    )
