"""Operator summary cards + reward-breakdown history wiring."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, html

from dashboard import config
from dashboard.components.operator_panel import build_operator_cards, build_operator_reward_breakdown_figure
from dashboard.data import loader
from dashboard.data.models import OperatorInfo


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("operator-cards", "children"),
        Output("operator-reward-breakdown-graph", "figure"),
        Input("run-dropdown", "value"),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        Input("timestep-slider", "value"),
        State("run-metadata-store", "data"),
    )
    def update_operator_views(run_id, epoch, rollout, timestep, metadata):
        if not run_id or epoch is None or rollout is None or timestep is None or not metadata:
            return [html.P("Select a run", className="placeholder-text")], go.Figure()

        operators = [OperatorInfo(**op) for op in metadata["operators"]]

        om_df = loader.load_operator_metrics(config.RUNS_DIR, run_id, epoch)
        om_now = loader.at_timestep(om_df, rollout, timestep)

        vdf = loader.load_vehicle_states(config.RUNS_DIR, run_id, epoch)
        vehicles_now = loader.at_timestep(vdf, rollout, timestep)

        rc_df = loader.load_reward_components(config.RUNS_DIR, run_id, epoch)
        rc_now = loader.at_timestep(rc_df, rollout, timestep)

        cards = build_operator_cards(operators, om_now, om_df, rc_now, rc_df, vehicles_now, vdf)

        if rc_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No reward-component data recorded for this run.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        else:
            rc_history = loader.for_rollout(rc_df, rollout)
            fig = build_operator_reward_breakdown_figure(rc_history, operators)

        return cards, fig
