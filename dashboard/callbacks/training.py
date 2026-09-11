"""Training/reward page wiring (requirements.md section 29)."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output

from dashboard import config
from dashboard.components.reward_charts import (
    build_reward_by_epoch_figure,
    build_reward_value_by_timestep_figure,
)
from dashboard.data import loader


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("training-rollout-dropdown", "options"),
        Output("training-rollout-dropdown", "value"),
        Input("training-run-dropdown", "value"),
    )
    def populate_rollouts(run_id):
        if not run_id:
            return [], None
        meta = loader.load_metadata(config.RUNS_DIR, run_id)
        options = [{"label": str(r), "value": r} for r in meta.rollout_ids]
        return options, meta.rollout_ids[0]

    @app.callback(
        Output("reward-by-epoch-graph", "figure"),
        Output("reward-value-by-timestep-graph", "figure"),
        Input("training-run-dropdown", "value"),
        Input("training-rollout-dropdown", "value"),
    )
    def update_graphs(run_id, rollout_id):
        if not run_id:
            return go.Figure(), go.Figure()
        meta = loader.load_metadata(config.RUNS_DIR, run_id)
        operators = meta.operators

        training_df = loader.load_training_metrics(config.RUNS_DIR, run_id)
        epoch_fig = build_reward_by_epoch_figure(training_df, operators)

        if rollout_id is None:
            return epoch_fig, go.Figure()

        # Reward/value by timestep is shown for the most recent epoch of the
        # selected rollout - the policy version most relevant to inspect.
        latest_epoch = meta.epoch_ids[-1]
        operator_metrics = loader.load_operator_metrics(config.RUNS_DIR, run_id, latest_epoch)
        rollout_metrics = operator_metrics[operator_metrics["rollout_id"] == rollout_id]
        timestep_fig = build_reward_value_by_timestep_figure(rollout_metrics, operators)

        return epoch_fig, timestep_fig
