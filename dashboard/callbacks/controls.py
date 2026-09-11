"""Run/epoch/rollout/timestep + playback wiring (sections 15-16).

This module only figures out *which* (run, epoch, rollout, timestep) is
selected. It never touches the map/fleet/vehicle/station panels - those
each independently react to the resulting dcc.Store / dropdown values in
their own callback modules, all reading through the cached loader so
re-filtering per panel stays cheap.
"""

from __future__ import annotations

from dataclasses import asdict

import dash
from dash import Input, Output, State, ctx, no_update

from dashboard import config
from dashboard.data import loader


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("epoch-dropdown", "options"),
        Output("epoch-dropdown", "value"),
        Output("rollout-dropdown", "options"),
        Output("rollout-dropdown", "value"),
        Output("run-metadata-store", "data"),
        Input("run-dropdown", "value"),
    )
    def on_run_change(run_id):
        if not run_id:
            return [], None, [], None, None
        meta = loader.load_metadata(config.RUNS_DIR, run_id)
        epoch_options = [{"label": str(e), "value": e} for e in meta.epoch_ids]
        rollout_options = [{"label": str(r), "value": r} for r in meta.rollout_ids]
        return (
            epoch_options,
            meta.epoch_ids[-1],
            rollout_options,
            meta.rollout_ids[0],
            asdict(meta),
        )

    @app.callback(
        Output("timestep-slider", "min"),
        Output("timestep-slider", "max"),
        Output("timestep-slider", "marks"),
        Input("run-metadata-store", "data"),
    )
    def on_metadata_change(metadata):
        if not metadata:
            return 0, 0, {}
        steps = metadata["steps_per_rollout"]
        return 0, steps - 1, {0: "0", steps - 1: str(steps - 1)}

    @app.callback(
        Output("timestep-slider", "value", allow_duplicate=True),
        Input("epoch-dropdown", "value"),
        Input("rollout-dropdown", "value"),
        prevent_initial_call=True,
    )
    def reset_timestep_on_selection_change(_epoch, _rollout):
        return 0

    @app.callback(
        Output("timestep-slider", "value", allow_duplicate=True),
        Input("btn-first", "n_clicks"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("btn-last", "n_clicks"),
        Input("playback-interval", "n_intervals"),
        State("timestep-slider", "value"),
        State("timestep-slider", "max"),
        prevent_initial_call=True,
    )
    def on_nav(_n_first, _n_prev, _n_next, _n_last, _n_intervals, value, max_value):
        triggered = ctx.triggered_id
        value = value or 0
        max_value = max_value or 0
        if triggered == "btn-first":
            return 0
        if triggered == "btn-last":
            return max_value
        if triggered == "btn-prev":
            return max(0, value - 1)
        if triggered in ("btn-next", "playback-interval"):
            return min(max_value, value + 1)
        return no_update

    @app.callback(
        Output("timestep-label", "children"),
        Input("timestep-slider", "value"),
        State("timestep-slider", "max"),
    )
    def on_timestep_label(value, max_value):
        if value is None or max_value is None:
            return ""
        return f"Timestep {value} / {max_value}"

    @app.callback(
        Output("playback-interval", "disabled"),
        Output("playback-interval", "interval"),
        Output("btn-play-pause", "children"),
        Output("playback-playing-store", "data"),
        Input("btn-play-pause", "n_clicks"),
        Input("speed-dropdown", "value"),
        State("playback-playing-store", "data"),
        prevent_initial_call=True,
    )
    def on_play_pause(_n_clicks, speed, playing):
        if ctx.triggered_id == "btn-play-pause":
            playing = not playing
        interval_ms = max(50, int(config.PLAYBACK_BASE_INTERVAL_MS / (speed or 1)))
        return (not playing), interval_ms, ("Pause" if playing else "Play"), playing
