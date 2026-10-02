"""Global replay controls: run/epoch/rollout/timestep + playback (sections 15-16)."""

from __future__ import annotations

from dash import dcc, html

from dashboard import config


def build_epoch_rollout_selectors() -> html.Div:
    return html.Div(
        className="control-group",
        children=[
            html.Div(
                className="control",
                children=[
                    html.Label("Epoch"),
                    dcc.Dropdown(id="epoch-dropdown", options=[], value=None, clearable=False),
                ],
            ),
            html.Div(
                className="control",
                children=[
                    html.Label("Rollout"),
                    dcc.Dropdown(id="rollout-dropdown", options=[], value=None, clearable=False),
                ],
            ),
        ],
    )


def build_timestep_controls() -> html.Div:
    return html.Div(
        className="timestep-controls",
        children=[
            html.Div(id="timestep-label", className="timestep-label"),
            # "mouseup" (not "drag") - six separate callbacks (operator
            # cards, vehicle map, fleet view, station status, and more) key
            # off this slider's value, each re-filtering a potentially large
            # cached per-epoch DataFrame and rebuilding figures. With
            # updatemode="drag" every intermediate position the mouse
            # passes through during a jump fires all of them, which is what
            # made large jumps on large runs feel laggy - "mouseup" fires
            # the callbacks once, at the final position, instead.
            dcc.Slider(id="timestep-slider", min=0, max=0, step=1, value=0, updatemode="mouseup"),
            html.Div(
                className="playback-buttons",
                children=[
                    html.Button("|<", id="btn-first", n_clicks=0, title="Jump to first timestep"),
                    html.Button("<", id="btn-prev", n_clicks=0, title="Previous timestep"),
                    html.Button("Play", id="btn-play-pause", n_clicks=0),
                    html.Button(">", id="btn-next", n_clicks=0, title="Next timestep"),
                    html.Button(">|", id="btn-last", n_clicks=0, title="Jump to last timestep"),
                    dcc.Dropdown(
                        id="speed-dropdown",
                        options=[{"label": f"{s}x", "value": s} for s in config.PLAYBACK_SPEEDS],
                        value=config.DEFAULT_PLAYBACK_SPEED,
                        clearable=False,
                        style={"width": "90px"},
                    ),
                ],
            ),
            dcc.Interval(
                id="playback-interval",
                interval=config.PLAYBACK_BASE_INTERVAL_MS,
                disabled=True,
            ),
        ],
    )
