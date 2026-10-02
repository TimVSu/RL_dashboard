"""Entrypoint: `python dashboard/app.py` -> http://localhost:8050

Reads recorded runs from config.RUNS_DIR (see README.md for the on-disk
contract) and never touches the SimPy simulation - it is a pure replay UI.
"""

from __future__ import annotations

import dash
from dash import Dash, dcc, html

from dashboard import config
from dashboard.callbacks import controls, fleet, operator, station, training, vehicle


def create_app() -> Dash:
    app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="app-header",
                children=[
                    html.H2("RL Auction Agent Dashboard"),
                    html.Nav(
                        children=[
                            dcc.Link(page["name"], href=page["path"], className="nav-link")
                            for page in dash.page_registry.values()
                        ]
                    ),
                ],
            ),
            dash.page_container,
        ],
    )

    controls.register(app)
    fleet.register(app)
    vehicle.register(app)
    station.register(app)
    operator.register(app)
    training.register(app)

    return app


app = create_app()


def main() -> None:
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)


if __name__ == "__main__":
    main()
