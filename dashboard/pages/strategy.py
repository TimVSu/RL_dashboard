"""Strategy analysis page - placeholder.

requirements.md section 30 describes this as a post-MVP addition (bid vs
SOC, bid vs station price, station selection distribution, etc.). It is
intentionally not built out yet, per the phased implementation order in
section 50 ("do not implement Phase 4/5 before Phase 1-3 are usable").
"""

from __future__ import annotations

import dash
from dash import html

dash.register_page(__name__, path="/strategy", name="Strategy")


def layout() -> html.Div:
    return html.Div(
        className="page placeholder-page",
        children=[
            html.H3("Strategy analysis"),
            html.P(
                "Not yet implemented. Planned: bid vs SOC, bid vs station "
                "price, bid vs distance, station selection distribution, "
                "no-assignment frequency by epoch. See requirements.md "
                "section 30."
            ),
        ],
    )
