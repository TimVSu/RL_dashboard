"""Epoch/run comparison page - placeholder.

requirements.md sections 31-32: side-by-side epoch/run comparison. Post-MVP;
not built out yet (see the phased implementation order in section 50).
"""

from __future__ import annotations

import dash
from dash import html

dash.register_page(__name__, path="/compare", name="Compare")


def layout() -> html.Div:
    return html.Div(
        className="page placeholder-page",
        children=[
            html.H3("Epoch / run comparison"),
            html.P(
                "Not yet implemented. Planned: side-by-side H3 map, fleet, "
                "and metrics for two epochs or two runs. See requirements.md "
                "sections 31-32."
            ),
        ],
    )
