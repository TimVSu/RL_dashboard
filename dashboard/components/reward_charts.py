"""Reward/value training views (requirements.md sections 9, 10, 29, 47).

Kept intentionally small for the MVP: reward by epoch, reward by operator,
reward by timestep for a rollout, and value by timestep. Reward components
(section 48) and policy/critic loss are future additions once the recorder
emits them - nothing here assumes they exist.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.data.models import OperatorInfo

_OPERATOR_COLORS = [
    "#1e3fd6",
    "#d6551e",
    "#1ea67a",
    "#8b1ed6",
    "#d6c31e",
    "#1ec3d6",
]


def _operator_name(operators: list[OperatorInfo], operator_id: int) -> str:
    for op in operators:
        if op.operator_id == operator_id:
            return op.name
    return f"Operator {operator_id}"


def build_reward_by_epoch_figure(
    training_metrics: pd.DataFrame, operators: list[OperatorInfo]
) -> go.Figure:
    fig = go.Figure()
    for i, op in enumerate(operators):
        subset = training_metrics[training_metrics["operator_id"] == op.operator_id].sort_values(
            "epoch"
        )
        fig.add_trace(
            go.Scatter(
                x=subset["epoch"],
                y=subset["total_reward"],
                mode="lines+markers",
                name=op.name,
                line={"color": _OPERATOR_COLORS[i % len(_OPERATOR_COLORS)]},
            )
        )
    fig.update_layout(
        title="Total reward by epoch",
        xaxis_title="Epoch",
        yaxis_title="Total reward",
        height=320,
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def build_reward_value_by_timestep_figure(
    operator_metrics: pd.DataFrame, operators: list[OperatorInfo]
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Reward by timestep", "Critic value by timestep"),
        vertical_spacing=0.1,
    )
    for i, op in enumerate(operators):
        subset = operator_metrics[operator_metrics["operator_id"] == op.operator_id].sort_values(
            "timestep"
        )
        color = _OPERATOR_COLORS[i % len(_OPERATOR_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=subset["timestep"], y=subset["reward"], mode="lines", name=op.name, line={"color": color}
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["timestep"],
                y=subset["value"],
                mode="lines",
                name=op.name,
                line={"color": color, "dash": "dot"},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_layout(
        height=420,
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig
