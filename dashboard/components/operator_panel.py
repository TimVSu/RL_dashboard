"""Operator-level view: per-step earnings/bids/reward, and reward-component
history over a rollout.

This is the primary "is training working, and if not, which part isn't"
view (requirements.md section 48 in spirit): reward alone doesn't say
whether an operator is failing to earn, over-paying in cost, or leaving too
many vehicles at low SOC - the component breakdown does. Built from two
tables: operator_metrics.parquet (always present - reward/value, reshaped
from the PPO buffers) and reward_components.parquet (optional - only
present for runs recorded after this table was added; see
dashboard/data/loader.load_reward_components). vehicle_states.parquet
supplies the per-step bidding summary, since bids/allocations aren't part
of either reward table.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import html
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

_WINNINGS_FACTOR_COLOR = "#1ea67a"
_LOW_SOC_COLOR = "#d6551e"
_COST_PENALTY_COLOR = "#8b1ed6"
_REWARD_COLOR = "#1a1a1a"


def _field(label: str, value: str) -> html.Div:
    return html.Div(
        className="detail-field",
        children=[html.Span(label, className="detail-label"), html.Span(value, className="detail-value")],
    )


def build_operator_cards(
    operators: list[OperatorInfo],
    operator_metrics_now: pd.DataFrame,
    operator_metrics_epoch: pd.DataFrame,
    reward_components_now: pd.DataFrame,
    reward_components_epoch: pd.DataFrame,
    vehicles_now: pd.DataFrame,
) -> html.Div:
    cards = []
    for i, op in enumerate(operators):
        color = _OPERATOR_COLORS[i % len(_OPERATOR_COLORS)]
        op_vehicles = vehicles_now[vehicles_now["operator_id"] == op.operator_id]
        fleet_size = len(op_vehicles)
        num_bidding = int(op_vehicles["is_bid"].sum()) if fleet_size else 0
        num_charging = int(op_vehicles["has_commitment"].sum()) if fleet_size else 0
        mean_soc = float(op_vehicles["soc"].mean()) if fleet_size else None
        total_bid_value = float(op_vehicles.loc[op_vehicles["is_bid"], "bid_price"].sum()) if fleet_size else 0.0

        metrics_row = operator_metrics_now[operator_metrics_now["operator_id"] == op.operator_id]
        reward = float(metrics_row["reward"].iloc[0]) if not metrics_row.empty else None
        value = float(metrics_row["value"].iloc[0]) if not metrics_row.empty else None

        # Summed across every rollout/timestep in the currently-selected
        # epoch (not just the current timestep/rollout) - a quick,
        # at-a-glance check for whether the epoch as a whole is trending up
        # across training, independent of where the timestep slider happens
        # to be pointed right now.
        epoch_rows = operator_metrics_epoch[operator_metrics_epoch["operator_id"] == op.operator_id]
        reward_sum = float(epoch_rows["reward"].sum()) if not epoch_rows.empty else None

        components_row = reward_components_now[reward_components_now["operator_id"] == op.operator_id]
        has_components = not components_row.empty

        # Epoch means (not sums) of the per-step reward components -
        # winnings_factor/low_soc_penalty/cost_penalty are each already a
        # per-step magnitude, not something that accumulates meaningfully
        # across steps, so the average over the epoch is what's directly
        # comparable to the per-step values in the reward-breakdown chart
        # below. Grouped together at the bottom of the card (see fields
        # list) so the three are easy to compare side by side to see which
        # one dominated the epoch.
        components_epoch_rows = reward_components_epoch[
            reward_components_epoch["operator_id"] == op.operator_id
        ]
        mean_winnings_factor = (
            float(components_epoch_rows["winnings_factor"].mean())
            if not components_epoch_rows.empty
            else None
        )
        mean_low_soc_penalty = (
            float(components_epoch_rows["low_soc_penalty"].mean())
            if not components_epoch_rows.empty
            else None
        )
        mean_cost_penalty = (
            float(components_epoch_rows["cost_penalty"].mean())
            if not components_epoch_rows.empty
            else None
        )

        fields = [
            _field("Fleet size", str(fleet_size)),
            _field("Mean SOC", f"{mean_soc:.0%}" if mean_soc is not None else "-"),
            _field("Bidding now", f"{num_bidding} / {fleet_size}"),
            _field("Charging (committed)", f"{num_charging} / {fleet_size}"),
            _field("Total bid value", f"{total_bid_value:.2f}"),
            _field("Critic value", f"{value:.3f}" if value is not None else "-"),
            _field("Reward sum (epoch)", f"{reward_sum:.3f}" if reward_sum is not None else "-"),
        ]

        if has_components:
            c = components_row.iloc[0]
            fields += [
                _field("Earnings", f"{c['earnings']:.3f}"),
                _field("Cost", f"{c['cost']:.3f}"),
                _field("Winnings factor", f"{c['winnings_factor']:.3f}"),
                _field("Low-SOC penalty", f"{c['low_soc_penalty']:.3f}"),
                _field("Cost penalty", f"{c['cost_penalty']:.3f}"),
                # Epoch means, grouped together at the end for easy visual
                # comparison of which component dominated this epoch.
                _field(
                    "Winnings factor (epoch mean)",
                    f"{mean_winnings_factor:.3f}" if mean_winnings_factor is not None else "-",
                ),
                _field(
                    "Low-SOC penalty (epoch mean)",
                    f"{mean_low_soc_penalty:.3f}" if mean_low_soc_penalty is not None else "-",
                ),
                _field(
                    "Cost penalty (epoch mean)",
                    f"{mean_cost_penalty:.3f}" if mean_cost_penalty is not None else "-",
                ),
            ]
        else:
            fields.append(
                html.P(
                    "Reward-component breakdown not recorded for this run.",
                    className="placeholder-text",
                )
            )

        cards.append(
            html.Div(
                className="operator-card",
                style={"borderTop": f"3px solid {color}"},
                children=[
                    html.H4(op.name),
                    html.Div(
                        className="operator-reward-headline",
                        children=[
                            html.Span("Reward", className="detail-label"),
                            html.Span(
                                f"{reward:.3f}" if reward is not None else "-",
                                className="operator-reward-value",
                            ),
                        ],
                    ),
                    *fields,
                ],
            )
        )

    return html.Div(className="operator-row", children=cards)


def build_operator_reward_breakdown_figure(
    reward_components_history: pd.DataFrame, operators: list[OperatorInfo]
) -> go.Figure:
    """Reward-component lines, one row per operator.

    Plots winnings_factor, low_soc_penalty and cost_penalty as their raw
    magnitudes (not negated, even though the reward formula subtracts the
    two penalties) alongside the resulting reward - these four are the
    actual terms of reward = winnings_factor - low_soc_penalty -
    cost_penalty, all on a comparable scale. earnings/cost are deliberately
    left out: they're raw currency-scale values that can run orders of
    magnitude larger than the other components, which used to flatten
    everything else on the same axis into invisibility.
    """
    fig = make_subplots(
        rows=max(len(operators), 1),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[op.name for op in operators] or ["No operators"],
        vertical_spacing=0.08,
    )
    series = (
        ("winnings_factor", "Winnings factor", _WINNINGS_FACTOR_COLOR, 1.5),
        ("low_soc_penalty", "Low-SOC penalty", _LOW_SOC_COLOR, 1.5),
        ("cost_penalty", "Cost penalty", _COST_PENALTY_COLOR, 1.5),
        ("reward", "Reward", _REWARD_COLOR, 2),
    )
    for i, op in enumerate(operators, start=1):
        subset = reward_components_history[
            reward_components_history["operator_id"] == op.operator_id
        ].sort_values("timestep")
        show_legend = i == 1
        for column, name, color, width in series:
            y = subset[column]
            fig.add_trace(
                go.Scatter(
                    x=subset["timestep"],
                    y=y,
                    name=name,
                    mode="lines+markers",
                    line={"color": color, "width": width},
                    legendgroup=column,
                    showlegend=show_legend,
                ),
                row=i,
                col=1,
            )

    fig.update_layout(
        height=220 * max(len(operators), 1) + 40,
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.05 / max(len(operators), 1)},
    )
    fig.update_xaxes(title_text="Timestep", row=max(len(operators), 1), col=1)
    return fig
