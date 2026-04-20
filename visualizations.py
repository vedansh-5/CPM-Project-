"""
RoadOpt AI — Visualization Helpers
Plotly-based charts: Gantt, Resource Heatmap, Cost Breakdown, Risk Matrix
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List


def create_gantt_chart(schedule: List[Dict], title: str = "Project Schedule") -> go.Figure:
    """Interactive Gantt chart from solver schedule."""
    df = pd.DataFrame(schedule)
    df["color"] = df["is_critical"].map({True: "Critical Path", False: "Non-Critical"})

    fig = px.timeline(
        df,
        x_start=df["start_week"].apply(lambda w: f"2026-W{w+1:02d}"),
        x_end=df["end_week"].apply(lambda w: f"2026-W{w+1:02d}"),
        y="task_name",
        color="color",
        color_discrete_map={"Critical Path": "#e74c3c", "Non-Critical": "#3498db"},
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=700, xaxis_title="Week", yaxis_title="", showlegend=True)
    return fig


def create_gantt_chart_bar(schedule: List[Dict], title: str = "Project Schedule — Gantt") -> go.Figure:
    """Horizontal bar Gantt (works without date parsing issues)."""
    df = pd.DataFrame(schedule).sort_values("start_week")

    colors = ["#e74c3c" if row["is_critical"] else "#3498db" for _, row in df.iterrows()]

    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["duration"]],
            y=[row["task_name"]],
            base=[row["start_week"]],
            orientation="h",
            marker_color="#e74c3c" if row["is_critical"] else "#3498db",
            name="Critical" if row["is_critical"] else "Normal",
            showlegend=False,
            hovertemplate=(
                f"<b>{row['task_name']}</b><br>"
                f"Start: Week {row['start_week']}<br>"
                f"End: Week {row['end_week']}<br>"
                f"Duration: {row['duration']} weeks<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Project Week",
        yaxis_title="",
        height=max(500, len(df) * 28),
        barmode="stack",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_resource_heatmap(
    resource_usage: Dict[int, Dict[str, int]],
    resource_caps: Dict[str, int],
) -> go.Figure:
    """Heatmap: weeks × resources, color = utilization %."""
    weeks = sorted(resource_usage.keys())
    resources = sorted(resource_caps.keys())

    z = []
    for res in resources:
        row = []
        cap = resource_caps[res]
        for w in weeks:
            usage = resource_usage[w].get(res, 0)
            pct = round(usage / cap * 100, 1) if cap > 0 else 0
            row.append(pct)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"W{w+1}" for w in weeks],
        y=[r.replace("_", " ").title() for r in resources],
        colorscale="RdYlGn_r",
        zmin=0,
        zmax=100,
        colorbar_title="Utilization %",
        hovertemplate="Week %{x}<br>%{y}: %{z:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Resource Utilization Heatmap",
        xaxis_title="Project Week",
        height=450,
    )
    return fig


def create_cost_breakdown(cost_data: Dict) -> go.Figure:
    """Donut chart of cost by resource type."""
    labels = [k.replace("_", " ").title() for k in cost_data["breakdown"]]
    values = list(cost_data["breakdown"].values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
    )])
    fig.update_layout(
        title=f"Cost Breakdown — Total: ₹{cost_data['total_cost']:,.0f}",
        height=450,
    )
    return fig


def create_risk_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of predicted delay per task."""
    df = risk_df.sort_values("predicted_delay_weeks", ascending=True)

    colors = []
    for _, row in df.iterrows():
        if "Low" in row["risk_level"]:
            colors.append("#27ae60")
        elif "Medium" in row["risk_level"]:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig = go.Figure(go.Bar(
        x=df["predicted_delay_weeks"],
        y=df["task_name"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.1f} weeks delay<extra></extra>",
    ))
    fig.update_layout(
        title="AI-Predicted Delay Risk per Task",
        xaxis_title="Predicted Delay (weeks)",
        yaxis_title="",
        height=max(450, len(df) * 25),
    )
    return fig


def create_feature_importance_chart(feature_imp: Dict[str, float]) -> go.Figure:
    """Bar chart of ML feature importances."""
    sorted_feats = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").title() for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#8e44ad",
    ))
    fig.update_layout(
        title="Delay Prediction — Feature Importance",
        xaxis_title="Importance",
        height=350,
    )
    return fig
