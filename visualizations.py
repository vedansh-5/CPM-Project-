"""
RoadOpt AI — Visualization Helpers
Plotly-based charts: Gantt, Resource Heatmap, Cost Breakdown, Risk Matrix
"""

import numpy as np
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


# ═══════════════════════════════════════════════════════════════════════
# NEW: Scenario Comparison Charts
# ═══════════════════════════════════════════════════════════════════════

def create_scenario_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing makespan & cost across scenarios."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Makespan (weeks)",
        x=comparison_df["Scenario"],
        y=comparison_df["Makespan (weeks)"],
        marker_color="#3498db",
        text=comparison_df["Makespan (weeks)"],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Predicted Delay (weeks)",
        x=comparison_df["Scenario"],
        y=comparison_df["Predicted Delay (weeks)"],
        marker_color="#e74c3c",
        text=comparison_df["Predicted Delay (weeks)"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Scenario Comparison — Makespan & Predicted Delay",
        barmode="group",
        xaxis_title="Scenario",
        yaxis_title="Weeks",
        height=450,
    )
    return fig


def create_scenario_risk_radar(comparison_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing risk dimensions across scenarios."""
    categories = ["Makespan", "Predicted Delay", "High-Risk Tasks", "External Risk"]

    fig = go.Figure()
    for _, row in comparison_df.iterrows():
        if row["Status"] == "INFEASIBLE":
            continue
        vals = [
            row["Makespan (weeks)"] / max(comparison_df["Makespan (weeks)"].max(), 1),
            row["Predicted Delay (weeks)"] / max(comparison_df["Predicted Delay (weeks)"].max(), 1),
            row["High-Risk Tasks"] / max(comparison_df["High-Risk Tasks"].max(), 1),
            row["Avg External Risk"] / max(comparison_df["Avg External Risk"].max(), 0.01),
        ]
        vals.append(vals[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Scenario"],
            opacity=0.6,
        ))

    fig.update_layout(
        title="Risk Radar — Scenario Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
        showlegend=True,
    )
    return fig


def create_external_risk_timeline(risk_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of weather / traffic / event risk over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk_df["week"], y=risk_df["weather_risk"],
        mode="lines", name="Weather Risk",
        fill="tozeroy", line=dict(color="#3498db"),
    ))
    fig.add_trace(go.Scatter(
        x=risk_df["week"], y=risk_df["traffic_risk"],
        mode="lines", name="Traffic Risk",
        fill="tonexty", line=dict(color="#e67e22"),
    ))
    fig.add_trace(go.Scatter(
        x=risk_df["week"], y=risk_df["event_risk"],
        mode="lines", name="Event Risk",
        fill="tonexty", line=dict(color="#e74c3c"),
    ))
    fig.update_layout(
        title="External Risk Timeline (Weather + Traffic + Events)",
        xaxis_title="Project Week",
        yaxis_title="Risk Score",
        height=400,
        yaxis=dict(range=[0, 1.2]),
    )
    return fig


def create_fuel_budget_chart(fuel_data: Dict) -> go.Figure:
    """Weekly fuel cost bar chart with budget cap line."""
    weeks = sorted(fuel_data["weekly_fuel_cost"].keys())
    costs = [fuel_data["weekly_fuel_cost"][w] for w in weeks]

    colors = ["#e74c3c" if c > fuel_data["weekly_cap"] else "#27ae60" for c in costs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"W{w+1}" for w in weeks],
        y=costs,
        marker_color=colors,
        name="Fuel Cost",
    ))
    fig.add_hline(
        y=fuel_data["weekly_cap"],
        line_dash="dash", line_color="red",
        annotation_text=f"Weekly Cap: ₹{fuel_data['weekly_cap']:,.0f}",
    )
    fig.update_layout(
        title=f"Weekly Fuel Cost — {fuel_data['status']} (Total: ₹{fuel_data['total_fuel_cost']:,.0f})",
        xaxis_title="Week",
        yaxis_title="Cost (₹)",
        height=400,
    )
    return fig


def create_milestone_timeline(ms_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of milestones: deadline vs projected finish."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ms_df["milestone"],
        x=ms_df["deadline_week"],
        name="Deadline",
        orientation="h",
        marker_color="#3498db",
        opacity=0.5,
    ))
    fig.add_trace(go.Bar(
        y=ms_df["milestone"],
        x=ms_df["projected_finish"],
        name="Projected Finish",
        orientation="h",
        marker_color=["#e74c3c" if r > 0 else "#27ae60" for r in ms_df["weeks_late"]],
    ))
    fig.update_layout(
        title="Priority Milestones — Deadline vs Projected",
        xaxis_title="Project Week",
        height=350,
        barmode="overlay",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ML Quality Charts
# ═══════════════════════════════════════════════════════════════════════

def create_model_benchmark_chart(bench_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing MAE & R² across models."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE (weeks)",
        x=bench_df["Model"],
        y=bench_df["MAE (weeks)"],
        marker_color="#e74c3c",
        text=bench_df["MAE (weeks)"],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="RMSE (weeks)",
        x=bench_df["Model"],
        y=bench_df["RMSE (weeks)"],
        marker_color="#f39c12",
        text=bench_df["RMSE (weeks)"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Model Benchmark — MAE & RMSE Comparison",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Error (weeks)",
        height=420,
    )
    return fig


def create_model_r2_chart(bench_df: pd.DataFrame) -> go.Figure:
    """Bar chart of R² scores."""
    colors = ["#27ae60" if r > 0.7 else ("#f39c12" if r > 0.4 else "#e74c3c") for r in bench_df["R²"]]
    fig = go.Figure(go.Bar(
        x=bench_df["Model"],
        y=bench_df["R²"],
        marker_color=colors,
        text=bench_df["R²"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Model Benchmark — R² Score",
        xaxis_title="Model",
        yaxis_title="R²",
        height=380,
        yaxis=dict(range=[0, 1]),
    )
    return fig


def create_uncertainty_chart(unc_df: pd.DataFrame) -> go.Figure:
    """Error bar chart showing confidence intervals per task."""
    df = unc_df.sort_values("delay_median", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["delay_median"],
        y=df["task_name"],
        mode="markers",
        marker=dict(size=10, color="#2980b9"),
        name="Median Prediction",
        error_x=dict(
            type="data",
            symmetric=False,
            array=(df["delay_upper_90"] - df["delay_median"]).tolist(),
            arrayminus=(df["delay_median"] - df["delay_lower_10"]).tolist(),
            color="#e74c3c",
            thickness=2,
            width=6,
        ),
    ))
    fig.update_layout(
        title="Delay Prediction with 80% Confidence Interval (P10–P90)",
        xaxis_title="Predicted Delay (weeks)",
        height=max(450, len(df) * 28),
    )
    return fig


def create_shap_bar_chart(global_importance: Dict[str, float]) -> go.Figure:
    """SHAP global importance bar chart."""
    sorted_feats = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").title() for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#8e44ad",
    ))
    fig.update_layout(
        title="SHAP — Global Feature Importance (mean |SHAP|)",
        xaxis_title="Mean |SHAP value|",
        height=380,
    )
    return fig


def create_shap_waterfall(explanation: Dict) -> go.Figure:
    """Waterfall chart for a single prediction's SHAP contributions."""
    contribs = explanation.get("contributions", {})
    if not contribs:
        return go.Figure()

    names = []
    values = []
    for feat, info in contribs.items():
        names.append(feat.replace("_", " ").title())
        if "shap_contribution" in info:
            values.append(info["shap_contribution"])
        else:
            values.append(info.get("importance", 0))

    # Sort by absolute value
    order = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
    ))
    base_val = explanation.get("base_value", 0)
    pred_val = explanation.get("prediction", 0)
    fig.update_layout(
        title=f"SHAP Explanation — {explanation['task_name']} (pred={pred_val:.1f}w, base={base_val:.1f}w)",
        xaxis_title="SHAP Contribution (weeks)",
        height=380,
    )
    return fig


def create_drift_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Bar chart of PSI values per feature with drift thresholds."""
    colors = []
    for _, row in drift_df.iterrows():
        if "No Drift" in row["Status"]:
            colors.append("#27ae60")
        elif "Minor" in row["Status"]:
            colors.append("#f39c12")
        elif "Moderate" in row["Status"]:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")

    fig = go.Figure(go.Bar(
        x=drift_df["Feature"],
        y=drift_df["PSI"],
        marker_color=colors,
        text=drift_df["PSI"],
        textposition="auto",
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                  annotation_text="Minor drift")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="Significant drift")
    fig.update_layout(
        title="Data Drift Detection — PSI per Feature",
        xaxis_title="Feature",
        yaxis_title="Population Stability Index (PSI)",
        height=420,
    )
    return fig
