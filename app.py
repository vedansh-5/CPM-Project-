"""
RoadOpt AI — Streamlit Dashboard
Main entry point: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from config import RESOURCE_POOLS, OBJECTIVE_WEIGHTS
from data_generator import generate_project_tasks, tasks_to_dataframe
from optimizer import compute_critical_path, solve_rcpsp, estimate_cost
from ai_predictor import train_delay_model, predict_task_delays
from visualizations import (
    create_gantt_chart_bar,
    create_resource_heatmap,
    create_cost_breakdown,
    create_risk_chart,
    create_feature_importance_chart,
)

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadOpt AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛣️ RoadOpt AI")
st.markdown("**AI-Driven Scheduling & Resource Allocation for Road Construction Projects**")
st.divider()

# ─── Sidebar: Configuration ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Project Configuration")

    st.subheader("Objective")
    objective = st.selectbox(
        "Optimization Goal",
        ["minimize_time", "minimize_cost", "balanced"],
        format_func=lambda x: {
            "minimize_time": "⏱️ Minimize Completion Time",
            "minimize_cost": "💰 Minimize Total Cost",
            "balanced": "⚖️ Balanced (Time + Resources)",
        }[x],
    )

    st.subheader("Resource Caps")
    resource_caps = {}
    for res_name, info in RESOURCE_POOLS.items():
        label = res_name.replace("_", " ").title()
        resource_caps[res_name] = st.slider(
            label,
            min_value=1,
            max_value=info["units"] * 3,
            value=info["units"],
            key=f"res_{res_name}",
        )

    st.subheader("AI Predictor Settings")
    crew_exp = st.slider("Crew Experience (1–5)", 1, 5, 3)
    mat_avail = st.slider("Material Availability", 0.3, 1.0, 0.85, 0.05)
    start_month = st.selectbox(
        "Project Start Month",
        list(range(1, 13)),
        format_func=lambda m: [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ][m - 1],
        index=3,
    )

    st.subheader("What-If Simulation")
    labor_shock = st.slider("Labor Shortage %", 0, 50, 0, 5)
    equipment_shock = st.slider("Equipment Breakdown %", 0, 50, 0, 5)

    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# ─── Main Logic ──────────────────────────────────────────────────────
if run_btn or "schedule" in st.session_state:

    # Apply what-if shocks
    adjusted_caps = dict(resource_caps)
    if labor_shock > 0:
        adjusted_caps["labor_crew"] = max(1, int(adjusted_caps["labor_crew"] * (1 - labor_shock / 100)))
    if equipment_shock > 0:
        for eq in ["excavator", "bulldozer", "asphalt_paver", "roller_compactor", "crane"]:
            adjusted_caps[eq] = max(1, int(adjusted_caps[eq] * (1 - equipment_shock / 100)))

    # Generate tasks
    with st.spinner("Generating project tasks..."):
        tasks = generate_project_tasks()
        tasks, cpm_makespan = compute_critical_path(tasks)

    # Solve RCPSP
    with st.spinner("Running AI-powered optimization (OR-Tools CP-SAT)..."):
        result = solve_rcpsp(
            tasks,
            resource_caps=adjusted_caps,
            objective=objective,
        )

    if result["status"] == "INFEASIBLE":
        st.error("❌ No feasible schedule found! Try increasing resource caps or reducing constraints.")
        st.stop()

    st.session_state["schedule"] = result

    # Cost
    cost_data = estimate_cost(result["schedule"])

    # AI Delay Prediction
    with st.spinner("Training AI delay predictor..."):
        metrics = train_delay_model()
        risk_df = predict_task_delays(
            tasks,
            resource_utilization=0.7,
            start_month=start_month,
            crew_experience=crew_exp,
            material_availability=mat_avail,
        )

    # ─── Dashboard Tabs ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Schedule", "🔧 Resources", "💰 Cost", "🤖 AI Risk", "📋 Data"
    ])

    # ── Tab 1: Schedule ──────────────────────────────────────────────
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Project Makespan", f"{result['makespan']} weeks")
        col2.metric("CPM Makespan (ideal)", f"{cpm_makespan} weeks")
        col3.metric("Total Tasks", len(tasks))
        col4.metric("Solver Status", result["status"])

        st.plotly_chart(
            create_gantt_chart_bar(result["schedule"]),
            use_container_width=True,
        )

        # Critical path highlight
        critical_tasks = [s for s in result["schedule"] if s["is_critical"]]
        st.subheader("🔴 Critical Path Tasks")
        st.dataframe(
            pd.DataFrame(critical_tasks)[["task_name", "start_week", "end_week", "duration"]],
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 2: Resources ─────────────────────────────────────────────
    with tab2:
        st.plotly_chart(
            create_resource_heatmap(result["resource_usage"], adjusted_caps),
            use_container_width=True,
        )

        # Peak utilization table
        st.subheader("Peak Resource Utilization")
        peak_data = []
        for res in adjusted_caps:
            peak = max(
                result["resource_usage"][w].get(res, 0) for w in result["resource_usage"]
            )
            cap = adjusted_caps[res]
            peak_data.append({
                "Resource": res.replace("_", " ").title(),
                "Capacity": cap,
                "Peak Usage": peak,
                "Peak %": f"{peak / cap * 100:.0f}%" if cap > 0 else "N/A",
                "Status": "🔴 Over" if peak > cap else ("🟡 High" if peak / cap > 0.8 else "🟢 OK"),
            })
        st.dataframe(pd.DataFrame(peak_data), use_container_width=True, hide_index=True)

    # ── Tab 3: Cost ──────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Total Estimated Cost", f"₹{cost_data['total_cost']:,.0f}")
            st.plotly_chart(create_cost_breakdown(cost_data), use_container_width=True)
        with col2:
            st.subheader("Cost by Resource Type")
            cost_df = pd.DataFrame([
                {"Resource": k.replace("_", " ").title(), "Cost (₹)": f"{v:,.0f}"}
                for k, v in cost_data["breakdown"].items()
            ]).sort_values("Cost (₹)", ascending=False)
            st.dataframe(cost_df, use_container_width=True, hide_index=True)

    # ── Tab 4: AI Risk ───────────────────────────────────────────────
    with tab4:
        col1, col2, col3 = st.columns(3)
        col1.metric("Model MAE", f"{metrics['mae']} weeks")
        col2.metric("Model R²", f"{metrics['r2']}")
        col3.metric("Training Samples", metrics["train_size"])

        st.plotly_chart(create_risk_chart(risk_df), use_container_width=True)
        st.plotly_chart(
            create_feature_importance_chart(metrics["feature_importance"]),
            use_container_width=True,
        )

        st.subheader("🔴 High-Risk Tasks")
        high_risk = risk_df[risk_df["risk_level"].str.contains("High")]
        if len(high_risk) > 0:
            st.dataframe(high_risk, use_container_width=True, hide_index=True)
        else:
            st.success("No high-risk tasks detected! ✅")

    # ── Tab 5: Raw Data ──────────────────────────────────────────────
    with tab5:
        st.subheader("Task Details")
        st.dataframe(tasks_to_dataframe(tasks), use_container_width=True, hide_index=True)

        st.subheader("Full Schedule")
        st.dataframe(pd.DataFrame(result["schedule"]), use_container_width=True, hide_index=True)

        st.subheader("Risk Predictions")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

else:
    # Landing page
    st.info("👈 Configure your project in the sidebar and click **Run Optimization**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Smart Scheduling")
        st.markdown("OR-Tools CP-SAT solver for resource-constrained scheduling with Critical Path analysis")
    with col2:
        st.markdown("### 🤖 AI Delay Prediction")
        st.markdown("Gradient Boosting model predicts delay risk using weather, complexity & resource data")
    with col3:
        st.markdown("### 🔧 What-If Analysis")
        st.markdown("Simulate labor shortages, equipment failures and see the impact in real-time")
