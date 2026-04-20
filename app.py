"""
RoadOpt AI — Streamlit Dashboard
Main entry point: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import io

from config import RESOURCE_POOLS, OBJECTIVE_WEIGHTS
from data_generator import generate_project_tasks, tasks_to_dataframe
from optimizer import compute_critical_path, solve_rcpsp, estimate_cost
from ai_predictor import train_delay_model, predict_task_delays
from live_data import (
    fetch_weather_forecast, weather_to_dataframe,
    fetch_traffic_conditions, traffic_summary_dataframe,
    fetch_calendar_events, events_to_dataframe,
    compute_weekly_external_risk,
)
from constraints import (
    run_all_constraint_checks, FuelBudget, ShiftPolicy,
    validate_shift_constraints, validate_fuel_budget,
    validate_time_windows, validate_priority_milestones,
)
from simulator import (
    SCENARIO_LIBRARY, Scenario, build_custom_scenario,
    run_scenario, run_comparison,
)
from ml_quality import (
    benchmark_models, benchmark_to_dataframe,
    train_quantile_models, predict_with_uncertainty,
    compute_shap_explanations, explain_single_prediction,
    detect_data_drift, simulate_drift_scenario,
)
from monte_carlo import run_monte_carlo
from evm import compute_evm
from crashing import compute_crash_tradeoff
from visualizations import (
    create_gantt_chart_bar,
    create_resource_heatmap,
    create_cost_breakdown,
    create_risk_chart,
    create_feature_importance_chart,
    create_scenario_comparison_bar,
    create_scenario_risk_radar,
    create_external_risk_timeline,
    create_fuel_budget_chart,
    create_milestone_timeline,
    create_model_benchmark_chart,
    create_model_r2_chart,
    create_uncertainty_chart,
    create_shap_bar_chart,
    create_shap_waterfall,
    create_drift_chart,
    create_monte_carlo_histogram,
    create_criticality_chart,
    create_evm_scurve,
    create_crash_tradeoff_chart,
    create_dag_chart,
    create_map_view,
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

    st.subheader("🏗️ Business Constraints")
    max_shift_hrs = st.slider("Max Shift Hours/Week", 36, 60, 48, 2)
    fuel_budget_lakhs = st.slider("Fuel Budget (₹ Lakhs)", 10, 100, 50, 5)
    overtime_mult = st.slider("Overtime Cost Multiplier", 1.0, 3.0, 1.5, 0.1)

    st.subheader("🔬 Scenario Simulator")
    run_simulator = st.checkbox("Run Scenario Comparison", value=False)
    selected_scenarios = st.multiselect(
        "Scenarios to Compare",
        options=list(SCENARIO_LIBRARY.keys()),
        default=["baseline", "monsoon_delay", "labor_crisis"],
        format_func=lambda k: SCENARIO_LIBRARY[k].name,
    )

    st.subheader("📈 EVM Settings")
    evm_progress = st.slider("Simulated Progress %", 10, 90, 50, 5)
    evm_cost_var = st.slider("Cost Variance Factor", 0.90, 1.20, 1.05, 0.01)

    st.subheader("📤 Data Import")
    uploaded_file = st.file_uploader("Upload Task CSV", type=["csv"])
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

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
    tab1, tab2, tab3, tab4, tab_ml, tab5, tab6, tab7, tab_mc, tab_evm, tab_crash, tab_map, tab8 = st.tabs([
        "📊 Schedule", "🔧 Resources", "💰 Cost", "🤖 AI Risk",
        "🧪 ML Quality", "🌦️ Live Data", "📏 Constraints", "🔬 Simulator",
        "🎲 Monte Carlo", "📈 EVM", "⚡ Crashing", "🗺️ Map", "📋 Data",
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

        # DAG Network Graph
        st.subheader("🕸️ Task Dependency Network (DAG)")
        st.plotly_chart(
            create_dag_chart(tasks, result["schedule"]),
            use_container_width=True,
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

    # ── Tab ML: ML Quality Suite ──────────────────────────────────
    with tab_ml:
        st.subheader("🧪 ML Quality Suite")
        st.markdown("Model benchmarking, uncertainty quantification, explainability & drift detection.")

        ml_sub1, ml_sub2, ml_sub3, ml_sub4 = st.tabs([
            "📊 Benchmarking", "📏 Uncertainty", "🔍 Explainability", "📉 Drift Detection",
        ])

        # ── Benchmarking ─────────────────────────────────────────────
        with ml_sub1:
            with st.spinner("Benchmarking models (XGBoost / LightGBM / CatBoost / sklearn)..."):
                bench = benchmark_models()
            bench_df = benchmark_to_dataframe(bench)

            st.success(f"🏆 Best Model: **{bench['best_model_name']}** — MAE={bench['model_results'][bench['best_model_name']]['mae']} weeks")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(create_model_benchmark_chart(bench_df), use_container_width=True)
            with c2:
                st.plotly_chart(create_model_r2_chart(bench_df), use_container_width=True)

            st.subheader("Full Comparison Table")
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

            # Cross-validation info
            if "CV MAE (mean)" in bench_df.columns:
                st.subheader("Cross-Validation (5-Fold)")
                cv_df = bench_df[["Model", "CV MAE (mean)", "CV MAE (std)"]].dropna()
                st.dataframe(cv_df, use_container_width=True, hide_index=True)

        # ── Uncertainty ──────────────────────────────────────────────
        with ml_sub2:
            with st.spinner("Training quantile regressors for confidence intervals..."):
                q_result = train_quantile_models()
                unc_df = predict_with_uncertainty(
                    tasks,
                    q_result["models"],
                    start_month=start_month,
                    crew_experience=crew_exp,
                    material_availability=mat_avail,
                )

            c1, c2, c3 = st.columns(3)
            avg_width = unc_df["interval_width"].mean()
            low_conf = len(unc_df[unc_df["confidence"].str.contains("Low")])
            c1.metric("Avg Interval Width", f"{avg_width:.1f} weeks")
            c2.metric("Low-Confidence Tasks", low_conf)
            c3.metric("Method", "Quantile Regression")

            st.plotly_chart(create_uncertainty_chart(unc_df), use_container_width=True)

            st.subheader("Detailed Predictions with Intervals")
            st.dataframe(
                unc_df[["task_name", "delay_lower_10", "delay_median", "delay_upper_90", "interval_width", "confidence"]],
                use_container_width=True, hide_index=True,
            )

        # ── Explainability ───────────────────────────────────────────
        with ml_sub3:
            with st.spinner("Computing SHAP explanations..."):
                if "bench" not in dir():
                    bench = benchmark_models()
                shap_result = compute_shap_explanations(
                    bench["best_model"], bench["X_train"], bench["X_test"],
                )

            st.info(f"Explainability method: **{shap_result['method']}**")
            st.plotly_chart(
                create_shap_bar_chart(shap_result["global_importance"]),
                use_container_width=True,
            )

            # Per-task explanation
            st.subheader("🔍 Explain Single Task")
            task_names = [t.name for t in tasks]
            selected_task_name = st.selectbox("Select task to explain", task_names)
            selected_task = [t for t in tasks if t.name == selected_task_name][0]

            explanation = explain_single_prediction(
                bench["best_model"], selected_task, tasks,
                start_month=start_month,
                crew_experience=crew_exp,
                material_availability=mat_avail,
                X_train=bench["X_train"],
            )
            st.plotly_chart(
                create_shap_waterfall(explanation),
                use_container_width=True,
            )

            # Show raw contributions
            contrib_rows = []
            for feat, info in explanation["contributions"].items():
                row = {"Feature": feat.replace("_", " ").title(), "Value": info["value"]}
                if "shap_contribution" in info:
                    row["SHAP Contribution"] = info["shap_contribution"]
                    row["Effect"] = info["direction"]
                else:
                    row["Importance"] = info.get("importance", 0)
                contrib_rows.append(row)
            st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True, hide_index=True)

        # ── Drift Detection ──────────────────────────────────────────
        with ml_sub4:
            st.markdown("Detect when incoming data distribution changes vs training data.")

            drift_scenario = st.selectbox(
                "Drift Scenario",
                ["none", "gradual", "sudden", "seasonal"],
                format_func=lambda x: {
                    "none": "✅ No Drift (control)",
                    "gradual": "📈 Gradual Drift (climate change)",
                    "sudden": "⚡ Sudden Drift (labor crisis)",
                    "seasonal": "🌧️ Seasonal Drift (monsoon)",
                }[x],
            )

            with st.spinner("Running drift detection..."):
                ref_data, cur_data = simulate_drift_scenario(drift_type=drift_scenario)
                drift_df = detect_data_drift(ref_data, cur_data)

            drifted = len(drift_df[~drift_df["Status"].str.contains("No Drift")])
            c1, c2, c3 = st.columns(3)
            c1.metric("Features Checked", len(drift_df))
            c2.metric("Features Drifted", drifted)
            c3.metric("Max PSI", f"{drift_df['PSI'].max():.3f}")

            st.plotly_chart(create_drift_chart(drift_df), use_container_width=True)

            st.subheader("Drift Report")
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

            if drifted > 0:
                st.warning(f"⚠️ {drifted} feature(s) show distribution drift — consider retraining the model.")
            else:
                st.success("✅ No significant data drift detected.")

    # ── Tab 5: Live Data (Weather, Traffic, Events) ────────────────
    with tab5:
        st.subheader("🌦️ External Risk Dashboard")
        st.markdown("Real-time weather, traffic, and event data affecting the project timeline.")

        with st.spinner("Fetching live data..."):
            ext_risk = compute_weekly_external_risk(
                start_month=start_month,
                num_weeks=max(result["makespan"], 12),
            )
            weather_data = fetch_weather_forecast(start_month=start_month, num_weeks=max(result["makespan"], 12))
            weather_df = weather_to_dataframe(weather_data)
            traffic_data = fetch_traffic_conditions(num_weeks=max(result["makespan"], 12))
            traffic_df = traffic_summary_dataframe(traffic_data)
            events = fetch_calendar_events(start_month=start_month, num_weeks=max(result["makespan"], 12))
            event_df = events_to_dataframe(events)

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        avg_risk = ext_risk["combined_risk"].mean()
        high_risk_weeks = len(ext_risk[ext_risk["combined_risk"] > 0.45])
        c1.metric("Avg Weekly Risk", f"{avg_risk:.2f}")
        c2.metric("High-Risk Weeks", high_risk_weeks)
        c3.metric("Upcoming Events", len(event_df))
        storm_weeks = len(weather_df[weather_df["condition"].isin(["Storm", "Rain"])])
        c4.metric("Storm/Rain Weeks", storm_weeks)

        st.plotly_chart(create_external_risk_timeline(ext_risk), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🌡️ Weather Forecast")
            st.dataframe(
                weather_df[["date", "condition", "temperature_c", "precipitation_mm", "work_impact"]],
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.subheader("🚗 Traffic Summary")
            st.dataframe(traffic_df, use_container_width=True, hide_index=True)

        if len(event_df) > 0:
            st.subheader("📅 Holidays, Strikes & Events")
            st.dataframe(
                event_df[["date", "event_name", "event_type", "severity", "affects_labor", "affects_transport"]],
                use_container_width=True, hide_index=True,
            )

    # ── Tab 6: Business Constraints ──────────────────────────────────
    with tab6:
        st.subheader("📏 Business Constraint Validation")
        st.markdown("Checks driver shifts, fuel budget, time windows, and priority milestones.")

        with st.spinner("Running constraint checks..."):
            shift_policy = ShiftPolicy(
                max_hours_per_week=max_shift_hrs,
                overtime_multiplier=overtime_mult,
            )
            fuel = FuelBudget(total_budget_inr=fuel_budget_lakhs * 100_000)
            constraint_rpt = run_all_constraint_checks(
                result["schedule"],
                makespan=result["makespan"],
                shift_policy=shift_policy,
                fuel_budget=fuel,
            )

        summary = constraint_rpt["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health", summary["health"])
        c2.metric("Shift Warnings", summary["shift_warnings"])
        c3.metric("Fuel Status", summary["fuel_status"])
        c4.metric("Total Penalties", f"₹{summary['total_penalties_inr']:,.0f}")

        # Fuel chart
        st.plotly_chart(
            create_fuel_budget_chart(constraint_rpt["fuel_report"]),
            use_container_width=True,
        )

        # Milestone timeline
        ms_df = constraint_rpt["milestone_report"]
        if len(ms_df) > 0:
            st.plotly_chart(create_milestone_timeline(ms_df), use_container_width=True)
            st.subheader("🏁 Priority Milestones")
            st.dataframe(
                ms_df[["milestone", "priority", "deadline_week", "projected_finish", "status", "penalty_inr"]],
                use_container_width=True, hide_index=True,
            )

        # Time windows
        tw_df = constraint_rpt["time_window_report"]
        if len(tw_df) > 0:
            st.subheader("⏰ Time Window Compliance")
            st.dataframe(tw_df, use_container_width=True, hide_index=True)

        # Shift report
        shift_df = constraint_rpt["shift_report"]
        if len(shift_df) > 0:
            st.subheader("👷 Shift Compliance")
            st.dataframe(shift_df, use_container_width=True, hide_index=True)

    # ── Tab 7: What-If Simulator ─────────────────────────────────────
    with tab7:
        st.subheader("🔬 What-If Scenario Simulator")
        st.markdown("Compare multiple plans under different disruption scenarios.")

        if run_simulator and len(selected_scenarios) >= 2:
            with st.spinner("Running scenario simulations..."):
                scenarios = [SCENARIO_LIBRARY[k] for k in selected_scenarios]
                comp_df = run_comparison(scenarios, base_resource_caps=resource_caps)

            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            feasible = comp_df[comp_df["Status"] != "INFEASIBLE"]
            if len(feasible) >= 2:
                st.plotly_chart(
                    create_scenario_comparison_bar(feasible),
                    use_container_width=True,
                )
                st.plotly_chart(
                    create_scenario_risk_radar(feasible),
                    use_container_width=True,
                )

            # Find best scenario
            if len(feasible) > 0:
                best = feasible.loc[feasible["Makespan (weeks)"].idxmin()]
                st.success(f"🏆 **Best Scenario:** {best['Scenario']} — {best['Makespan (weeks)']} weeks makespan")

        elif run_simulator:
            st.warning("Select at least 2 scenarios to compare.")
        else:
            st.info("☑️ Enable **Run Scenario Comparison** in the sidebar and select scenarios to compare.")

            # Show scenario library
            st.subheader("📚 Available Scenarios")
            lib_data = []
            for key, sc in SCENARIO_LIBRARY.items():
                lib_data.append({
                    "Key": key,
                    "Name": sc.name,
                    "Description": sc.description,
                    "Labor Shock": f"{sc.labor_shock_pct}%",
                    "Equipment Shock": f"{sc.equipment_shock_pct}%",
                    "Start Month": sc.start_month,
                })
            st.dataframe(pd.DataFrame(lib_data), use_container_width=True, hide_index=True)

    # ── Tab: Monte Carlo Simulation ────────────────────────────────────
    with tab_mc:
        st.subheader("🎲 Monte Carlo Probabilistic Scheduling")
        st.markdown("PERT-distributed durations × 1,000 simulations → confidence intervals.")

        mc_sims = st.select_slider("Simulations", [500, 1000, 2000, 5000], value=1000)
        with st.spinner(f"Running {mc_sims} Monte Carlo simulations..."):
            mc = run_monte_carlo(n_simulations=mc_sims)

        p = mc["percentiles"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P50 (Median)", f"{p['P50']} weeks")
        c2.metric("P80", f"{p['P80']} weeks")
        c3.metric("P90", f"{p['P90']} weeks")
        c4.metric("Mean ± Std", f"{p['Mean']} ± {p['Std']}")

        st.plotly_chart(create_monte_carlo_histogram(mc["makespans"], p), use_container_width=True)
        st.plotly_chart(create_criticality_chart(mc["task_stats"]), use_container_width=True)

        st.subheader("Per-Task PERT Statistics")
        st.dataframe(mc["task_stats"], use_container_width=True, hide_index=True)

    # ── Tab: Earned Value Management ──────────────────────────────────
    with tab_evm:
        st.subheader("📈 Earned Value Management (EVM)")
        st.markdown("S-Curve analysis with CPI, SPI, EAC, ETC indicators.")

        with st.spinner("Computing EVM metrics..."):
            evm_data = compute_evm(
                result["schedule"], result["makespan"],
                progress_pct=evm_progress / 100, cost_variance_factor=evm_cost_var,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CPI", f"{evm_data['CPI']}", delta=evm_data["cost_status"])
        c2.metric("SPI", f"{evm_data['SPI']}", delta=evm_data["schedule_status"])
        c3.metric("EAC", f"₹{evm_data['EAC']:,.0f}")
        c4.metric("BAC", f"₹{evm_data['BAC']:,.0f}")

        st.plotly_chart(create_evm_scurve(evm_data), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cost Variance (CV)", f"₹{evm_data['CV']:,.0f}")
        c2.metric("Schedule Variance (SV)", f"₹{evm_data['SV']:,.0f}")
        c3.metric("Estimate to Complete (ETC)", f"₹{evm_data['ETC']:,.0f}")
        c4.metric("Variance at Completion (VAC)", f"₹{evm_data['VAC']:,.0f}")

        st.subheader("📖 EVM Glossary")
        st.markdown("""
        | Metric | Formula | Meaning |
        |---|---|---|
        | **CPI** | EV / AC | Cost efficiency (>1 = under budget) |
        | **SPI** | EV / PV | Schedule efficiency (>1 = ahead) |
        | **EAC** | BAC / CPI | Estimated total cost at completion |
        | **ETC** | EAC − AC | Remaining cost to finish |
        | **VAC** | BAC − EAC | Expected budget surplus/deficit |
        """)

    # ── Tab: Schedule Crashing ────────────────────────────────────────
    with tab_crash:
        st.subheader("⚡ Schedule Crashing — Cost-Time Tradeoff")
        st.markdown("Progressively crash critical tasks to reduce makespan at additional cost.")

        with st.spinner("Computing crash tradeoff..."):
            crash = compute_crash_tradeoff(max_steps=10)

        if len(crash["tradeoff_curve"]) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Makespan", f"{crash['base_makespan']} weeks")
            c2.metric("Best Crashed Makespan", f"{crash['best_makespan']} weeks")
            c3.metric("Total Crash Cost", f"₹{crash['total_crash_cost']:,.0f}")

            st.plotly_chart(create_crash_tradeoff_chart(crash["tradeoff_curve"]), use_container_width=True)

            st.subheader("Crash Steps")
            st.dataframe(crash["tradeoff_curve"], use_container_width=True, hide_index=True)

            st.subheader("Crash Plan — Available Tasks")
            st.dataframe(crash["crash_plan"], use_container_width=True, hide_index=True)
        else:
            st.warning("Could not compute crash tradeoff.")

    # ── Tab: Map View ────────────────────────────────────────────────
    with tab_map:
        st.subheader("🗺️ Geospatial View — NH-48 Route")
        st.markdown("Tasks plotted along the highway alignment. Red = Critical Path.")
        st.plotly_chart(
            create_map_view(tasks, result["schedule"]),
            use_container_width=True,
        )

    # ── Tab: Raw Data ────────────────────────────────────────────────
    with tab8:
        st.subheader("Task Details")
        st.dataframe(tasks_to_dataframe(tasks), use_container_width=True, hide_index=True)

        st.subheader("Full Schedule")
        st.dataframe(pd.DataFrame(result["schedule"]), use_container_width=True, hide_index=True)

        st.subheader("Risk Predictions")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        # CSV Export
        st.subheader("📥 Export Data")
        csv_buf = io.StringIO()
        pd.DataFrame(result["schedule"]).to_csv(csv_buf, index=False)
        st.download_button("Download Schedule CSV", csv_buf.getvalue(),
                           file_name="roadopt_schedule.csv", mime="text/csv")

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

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("### 🎲 Monte Carlo Simulation")
        st.markdown("PERT distributions + 1000 simulations → P50/P80/P90 confidence intervals")
    with col5:
        st.markdown("### 📈 Earned Value Management")
        st.markdown("S-Curve, CPI, SPI, EAC — full EVM dashboard for project cost & schedule control")
    with col6:
        st.markdown("### ⚡ Schedule Crashing")
        st.markdown("Cost-time tradeoff analysis to accelerate critical path tasks")
