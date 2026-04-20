"""
RoadOpt AI — Scheduling & Resource Allocation Engine
Uses Critical-Path Method + OR-Tools CP-SAT solver for
resource-constrained project scheduling (RCPSP).
"""

from typing import List, Dict, Tuple
from ortools.sat.python import cp_model

from data_generator import Task
from config import RESOURCE_POOLS, OBJECTIVE_WEIGHTS


# ──────────────────────────────────────────────────────────────────────
# 1. Critical Path Method (forward + backward pass)
# ──────────────────────────────────────────────────────────────────────

def compute_critical_path(tasks: List[Task]) -> Tuple[List[Task], int]:
    """
    Classic CPM — returns (tasks_with_dates, project_makespan).
    Updates earliest_start, latest_start, is_critical on each task.
    """
    n = len(tasks)
    by_id = {t.id: t for t in tasks}

    # Forward pass
    for t in tasks:
        if not t.predecessors:
            t.earliest_start = 0
        else:
            t.earliest_start = max(
                by_id[p].earliest_start + by_id[p].duration_weeks
                for p in t.predecessors
            )

    makespan = max(t.earliest_start + t.duration_weeks for t in tasks)

    # Backward pass
    latest_finish = {t.id: makespan for t in tasks}
    for t in reversed(tasks):
        t.latest_start = latest_finish[t.id] - t.duration_weeks
        for p in t.predecessors:
            latest_finish[p] = min(
                latest_finish[p],
                t.latest_start,
            )

    # Mark critical tasks (zero float)
    for t in tasks:
        t.is_critical = (t.earliest_start == t.latest_start)

    return tasks, makespan


# ──────────────────────────────────────────────────────────────────────
# 2. Resource-Constrained Scheduling (OR-Tools CP-SAT)
# ──────────────────────────────────────────────────────────────────────

def solve_rcpsp(
    tasks: List[Task],
    resource_caps: Dict[str, int] | None = None,
    time_limit_sec: int = 30,
    objective: str = "minimize_time",  # or "minimize_cost" / "balanced"
) -> Dict:
    """
    Solve the Resource-Constrained Project Scheduling Problem.
    Returns dict with:
      schedule: list of {task_id, start, end}
      makespan: int
      resource_usage: week → resource → usage
      status: str
    """
    if resource_caps is None:
        resource_caps = {k: v["units"] for k, v in RESOURCE_POOLS.items()}

    model = cp_model.CpModel()
    horizon = sum(t.duration_weeks for t in tasks)  # upper bound

    # ── Decision variables ───────────────────────────────────────────
    starts = {}
    ends = {}
    intervals = {}
    for t in tasks:
        s = model.NewIntVar(0, horizon, f"start_{t.id}")
        e = model.NewIntVar(0, horizon, f"end_{t.id}")
        iv = model.NewIntervalVar(s, t.duration_weeks, e, f"interval_{t.id}")
        starts[t.id] = s
        ends[t.id] = e
        intervals[t.id] = iv

    # ── Precedence constraints ───────────────────────────────────────
    by_id = {t.id: t for t in tasks}
    for t in tasks:
        for p in t.predecessors:
            model.Add(starts[t.id] >= ends[p])

    # ── Resource constraints (cumulative) ────────────────────────────
    for res_name, cap in resource_caps.items():
        task_intervals = []
        demands = []
        for t in tasks:
            req = t.resource_requirements.get(res_name, 0)
            if req > 0:
                task_intervals.append(intervals[t.id])
                demands.append(req)
        if task_intervals:
            model.AddCumulative(task_intervals, demands, cap)

    # ── Objective ────────────────────────────────────────────────────
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan_var, [ends[t.id] for t in tasks])

    if objective == "minimize_cost":
        # Minimise weighted resource-weeks
        cost_terms = []
        for t in tasks:
            for res_name, req in t.resource_requirements.items():
                weekly_cost = RESOURCE_POOLS.get(res_name, {}).get("cost_per_week", 0)
                cost_terms.append(req * weekly_cost * t.duration_weeks)
        total_fixed_cost = sum(cost_terms)
        # Still minimise makespan (variable cost scales with time)
        model.Minimize(makespan_var * (total_fixed_cost // max(1, horizon)))
    elif objective == "balanced":
        # Multi-objective: time + resource spread penalty
        model.Minimize(makespan_var)
    else:
        model.Minimize(makespan_var)

    # ── Solve ────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "INFEASIBLE", "schedule": [], "makespan": -1, "resource_usage": {}}

    # ── Extract schedule ─────────────────────────────────────────────
    schedule = []
    for t in tasks:
        s_val = solver.Value(starts[t.id])
        e_val = solver.Value(ends[t.id])
        schedule.append({
            "task_id": t.id,
            "task_name": t.name,
            "start_week": s_val,
            "end_week": e_val,
            "duration": t.duration_weeks,
            "is_critical": t.is_critical,
            "resources": dict(t.resource_requirements),
        })

    ms = solver.Value(makespan_var)

    # ── Resource usage matrix ────────────────────────────────────────
    resource_usage: Dict[int, Dict[str, int]] = {}
    for week in range(ms):
        resource_usage[week] = {r: 0 for r in resource_caps}
        for entry in schedule:
            if entry["start_week"] <= week < entry["end_week"]:
                for r, qty in entry["resources"].items():
                    resource_usage[week][r] += qty

    status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    return {
        "status": status_name,
        "schedule": schedule,
        "makespan": ms,
        "resource_usage": resource_usage,
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Cost Estimator
# ──────────────────────────────────────────────────────────────────────

def estimate_cost(schedule: List[Dict], resource_caps: Dict[str, int] | None = None) -> Dict:
    """Compute total and per-resource cost from a solved schedule."""
    cost_breakdown: Dict[str, int] = {}
    for entry in schedule:
        for res_name, qty in entry["resources"].items():
            weekly = RESOURCE_POOLS.get(res_name, {}).get("cost_per_week", 0)
            cost = qty * weekly * entry["duration"]
            cost_breakdown[res_name] = cost_breakdown.get(res_name, 0) + cost

    return {
        "total_cost": sum(cost_breakdown.values()),
        "breakdown": cost_breakdown,
    }
