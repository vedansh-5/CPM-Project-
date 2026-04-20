"""
RoadOpt AI — Business Constraints Engine
Enforces real-world business rules on top of the OR-Tools schedule:
  • Driver / crew shift limits (max hours per day/week)
  • Fuel budget caps
  • Delivery time windows per task
  • Priority customers / milestones
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import RESOURCE_POOLS


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ShiftPolicy:
    """Driver / crew shift limits."""
    max_hours_per_day: float = 10.0
    max_hours_per_week: float = 48.0
    mandatory_rest_hours: float = 10.0   # between shifts
    max_consecutive_days: int = 6        # then 1 day off
    overtime_multiplier: float = 1.5     # cost multiplier for OT
    night_shift_allowed: bool = False


@dataclass
class FuelBudget:
    """Fuel / energy budget constraints."""
    total_budget_inr: float = 5_000_000.0        # ₹50 lakh
    weekly_cap_inr: float = 150_000.0             # ₹1.5 lakh/week
    fuel_price_per_litre: float = 95.0            # diesel ₹/L
    avg_consumption: Dict[str, float] = field(default_factory=lambda: {
        # litres per operating hour for each equipment
        "excavator": 25.0,
        "bulldozer": 30.0,
        "dump_truck": 18.0,
        "asphalt_paver": 20.0,
        "roller_compactor": 12.0,
        "crane": 22.0,
        "concrete_mixer": 8.0,
    })
    operating_hours_per_week: float = 48.0


@dataclass
class TimeWindow:
    """Delivery / completion time window for a task."""
    task_id: int
    task_name: str
    earliest_start_week: int = 0
    latest_finish_week: int = 999
    is_hard_constraint: bool = True      # hard = must obey, soft = penalize
    penalty_per_week_late: float = 100_000.0  # ₹ per week if soft


@dataclass
class PriorityCustomer:
    """Priority milestone / customer-linked tasks."""
    milestone_name: str
    task_ids: List[int]
    deadline_week: int
    priority_level: int = 1              # 1 = highest
    contractual_penalty_inr: float = 500_000.0  # per week late
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_SHIFT_POLICY = ShiftPolicy()

DEFAULT_FUEL_BUDGET = FuelBudget()

# Default time windows (some tasks have hard deadlines)
DEFAULT_TIME_WINDOWS = [
    TimeWindow(2,  "Design & Engineering Approval",    0, 12, True),
    TimeWindow(3,  "Land Acquisition & Clearance",     0, 18, True),
    TimeWindow(14, "Bituminous Concrete (BC) Surface", 0, 42, False, 200_000),
    TimeWindow(16, "Bridge Superstructure",            0, 45, True),
    TimeWindow(23, "Final Inspection & Handover",      0, 50, True),
]

# Default priority milestones
DEFAULT_MILESTONES = [
    PriorityCustomer(
        "Phase 1 Complete — Site Ready",
        task_ids=[3, 4, 5, 6, 7],
        deadline_week=16,
        priority_level=2,
        contractual_penalty_inr=300_000,
        description="Land cleared, utilities relocated, site prepped",
    ),
    PriorityCustomer(
        "Bridge Foundation Done",
        task_ids=[15],
        deadline_week=30,
        priority_level=1,
        contractual_penalty_inr=800_000,
        description="Critical structure — delays cascade to entire project",
    ),
    PriorityCustomer(
        "Road Surface Complete",
        task_ids=[13, 14],
        deadline_week=40,
        priority_level=1,
        contractual_penalty_inr=500_000,
        description="Surface layer must be done before monsoon window closes",
    ),
    PriorityCustomer(
        "Final Handover to NHAI",
        task_ids=[23],
        deadline_week=50,
        priority_level=1,
        contractual_penalty_inr=1_000_000,
        description="Contractual obligation — National Highways Authority of India",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# CONSTRAINT VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def validate_shift_constraints(
    schedule: List[Dict],
    policy: ShiftPolicy | None = None,
) -> pd.DataFrame:
    """
    Check each task against shift-limit policy.
    Returns a DataFrame of violations / warnings.
    """
    if policy is None:
        policy = DEFAULT_SHIFT_POLICY

    hours_per_week = policy.max_hours_per_week
    max_consec = policy.max_consecutive_days

    rows = []
    for entry in schedule:
        duration_weeks = entry["duration"]
        labor_count = entry["resources"].get("labor_crew", 0)

        if labor_count == 0:
            status = "✅ OK"
            detail = "No labor crew assigned"
            overtime_cost = 0
        else:
            # Total labor-hours needed
            total_hours = duration_weeks * hours_per_week * labor_count
            # Check if overtime is needed (capacity = crews × max_hours)
            capacity_hours = labor_count * hours_per_week * duration_weeks
            needed_hours = total_hours  # simplified

            if duration_weeks > max_consec:
                rest_weeks = math.ceil(duration_weeks / max_consec)
                effective_weeks = duration_weeks - rest_weeks
            else:
                effective_weeks = duration_weeks
                rest_weeks = 0

            if effective_weeks < duration_weeks * 0.7:
                status = "⚠️ Warning"
                detail = f"Need {rest_weeks} rest day(s); may require extra crew"
                overtime_cost = int(rest_weeks * labor_count * RESOURCE_POOLS["labor_crew"]["cost_per_week"] * (policy.overtime_multiplier - 1))
            else:
                status = "✅ OK"
                detail = f"{labor_count} crews × {hours_per_week}h/wk = adequate"
                overtime_cost = 0

        rows.append({
            "task_name": entry["task_name"],
            "duration_weeks": duration_weeks,
            "labor_crews": labor_count,
            "shift_status": status,
            "detail": detail,
            "overtime_cost_inr": overtime_cost,
        })

    return pd.DataFrame(rows)


def validate_fuel_budget(
    schedule: List[Dict],
    budget: FuelBudget | None = None,
    makespan: int = 52,
) -> Dict:
    """
    Compute fuel consumption per week and check against budget.
    Returns dict with weekly fuel costs, violations, and summary.
    """
    if budget is None:
        budget = DEFAULT_FUEL_BUDGET

    weekly_fuel = {}
    for week in range(makespan):
        weekly_fuel[week] = 0.0

    for entry in schedule:
        for week in range(entry["start_week"], entry["end_week"]):
            for equip, qty in entry["resources"].items():
                if equip in budget.avg_consumption:
                    litres = qty * budget.avg_consumption[equip] * budget.operating_hours_per_week
                    cost = litres * budget.fuel_price_per_litre
                    weekly_fuel[week] = weekly_fuel.get(week, 0) + cost

    total_fuel_cost = sum(weekly_fuel.values())
    over_budget_weeks = [w for w, c in weekly_fuel.items() if c > budget.weekly_cap_inr]
    budget_utilization = total_fuel_cost / budget.total_budget_inr * 100 if budget.total_budget_inr > 0 else 0

    return {
        "weekly_fuel_cost": weekly_fuel,
        "total_fuel_cost": round(total_fuel_cost),
        "budget_total": budget.total_budget_inr,
        "budget_utilization_pct": round(budget_utilization, 1),
        "over_budget": total_fuel_cost > budget.total_budget_inr,
        "over_budget_weeks": over_budget_weeks,
        "weekly_cap": budget.weekly_cap_inr,
        "status": "🔴 Over Budget" if total_fuel_cost > budget.total_budget_inr else (
            "🟡 Near Limit" if budget_utilization > 80 else "🟢 Within Budget"
        ),
    }


def validate_time_windows(
    schedule: List[Dict],
    windows: List[TimeWindow] | None = None,
) -> pd.DataFrame:
    """
    Check schedule against time-window constraints.
    Returns DataFrame with compliance status.
    """
    if windows is None:
        windows = DEFAULT_TIME_WINDOWS

    sched_by_id = {entry["task_id"]: entry for entry in schedule}

    rows = []
    for tw in windows:
        entry = sched_by_id.get(tw.task_id)
        if entry is None:
            rows.append({
                "task_id": tw.task_id,
                "task_name": tw.task_name,
                "earliest_start": tw.earliest_start_week,
                "latest_finish": tw.latest_finish_week,
                "actual_start": "—",
                "actual_finish": "—",
                "constraint_type": "Hard" if tw.is_hard_constraint else "Soft",
                "status": "⚪ Not Scheduled",
                "weeks_late": 0,
                "penalty_inr": 0,
            })
            continue

        actual_end = entry["end_week"]
        actual_start = entry["start_week"]

        weeks_late = max(0, actual_end - tw.latest_finish_week)
        early_violation = actual_start < tw.earliest_start_week

        if weeks_late > 0:
            if tw.is_hard_constraint:
                status = f"🔴 VIOLATED — {weeks_late}w late"
            else:
                status = f"🟡 Soft breach — {weeks_late}w late"
            penalty = weeks_late * tw.penalty_per_week_late if not tw.is_hard_constraint else 0
        elif early_violation:
            status = "⚠️ Started too early"
            penalty = 0
            weeks_late = 0
        else:
            status = "✅ On Track"
            penalty = 0

        rows.append({
            "task_id": tw.task_id,
            "task_name": tw.task_name,
            "earliest_start": tw.earliest_start_week,
            "latest_finish": tw.latest_finish_week,
            "actual_start": actual_start,
            "actual_finish": actual_end,
            "constraint_type": "Hard" if tw.is_hard_constraint else "Soft",
            "status": status,
            "weeks_late": weeks_late,
            "penalty_inr": int(penalty),
        })

    return pd.DataFrame(rows)


def validate_priority_milestones(
    schedule: List[Dict],
    milestones: List[PriorityCustomer] | None = None,
) -> pd.DataFrame:
    """
    Check priority milestones against the schedule.
    """
    if milestones is None:
        milestones = DEFAULT_MILESTONES

    sched_by_id = {entry["task_id"]: entry for entry in schedule}

    rows = []
    for ms in milestones:
        latest_finish = 0
        all_scheduled = True
        for tid in ms.task_ids:
            entry = sched_by_id.get(tid)
            if entry:
                latest_finish = max(latest_finish, entry["end_week"])
            else:
                all_scheduled = False

        weeks_late = max(0, latest_finish - ms.deadline_week)
        penalty = weeks_late * ms.contractual_penalty_inr

        if not all_scheduled:
            status = "⚪ Incomplete schedule"
        elif weeks_late > 0:
            status = f"🔴 {weeks_late}w late — ₹{penalty:,.0f} penalty"
        elif latest_finish == ms.deadline_week:
            status = "🟡 Exactly on deadline"
        else:
            buffer = ms.deadline_week - latest_finish
            status = f"✅ On track ({buffer}w buffer)"

        rows.append({
            "milestone": ms.milestone_name,
            "priority": f"P{ms.priority_level}",
            "deadline_week": ms.deadline_week,
            "projected_finish": latest_finish,
            "weeks_late": weeks_late,
            "penalty_inr": int(penalty),
            "status": status,
            "description": ms.description,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# COMBINED CONSTRAINT REPORT
# ═══════════════════════════════════════════════════════════════════════

def run_all_constraint_checks(
    schedule: List[Dict],
    makespan: int = 52,
    shift_policy: ShiftPolicy | None = None,
    fuel_budget: FuelBudget | None = None,
    time_windows: List[TimeWindow] | None = None,
    milestones: List[PriorityCustomer] | None = None,
) -> Dict:
    """
    Run all business constraint validations and return a combined report.
    """
    shift_df = validate_shift_constraints(schedule, shift_policy)
    fuel_data = validate_fuel_budget(schedule, fuel_budget, makespan)
    tw_df = validate_time_windows(schedule, time_windows)
    ms_df = validate_priority_milestones(schedule, milestones)

    # Summary
    shift_violations = len(shift_df[shift_df["shift_status"].str.contains("Warning")])
    tw_violations = len(tw_df[tw_df["status"].str.contains("VIOLATED|breach", case=False)])
    ms_violations = len(ms_df[ms_df["weeks_late"] > 0])
    total_penalties = tw_df["penalty_inr"].sum() + ms_df["penalty_inr"].sum()
    overtime_cost = shift_df["overtime_cost_inr"].sum()

    health = "🟢 All Clear"
    if shift_violations > 0 or fuel_data["over_budget"]:
        health = "🟡 Warnings Present"
    if tw_violations > 0 or ms_violations > 0:
        health = "🔴 Violations Detected"

    return {
        "shift_report": shift_df,
        "fuel_report": fuel_data,
        "time_window_report": tw_df,
        "milestone_report": ms_df,
        "summary": {
            "health": health,
            "shift_warnings": shift_violations,
            "fuel_status": fuel_data["status"],
            "tw_violations": tw_violations,
            "ms_violations": ms_violations,
            "total_penalties_inr": int(total_penalties),
            "total_overtime_cost_inr": int(overtime_cost),
        },
    }
