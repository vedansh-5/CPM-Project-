"""
RoadOpt AI — Synthetic Data Generator
Generates realistic road-construction project data:
  • Task list with dependencies, durations, resource needs
  • Historical project records for ML training
"""

import random
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

from config import RESOURCE_POOLS, MONTHLY_WEATHER_RISK

# ─── Road Construction Task Templates ────────────────────────────────
# Each tuple: (name, base_duration_weeks, complexity, resource_requirements)
TASK_TEMPLATES = [
    # Phase 1 – Survey & Planning
    ("Site Survey & Soil Testing",           3,  2, {"surveyor_team": 2, "labor_crew": 1}),
    ("Environmental Impact Assessment",      4,  3, {"surveyor_team": 1}),
    ("Design & Engineering Approval",        3,  4, {}),
    ("Land Acquisition & Clearance",         6,  5, {"labor_crew": 2, "bulldozer": 1}),

    # Phase 2 – Site Preparation
    ("Site Clearing & Grubbing",             3,  2, {"bulldozer": 2, "labor_crew": 3, "dump_truck": 2}),
    ("Demolition of Existing Structures",    2,  3, {"excavator": 2, "labor_crew": 2, "dump_truck": 2}),
    ("Temporary Drainage Setup",             2,  2, {"labor_crew": 2, "excavator": 1}),
    ("Utility Relocation",                   4,  4, {"labor_crew": 2, "crane": 1, "excavator": 1}),

    # Phase 3 – Earthwork
    ("Embankment Construction",              5,  3, {"excavator": 3, "bulldozer": 2, "dump_truck": 4, "roller_compactor": 2, "labor_crew": 4}),
    ("Cut & Fill Operations",                4,  3, {"excavator": 2, "bulldozer": 2, "dump_truck": 3, "labor_crew": 3}),
    ("Subgrade Preparation",                 3,  3, {"roller_compactor": 2, "labor_crew": 3, "bulldozer": 1}),

    # Phase 4 – Pavement Layers
    ("Granular Sub-Base (GSB) Layer",        3,  3, {"dump_truck": 3, "roller_compactor": 2, "labor_crew": 3}),
    ("Wet Mix Macadam (WMM) Layer",          3,  3, {"dump_truck": 3, "roller_compactor": 2, "labor_crew": 3}),
    ("Dense Bituminous Macadam (DBM)",       3,  4, {"asphalt_paver": 1, "roller_compactor": 2, "dump_truck": 2, "labor_crew": 3}),
    ("Bituminous Concrete (BC) Surface",     3,  4, {"asphalt_paver": 2, "roller_compactor": 2, "dump_truck": 2, "labor_crew": 4}),

    # Phase 5 – Structures
    ("Bridge / Flyover Foundation",          6,  5, {"crane": 2, "concrete_mixer": 3, "labor_crew": 5, "excavator": 2}),
    ("Bridge Superstructure",                8,  5, {"crane": 2, "concrete_mixer": 2, "labor_crew": 4}),
    ("Culvert & Cross-Drainage",             4,  3, {"excavator": 1, "concrete_mixer": 2, "labor_crew": 3}),
    ("Retaining Wall Construction",          4,  4, {"concrete_mixer": 2, "crane": 1, "labor_crew": 3}),

    # Phase 6 – Finishing
    ("Road Marking & Signage",               2,  2, {"labor_crew": 2}),
    ("Guardrail & Barrier Installation",     2,  2, {"labor_crew": 2, "crane": 1}),
    ("Street Lighting Installation",         3,  3, {"labor_crew": 2, "crane": 1}),
    ("Landscaping & Plantation",             2,  1, {"labor_crew": 2}),
    ("Final Inspection & Handover",          2,  3, {"surveyor_team": 2}),
]

# Dependencies: index → list of predecessor indices (0-based into TASK_TEMPLATES)
DEPENDENCY_MAP: Dict[int, List[int]] = {
    1:  [0],
    2:  [0, 1],
    3:  [2],
    4:  [3],
    5:  [3],
    6:  [3],
    7:  [3],
    8:  [4, 5, 6],
    9:  [4, 5],
    10: [8, 9],
    11: [10],
    12: [11],
    13: [12],
    14: [13],
    15: [7, 10],
    16: [15],
    17: [10],
    18: [10],
    19: [14, 16],
    20: [14],
    21: [14],
    22: [14],
    23: [19, 20, 21, 22],
}


@dataclass
class Task:
    id: int
    name: str
    duration_weeks: int
    complexity: int  # 1-5
    resource_requirements: Dict[str, int]
    predecessors: List[int] = field(default_factory=list)
    earliest_start: int = 0
    latest_start: int = 0
    is_critical: bool = False


def generate_project_tasks(
    duration_noise: float = 0.2,
    seed: int = 42,
) -> List[Task]:
    """Generate a full set of road-project tasks with slight random variation."""
    rng = random.Random(seed)
    tasks: List[Task] = []
    for idx, (name, dur, comp, res) in enumerate(TASK_TEMPLATES):
        noise = rng.uniform(-duration_noise, duration_noise)
        actual_dur = max(1, round(dur * (1 + noise)))
        tasks.append(Task(
            id=idx,
            name=name,
            duration_weeks=actual_dur,
            complexity=comp,
            resource_requirements=dict(res),
            predecessors=list(DEPENDENCY_MAP.get(idx, [])),
        ))
    return tasks


def generate_historical_data(n_records: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic historical project-task records for ML training.
    Features: task_complexity, resource_utilization, weather_risk,
              dependency_depth, is_critical_path, crew_experience,
              material_availability
    Target:   delay_weeks (0 = on-time, >0 = delayed)
    """
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_records):
        complexity = rng.randint(1, 6)
        res_util = round(rng.uniform(0.3, 1.0), 2)
        month = rng.randint(0, 12)
        weather = MONTHLY_WEATHER_RISK[month % 12]
        dep_depth = rng.randint(0, 8)
        critical = rng.choice([0, 1])
        experience = rng.randint(1, 6)
        material = round(rng.uniform(0.4, 1.0), 2)

        # Simulate delay: higher complexity, weather, low material → more delay
        delay_score = (
            0.25 * complexity / 5
            + 0.20 * (1 - material)
            + 0.25 * weather
            + 0.10 * res_util
            + 0.10 * (dep_depth / 8)
            + 0.05 * critical
            + 0.05 * (1 - experience / 5)
        )
        noise = rng.normal(0, 0.08)
        delay_prob = np.clip(delay_score + noise, 0, 1)
        delay_weeks = int(np.round(delay_prob * 6))  # 0-6 weeks delay

        rows.append({
            "task_complexity": complexity,
            "resource_utilization": res_util,
            "weather_risk": weather,
            "dependency_depth": dep_depth,
            "is_critical_path": critical,
            "crew_experience": experience,
            "material_availability": material,
            "delay_weeks": delay_weeks,
        })

    return pd.DataFrame(rows)


def tasks_to_dataframe(tasks: List[Task]) -> pd.DataFrame:
    """Convert task list to a DataFrame for display / export."""
    records = []
    for t in tasks:
        rec = {
            "id": t.id,
            "name": t.name,
            "duration_weeks": t.duration_weeks,
            "complexity": t.complexity,
            "predecessors": ", ".join(str(p) for p in t.predecessors) if t.predecessors else "—",
            "earliest_start": t.earliest_start,
            "latest_start": t.latest_start,
            "is_critical": t.is_critical,
        }
        for res_name in RESOURCE_POOLS:
            rec[f"res_{res_name}"] = t.resource_requirements.get(res_name, 0)
        records.append(rec)
    return pd.DataFrame(records)
