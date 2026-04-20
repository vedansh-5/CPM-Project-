"""
RoadOpt AI — Global Configuration
"""

# ─── Project Defaults ────────────────────────────────────────────────
DEFAULT_PROJECT_NAME = "National Highway NH-48 Expansion"
DEFAULT_HORIZON_WEEKS = 52  # 1 year
DEFAULT_WORK_HOURS_PER_WEEK = 48

# ─── Resource Pools (defaults) ───────────────────────────────────────
RESOURCE_POOLS = {
    "labor_crew":       {"units": 8,  "cost_per_week": 50_000},
    "excavator":        {"units": 4,  "cost_per_week": 120_000},
    "bulldozer":        {"units": 3,  "cost_per_week": 100_000},
    "asphalt_paver":    {"units": 2,  "cost_per_week": 150_000},
    "roller_compactor": {"units": 3,  "cost_per_week": 80_000},
    "dump_truck":       {"units": 6,  "cost_per_week": 40_000},
    "concrete_mixer":   {"units": 3,  "cost_per_week": 60_000},
    "crane":            {"units": 2,  "cost_per_week": 200_000},
    "surveyor_team":    {"units": 2,  "cost_per_week": 70_000},
}

# ─── Weather Risk (monthly probability of delay, index 0 = Jan) ──────
MONTHLY_WEATHER_RISK = [
    0.10, 0.08, 0.05, 0.05, 0.15, 0.30,  # Jan–Jun  (monsoon spike Jun)
    0.45, 0.50, 0.35, 0.10, 0.05, 0.08,  # Jul–Dec
]

# ─── Objective Weights (sum to 1.0) ──────────────────────────────────
OBJECTIVE_WEIGHTS = {
    "minimize_time": 0.35,
    "minimize_cost": 0.30,
    "balance_resources": 0.20,
    "reduce_risk": 0.15,
}

# ─── AI Delay Predictor ─────────────────────────────────────────────
DELAY_MODEL_FEATURES = [
    "task_complexity",       # 1-5
    "resource_utilization",  # 0.0–1.0
    "weather_risk",          # 0.0–1.0
    "dependency_depth",      # int
    "is_critical_path",      # 0/1
    "crew_experience",       # 1-5
    "material_availability", # 0.0–1.0
]
