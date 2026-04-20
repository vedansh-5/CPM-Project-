# 🛣️ RoadOpt AI

**AI-Driven Scheduling & Resource Allocation for Road Construction Projects**

## Features

| Module | What it does |
|---|---|
| **Critical Path Method** | Forward/backward pass to identify critical tasks & project makespan |
| **OR-Tools CP-SAT Solver** | Resource-constrained scheduling (RCPSP) with configurable objectives |
| **AI Delay Predictor** | Gradient Boosting model predicts delay risk per task |
| **What-If Simulation** | Simulate labor shortages & equipment breakdowns |
| **Interactive Dashboard** | Gantt chart, resource heatmap, cost breakdown, risk analysis |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py
```

## Project Structure

```
roadopt-ai/
├── app.py              # Streamlit dashboard (main entry)
├── config.py           # Global configuration & defaults
├── data_generator.py   # Synthetic task & historical data
├── optimizer.py        # CPM + OR-Tools RCPSP solver
├── ai_predictor.py     # ML delay-risk predictor
├── visualizations.py   # Plotly chart builders
├── requirements.txt    # Python dependencies
└── README.md
```

## Architecture

```
User (Streamlit UI)
  │
  ├──▶ config.py          (parameters)
  ├──▶ data_generator.py  (tasks + synthetic history)
  ├──▶ optimizer.py        (CPM → RCPSP solve)
  ├──▶ ai_predictor.py     (train model → predict delays)
  └──▶ visualizations.py   (Gantt, heatmap, charts)
```

## Team (18 Members) — Suggested Split

| Team | Size | Responsibility |
|---|---|---|
| Data & Domain | 4 | Task templates, dependencies, synthetic data, domain research |
| Optimization | 4 | CPM, OR-Tools solver, objective tuning, constraint modeling |
| AI/ML | 3 | Delay model training, feature engineering, evaluation |
| Frontend/UI | 3 | Streamlit dashboard, Plotly charts, UX |
| Testing & Docs | 2 | Unit tests, integration tests, documentation |
| Presentation | 2 | PPT, demo script, rehearsal coordination |

## License

Academic use only — Group Project.
