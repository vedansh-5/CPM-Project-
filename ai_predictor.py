"""
RoadOpt AI — AI Delay Risk Predictor
Trains a Gradient-Boosted model on synthetic historical data
to predict likely delay (weeks) for each task.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from data_generator import generate_historical_data, Task
from config import MONTHLY_WEATHER_RISK, DELAY_MODEL_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "delay_model.joblib")


def train_delay_model(n_records: int = 3000, seed: int = 42) -> dict:
    """Train & persist a delay-prediction model. Returns metrics."""
    df = generate_historical_data(n_records=n_records, seed=seed)

    X = df[DELAY_MODEL_FEATURES]
    y = df["delay_weeks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Primary model
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)

    # Compare all 4 models
    models = {
        "Gradient Boosting": model,
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=seed),
        "Linear Regression": LinearRegression(),
        "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=1.0, epsilon=0.2))]),
    }
    model_comparison = {}
    for name, m in models.items():
        if name != "Gradient Boosting":
            m.fit(X_train, y_train)
        preds = m.predict(X_test)
        model_comparison[name] = {
            "mae": round(mean_absolute_error(y_test, preds), 3),
            "r2": round(r2_score(y_test, preds), 3),
        }

    feature_imp = dict(zip(DELAY_MODEL_FEATURES, model.feature_importances_))

    return {
        "mae": round(mae, 3),
        "r2": round(r2, 3),
        "feature_importance": feature_imp,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_comparison": model_comparison,
    }


def load_delay_model():
    """Load persisted model (train first if missing)."""
    if not os.path.exists(MODEL_PATH):
        train_delay_model()
    return joblib.load(MODEL_PATH)


def predict_task_delays(
    tasks: list[Task],
    resource_utilization: float = 0.7,
    start_month: int = 1,
    crew_experience: int = 3,
    material_availability: float = 0.85,
) -> pd.DataFrame:
    """
    Predict delay risk for each task.
    Returns DataFrame with columns: task_id, task_name, predicted_delay, risk_level
    """
    model = load_delay_model()

    rows = []
    for t in tasks:
        dep_depth = _dependency_depth(t, {tk.id: tk for tk in tasks})
        month = (start_month + t.earliest_start // 4) % 12
        weather = MONTHLY_WEATHER_RISK[month]

        features = pd.DataFrame([[
            t.complexity,
            resource_utilization,
            weather,
            dep_depth,
            int(t.is_critical),
            crew_experience,
            material_availability,
        ]], columns=DELAY_MODEL_FEATURES)

        pred = model.predict(features)[0]
        pred = max(0, round(pred, 1))

        if pred <= 0.5:
            risk = "🟢 Low"
        elif pred <= 2.0:
            risk = "🟡 Medium"
        else:
            risk = "🔴 High"

        rows.append({
            "task_id": t.id,
            "task_name": t.name,
            "predicted_delay_weeks": pred,
            "risk_level": risk,
            "weather_risk": round(weather, 2),
        })

    return pd.DataFrame(rows)


def _dependency_depth(task: Task, by_id: dict, cache: dict | None = None) -> int:
    if cache is None:
        cache = {}
    if task.id in cache:
        return cache[task.id]
    if not task.predecessors:
        cache[task.id] = 0
        return 0
    depth = 1 + max(_dependency_depth(by_id[p], by_id, cache) for p in task.predecessors)
    cache[task.id] = depth
    return depth
