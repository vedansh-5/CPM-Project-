"""
RoadOpt AI — ML Quality Suite
Advanced ML capabilities:
  1. Model Benchmarking  — XGBoost / LightGBM / CatBoost + sklearn baselines
  2. Uncertainty          — Confidence intervals via quantile regression + bootstrapping
  3. Explainability       — SHAP global & per-prediction explanations
  4. Data Drift Detection — PSI + KS-test on feature distributions
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from data_generator import generate_historical_data, Task
from config import DELAY_MODEL_FEATURES, MONTHLY_WEATHER_RISK

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════

def _get_all_models(seed: int = 42) -> Dict[str, object]:
    """Return all candidate models (including boost libraries if installed)."""
    models = {
        "Gradient Boosting (sklearn)": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=seed,
        ),
        "Linear Regression": LinearRegression(),
        "SVR (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=2.0, epsilon=0.15, kernel="rbf")),
        ]),
    }

    # XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            random_state=seed, verbosity=0, n_jobs=-1,
        )
    except ImportError:
        pass

    # LightGBM
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=250, depth=6, learning_rate=0.08,
            random_seed=seed, verbose=0,
        )
    except ImportError:
        pass

    return models


def benchmark_models(n_records: int = 3000, seed: int = 42, cv_folds: int = 5) -> Dict:
    """
    Train & evaluate all available models.
    Returns:
      - model_results: {name: {mae, rmse, r2, cv_mae_mean, cv_mae_std}}
      - best_model_name: str
      - best_model: fitted model
      - X_test, y_test, y_preds: for downstream use
    """
    df = generate_historical_data(n_records=n_records, seed=seed)
    X = df[DELAY_MODEL_FEATURES]
    y = df["delay_weeks"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    models = _get_all_models(seed)
    results = {}
    fitted = {}
    preds_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        preds_dict[name] = y_pred

        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                        scoring="neg_mean_absolute_error")
            cv_mae_mean = round(-cv_scores.mean(), 3)
            cv_mae_std = round(cv_scores.std(), 3)
        except Exception:
            cv_mae_mean = None
            cv_mae_std = None

        results[name] = {
            "mae": round(mean_absolute_error(y_test, y_pred), 3),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            "r2": round(r2_score(y_test, y_pred), 3),
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
        }

    # Pick best by MAE
    best_name = min(results, key=lambda k: results[k]["mae"])

    return {
        "model_results": results,
        "best_model_name": best_name,
        "best_model": fitted[best_name],
        "all_fitted": fitted,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_preds": preds_dict,
    }


def benchmark_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert benchmark results to a display DataFrame."""
    rows = []
    for name, metrics in results["model_results"].items():
        row = {
            "Model": name,
            "MAE (weeks)": metrics["mae"],
            "RMSE (weeks)": metrics["rmse"],
            "R²": metrics["r2"],
        }
        if metrics["cv_mae_mean"] is not None:
            row["CV MAE (mean)"] = metrics["cv_mae_mean"]
            row["CV MAE (std)"] = metrics["cv_mae_std"]
        if name == results["best_model_name"]:
            row["Model"] = f"🏆 {name}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE (weeks)")


# ═══════════════════════════════════════════════════════════════════════
# 2. UNCERTAINTY / CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════

def train_quantile_models(
    n_records: int = 3000, seed: int = 42,
) -> Dict:
    """
    Train quantile regressors for prediction intervals.
    Returns lower (10th), median (50th), upper (90th) models.
    """
    df = generate_historical_data(n_records=n_records, seed=seed)
    X = df[DELAY_MODEL_FEATURES]
    y = df["delay_weeks"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    quantile_models = {}
    for alpha, label in [(0.1, "lower_10"), (0.5, "median"), (0.9, "upper_90")]:
        m = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            loss="quantile", alpha=alpha, random_state=seed,
        )
        m.fit(X_train, y_train)
        quantile_models[label] = m

    return {
        "models": quantile_models,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict_with_uncertainty(
    tasks: List[Task],
    quantile_models: Dict[str, object],
    resource_utilization: float = 0.7,
    start_month: int = 1,
    crew_experience: int = 3,
    material_availability: float = 0.85,
) -> pd.DataFrame:
    """
    Predict delay with 80% confidence interval (10th – 90th percentile).
    """
    by_id = {t.id: t for t in tasks}
    rows = []
    for t in tasks:
        dep_depth = _dep_depth(t, by_id)
        month = (start_month + t.earliest_start // 4) % 12
        weather = MONTHLY_WEATHER_RISK[month]

        feat = pd.DataFrame([[
            t.complexity, resource_utilization, weather,
            dep_depth, int(t.is_critical), crew_experience, material_availability,
        ]], columns=DELAY_MODEL_FEATURES)

        lo = max(0, round(quantile_models["lower_10"].predict(feat)[0], 1))
        med = max(0, round(quantile_models["median"].predict(feat)[0], 1))
        hi = max(0, round(quantile_models["upper_90"].predict(feat)[0], 1))
        width = round(hi - lo, 1)

        confidence = "🟢 High" if width < 1.5 else ("🟡 Medium" if width < 3.0 else "🔴 Low")

        rows.append({
            "task_id": t.id,
            "task_name": t.name,
            "delay_lower_10": lo,
            "delay_median": med,
            "delay_upper_90": hi,
            "interval_width": width,
            "confidence": confidence,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 3. EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict:
    """
    Compute SHAP values for the model.
    Falls back to permutation-based if shap is not installed.
    Returns: {shap_values, feature_names, global_importance, expected_value}
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_value = explainer.expected_value

        global_imp = dict(zip(
            DELAY_MODEL_FEATURES,
            np.abs(shap_values).mean(axis=0).round(4),
        ))

        return {
            "shap_values": shap_values,
            "X_test": X_test,
            "feature_names": DELAY_MODEL_FEATURES,
            "global_importance": global_imp,
            "expected_value": float(expected_value) if np.isscalar(expected_value) else float(expected_value[0]),
            "method": "TreeSHAP",
        }
    except ImportError:
        # Fallback: permutation importance
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, model.predict(X_test),
                                       n_repeats=10, random_state=42)
        global_imp = dict(zip(DELAY_MODEL_FEATURES, perm.importances_mean.round(4)))
        return {
            "shap_values": None,
            "X_test": X_test,
            "feature_names": DELAY_MODEL_FEATURES,
            "global_importance": global_imp,
            "expected_value": float(model.predict(X_test).mean()),
            "method": "Permutation Importance (shap not installed)",
        }
    except Exception:
        # Model not tree-based (e.g. SVR pipeline)
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, model.predict(X_test),
                                       n_repeats=10, random_state=42)
        global_imp = dict(zip(DELAY_MODEL_FEATURES, perm.importances_mean.round(4)))
        return {
            "shap_values": None,
            "X_test": X_test,
            "feature_names": DELAY_MODEL_FEATURES,
            "global_importance": global_imp,
            "expected_value": float(model.predict(X_test).mean()),
            "method": "Permutation Importance (non-tree model)",
        }


def explain_single_prediction(
    model,
    task: Task,
    tasks: List[Task],
    resource_utilization: float = 0.7,
    start_month: int = 1,
    crew_experience: int = 3,
    material_availability: float = 0.85,
    X_train: pd.DataFrame | None = None,
) -> Dict:
    """
    Explain a single task's delay prediction.
    Returns feature contributions.
    """
    by_id = {t.id: t for t in tasks}
    dep_depth = _dep_depth(task, by_id)
    month = (start_month + task.earliest_start // 4) % 12
    weather = MONTHLY_WEATHER_RISK[month]

    feat = pd.DataFrame([[
        task.complexity, resource_utilization, weather,
        dep_depth, int(task.is_critical), crew_experience, material_availability,
    ]], columns=DELAY_MODEL_FEATURES)

    prediction = model.predict(feat)[0]

    contributions = {}
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(feat)[0]
        base = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
        for i, f in enumerate(DELAY_MODEL_FEATURES):
            contributions[f] = {
                "value": float(feat.iloc[0, i]),
                "shap_contribution": round(float(sv[i]), 4),
                "direction": "↑ increases delay" if sv[i] > 0 else "↓ decreases delay",
            }
        return {
            "task_name": task.name,
            "prediction": round(prediction, 2),
            "base_value": round(base, 2),
            "contributions": contributions,
            "method": "SHAP",
        }
    except Exception:
        # Fallback: feature value + model importance
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        else:
            imps = np.ones(len(DELAY_MODEL_FEATURES)) / len(DELAY_MODEL_FEATURES)
        for i, f in enumerate(DELAY_MODEL_FEATURES):
            contributions[f] = {
                "value": float(feat.iloc[0, i]),
                "importance": round(float(imps[i]), 4),
            }
        return {
            "task_name": task.name,
            "prediction": round(prediction, 2),
            "contributions": contributions,
            "method": "Feature Importance (fallback)",
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. DATA DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-6
    breakpoints = np.linspace(
        min(expected.min(), actual.min()) - eps,
        max(expected.max(), actual.max()) + eps,
        bins + 1,
    )
    exp_hist = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_hist = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    exp_hist = np.clip(exp_hist, eps, None)
    act_hist = np.clip(act_hist, eps, None)
    psi_val = np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist))
    return round(float(psi_val), 4)


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compare feature distributions between reference (training) and current data.
    Uses PSI + KS-test.
    Returns DataFrame with drift metrics per feature.
    """
    from scipy.stats import ks_2samp

    if features is None:
        features = DELAY_MODEL_FEATURES

    rows = []
    for feat in features:
        if feat not in reference_data.columns or feat not in current_data.columns:
            continue

        ref = reference_data[feat].dropna().values
        cur = current_data[feat].dropna().values

        # PSI
        psi_val = _psi(ref, cur)

        # KS test
        ks_stat, ks_p = ks_2samp(ref, cur)

        # Drift severity
        if psi_val < 0.1 and ks_p > 0.05:
            status = "🟢 No Drift"
        elif psi_val < 0.2:
            status = "🟡 Minor Drift"
        elif psi_val < 0.3:
            status = "🟠 Moderate Drift"
        else:
            status = "🔴 Significant Drift"

        rows.append({
            "Feature": feat.replace("_", " ").title(),
            "Ref Mean": round(float(ref.mean()), 3),
            "Ref Std": round(float(ref.std()), 3),
            "Current Mean": round(float(cur.mean()), 3),
            "Current Std": round(float(cur.std()), 3),
            "PSI": psi_val,
            "KS Statistic": round(float(ks_stat), 4),
            "KS p-value": round(float(ks_p), 4),
            "Status": status,
        })

    return pd.DataFrame(rows)


def simulate_drift_scenario(
    seed: int = 42,
    drift_type: str = "none",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate reference + drifted data for demonstration.
    drift_type: "none", "gradual", "sudden", "seasonal"
    """
    ref = generate_historical_data(n_records=2000, seed=seed)

    if drift_type == "none":
        cur = generate_historical_data(n_records=500, seed=seed + 100)
    elif drift_type == "gradual":
        cur = generate_historical_data(n_records=500, seed=seed + 100)
        # Gradually shift weather risk upward (climate change sim)
        cur["weather_risk"] = np.clip(cur["weather_risk"] + 0.15, 0, 1)
        cur["material_availability"] = np.clip(cur["material_availability"] - 0.10, 0.2, 1)
    elif drift_type == "sudden":
        cur = generate_historical_data(n_records=500, seed=seed + 100)
        # Sudden labor crisis
        cur["crew_experience"] = np.clip(cur["crew_experience"] - 2, 1, 5)
        cur["resource_utilization"] = np.clip(cur["resource_utilization"] + 0.20, 0.3, 1)
        cur["task_complexity"] = np.clip(cur["task_complexity"] + 1, 1, 5)
    elif drift_type == "seasonal":
        cur = generate_historical_data(n_records=500, seed=seed + 100)
        # Monsoon-like shift
        cur["weather_risk"] = np.clip(cur["weather_risk"] + 0.25, 0, 1)
    else:
        cur = generate_historical_data(n_records=500, seed=seed + 100)

    return ref, cur


# ═══════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════

def _dep_depth(task: Task, by_id: dict, cache: dict | None = None) -> int:
    if cache is None:
        cache = {}
    if task.id in cache:
        return cache[task.id]
    if not task.predecessors:
        cache[task.id] = 0
        return 0
    depth = 1 + max(_dep_depth(by_id[p], by_id, cache) for p in task.predecessors)
    cache[task.id] = depth
    return depth
