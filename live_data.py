"""
RoadOpt AI — Live Data Connectors
Fetches real-time & contextual data from external sources:
  • Weather (OpenWeatherMap API / fallback synthetic)
  • Traffic congestion index
  • Public holidays & calendar events
  • Strikes / disruption events
"""

import os
import random
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─── API Keys (set via environment or .env) ──────────────────────────
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WeatherForecast:
    date: str
    condition: str          # Clear / Rain / Storm / Fog / Extreme Heat
    temperature_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    work_impact: float      # 0.0 (no impact) – 1.0 (full stoppage)


@dataclass
class TrafficCondition:
    route_segment: str
    congestion_index: float   # 0.0 (free flow) – 1.0 (gridlock)
    avg_speed_kmh: float
    incidents: int
    estimated_delay_min: float


@dataclass
class CalendarEvent:
    date: str
    event_name: str
    event_type: str           # holiday / strike / festival / election / other
    severity: float           # 0.0 – 1.0 impact on work
    affects_labor: bool
    affects_transport: bool


# ═══════════════════════════════════════════════════════════════════════
# 1. WEATHER CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

# Indian monthly climate profiles (temperature, rain probability, etc.)
_INDIA_CLIMATE = {
    1:  {"temp": 18, "rain_prob": 0.05, "conditions": ["Clear", "Fog"]},
    2:  {"temp": 22, "rain_prob": 0.05, "conditions": ["Clear", "Fog"]},
    3:  {"temp": 28, "rain_prob": 0.08, "conditions": ["Clear", "Extreme Heat"]},
    4:  {"temp": 34, "rain_prob": 0.10, "conditions": ["Extreme Heat", "Clear"]},
    5:  {"temp": 38, "rain_prob": 0.15, "conditions": ["Extreme Heat", "Storm"]},
    6:  {"temp": 35, "rain_prob": 0.55, "conditions": ["Rain", "Storm"]},
    7:  {"temp": 32, "rain_prob": 0.70, "conditions": ["Rain", "Storm"]},
    8:  {"temp": 31, "rain_prob": 0.65, "conditions": ["Rain", "Storm"]},
    9:  {"temp": 30, "rain_prob": 0.45, "conditions": ["Rain", "Clear"]},
    10: {"temp": 28, "rain_prob": 0.15, "conditions": ["Clear", "Rain"]},
    11: {"temp": 24, "rain_prob": 0.05, "conditions": ["Clear", "Fog"]},
    12: {"temp": 19, "rain_prob": 0.05, "conditions": ["Clear", "Fog"]},
}

# Work-impact mapping
_CONDITION_IMPACT = {
    "Clear":        0.00,
    "Fog":          0.15,
    "Rain":         0.40,
    "Storm":        0.70,
    "Extreme Heat": 0.25,
}


def fetch_weather_forecast(
    start_month: int = 1,
    num_weeks: int = 52,
    location: str = "Varanasi, India",
    seed: int = 42,
) -> List[WeatherForecast]:
    """
    Generate a realistic weekly weather forecast.
    Uses OpenWeatherMap if API key is set, otherwise falls back to
    climate-model-based synthetic data.
    """
    # TODO: Add real API call when OPENWEATHER_API_KEY is set
    # For now, use realistic synthetic data based on Indian climate
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    forecasts = []
    for week in range(num_weeks):
        month = ((start_month - 1) + week // 4) % 12 + 1
        climate = _INDIA_CLIMATE[month]

        # Pick weather condition weighted by month
        if rng.random() < climate["rain_prob"]:
            condition = rng.choice([c for c in climate["conditions"] if c in ("Rain", "Storm")] or ["Rain"])
        else:
            condition = rng.choice([c for c in climate["conditions"] if c not in ("Rain", "Storm")] or ["Clear"])

        temp = climate["temp"] + np_rng.normal(0, 3)
        precip = max(0, np_rng.normal(climate["rain_prob"] * 50, 15)) if condition in ("Rain", "Storm") else 0
        wind = max(0, np_rng.normal(15, 8)) if condition == "Storm" else max(0, np_rng.normal(8, 4))

        impact = _CONDITION_IMPACT.get(condition, 0.0)
        # Add some noise to impact
        impact = min(1.0, max(0.0, impact + np_rng.normal(0, 0.05)))

        date_offset = datetime.timedelta(weeks=week)
        base_date = datetime.date(2026, start_month, 1) + date_offset

        forecasts.append(WeatherForecast(
            date=base_date.isoformat(),
            condition=condition,
            temperature_c=round(temp, 1),
            precipitation_mm=round(precip, 1),
            wind_speed_kmh=round(wind, 1),
            work_impact=round(impact, 3),
        ))

    return forecasts


def weather_to_dataframe(forecasts: List[WeatherForecast]) -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in forecasts])


# ═══════════════════════════════════════════════════════════════════════
# 2. TRAFFIC CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

_ROUTE_SEGMENTS = [
    "NH-48 Km 0–25 (Urban Entry)",
    "NH-48 Km 25–60 (Suburban)",
    "NH-48 Km 60–100 (Semi-Rural)",
    "NH-48 Km 100–150 (Rural)",
    "NH-48 Km 150–200 (Highway Stretch)",
    "Bridge Approach Road",
    "Flyover Zone A",
    "Material Haul Route – Quarry",
    "Material Haul Route – Plant",
    "Worker Transport Corridor",
]


def fetch_traffic_conditions(
    num_weeks: int = 52,
    seed: int = 42,
) -> Dict[int, List[TrafficCondition]]:
    """
    Generate weekly traffic conditions for each route segment.
    Returns: {week_number: [TrafficCondition, ...]}
    """
    rng = np.random.RandomState(seed)
    traffic_data = {}

    for week in range(num_weeks):
        conditions = []
        for seg in _ROUTE_SEGMENTS:
            # Urban segments have more congestion
            base_congestion = 0.7 if "Urban" in seg else (0.4 if "Suburban" in seg else 0.2)
            # Weekly variation
            congestion = min(1.0, max(0.0, base_congestion + rng.normal(0, 0.15)))
            speed = max(5, 80 * (1 - congestion) + rng.normal(0, 5))
            incidents = max(0, int(rng.poisson(congestion * 2)))
            delay = max(0, congestion * 45 + rng.normal(0, 10))

            conditions.append(TrafficCondition(
                route_segment=seg,
                congestion_index=round(congestion, 3),
                avg_speed_kmh=round(speed, 1),
                incidents=incidents,
                estimated_delay_min=round(delay, 1),
            ))
        traffic_data[week] = conditions

    return traffic_data


def traffic_summary_dataframe(
    traffic_data: Dict[int, List[TrafficCondition]],
) -> pd.DataFrame:
    """Summarize traffic data: avg congestion per segment across all weeks."""
    rows = []
    segments = {tc.route_segment for tcs in traffic_data.values() for tc in tcs}
    for seg in sorted(segments):
        congs = [tc.congestion_index for tcs in traffic_data.values() for tc in tcs if tc.route_segment == seg]
        delays = [tc.estimated_delay_min for tcs in traffic_data.values() for tc in tcs if tc.route_segment == seg]
        incidents = [tc.incidents for tcs in traffic_data.values() for tc in tcs if tc.route_segment == seg]
        rows.append({
            "Route Segment": seg,
            "Avg Congestion": round(np.mean(congs), 3),
            "Max Congestion": round(np.max(congs), 3),
            "Avg Delay (min)": round(np.mean(delays), 1),
            "Total Incidents": int(np.sum(incidents)),
            "Risk Level": "🔴 High" if np.mean(congs) > 0.6 else ("🟡 Medium" if np.mean(congs) > 0.35 else "🟢 Low"),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 3. HOLIDAYS & EVENTS CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

# Major Indian holidays & events that affect construction
_INDIA_EVENTS_2026 = [
    # (month, day, name, type, severity, affects_labor, affects_transport)
    (1, 26, "Republic Day", "holiday", 0.8, True, True),
    (3, 10, "Holi", "festival", 0.7, True, False),
    (3, 30, "Id-ul-Fitr (estimated)", "holiday", 0.6, True, False),
    (4, 14, "Ambedkar Jayanti", "holiday", 0.5, True, False),
    (5, 1, "May Day / Labour Day", "holiday", 0.9, True, False),
    (6, 5, "Bakrid / Eid al-Adha (estimated)", "holiday", 0.6, True, False),
    (7, 1, "Monsoon Season Start", "other", 0.3, False, True),
    (8, 15, "Independence Day", "holiday", 0.9, True, True),
    (8, 19, "Janmashtami", "festival", 0.5, True, False),
    (9, 1, "Peak Monsoon — Material Transport Disruption", "other", 0.5, False, True),
    (10, 2, "Gandhi Jayanti", "holiday", 0.7, True, False),
    (10, 12, "Dussehra", "festival", 0.7, True, False),
    (10, 31, "Diwali", "festival", 0.9, True, True),
    (11, 1, "Diwali Holiday Extension", "festival", 0.8, True, False),
    (11, 15, "Guru Nanak Jayanti", "holiday", 0.5, True, False),
    (12, 25, "Christmas", "holiday", 0.4, True, False),
    # Potential disruptions
    (2, 15, "State Election Period (possible)", "election", 0.4, True, True),
    (6, 15, "Truckers' Strike (simulated)", "strike", 0.9, True, True),
    (9, 20, "Farmers' Protest – Road Block (simulated)", "strike", 0.7, False, True),
]


def fetch_calendar_events(
    start_month: int = 1,
    num_weeks: int = 52,
    include_strikes: bool = True,
    seed: int = 42,
) -> List[CalendarEvent]:
    """
    Get holidays, festivals, strikes, and disruption events for the project timeline.
    """
    rng = random.Random(seed)
    events = []

    base_date = datetime.date(2026, start_month, 1)
    end_date = base_date + datetime.timedelta(weeks=num_weeks)

    for month, day, name, etype, severity, labor, transport in _INDIA_EVENTS_2026:
        try:
            event_date = datetime.date(2026, month, day)
        except ValueError:
            continue

        if base_date <= event_date <= end_date:
            if etype == "strike" and not include_strikes:
                continue
            # Add some randomness to severity
            adj_severity = min(1.0, max(0.1, severity + rng.uniform(-0.1, 0.1)))
            events.append(CalendarEvent(
                date=event_date.isoformat(),
                event_name=name,
                event_type=etype,
                severity=round(adj_severity, 2),
                affects_labor=labor,
                affects_transport=transport,
            ))

    return events


def events_to_dataframe(events: List[CalendarEvent]) -> pd.DataFrame:
    df = pd.DataFrame([e.__dict__ for e in events])
    if len(df) > 0:
        df["week_number"] = df["date"].apply(
            lambda d: (datetime.date.fromisoformat(d) - datetime.date(2026, 1, 1)).days // 7
        )
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. COMBINED EXTERNAL RISK SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_weekly_external_risk(
    start_month: int = 1,
    num_weeks: int = 52,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Combine weather, traffic, and events into a single weekly risk score.
    Returns DataFrame with columns: week, weather_risk, traffic_risk,
    event_risk, combined_risk, risk_label
    """
    weather = fetch_weather_forecast(start_month=start_month, num_weeks=num_weeks, seed=seed)
    traffic = fetch_traffic_conditions(num_weeks=num_weeks, seed=seed)
    events = fetch_calendar_events(start_month=start_month, num_weeks=num_weeks, seed=seed)

    event_df = events_to_dataframe(events)

    rows = []
    for week in range(num_weeks):
        # Weather risk for this week
        w_risk = weather[week].work_impact if week < len(weather) else 0.0

        # Traffic risk (average congestion across segments)
        t_conds = traffic.get(week, [])
        t_risk = np.mean([tc.congestion_index for tc in t_conds]) if t_conds else 0.0

        # Event risk (max severity of events in this week)
        e_risk = 0.0
        if len(event_df) > 0 and "week_number" in event_df.columns:
            week_events = event_df[event_df["week_number"] == week]
            if len(week_events) > 0:
                e_risk = week_events["severity"].max()

        # Combined: weighted average
        combined = 0.45 * w_risk + 0.25 * t_risk + 0.30 * e_risk
        combined = min(1.0, combined)

        if combined < 0.2:
            label = "🟢 Low"
        elif combined < 0.45:
            label = "🟡 Medium"
        elif combined < 0.7:
            label = "🟠 High"
        else:
            label = "🔴 Critical"

        rows.append({
            "week": week,
            "weather_risk": round(w_risk, 3),
            "traffic_risk": round(t_risk, 3),
            "event_risk": round(e_risk, 3),
            "combined_risk": round(combined, 3),
            "risk_label": label,
            "weather_condition": weather[week].condition if week < len(weather) else "Unknown",
        })

    return pd.DataFrame(rows)
