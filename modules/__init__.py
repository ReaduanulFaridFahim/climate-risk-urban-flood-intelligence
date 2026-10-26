"""
DeltaShield: Bangladesh Unified Climate & Multi-Sector Intelligence Suite
Four Pillars:
  1. agroclaim   : Parametric Crop Insurance & Automated Payout Engine
  2. evacroute   : Submerged Road Networks, Island Isolation & Shelter Routing
  3. climadeval  : 30-Year Real Estate & Banking ESG Climate Credit Rating (AAA to CCC)
  4. deltasim    : National Climate Adaptation Policy & Budget Simulation Sandbox
"""

ALIASES = {
    'chittagong': 'Chattogram',
    'bogra': 'Bogura',
    'comilla': 'Cumilla',
    'barisal': 'Barishal',
    'jessore': 'Jashore',
    'chapai': 'Chapai Nawabganj',
    'moulvibazar': 'Maulvibazar',
    'dacca': 'Dhaka'
}

def normalize_district_name(name: str) -> str:
    cleaned = name.strip()
    return ALIASES.get(cleaned.lower(), cleaned)

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_json_path = BASE_DIR / "data" / "processed" / "bangladesh_64_districts_live_weather.json"

BANGLADESH_DISTRICT_PROFILES = {}
if _json_path.exists():
    with open(_json_path, encoding="utf-8") as _f:
        _dists = json.load(_f)
        for _d in _dists:
            BANGLADESH_DISTRICT_PROFILES[_d["name"]] = {
                "name": _d["name"],
                "division": _d.get("division", "Unknown"),
                "population": int(_d.get("pop", 1500000)),
                "area_sq_km": float(_d.get("area", 2000.0)),
                "elevation_amsl_m": float(_d.get("elev", 10.0)),
                "base_inundation_pct": float(_d.get("baseInund", 35.0)),
                "water_recurrence_pct": float(_d.get("waterRecurr", 40.0)),
                "zone": str(_d.get("zone", "General Floodplain")),
                "lat": float(_d.get("lat", 23.8)),
                "lon": float(_d.get("lon", 90.4)),
            }

from .weather_climate_ai import train_and_benchmark_models, predict_future_2months
from .automated_alerts import evaluate_district_disaster_alert, auto_evaluate_all_districts

__version__ = "1.3.0"
__all__ = [
    "normalize_district_name",
    "BANGLADESH_DISTRICT_PROFILES",
    "train_and_benchmark_models",
    "predict_future_2months",
    "evaluate_district_disaster_alert",
    "auto_evaluate_all_districts",
]

