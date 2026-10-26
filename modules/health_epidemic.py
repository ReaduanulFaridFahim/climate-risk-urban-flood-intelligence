"""
DeltaShield HealthEpidemic Intelligence Module
==============================================
Predicts climate-driven disease outbreaks (Dengue, Cholera, Waterborne Diarrheal illness)
and forecasts hospital surge capacity (bed occupancy, IV saline requirements, ORS packets)
leveraging stagnant water recurrence (JRC), temperature/humidity, and population density (WorldPop).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List
from modules import normalize_district_name, BANGLADESH_DISTRICT_PROFILES

def assess_epidemic_risk(district_name: str, flood_inundation_pct: float = None, temp_c: float = None, humidity_pct: float = None) -> Dict[str, Any]:
    std_name = normalize_district_name(district_name)
    prof = BANGLADESH_DISTRICT_PROFILES.get(std_name, BANGLADESH_DISTRICT_PROFILES["Dhaka"])

    pop = prof["population"]
    water_recurr = prof["water_recurrence_pct"]
    base_inund = flood_inundation_pct if flood_inundation_pct is not None else prof["base_inundation_pct"]
    temp = temp_c if temp_c is not None else 31.5
    hum = humidity_pct if humidity_pct is not None else 65.0

    is_urban = prof["zone"] == "Urban/Commercial" or std_name in ["Dhaka", "Chattogram", "Gazipur", "Narayanganj"]
    is_haor = "haor" in prof["zone"].lower()
    is_coastal = "coastal" in prof["zone"].lower()

    # Dengue Outbreak Index (0 - 100)
    # Aedes mosquito thrives in 26-32°C, humidity > 60%, urban water accumulation
    temp_factor_dengue = max(0.0, 1.0 - abs(temp - 29.0) / 7.0)
    hum_factor_dengue = min(1.0, hum / 75.0)
    urban_factor = 1.6 if is_urban else 0.7
    stagnation_factor = min(1.0, (water_recurr * 0.4 + base_inund * 0.6) / 60.0)

    dengue_score = min(98.0, max(5.0, (temp_factor_dengue * 35.0 + hum_factor_dengue * 25.0 + stagnation_factor * 40.0) * urban_factor))

    # Cholera / Waterborne Diarrhea Index (0 - 100)
    # Thrives in warm contaminated flood water, submerged latrines and shallow tube wells
    water_exposure = (base_inund * 0.6 + water_recurr * 0.4)
    rural_vuln = 1.4 if (is_haor or is_coastal or not is_urban) else 0.8
    cholera_score = min(96.0, max(8.0, (water_exposure * 0.75 + (temp / 35.0) * 25.0) * rural_vuln))

    # Hospital Surge Calculations
    hospital_bed_capacity = max(150, int(pop / 1200)) # ~1 bed per 1200 persons baseline
    projected_weekly_admissions = int((pop * 0.00035 * (dengue_score / 50.0)) + (pop * 0.00055 * (cholera_score / 50.0)))
    bed_occupancy_pct = min(180.0, round((projected_weekly_admissions / hospital_bed_capacity) * 100.0, 1))

    # Medical Supply Logistics Requirements
    iv_saline_bags_needed = int(projected_weekly_admissions * 14) # ~2 bags/day for 7 days
    ors_packets_needed = int(projected_weekly_admissions * 45)
    water_purification_tablets = int(pop * 0.15 * 10) # 10 tabs per displaced citizen

    # Classification
    composite_risk = (dengue_score * 0.45 + cholera_score * 0.55)
    if composite_risk >= 70.0:
        alert_tier = "🔴 CRITICAL HEALTH EPIDEMIC WATCH"
        alert_color = "#ef4444"
        directive = "Deploy army mobile medical units, dispatch emergency IV saline stockpiles, chlorinate all local tube wells."
    elif composite_risk >= 45.0:
        alert_tier = "🟠 ELEVATED HEALTH VULNERABILITY"
        alert_color = "#f59e0b"
        directive = "Distribute ORS and water purification tablets; activate upazila health complex fever clinics."
    else:
        alert_tier = "🟢 STABLE MEDICAL BASELINE"
        alert_color = "#10b981"
        directive = "Routine vector spraying and standard outpatient surveillance."

    return {
        "district": std_name,
        "population": pop,
        "dengue_risk_score": round(dengue_score, 1),
        "cholera_risk_score": round(cholera_score, 1),
        "composite_health_risk": round(composite_risk, 1),
        "alert_tier": alert_tier,
        "alert_color": alert_color,
        "directive": directive,
        "hospital_capacity": {
            "total_beds": hospital_bed_capacity,
            "projected_weekly_patients": projected_weekly_admissions,
            "bed_occupancy_pct": bed_occupancy_pct,
            "status": "OVERFLOW SURGE DANGER" if bed_occupancy_pct > 100.0 else "CAPACITY MANAGEABLE"
        },
        "supply_logistics": {
            "iv_saline_bags_1000ml": iv_saline_bags_needed,
            "ors_hydration_packets": ors_packets_needed,
            "halogen_water_purification_tabs": water_purification_tablets
        }
    }
