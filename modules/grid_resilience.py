"""
DeltaShield GridResilience Module
=================================
Models critical infrastructure vulnerability during extreme weather:
  - 132kV / 33kV Electrical Substation flood inundation risk
  - Telecom Base Transceiver Station (BTS) tower blackout probability
  - Overhead transmission line storm-trip probability (wind gust limits)
  - Emergency backup diesel generator hours required for hospitals & cyclone shelters
"""

from __future__ import annotations
from typing import Any, Dict
from modules import normalize_district_name, BANGLADESH_DISTRICT_PROFILES

def assess_grid_vulnerability(district_name: str, wind_speed_kmh: float = None, flood_depth_m: float = None) -> Dict[str, Any]:
    std_name = normalize_district_name(district_name)
    prof = BANGLADESH_DISTRICT_PROFILES.get(std_name, BANGLADESH_DISTRICT_PROFILES["Dhaka"])

    pop = prof["population"]
    elev = prof["elevation_amsl_m"]
    base_inund = prof["base_inundation_pct"]
    wind = wind_speed_kmh if wind_speed_kmh is not None else 18.0
    water_rise = flood_depth_m if flood_depth_m is not None else 1.2

    # Estimated Substation Infrastructure in District
    # 1 major 132/33kV grid substation per ~300k persons
    num_substations = max(2, int(pop / 300000))
    num_telecom_towers = max(45, int(pop / 6500))

    # Substation Flood Risk
    # Substations typically built at base elevation + 1.5m plinth
    effective_plinth_height = 2.0
    substation_flood_prob = min(95.0, max(5.0, ((water_rise / effective_plinth_height) * (base_inund / 40.0) * (15.0 / max(3.0, elev))) * 40.0))
    substations_at_risk = int(round(num_substations * (substation_flood_prob / 100.0)))

    # Overhead Line Wind Trip Risk (conductor galloping & tree falls on feeders)
    # Wind threshold: >= 45 km/h begins feeder trips, >= 75 km/h snaps poles
    wind_trip_prob = min(98.0, max(2.0, (wind / 75.0) ** 1.8 * 60.0))

    # Overall Blackout Probability
    blackout_prob = min(99.0, max(4.0, (substation_flood_prob * 0.55 + wind_trip_prob * 0.45)))
    telecom_towers_offline = int(round(num_telecom_towers * (blackout_prob / 100.0 * 0.75)))

    # Generator Fuel Logistics for Hospitals & Shelters
    critical_facilities = max(5, int(pop / 85000))
    daily_diesel_liters_per_facility = 220 # ~9-10 L/hr continuous load
    total_daily_diesel_needed_liters = critical_facilities * daily_diesel_liters_per_facility
    estimated_outage_duration_hours = max(2, int(round((blackout_prob / 100.0) * 72.0)))

    status = "CRITICAL BLACKOUT ALERT" if blackout_prob >= 65.0 else ("ELEVATED GRID STRAIN" if blackout_prob >= 35.0 else "GRID STABLE")

    return {
        "district": std_name,
        "grid_status": status,
        "blackout_probability_pct": round(blackout_prob, 1),
        "substation_vulnerability": {
            "total_substations": num_substations,
            "substations_at_flood_risk": substations_at_risk,
            "flood_submersion_probability_pct": round(substation_flood_prob, 1)
        },
        "telecom_infrastructure": {
            "total_bts_towers": num_telecom_towers,
            "projected_towers_offline": telecom_towers_offline,
            "cellular_coverage_loss_pct": round((telecom_towers_offline / num_telecom_towers) * 100.0, 1)
        },
        "backup_power_logistics": {
            "critical_facilities_monitored": critical_facilities,
            "projected_outage_hours": estimated_outage_duration_hours,
            "daily_diesel_liters": total_daily_diesel_needed_liters,
            "daily_diesel_fuel_liters": total_daily_diesel_needed_liters,
            "strategic_fuel_depot": f"{std_name} Central Petroleum Depot Tank #02"
        }
    }
