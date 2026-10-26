"""
DeltaShield RiverbankErosion & Char Displacement Module
======================================================
Analyzes dynamic river morphology, alluvion (land deposition) vs diluvion (river erosion),
and projects climate refugee displacement counts across riparian river districts.
"""

from __future__ import annotations
from typing import Any, Dict
from modules import normalize_district_name, BANGLADESH_DISTRICT_PROFILES

RIPARIAN_ACTIVE_EROSION_ZONES = {
    "Sirajganj": {"river": "Jamuna", "annual_erosion_ha": 420.0, "vulnerability": "Extreme", "char_pop": 210000},
    "Kurigram": {"river": "Brahmaputra / Dharla", "annual_erosion_ha": 380.0, "vulnerability": "Extreme", "char_pop": 260000},
    "Gaibandha": {"river": "Jamuna / Teesta", "annual_erosion_ha": 290.0, "vulnerability": "High", "char_pop": 180000},
    "Bogura": {"river": "Jamuna", "annual_erosion_ha": 210.0, "vulnerability": "High", "char_pop": 140000},
    "Jamalpur": {"river": "Old Brahmaputra / Jamuna", "annual_erosion_ha": 190.0, "vulnerability": "Moderate", "char_pop": 120000},
    "Tangail": {"river": "Jamuna", "annual_erosion_ha": 220.0, "vulnerability": "High", "char_pop": 150000},
    "Manikganj": {"river": "Padma / Dhaleshwari", "annual_erosion_ha": 170.0, "vulnerability": "High", "char_pop": 95000},
    "Faridpur": {"river": "Padma", "annual_erosion_ha": 240.0, "vulnerability": "High", "char_pop": 160000},
    "Rajbari": {"river": "Padma", "annual_erosion_ha": 230.0, "vulnerability": "High", "char_pop": 110000},
    "Shariatpur": {"river": "Padma / Meghna", "annual_erosion_ha": 310.0, "vulnerability": "Extreme", "char_pop": 175000},
    "Chandpur": {"river": "Meghna / Padma Confluence", "annual_erosion_ha": 340.0, "vulnerability": "Extreme", "char_pop": 190000},
    "Bhola": {"river": "Lower Meghna Estuary", "annual_erosion_ha": 480.0, "vulnerability": "Critical", "char_pop": 320000},
}

def assess_riverbank_erosion(district_name: str, monsoon_water_shock_pct: float = 0.0) -> Dict[str, Any]:
    std_name = normalize_district_name(district_name)
    prof = BANGLADESH_DISTRICT_PROFILES.get(std_name, BANGLADESH_DISTRICT_PROFILES["Dhaka"])

    riparian_data = RIPARIAN_ACTIVE_EROSION_ZONES.get(std_name, None)
    
    if riparian_data:
        river = riparian_data["river"]
        base_erosion_ha = riparian_data["annual_erosion_ha"]
        vuln_rating = riparian_data["vulnerability"]
        char_population = riparian_data["char_pop"]
    else:
        river = "Inland Tributaries / Minor Channels"
        base_erosion_ha = round(prof["area_sq_km"] * 0.02, 1)
        vuln_rating = "Low to Baseline"
        char_population = int(prof["population"] * 0.03)

    # Dynamic multiplier based on monsoon discharge surge (+0 to +100%)
    shock_mult = 1.0 + (monsoon_water_shock_pct / 100.0) * 0.65
    projected_land_lost_ha = round(base_erosion_ha * shock_mult, 1)
    projected_land_lost_sq_km = round(projected_land_lost_ha / 100.0, 2)

    # Climate Refugee Displacement Estimate
    # Char average population density ~ 5-7 persons per hectare of lost land
    displaced_citizens = int(round(projected_land_lost_ha * 6.2))
    homes_destroyed = int(round(displaced_citizens / 4.8)) # ~4.8 persons per rural household

    # Resettlement Logistics & Cost
    resettlement_cost_usd = round(displaced_citizens * 450.0, 2) # ~$450 per person emergency re-housing grant
    resettlement_corridor = f"{std_name} Inland High-Embankment Cluster Village (Guchhogram #03)"

    return {
        "district": std_name,
        "primary_river_system": river,
        "erosion_vulnerability_rating": vuln_rating,
        "char_population_exposed": char_population,
        "projected_land_loss_ha": projected_land_lost_ha,
        "projected_land_loss_sq_km": projected_land_lost_sq_km,
        "displacement_estimates": {
            "climate_refugees_displaced": displaced_citizens,
            "homesteads_submerged": homes_destroyed,
            "immediate_resettlement_grant_usd": resettlement_cost_usd
        },
        "resettlement_logistics": {
            "designated_host_zone": resettlement_corridor,
            "embankment_protection_directive": "Deploy geo-synthetic sand containers (Geo-Bags) along active river toe."
        }
    }
