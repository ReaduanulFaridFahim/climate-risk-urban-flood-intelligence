"""
DeltaShield AgriSupply & Food Security Module
=============================================
Identifies flood-severed arterial trucking highways connecting agricultural producer districts
to Dhaka/Chattogram urban consumer markets, calculates detour logistics (+km, fuel),
and estimates urban commodity inflation spikes.
"""

from __future__ import annotations
from typing import Any, Dict, List
from modules import normalize_district_name, BANGLADESH_DISTRICT_PROFILES

HIGHWAY_CORRIDORS = {
    "N1": {"name": "Dhaka - Chattogram - Cox's Bazar National Highway", "districts": ["Dhaka", "Narayanganj", "Munshiganj", "Cumilla", "Feni", "Chattogram", "Cox's Bazar"]},
    "N2": {"name": "Dhaka - Sylhet High-Speed Arterial Highway", "districts": ["Dhaka", "Narsingdi", "Brahmanbaria", "Kishoreganj", "Habiganj", "Moulvibazar", "Sylhet"]},
    "N5": {"name": "Dhaka - Bangabandhu Bridge - Bogura - Rangpur Highway", "districts": ["Dhaka", "Manikganj", "Tangail", "Sirajganj", "Bogura", "Gaibandha", "Rangpur"]},
    "N8": {"name": "Dhaka - Padma Bridge - Barishal - Patuakhali Expressway", "districts": ["Dhaka", "Munshiganj", "Madaripur", "Shariatpur", "Barishal", "Patuakhali"]},
}

COMMODITY_HUBS = {
    "Bogura": {"staple": "Potato & Winter Vegetables", "daily_trucks_to_capital": 350, "metric_tons_daily": 5200},
    "Munshiganj": {"staple": "Cold Storage Potato & Vegetables", "daily_trucks_to_capital": 420, "metric_tons_daily": 6300},
    "Dinajpur": {"staple": "Aromatic & Coarse Rice", "daily_trucks_to_capital": 280, "metric_tons_daily": 4200},
    "Naogaon": {"staple": "Boro & Miniket Rice", "daily_trucks_to_capital": 310, "metric_tons_daily": 4650},
    "Jashore": {"staple": "Vegetables, Pulses & Cut Flowers", "daily_trucks_to_capital": 240, "metric_tons_daily": 3600},
    "Chattogram": {"staple": "Imported Wheat, Lentils & Edible Oil", "daily_trucks_to_capital": 550, "metric_tons_daily": 9500},
    "Sunamganj": {"staple": "Haor Boro Harvest & Inland Fish", "daily_trucks_to_capital": 180, "metric_tons_daily": 2500},
}

def analyze_supply_chain_bottlenecks(district_name: str, water_rise_m: float = 1.0) -> Dict[str, Any]:
    std_name = normalize_district_name(district_name)
    prof = BANGLADESH_DISTRICT_PROFILES.get(std_name, BANGLADESH_DISTRICT_PROFILES["Dhaka"])

    hub_info = COMMODITY_HUBS.get(std_name, {
        "staple": "Local Mixed Grains & Produce",
        "daily_trucks_to_capital": 90,
        "metric_tons_daily": 1200
    })

    # Submersion calculation of connecting primary highways
    elev = prof["elevation_amsl_m"]
    inund = prof["base_inundation_pct"]
    chokepoint_risk_pct = min(96.0, max(5.0, (water_rise_m / 1.5) * (inund / 40.0) * (18.0 / max(3.0, elev)) * 50.0))

    is_severed = chokepoint_risk_pct >= 48.0
    status = "🔴 ARTERIAL HIGHWAY SUBMERGED (CHOKEPOINT SEVERED)" if is_severed else ("🟠 WATERLOGGED SLOW-DOWN" if chokepoint_risk_pct >= 25.0 else "🟢 HIGHWAY FULLY OPEN")

    # Bypass reroute logistics
    standard_transit_hours = 6.5
    detour_extra_km = int(round((chokepoint_risk_pct / 100.0) * 85.0)) if is_severed else 0
    detour_extra_hours = round(detour_extra_km / 35.0, 1) # ~35 km/h truck detour speed
    extra_diesel_liters_per_truck = int(round(detour_extra_km * 0.38)) # ~0.38 L/km heavy truck

    # Urban Food Inflation Threat in Dhaka/Chattogram
    if is_severed:
        inflation_spike_pct = round(min(45.0, (chokepoint_risk_pct / 100.0) * 32.0), 1)
    else:
        inflation_spike_pct = round((chokepoint_risk_pct / 100.0) * 8.0, 1)

    return {
        "district": std_name,
        "commodity_specialization": hub_info["staple"],
        "daily_volume": {
            "truckloads_to_metropolitan": hub_info["daily_trucks_to_capital"],
            "metric_tons_dispatched": hub_info["metric_tons_daily"]
        },
        "highway_chokepoint_analysis": {
            "chokepoint_status": status,
            "highway_submersion_probability_pct": round(chokepoint_risk_pct, 1),
            "is_corridor_severed": is_severed
        },
        "freight_rerouting_logistics": {
            "standard_transit_time": f"{standard_transit_hours} Hours",
            "detour_additional_distance_km": detour_extra_km,
            "detour_additional_transit_hours": detour_extra_hours,
            "diesel_fuel_penalty_per_truck_liters": extra_diesel_liters_per_truck,
            "recommended_bypass": f"Divert heavy freight via {std_name} Elevated Western Feeder Bypass Link"
        },
        "urban_market_impact": {
            "projected_commodity_price_inflation_pct": f"+{inflation_spike_pct}%",
            "at_risk_metropolitan_hub": "Dhaka Mega-City Wholesale Kitchen Markets (Kawran Bazar / Shyambazar)"
        }
    }
