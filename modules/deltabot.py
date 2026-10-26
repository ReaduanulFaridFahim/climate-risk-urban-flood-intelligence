"""
DeltaShield DeltaBot AI Climate Copilot
=======================================
Conversational AI engine that processes natural language queries in English and Bengali,
cross-references the 64-district live database and all 8 climate pillars,
and outputs verified intelligence and operational directives.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict
from modules import normalize_district_name, BANGLADESH_DISTRICT_PROFILES
from modules.agroclaim import assess_district_crop_risk
from modules.evacroute import assess_isolation_risk
from modules.climadeval import compute_climadeval_rating
from modules.deltasim import run_policy_simulation
from modules.weather_climate_ai import predict_future_2months
from modules.health_epidemic import assess_epidemic_risk
from modules.grid_resilience import assess_grid_vulnerability
from modules.riverbank_erosion import assess_riverbank_erosion
from modules.agri_supply import analyze_supply_chain_bottlenecks

def ask_deltabot(query: str) -> Dict[str, Any]:
    q = query.lower().strip()
    
    # Identify district mentioned
    target_district = "Dhaka"
    for name in BANGLADESH_DISTRICT_PROFILES.keys():
        if name.lower() in q:
            target_district = name
            break
        # Common aliases
        if "chittagong" in q: target_district = "Chattogram"; break
        if "cox" in q: target_district = "Cox's Bazar"; break
        if "bogra" in q: target_district = "Bogura"; break
        if "comilla" in q: target_district = "Cumilla"; break
        if "barisal" in q: target_district = "Barishal"; break
        if "jessore" in q: target_district = "Jashore"; break

    # Route based on user intent
    if any(k in q for k in ["crop", "insurance", "agro", "payout", "rice", "farmer", "boro", "aman", "ক্ষতিপূরণ", "কৃষি"]):
        res = assess_district_crop_risk(target_district, rainfall_anomaly_pct=35.0)
        payout_txt = f"${res['payout_amount_usd']:,.2f} USD" if res['payout_triggered'] else "$0.00 USD (Threshold <45%)"
        answer = (
            f"🌾 [AgroClaim Intel] In {target_district}, total standing cropland is {res['total_cropland_ha']:,} hectares. "
            f"At 48% simulated deluge inundation, {res['submerged_cropland_ha']:,} ha are submerged with estimated farmer economic loss of ${res['farmer_loss_usd']:,.2f} USD. "
            f"Parametric mobile payout trigger: {'🚨 APPROVED' if res['payout_triggered'] else 'SAFE'}. Total payout: {payout_txt}."
        )
        pillar = "AgroClaim (Crop Insurance)"

    elif any(k in q for k in ["shelter", "evac", "escape", "road", "isolate", "আশ্রয়", "উদ্ধার", "রাস্তা"]):
        res = assess_isolation_risk(target_district, water_rise_m=1.2)
        answer = (
            f"🚑 [EvacRoute Intel] In {target_district} (Mean Elevation: {res['elevation_amsl_m']}m AMSL), a +1.2m water rise submerges {res['submerged_roads_km']:,} km of roads ({res['submersion_pct']}%). "
            f"Isolation Score: {res['island_isolation_index']}/100 ({res['isolation_status']}). "
            f"Nearest designated shelter is {res['shelter']['name']} ({res['shelter']['type']}, capacity {res['shelter']['capacity']}). "
            f"Directive: {res['shelter']['routing_directive']}"
        )
        pillar = "EvacRoute (Logistics)"

    elif any(k in q for k in ["property", "bank", "house", "esg", "deval", "দাম", "বাড়ি", "ব্যাংক"]):
        res = compute_climadeval_rating(target_district, initial_property_value_usd=500000.0)
        answer = (
            f"🏢 [ClimaDeval Intel] {target_district} holds an Institutional ESG Climate Rating of [{res['climadeval_rating']}] ({res['rating_description']}). "
            f"Composite Physical Risk: {res['composite_risk_score']}/100. Over a 30-year horizon (SSP5-8.5), projected asset haircut is -{res['estimated_30yr_depreciation_pct']}% (-${res['projected_value_loss_usd']:,.2f} USD). "
            f"Retained 2050 Value: ${res['projected_retained_value_usd']:,.2f} USD. Underwriting covenant: {res['recommended_underwriting_covenant']}."
        )
        pillar = "ClimaDeval (Real Estate ESG)"

    elif any(k in q for k in ["health", "hospital", "dengue", "cholera", "disease", "ডেঙ্গু", "কলেরা", "হাসপাতাল"]):
        res = assess_epidemic_risk(target_district)
        answer = (
            f"🏥 [HealthEpidemic Intel] {target_district} Health Risk Index: Dengue Score {res['dengue_risk_score']}/100, Cholera Score {res['cholera_risk_score']}/100. "
            f"Alert: {res['alert_tier']}. Hospital bed occupancy projected at {res['hospital_capacity']['bed_occupancy_pct']}% ({res['hospital_capacity']['projected_weekly_patients']} weekly patients). "
            f"Medical supply requirements: {res['supply_logistics']['iv_saline_bags_1000ml']:,} IV saline bags & {res['supply_logistics']['ors_hydration_packets']:,} ORS packets. "
            f"Directive: {res['directive']}"
        )
        pillar = "HealthEpidemic (Disease & Hospitals)"

    elif any(k in q for k in ["power", "grid", "electricity", "blackout", "substation", "tower", "বিদ্যুৎ", "ব্ল্যাকআউট"]):
        res = assess_grid_vulnerability(target_district)
        answer = (
            f"⚡ [GridResilience Intel] In {target_district}, blackout probability is {res['blackout_probability_pct']}% ({res['grid_status']}). "
            f"{res['substation_vulnerability']['substations_at_flood_risk']} of {res['substation_vulnerability']['total_substations']} major substations face inundation risk. "
            f"Projected cellular BTS towers offline: {res['telecom_infrastructure']['projected_towers_offline']} ({res['telecom_infrastructure']['cellular_coverage_loss_pct']}% coverage drop). "
            f"Critical hospitals require {res['backup_power_logistics']['daily_diesel_liters']:,} liters of backup generator diesel daily."
        )
        pillar = "GridResilience (Infrastructure)"

    elif any(k in q for k in ["erosion", "river", "char", "displace", "ভাঙন", "নদী", "চর"]):
        res = assess_riverbank_erosion(target_district)
        answer = (
            f"🌊 [RiverbankErosion Intel] Active river system in {target_district}: {res['primary_river_system']}. "
            f"Vulnerability: {res['erosion_vulnerability_rating']}. Projected annual land loss: {res['projected_land_loss_ha']} hectares ({res['projected_land_loss_sq_km']} sq km). "
            f"Climate refugees projected: {res['displacement_estimates']['climate_refugees_displaced']:,} citizens ({res['displacement_estimates']['homesteads_submerged']:,} homes lost). "
            f"Host relocation zone: {res['resettlement_logistics']['designated_host_zone']}."
        )
        pillar = "RiverbankErosion (Char Displaced)"

    elif any(k in q for k in ["food", "supply", "highway", "truck", "inflation", "খাদ্য", "ট্রাক", "হাইওয়ে"]):
        res = analyze_supply_chain_bottlenecks(target_district)
        answer = (
            f"🍞 [AgriSupply Intel] {target_district} specializes in {res['commodity_specialization']}, dispatching {res['daily_volume']['truckloads_to_metropolitan']} trucks daily ({res['daily_volume']['metric_tons_dispatched']} tons). "
            f"Highway Status: {res['highway_chokepoint_analysis']['chokepoint_status']}. "
            f"Truck detour required: +{res['freight_rerouting_logistics']['detour_additional_distance_km']} km (+{res['freight_rerouting_logistics']['detour_additional_transit_hours']} hrs). "
            f"Urban market price inflation in Dhaka kitchen markets projected at: {res['urban_market_impact']['projected_commodity_price_inflation_pct']}."
        )
        pillar = "AgriSupply (Food Logistics)"

    elif any(k in q for k in ["budget", "sim", "policy", "million", "বাজেট"]):
        res = run_policy_simulation(500.0)
        answer = (
            f"🎮 [DeltaSim Policy Sandbox] National Adaptation Allocation of $500M USD achieves a Resilience Score of {res['national_resilience_score']}/100 ({res['evaluation']}). "
            f"Protects {res['mitigated_2050']['lives_protected']:,} citizens and avoids ${res['mitigated_2050']['damage_avoided_usd_billions']} Billion USD in severe damages by 2050."
        )
        pillar = "DeltaSim (Budget Policy)"

    elif any(k in q for k in ["forecast", "future", "weather", "rain", "temperature", "আবহাওয়া", "বৃষ্টি"]):
        res = predict_future_2months(target_district, days=60)
        answer = (
            f"🌤️ [Weather & Climate AI] 60-day prediction for {target_district} ({res['division']}): Expected temp range: {res['summary']['avg_temp_min_c']}°C - {res['summary']['avg_temp_max_c']}°C. "
            f"Projected cumulative rainfall: {res['summary']['total_precipitation_mm']} mm (Peak Single Day: {res['summary']['max_single_day_rain_mm']} mm). "
            f"Dense fog days: {res['summary']['dense_fog_days']}. Peak flood threat score: {res['summary']['peak_flood_risk_score']}/100."
        )
        pillar = "Weather & Climate AI (60-Day)"

    else:
        answer = (
            f"🛡️ [DeltaShield Intelligence Engine] Online and monitoring all 64 districts. "
            f"Target District Selected: {target_district} ({BANGLADESH_DISTRICT_PROFILES[target_district]['division']} Division). "
            f"You can ask me about: (1) Crop Insurance Payouts, (2) Submerged Shelter Evacuation, (3) Real Estate ESG Ratings, "
            f"(4) Health Epidemics (Dengue/Cholera), (5) Power Grid & Telecom Outages, (6) Riverbank Erosion, (7) Highway Food Bottlenecks, or (8) 60-Day Weather Predictions."
        )
        pillar = "Unified Command"

    return {
        "query": query,
        "target_district": target_district,
        "pillar": pillar,
        "response": answer
    }
