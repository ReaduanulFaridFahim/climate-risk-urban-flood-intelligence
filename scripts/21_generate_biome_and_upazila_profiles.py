"""
========================================================================================
21_generate_biome_and_upazila_profiles.py
BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - REGIONAL BIOMES & ECOLOGICAL ZONES ENGINE
Categorizes all 64 districts into Bangladesh's 5 macro-ecological biomes:
  1. Haor Basin Depression Zone (Northeastern flash flood & waterlogging)
  2. Coastal Polders & Mangrove Estuary (Salinity, storm surge & tidal inundation)
  3. Chattogram Hill Tracts & Ridge Zone (Steep slope, landslide & flash surge)
  4. Northwest Barind Arid Zone (High LST, groundwater depletion & drought)
  5. Active Brahmaputra-Meghna Central Floodplain (Riverbank erosion & monsoon inundation)
Computes regional exposure totals, service deficits, and policy action packages.
Author: Readuanul Farid Fahim | Department of Computer Science & Engineering, North South University
========================================================================================
"""

import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(DATA_DIR, exist_ok=True)

# 5 Regional Ecological Biomes of Bangladesh
BIOME_DISTRICT_MAPPING = {
    "Haor Basin Depression Zone": [
        "Sunamganj", "Kishoreganj", "Habiganj", "Netrokona", "Sylhet", "Moulvibazar", "Brahmanbaria"
    ],
    "Coastal Polders & Mangrove Estuary": [
        "Barguna", "Patuakhali", "Bhola", "Bagerhat", "Khulna", "Satkhira", "Pirojpur",
        "Jhalokati", "Noakhali", "Lakshmipur", "Feni", "Cox's Bazar", "Chittagong"
    ],
    "Chattogram Hill Tracts & Ridge Zone": [
        "Bandarban", "Rangamati", "Khagrachhari"
    ],
    "Northwest Barind Arid Zone": [
        "Rajshahi", "Naogaon", "Chapai Nawabganj", "Dinajpur", "Panchagarh",
        "Thakurgaon", "Joypurhat", "Natore"
    ]
}


def assign_biome(district_name: str) -> str:
    for biome, districts in BIOME_DISTRICT_MAPPING.items():
        if district_name in districts:
            return biome
    return "Active Brahmaputra-Meghna Central Floodplain"


def generate_biome_profiles():
    print("=" * 90)
    print("BCIP STEP: GENERATING REGIONAL ECOLOGICAL BIOMES & POLICY PROFILES")
    print("=" * 90)

    rankings_path = os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv')
    if not os.path.exists(rankings_path):
        raise FileNotFoundError(f"Missing master rankings at {rankings_path}")

    df = pd.read_csv(rankings_path)
    df['EcologicalBiome'] = df['DistrictName'].apply(assign_biome)

    # Save biome column back to rankings dataset
    df.to_csv(rankings_path, index=False)
    print(f"[+] Injected 'EcologicalBiome' attribute into 64-district dataset.")

    # Compute Regional Biome Aggregations
    biome_records = []
    total_nat_pop = df['Population'].sum()
    total_nat_area = df['AreaSqKm'].sum()

    for biome_name, group in df.groupby('EcologicalBiome'):
        n_dist = len(group)
        area_sum = group['AreaSqKm'].sum()
        pop_sum = group['Population'].sum()
        
        # Exposure metrics
        trans_dist = group['IsTransitioning_MidSSP585'].sum()
        trans_area = group[group['IsTransitioning_MidSSP585'] == 1]['AreaSqKm'].sum()
        trans_pop = group['ExposedPop_MidSSP585'].sum()
        built_sum = group['BuiltUpAreaSqKm'].sum()
        cropland_sum = group['CroplandAreaSqKm'].sum()
        
        # Service & Vulnerability averages
        mean_health_dist = group['DistanceToHealthFacility_Km'].mean()
        mean_road_dens = group['RoadDensityKmPerSqKm'].mean()
        mean_bcip_score = group['Score_Model4_IntegratedBCIP'].mean()
        mean_exposure_score = group['Score_Model1_ExposureOnly'].mean()
        
        # Robust Priority Distribution
        tier1_count = group['BCIP_RobustnessTier'].str.contains('Tier 1').sum()
        tier2_count = group['BCIP_RobustnessTier'].str.contains('Tier 2').sum()

        # Strategic Policy Intervention Recommendation
        if "Haor" in biome_name:
            strategy = "Pre-monsoon early flash flood warning; elevated Mujib Killas; floating medical boats for Boro harvest."
        elif "Coastal" in biome_name:
            strategy = "Polder embankment revetment; tidal sluice gate automation; solar-powered desalination stations."
        elif "Hill Tracts" in biome_name:
            strategy = "Slope stability monitoring; community landslide sanctuaries; forest catchment canopy restoration."
        elif "Barind" in biome_name:
            strategy = "Managed aquifer recharge; drought-tolerant Aman rice cultivars; extreme heatwave cooling shelters."
        else:
            strategy = "Active riverbank erosion revetment; modular floating schools; seasonal flood displacement insurance."

        biome_records.append({
            "EcologicalBiome": biome_name,
            "DistrictCount": n_dist,
            "TotalAreaSqKm": round(area_sum, 1),
            "AreaNationalPct": round((area_sum / total_nat_area) * 100.0, 2),
            "TotalPopulation": int(pop_sum),
            "PopulationNationalPct": round((pop_sum / total_nat_pop) * 100.0, 2),
            "DistrictsTransitioningSSP585": int(trans_dist),
            "TransitionAreaSqKm": round(trans_area, 1),
            "ExposedPopulationSSP585": int(trans_pop),
            "TotalBuiltUpSqKm": round(built_sum, 1),
            "TotalCroplandSqKm": round(cropland_sum, 1),
            "MeanDistanceToHospitalKm": round(mean_health_dist, 2),
            "MeanRoadDensityKmPerSqKm": round(mean_road_dens, 2),
            "MeanBCIPPriorityScore": round(mean_bcip_score, 4),
            "MeanExposureOnlyScore": round(mean_exposure_score, 4),
            "Tier1_RobustDistrictsCount": int(tier1_count),
            "Tier2_ClimateSensitiveCount": int(tier2_count),
            "RecommendedAdaptationStrategy": strategy
        })

    df_biome = pd.DataFrame(biome_records).sort_values('MeanBCIPPriorityScore', ascending=False)

    out_csv = os.path.join(DATA_DIR, 'bcip_regional_biomes_summary.csv')
    out_json = os.path.join(DATA_DIR, 'bcip_regional_biomes_summary.json')

    df_biome.to_csv(out_csv, index=False)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(df_biome.to_dict(orient="records"), f, indent=2)

    print(f"[+] Saved Regional Biome Summaries:")
    print(f"    - CSV: {out_csv}")
    print(f"    - JSON: {out_json}")

    print("\n--- ECOLOGICAL BIOME PROFILES SUMMARY ---")
    for _, r in df_biome.iterrows():
        print(f"• {r['EcologicalBiome']:<46} | Dists: {r['DistrictCount']:>2} | Area: {r['TotalAreaSqKm']:>8,.1f} km² ({r['AreaNationalPct']:>4.1f}%) | Pop: {r['TotalPopulation']:>11,} | Mean BCIP: {r['MeanBCIPPriorityScore']:.4f}")

    return df_biome


if __name__ == '__main__':
    generate_biome_profiles()
