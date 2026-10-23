"""
========================================================================================
18_climate_transition_and_scenario_agreement_engine.py
BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - SCIENTIFIC CLIMATE TRANSITION ENGINE
Computes 4 baseline-to-future transition matrices, scenario agreement/disagreement,
and multi-dimensional exposure overlays (Population, Built-up, Agriculture, Services).
Author: Readuanul Farid Fahim | Department of Computer Science & Engineering, North South University
========================================================================================
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPATIAL_DIR = os.path.join(BASE_DIR, 'data', 'spatial')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPATIAL_DIR, exist_ok=True)

# Standardized Köppen-Geiger classes in Bangladesh
KOPPEN_CLASSES = [
    "Am (Tropical Monsoon)",
    "Aw (Tropical Savannah)",
    "Cwa (Temperate, Dry Winter, Hot Summer)",
    "Cwb (Temperate, Dry Winter, Warm Summer)"
]

CLASS_SHORT_MAP = {
    "Am (Tropical Monsoon)": "Am",
    "Aw (Tropical Savannah)": "Aw",
    "Cwa (Temperate, Dry Winter, Hot Summer)": "Cwa",
    "Cwb (Temperate, Dry Winter, Warm Summer)": "Cwb"
}


def compute_transition_matrix(df: pd.DataFrame, base_col: str, fut_col: str, area_col: str = 'AreaSqKm') -> dict:
    """
    Computes a rigorous 4x4 transition matrix in both Area (km²) and Percentage (%),
    along with class-level gains, losses, and net changes.
    """
    total_national_area = df[area_col].sum()
    matrix_km2 = {c_from: {c_to: 0.0 for c_to in KOPPEN_CLASSES} for c_from in KOPPEN_CLASSES}
    
    for _, row in df.iterrows():
        c_from = row[base_col]
        c_to = row[fut_col]
        area = float(row[area_col])
        if c_from in matrix_km2 and c_to in matrix_km2[c_from]:
            matrix_km2[c_from][c_to] += area
            
    # Compute Percentage Matrix & Class Balance
    matrix_pct = {c_from: {c_to: 0.0 for c_to in KOPPEN_CLASSES} for c_from in KOPPEN_CLASSES}
    total_area_changed = 0.0
    
    for c_from in KOPPEN_CLASSES:
        for c_to in KOPPEN_CLASSES:
            val = matrix_km2[c_from][c_to]
            matrix_pct[c_from][c_to] = round((val / total_national_area) * 100.0, 3)
            if c_from != c_to:
                total_area_changed += val

    # Area gained, lost, and net change per class
    class_stats = {}
    for c in KOPPEN_CLASSES:
        baseline_area = df[df[base_col] == c][area_col].sum()
        future_area = df[df[fut_col] == c][area_col].sum()
        area_lost = sum(matrix_km2[c][c_other] for c_other in KOPPEN_CLASSES if c_other != c)
        area_gained = sum(matrix_km2[c_other][c] for c_other in KOPPEN_CLASSES if c_other != c)
        net_change_km2 = future_area - baseline_area
        net_change_pct = (net_change_km2 / baseline_area * 100.0) if baseline_area > 0 else 0.0
        
        class_stats[c] = {
            "BaselineAreaKm2": round(baseline_area, 2),
            "FutureAreaKm2": round(future_area, 2),
            "AreaLostKm2": round(area_lost, 2),
            "AreaGainedKm2": round(area_gained, 2),
            "NetChangeKm2": round(net_change_km2, 2),
            "NetChangePct": round(net_change_pct, 2)
        }

    return {
        "TotalNationalAreaKm2": round(total_national_area, 2),
        "TotalAreaChangedKm2": round(total_area_changed, 2),
        "TotalAreaChangedPct": round((total_area_changed / total_national_area) * 100.0, 2),
        "TotalAreaStableKm2": round(total_national_area - total_area_changed, 2),
        "TotalAreaStablePct": round(((total_national_area - total_area_changed) / total_national_area) * 100.0, 2),
        "MatrixKm2": matrix_km2,
        "MatrixPct": matrix_pct,
        "ClassStats": class_stats
    }


def execute_climate_transition_and_exposure_engine():
    print("=" * 90)
    print("BCIP STEP 2: CLIMATE TRANSITION MATRICES & MULTI-DIMENSIONAL EXPOSURE ENGINE")
    print("=" * 90)

    # Load 49-layer geodatabase
    layers_path = os.path.join(DATA_DIR, 'bangladesh_64_districts_comprehensive_49layers.csv')
    if not os.path.exists(layers_path):
        raise FileNotFoundError(f"Missing master 49 layers dataset at {layers_path}")
        
    df = pd.read_csv(layers_path)
    n_dists = len(df)
    total_pop = df['Population'].sum()
    total_area = df['AreaSqKm'].sum()
    print(f"[+] Loaded 64-District Geodatabase: {n_dists} districts, {total_area:,.1f} km², {total_pop:,.0f} population.")

    # ---------------------------------------------------------------------------------
    # 1. COMPUTE 2 BASELINE-TO-FUTURE TRANSITION ANALYSES (MID-CENTURY 2041-2070)
    # ---------------------------------------------------------------------------------
    comparisons = [
        ("Comparison_A_SSP245_2050", "Baseline (1991–2020) vs SSP2-4.5 (2041–2070)", "Koppen_Baseline", "Koppen_SSP245_2050"),
        ("Comparison_B_SSP585_2050", "Baseline (1991–2020) vs SSP5-8.5 (2041–2070)", "Koppen_Baseline", "Koppen_SSP585_2050")
    ]

    matrices_output = {}
    summary_rows = []

    for comp_id, comp_title, base_col, fut_col in comparisons:
        res = compute_transition_matrix(df, base_col, fut_col)
        matrices_output[comp_id] = {
            "Title": comp_title,
            "BaselineCol": base_col,
            "FutureCol": fut_col,
            "Results": res
        }
        
        summary_rows.append({
            "ComparisonID": comp_id,
            "ComparisonTitle": comp_title,
            "TotalNationalAreaKm2": res["TotalNationalAreaKm2"],
            "TotalAreaChangedKm2": res["TotalAreaChangedKm2"],
            "TotalAreaChangedPct": res["TotalAreaChangedPct"],
            "TotalAreaStableKm2": res["TotalAreaStableKm2"],
            "TotalAreaStablePct": res["TotalAreaStablePct"],
            "Am_NetChangeKm2": res["ClassStats"]["Am (Tropical Monsoon)"]["NetChangeKm2"],
            "Aw_NetChangeKm2": res["ClassStats"]["Aw (Tropical Savannah)"]["NetChangeKm2"],
            "Cwa_NetChangeKm2": res["ClassStats"]["Cwa (Temperate, Dry Winter, Hot Summer)"]["NetChangeKm2"],
            "Cwb_NetChangeKm2": res["ClassStats"]["Cwb (Temperate, Dry Winter, Warm Summer)"]["NetChangeKm2"]
        })
        
        print(f"\n[+] {comp_title}:")
        print(f"    • Total Area Changing Climate Class: {res['TotalAreaChangedKm2']:,.1f} km² ({res['TotalAreaChangedPct']}%)")
        print(f"    • Am (Monsoon) Net Change: {res['ClassStats']['Am (Tropical Monsoon)']['NetChangeKm2']:+,.1f} km²")
        print(f"    • Cwa (Temperate) Net Change: {res['ClassStats']['Cwa (Temperate, Dry Winter, Hot Summer)']['NetChangeKm2']:+,.1f} km²")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(DATA_DIR, 'bcip_climate_transitions_national_summary.csv'), index=False)

    with open(os.path.join(DATA_DIR, 'bcip_climate_transition_matrices.json'), 'w', encoding='utf-8') as f:
        json.dump(matrices_output, f, indent=2)

    # ---------------------------------------------------------------------------------
    # 2. SCENARIO AGREEMENT & DISAGREEMENT ANALYSIS (MID-CENTURY)
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING SCENARIO AGREEMENT (MID-CENTURY 2041–2070)")
    print("=" * 90)

    agreement_records = []
    
    for _, r in df.iterrows():
        b = r['Koppen_Baseline']
        s2_50 = r['Koppen_SSP245_2050']
        s5_50 = r['Koppen_SSP585_2050']
        
        # Mid-Century Classification
        if s2_50 == s5_50:
            if s2_50 == b:
                mid_status = "High Agreement (Stable Class)"
            else:
                mid_status = "High Agreement (Identical Transition)"
        else:
            if s2_50 == b or s5_50 == b:
                mid_status = "Scenario Divergence (One Transitions, One Stable)"
            else:
                mid_status = "Scenario Divergence (Different Transition Classes)"

        agreement_records.append({
            "DistrictID": r['DistrictID'],
            "DistrictName": r['DistrictName'],
            "Division": r['Division'],
            "AreaSqKm": r['AreaSqKm'],
            "Population": r['Population'],
            "MidCentury_SSP245": s2_50,
            "MidCentury_SSP585": s5_50,
            "MidCentury_AgreementStatus": mid_status
        })

    df_agree = pd.DataFrame(agreement_records)
    df_agree.to_csv(os.path.join(DATA_DIR, 'bcip_scenario_agreement_analysis.csv'), index=False)

    print("\n--- Mid-Century (2041–2070) Scenario Agreement Breakdown ---")
    mid_counts = df_agree['MidCentury_AgreementStatus'].value_counts()
    for status, cnt in mid_counts.items():
        area_sum = df_agree[df_agree['MidCentury_AgreementStatus'] == status]['AreaSqKm'].sum()
        pct = (area_sum / total_area) * 100.0
        print(f"  • {status:<50}: {cnt:>2} districts | {area_sum:>12,.1f} km² ({pct:>5.1f}%)")

    # ---------------------------------------------------------------------------------
    # 3. MULTI-DIMENSIONAL EXPOSURE & CONTEXT OVERLAY CALCULATION
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING MULTI-DIMENSIONAL EXPOSURE METRICS (64 DISTRICTS)")
    print("=" * 90)

    # Merge agreement status into master dataset
    df_enriched = df.copy()
    df_enriched['MidCentury_AgreementStatus'] = df_agree['MidCentury_AgreementStatus']

    # Transition Indicator (I_i): Does the district change class under Comparison B (SSP5-8.5 Mid-Century)?
    df_enriched['IsTransitioning_MidSSP585'] = (df_enriched['Koppen_Baseline'] != df_enriched['Koppen_SSP585_2050']).astype(int)

    # 1. Population Exposure: E_z = sum(P_i * I_i)
    df_enriched['ExposedPop_MidSSP585'] = df_enriched['Population'] * df_enriched['IsTransitioning_MidSSP585']
    df_enriched['ExposedPopPct_MidSSP585'] = (df_enriched['ExposedPop_MidSSP585'] / df_enriched['Population']) * 100.0

    # 2. Built-Up Asset Exposure (km² of physical built surface in transition zones)
    df_enriched['BuiltUpAreaSqKm'] = df_enriched['GHSL_BuiltUpSurface_2020_SqKm']
    df_enriched['ExposedBuiltUpKm2_MidSSP585'] = df_enriched['BuiltUpAreaSqKm'] * df_enriched['IsTransitioning_MidSSP585']

    # 3. Agricultural Land Exposure (ESA WorldCover Cropland proxy: ~65% of non-built-up land)
    df_enriched['CroplandAreaSqKm'] = (df_enriched['AreaSqKm'] - df_enriched['BuiltUpAreaSqKm']) * 0.65
    df_enriched['ExposedCroplandKm2_MidSSP585'] = df_enriched['CroplandAreaSqKm'] * df_enriched['IsTransitioning_MidSSP585']

    # 4. Critical Service Accessibility Deficits
    # Road Density
    df_enriched['RoadDensityKmPerSqKm'] = df_enriched['HOTOSM_RoadNetwork_TotalKm'] / df_enriched['AreaSqKm']
    # Healthcare density per 100k people
    df_enriched['HealthFacilitiesPer100k'] = (df_enriched['HOTOSM_HealthFacilities'] / df_enriched['Population']) * 100000.0
    df_enriched['EducationSheltersPer100k'] = (df_enriched['HOTOSM_EducationShelters'] / df_enriched['Population']) * 100000.0

    # 5. Compound Vulnerability Index (CVI) components
    # Standardize individual dimensions for multi-criteria screening
    def min_max(series):
        s_min = series.min()
        s_max = series.max()
        return (series - s_min) / (s_max - s_min + 1e-8)

    # C_z: Climate Transition Intensity (Transition Flag + High Agreement)
    c_z = df_enriched['IsTransitioning_MidSSP585'] * 0.7 + (df_enriched['MidCentury_AgreementStatus'].str.contains('High Agreement')).astype(float) * 0.3
    df_enriched['Score_ClimateTransition_Cz'] = min_max(c_z)

    # P_z: Human Population Exposure Intensity
    df_enriched['Score_PopulationExposure_Pz'] = min_max(df_enriched['Population'] / df_enriched['AreaSqKm'])

    # A_z: Physical Asset & Agrarian Exposure Intensity
    a_z = min_max(df_enriched['BuiltUpAreaSqKm']) * 0.5 + min_max(df_enriched['CroplandAreaSqKm']) * 0.5
    df_enriched['Score_AssetExposure_Az'] = a_z

    # H_z: Service Accessibility Deficit (High Distance + Low Road Density + Low Facility Density)
    h_z = (
        min_max(df_enriched['DistanceToHealthFacility_Km']) * 0.4 +
        (1.0 - min_max(df_enriched['RoadDensityKmPerSqKm'])) * 0.3 +
        (1.0 - min_max(df_enriched['HealthFacilitiesPer100k'])) * 0.3
    )
    df_enriched['Score_ServiceDeficit_Hz'] = min_max(h_z)

    # Save comprehensive district table
    out_csv = os.path.join(DATA_DIR, 'bcip_district_climate_transitions_and_exposure.csv')
    df_enriched.to_csv(out_csv, index=False)
    print(f"[+] Saved complete 64-District Climate Transition & Exposure Dataset: {out_csv}")

    # ---------------------------------------------------------------------------------
    # 4. EXPORT QGIS / ARCGIS READY GEOJSON
    # ---------------------------------------------------------------------------------
    features = []
    for _, row in df_enriched.iterrows():
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        
        # Approximate boundary polygon (~22 km bounding box around centroid)
        delta_lat = 0.20
        delta_lon = 0.22
        coords = [[
            [round(lon - delta_lon, 5), round(lat - delta_lat, 5)],
            [round(lon + delta_lon, 5), round(lat - delta_lat, 5)],
            [round(lon + delta_lon, 5), round(lat + delta_lat, 5)],
            [round(lon - delta_lon, 5), round(lat + delta_lat, 5)],
            [round(lon - delta_lon, 5), round(lat - delta_lat, 5)]
        ]]

        props = {
            "DistrictID": int(row['DistrictID']),
            "DistrictName": str(row['DistrictName']),
            "Division": str(row['Division']),
            "AreaSqKm": float(row['AreaSqKm']),
            "Population": int(row['Population']),
            "Koppen_Baseline": str(row['Koppen_Baseline']),
            "Koppen_SSP245_2050": str(row['Koppen_SSP245_2050']),
            "Koppen_SSP585_2050": str(row['Koppen_SSP585_2050']),
            "MidCentury_AgreementStatus": str(row['MidCentury_AgreementStatus']),
            "IsTransitioning_MidSSP585": int(row['IsTransitioning_MidSSP585']),
            "ExposedPop_MidSSP585": int(row['ExposedPop_MidSSP585']),
            "ExposedBuiltUpKm2_MidSSP585": float(round(row['ExposedBuiltUpKm2_MidSSP585'], 2)),
            "ExposedCroplandKm2_MidSSP585": float(round(row['ExposedCroplandKm2_MidSSP585'], 2)),
            "DistanceToHealthFacility_Km": float(round(row['DistanceToHealthFacility_Km'], 2)),
            "RoadDensityKmPerSqKm": float(round(row['RoadDensityKmPerSqKm'], 2)),
            "Score_ClimateTransition_Cz": float(round(row['Score_ClimateTransition_Cz'], 4)),
            "Score_PopulationExposure_Pz": float(round(row['Score_PopulationExposure_Pz'], 4)),
            "Score_AssetExposure_Az": float(round(row['Score_AssetExposure_Az'], 4)),
            "Score_ServiceDeficit_Hz": float(round(row['Score_ServiceDeficit_Hz'], 4))
        }

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            },
            "properties": props
        })

    geojson_collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }

    out_geojson = os.path.join(SPATIAL_DIR, 'bcip_district_climate_transitions.geojson')
    with open(out_geojson, 'w', encoding='utf-8') as f:
        json.dump(geojson_collection, f, indent=2)

    print(f"[+] Saved QGIS/ArcGIS Ready GeoJSON Layer: {out_geojson}")
    print("=" * 90)


if __name__ == '__main__':
    execute_climate_transition_and_exposure_engine()
