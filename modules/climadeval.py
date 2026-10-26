"""
========================================================================================
modules/climadeval.py
PILLAR 3: CLIMADEVAL - 30-YEAR REAL ESTATE & ESG CLIMATE CREDIT RATING ENGINE
========================================================================================
Evaluates 30-year multi-decadal physical asset risk (2025–2050), assigns an institutional
credit grade (AAA to CCC), calculates 30-year property asset value depreciation risk,
and produces structured institutional due diligence appraisal metrics.
========================================================================================
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def _load_data():
    fut_path = os.path.join(DATA_DIR, 'bangladesh_future_projections_2030_2050_all64.csv')
    layers_path = os.path.join(DATA_DIR, 'bangladesh_64_districts_comprehensive_49layers.csv')
    df_fut = pd.read_csv(fut_path) if os.path.exists(fut_path) else pd.DataFrame()
    df_layers = pd.read_csv(layers_path) if os.path.exists(layers_path) else pd.DataFrame()
    return df_fut, df_layers


def compute_climadeval_rating(district_name: str, initial_property_value_usd: float = 500000.0) -> dict:
    from modules import normalize_district_name
    district_name = normalize_district_name(district_name)
    df_fut, df_layers = _load_data()
    if df_layers.empty:
        return {'status': 'error', 'message': 'Dataset not found.'}

    matched_layer = df_layers[df_layers['DistrictName'].str.lower() == district_name.strip().lower()]
    if matched_layer.empty:
        return {'status': 'error', 'message': f"District '{district_name}' not found."}

    layer_row = matched_layer.iloc[0]
    elevation = float(layer_row.get('SRTM_Elevation_M', 10.0))
    water_recurrence = float(layer_row.get('JRC_WaterRecurrence_Pct', 30.0))
    built_surface_sqkm = float(layer_row.get('GHSL_BuiltUpSurface_2020_SqKm', 80.0))
    wealth_index = float(layer_row.get('Meta_RWI_Score', 0.20))

    # Future 2050 Long-Term Horizon projections
    if not df_fut.empty:
        d_fut = df_fut[(df_fut['DistrictName'].str.lower() == district_name.strip().lower()) & (df_fut['HorizonYear'] == 2050)]
        if not d_fut.empty:
            inund_2050 = float(d_fut.iloc[0].get('PredictedInundatedPercentage', 55.0))
            summer_temp_2050 = float(d_fut.iloc[0].get('ProjectedMaxHeatwaveTempC', 42.5))
        else:
            inund_2050 = 55.0
            summer_temp_2050 = 42.0
    else:
        inund_2050 = 55.0
        summer_temp_2050 = 42.0

    # Composite Physical Risk Score (0 = Zero Risk, 100 = Maximum Ruin Risk)
    risk_score = (
        (inund_2050 * 0.40) +
        (water_recurrence * 0.25) +
        (max(0.0, (15.0 - elevation) * 3.0) * 0.20) +
        (max(0.0, (summer_temp_2050 - 38.0) * 4.0) * 0.15)
    )
    risk_score = float(np.clip(risk_score, 5.0, 95.0))

    # Institutional Grade (AAA to CCC)
    if risk_score <= 25.0:
        rating = "AAA"
        rating_desc = "Prime Climate Resilience - Negligible 30-Year Physical Hazard"
        color = "#16a34a"
        depreciation_pct = float(np.clip(risk_score * 0.15, 1.0, 5.0))
    elif risk_score <= 35.0:
        rating = "AA"
        rating_desc = "High Grade - Superior Adaptive Buffer against Monsoon Shocks"
        color = "#22c55e"
        depreciation_pct = float(np.clip(risk_score * 0.22, 5.0, 9.0))
    elif risk_score <= 45.0:
        rating = "A"
        rating_desc = "Upper Medium - Moderate Exposure; Standard Insurance Required"
        color = "#38bdf8"
        depreciation_pct = float(np.clip(risk_score * 0.32, 9.0, 15.0))
    elif risk_score <= 58.0:
        rating = "BBB"
        rating_desc = "Investment Grade Baseline - Elevated Drainage Stress; Embankments Recommended"
        color = "#facc15"
        depreciation_pct = float(np.clip(risk_score * 0.45, 15.0, 24.0))
    elif risk_score <= 70.0:
        rating = "BB"
        rating_desc = "Speculative Climate Grade - Substantial 2050 Submersion Exposure"
        color = "#fb923c"
        depreciation_pct = float(np.clip(risk_score * 0.58, 24.0, 36.0))
    elif risk_score <= 82.0:
        rating = "B"
        rating_desc = "High Vulnerability - Critical Physical Devaluation Expected"
        color = "#f87171"
        depreciation_pct = float(np.clip(risk_score * 0.70, 36.0, 50.0))
    else:
        rating = "CCC"
        rating_desc = "Extreme Risk - Severe Recurring Inundation; High Mortgage Default Risk"
        color = "#dc2626"
        depreciation_pct = float(np.clip(risk_score * 0.85, 50.0, 72.0))

    estimated_depreciation_usd = round(initial_property_value_usd * (depreciation_pct / 100.0), 2)
    retained_property_value_usd = round(initial_property_value_usd - estimated_depreciation_usd, 2)

    return {
        'district': layer_row['DistrictName'],
        'division': layer_row.get('Division', 'Unknown'),
        'climadeval_rating': rating,
        'rating_description': rating_desc,
        'rating_color': color,
        'composite_risk_score': round(risk_score, 1),
        'projected_inundation_2050_pct': round(inund_2050, 1),
        'projected_peak_heat_2050_c': round(summer_temp_2050, 1),
        'elevation_amsl_m': elevation,
        'wealth_index_score': wealth_index,
        'initial_property_value_usd': initial_property_value_usd,
        'estimated_30yr_depreciation_pct': round(depreciation_pct, 1),
        'projected_value_loss_usd': estimated_depreciation_usd,
        'projected_retained_value_usd': retained_property_value_usd,
        'recommended_underwriting_covenant': (
            "Standard Underwriting Approved" if rating in ['AAA', 'AA', 'A']
            else "Mandatory Flood Mitigation Escrow (+1.5% Cap Rate)" if rating == 'BBB'
            else "High Risk Premium Required (+3.2% Loan Surcharge, Max 15-Year Term)"
        )
    }


def get_national_esg_leaderboard(n: int = 10) -> pd.DataFrame:
    df_fut, df_layers = _load_data()
    if df_layers.empty:
        return pd.DataFrame()

    out = []
    for _, row in df_layers.iterrows():
        res = compute_climadeval_rating(row['DistrictName'])
        out.append({
            'District': res['district'],
            'Division': res['division'],
            'Rating': res['climadeval_rating'],
            'Risk_Score': res['composite_risk_score'],
            'Depreciation_30Yr_Pct': res['estimated_30yr_depreciation_pct'],
            'Inundation_2050_Pct': res['projected_inundation_2050_pct'],
            'Covenant': res['recommended_underwriting_covenant']
        })
    df_out = pd.DataFrame(out)
    return df_out.sort_values(by='Risk_Score', ascending=False).head(n).reset_index(drop=True)
