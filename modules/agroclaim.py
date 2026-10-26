"""
========================================================================================
modules/agroclaim.py
PILLAR 1: AGROCLAIM - PARAMETRIC CROP INSURANCE & SMART PAYOUT ENGINE
========================================================================================
Calculates seasonal crop standing exposure (Boro, Aman, Aus rice in hectares),
evaluates 40-year JRC satellite flood recurrence, calculates fair actuarial insurance
premiums ($/ha), and simulates zero-paperwork automated mobile payout triggers.
========================================================================================
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def _load_data():
    layers_path = os.path.join(DATA_DIR, 'bangladesh_64_districts_comprehensive_49layers.csv')
    preds_path = os.path.join(DATA_DIR, 'bangladesh_national_flood_predictions.csv')
    
    df_layers = pd.read_csv(layers_path) if os.path.exists(layers_path) else pd.DataFrame()
    df_preds = pd.read_csv(preds_path) if os.path.exists(preds_path) else pd.DataFrame()
    return df_layers, df_preds


def assess_district_crop_risk(district_name: str, rainfall_anomaly_pct: float = 0.0) -> dict:
    from modules import normalize_district_name
    district_name = normalize_district_name(district_name)
    df_layers, df_preds = _load_data()
    if df_layers.empty:
        return {'status': 'error', 'message': 'Dataset not found.'}
        
    matched = df_layers[df_layers['DistrictName'].str.lower() == district_name.strip().lower()]
    if matched.empty:
        return {'status': 'error', 'message': f"District '{district_name}' not found."}
        
    row = matched.iloc[0]
    matched_p = df_preds[df_preds['DistrictName'].str.lower() == district_name.strip().lower()]
    pred_row = matched_p.iloc[0] if not matched_p.empty else {}

    area_sqkm = float(row.get('AreaSqKm', 2000.0))
    total_cropland_ha = round(area_sqkm * 100.0 * 0.65, 1)

    boro_ndvi = float(row.get('Rice_NDVI_Boro', 0.58))
    aman_ndvi = float(row.get('Rice_NDVI_Aman', 0.62))
    aus_ndvi = float(row.get('Rice_NDVI_Aus', 0.45))

    boro_ha = round(total_cropland_ha * 0.50 * (boro_ndvi / 0.60), 1)
    aman_ha = round(total_cropland_ha * 0.36 * (aman_ndvi / 0.60), 1)
    aus_ha = round(total_cropland_ha * 0.14 * (aus_ndvi / 0.60), 1)

    base_inund = float(pred_row.get('PredictedInundatedPercentage', 32.5))
    water_recurrence = float(row.get('JRC_WaterRecurrence_Pct', 28.0))
    
    shock_factor = 1.0 + (rainfall_anomaly_pct / 100.0) * 0.50
    sim_inund_pct = float(np.clip(base_inund * shock_factor, 2.0, 95.0))
    submerged_cropland_ha = round((sim_inund_pct / 100.0) * total_cropland_ha, 1)

    # 3.4 tons/ha yield, $320/ton world market average
    avg_yield_tons = 3.45
    paddy_usd_ton = 320.0
    crop_loss_tons = round(submerged_cropland_ha * avg_yield_tons * 0.85, 1)
    economic_loss_usd = round(crop_loss_tons * paddy_usd_ton, 2)

    # Actuarial pure premium calculation ($/ha)
    base_rate = (water_recurrence / 100.0) * 0.08 * 1.25
    premium_rate_pct = float(np.clip(base_rate * 100.0, 3.5, 16.5))
    premium_usd_per_ha = round(premium_rate_pct * 12.0, 2)

    # Parametric Trigger threshold (45% Inundation)
    trigger_threshold = 45.0
    payout_triggered = sim_inund_pct >= trigger_threshold
    payout_pct = 0.0
    if payout_triggered:
        payout_pct = min(100.0, (sim_inund_pct - trigger_threshold) * 2.5 + 40.0)

    total_insured_usd = total_cropland_ha * avg_yield_tons * paddy_usd_ton * 0.50
    total_payout_usd = round((payout_pct / 100.0) * total_insured_usd, 2)

    return {
        'district': row['DistrictName'],
        'division': row.get('Division', 'Unknown'),
        'total_cropland_ha': total_cropland_ha,
        'crops': {
            'boro_rice_ha': boro_ha,
            'aman_rice_ha': aman_ha,
            'aus_rice_ha': aus_ha
        },
        'inundation_pct': round(sim_inund_pct, 1),
        'submerged_cropland_ha': submerged_cropland_ha,
        'lost_production_tons': crop_loss_tons,
        'farmer_loss_usd': economic_loss_usd,
        'premium_rate_pct': round(premium_rate_pct, 2),
        'premium_usd_per_ha': premium_usd_per_ha,
        'trigger_threshold_pct': trigger_threshold,
        'payout_triggered': payout_triggered,
        'payout_pct': round(payout_pct, 1),
        'payout_amount_usd': total_payout_usd
    }


def get_national_crop_summary(n: int = 10) -> pd.DataFrame:
    df_layers, _ = _load_data()
    if df_layers.empty:
        return pd.DataFrame()

    out = []
    for _, row in df_layers.iterrows():
        res = assess_district_crop_risk(row['DistrictName'], rainfall_anomaly_pct=20.0)
        out.append({
            'District': res['district'],
            'Division': res['division'],
            'Cropland_Ha': res['total_cropland_ha'],
            'Submerged_Ha': res['submerged_cropland_ha'],
            'Loss_USD_M': round(res['farmer_loss_usd'] / 1e6, 2),
            'Premium_USD_Ha': res['premium_usd_per_ha'],
            'Payout': '🚨 TRIGGERED' if res['payout_triggered'] else '✅ SAFE'
        })
    df = pd.DataFrame(out)
    return df.sort_values(by='Loss_USD_M', ascending=False).head(n).reset_index(drop=True)
