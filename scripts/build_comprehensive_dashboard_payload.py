"""
scripts/build_comprehensive_dashboard_payload.py
Aggregates each and every dataset in data/processed into dashboard/data.json
"""

import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')
os.makedirs(DASHBOARD_DIR, exist_ok=True)

def clean_val(v):
    if pd.isna(v) or v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if np.isinf(v) or np.isnan(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

print("[1/6] Loading Core 64 Districts, 49 Layers & Rankings...")
df_rank = load_csv('bcip_model_comparison_and_rankings.csv')
df_49 = load_csv('bangladesh_64_districts_comprehensive_49layers.csv')
df_multi = load_csv('bangladesh_multi_hazard_ai_dataset.csv')
df_trans = load_csv('bcip_district_climate_transitions_and_exposure.csv')

# Merge district datasets
merged_df = df_rank.copy()
if not df_49.empty and 'DistrictName' in df_49.columns:
    cols_to_add = [c for c in df_49.columns if c not in merged_df.columns or c == 'DistrictName']
    merged_df = pd.merge(merged_df, df_49[cols_to_add], on='DistrictName', how='left')

if not df_multi.empty and 'DistrictName' in df_multi.columns:
    cols_to_add = [c for c in df_multi.columns if c not in merged_df.columns or c == 'DistrictName']
    merged_df = pd.merge(merged_df, df_multi[cols_to_add], on='DistrictName', how='left')

districts_records = []
for r in merged_df.to_dict(orient='records'):
    cleaned_r = {k: clean_val(v) for k, v in r.items()}
    districts_records.append(cleaned_r)

print(f"[+] Processed {len(districts_records)} comprehensive districts with {len(merged_df.columns)} indicators each.")

print("[2/6] Loading Future Projections (2030 & 2050)...")
df_future = load_csv('bangladesh_future_projections_2030_2050_all64.csv')
future_records = [{k: clean_val(v) for k, v in r.items()} for r in df_future.to_dict(orient='records')] if not df_future.empty else []

print("[3/6] Loading UN HDX Humanitarian District Data...")
df_hdx = load_csv('bangladesh_hdx_humanitarian_usecases_district_summary.csv')
hdx_records = [{k: clean_val(v) for k, v in r.items()} for r in df_hdx.to_dict(orient='records')] if not df_hdx.empty else []

print("[4/6] Loading AI/ML Model Benchmarks & Feature Importance...")
df_models = load_csv('production_model_benchmark_report.csv')
models_records = [{k: clean_val(v) for k, v in r.items()} for r in df_models.to_dict(orient='records')] if not df_models.empty else []

df_feat = load_csv('nationwide_flood_feature_importance.csv')
feature_importance = [{k: clean_val(v) for k, v in r.items()} for r in df_feat.to_dict(orient='records')] if not df_feat.empty else []

print("[5/6] Loading 111-Dataset Master Catalog...")
cat_json = load_json('bcip_master_catalog.json')
cat_list = cat_json if isinstance(cat_json, list) else cat_json.get('datasets', [])
if not cat_list:
    df_cat = load_csv('bcip_master_catalog.csv')
    if not df_cat.empty:
        cat_list = [{k: clean_val(v) for k, v in r.items()} for r in df_cat.to_dict(orient='records')]

print("[6/6] Loading 60-Day Forecast & Live Satellite Telemetry...")
forecast_60 = load_json('bangladesh_60day_weather_forecast.json')
satellite_telemetry = load_json('bangladesh_satellite_live_telemetry.json')
live_alerts = load_json('bangladesh_live_alerts.json')
biomes_summary = load_json('bcip_regional_biomes_summary.json')

payload = {
    "districts": districts_records,
    "total_districts": len(districts_records),
    "future_projections": future_records,
    "humanitarian_hdx": hdx_records,
    "models_benchmark": models_records,
    "feature_importance": feature_importance,
    "catalog_111": cat_list,
    "forecast_60day": forecast_60,
    "satellite_telemetry": satellite_telemetry,
    "live_alerts": live_alerts,
    "biomes_summary": biomes_summary
}

out_file = os.path.join(DASHBOARD_DIR, 'data.json')
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(payload, f)

size_mb = os.path.getsize(out_file) / (1024 * 1024)
print(f"\n[SUCCESS] Built comprehensive dashboard payload: {out_file} ({size_mb:.2f} MB)")
