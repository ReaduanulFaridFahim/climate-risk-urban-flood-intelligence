"""
scripts/23_build_website_payload.py
MASTER ALL-DATA CONNECTOR & BUNDLER FOR PRODUCTION WEB PLATFORM
Compiles every single dataset, spatial layer, model benchmark, feature importance,
satellite telemetry feed, live alert, and CMIP6 transition matrix into a high-speed unified payload.
"""

import os
import json
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPATIAL_DIR = os.path.join(BASE_DIR, 'data', 'spatial')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DATA_DIR = os.path.join(BASE_DIR, 'web', 'static', 'data')

os.makedirs(STATIC_DATA_DIR, exist_ok=True)


def build_master_all_data_payload():
    print("=" * 80)
    print("[+] CONNECTING ALL DATASETS TO WEBSITE PAYLOAD...")
    print("=" * 80)

    # Sync model reports & spatial GeoJSONs into DATA_DIR for instant direct download
    files_to_sync = [
        (os.path.join(MODELS_DIR, 'nationwide_flood_feature_importance.csv'), os.path.join(DATA_DIR, 'nationwide_flood_feature_importance.csv')),
        (os.path.join(MODELS_DIR, 'production_model_benchmark_report.csv'), os.path.join(DATA_DIR, 'production_model_benchmark_report.csv')),
        (os.path.join(MODELS_DIR, 'production_model_registry.json'), os.path.join(DATA_DIR, 'production_model_registry.json')),
        (os.path.join(MODELS_DIR, 'future_forecast_metrics_summary.json'), os.path.join(DATA_DIR, 'future_forecast_metrics_summary.json')),
        (os.path.join(STATIC_DATA_DIR, 'bangladesh_districts.geojson'), os.path.join(DATA_DIR, 'bangladesh_districts.geojson')),
        (os.path.join(SPATIAL_DIR, 'bangladesh_64_districts_full_49layers.geojson'), os.path.join(DATA_DIR, 'bangladesh_64_districts_full_49layers.geojson')),
        (os.path.join(SPATIAL_DIR, 'bcip_district_climate_transitions.geojson'), os.path.join(DATA_DIR, 'bcip_district_climate_transitions.geojson')),
    ]
    for src, dst in files_to_sync:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"[SYNC] Copied {os.path.basename(src)} -> {dst}")

    # 1. 64 Districts with complete 49 layers + flood + CMIP6 + rankings + HDX
    df_hdx = pd.read_csv(os.path.join(DATA_DIR, 'bangladesh_hdx_humanitarian_usecases_district_summary.csv'))
    df_nat = pd.read_csv(os.path.join(DATA_DIR, 'bangladesh_national_flood_predictions.csv'))
    df_49 = pd.read_csv(os.path.join(DATA_DIR, 'bangladesh_64_districts_comprehensive_49layers.csv'))
    df_bcip = pd.read_csv(os.path.join(DATA_DIR, 'bcip_district_climate_transitions_and_exposure.csv'))
    df_rank = pd.read_csv(os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv'))

    # Merge all attributes by DistrictID
    df_m = df_hdx.merge(df_nat.drop(columns=['DistrictName', 'Division', 'Population', 'AreaSqKm'], errors='ignore'), on='DistrictID')
    df_m = df_m.merge(df_49.drop(columns=['DistrictName', 'Division'], errors='ignore'), on='DistrictID', suffixes=('', '_49'))
    
    # Add BCIP transition and model rankings
    bcip_cols = [c for c in df_bcip.columns if c not in df_m.columns and c not in ['DistrictName', 'Division']]
    df_m = df_m.merge(df_bcip[['DistrictID'] + bcip_cols], on='DistrictID', how='left')

    rank_cols = [c for c in df_rank.columns if c not in df_m.columns and c not in ['DistrictName', 'Division']]
    df_m = df_m.merge(df_rank[['DistrictID'] + rank_cols], on='DistrictID', how='left')

    districts = df_m.to_dict(orient="records")
    print(f"[+] Connected 64 districts across {len(df_m.columns)} combined spatial & analytical variables.")

    # 2. CMIP6 Bioclimatic Transitions
    df_cmip_sum = pd.read_csv(os.path.join(DATA_DIR, 'bcip_climate_transitions_national_summary.csv'))
    with open(os.path.join(DATA_DIR, 'bcip_climate_transition_matrices.json'), 'r', encoding='utf-8') as f:
        cmip_matrices = json.load(f)
    with open(os.path.join(DATA_DIR, 'bcip_regional_biomes_summary.json'), 'r', encoding='utf-8') as f:
        regional_biomes = json.load(f)
    df_agree = pd.read_csv(os.path.join(DATA_DIR, 'bcip_scenario_agreement_analysis.csv'))

    cmip6_data = {
        "summary": df_cmip_sum.to_dict(orient="records"),
        "matrices": cmip_matrices,
        "regional_biomes": regional_biomes,
        "scenario_agreement": df_agree.to_dict(orient="records")
    }

    # 3. Enterprise AI Models Benchmark & Feature Importance
    df_models = pd.read_csv(os.path.join(MODELS_DIR, 'production_model_benchmark_report.csv'))
    df_fi = pd.read_csv(os.path.join(MODELS_DIR, 'nationwide_flood_feature_importance.csv'))
    with open(os.path.join(DATA_DIR, 'bcip_spearman_rank_correlations.json'), 'r', encoding='utf-8') as f:
        spearman_corr = json.load(f)
    df_top15 = pd.read_csv(os.path.join(DATA_DIR, 'bcip_top15_robust_priority_districts.csv'))

    ai_models_data = {
        "benchmark_report": df_models.to_dict(orient="records"),
        "feature_importances": df_fi.to_dict(orient="records"),
        "spearman_correlations": spearman_corr,
        "top15_robust_districts": df_top15.to_dict(orient="records")
    }

    # 4. Future Inundation Projections (2030, 2040, 2050)
    df_fut = pd.read_csv(os.path.join(DATA_DIR, 'bangladesh_future_projections_2030_2050_all64.csv'))
    future_data = df_fut.to_dict(orient="records")

    # 5. 60-Day Sub-Seasonal Weather Forecasts (All 8 Divisions)
    with open(os.path.join(DATA_DIR, 'bangladesh_60day_weather_forecast.json'), 'r', encoding='utf-8') as f:
        forecast_60d = json.load(f)

    # 6. UN HDX 62-Dataset Linkage Manifest
    with open(os.path.join(DATA_DIR, 'hdx_dataset_linkage_manifest.json'), 'r', encoding='utf-8') as f:
        hdx_manifest = json.load(f)

    # 7. Complete 111-Dataset Master Catalog
    df_cat = pd.read_csv(os.path.join(DATA_DIR, 'bcip_master_catalog.csv'))
    master_catalog = df_cat.to_dict(orient="records")

    # 8. Live Real-Time Feeds: Alerts, Satellite Telemetry, Dispatches
    with open(os.path.join(DATA_DIR, 'bangladesh_live_alerts.json'), 'r', encoding='utf-8') as f:
        live_alerts = json.load(f)
    with open(os.path.join(DATA_DIR, 'bangladesh_satellite_live_telemetry.json'), 'r', encoding='utf-8') as f:
        satellite_telemetry = json.load(f)
    with open(os.path.join(DATA_DIR, 'bangladesh_automated_live_dispatches.json'), 'r', encoding='utf-8') as f:
        live_dispatches = json.load(f)

    # 9. Chattogram Metropolitan Area (CMA) 41 Wards
    df_area = pd.read_csv(os.path.join(DATA_DIR, 'dim_area.csv'))
    df_flood = pd.read_csv(os.path.join(DATA_DIR, 'fact_flood_extent.csv'))
    cma_merged = df_flood[df_flood['EventKey'] == 'EVT_2024_08_MONSOON'].merge(df_area, on='AreaKey')
    if cma_merged.empty:
        cma_merged = df_flood.merge(df_area, on='AreaKey').groupby('WardName').first().reset_index()

    cma_wards = cma_merged[['WardID', 'WardName', 'AreaSqKm', 'MeanElevationM', 'InundatedAreaSqKm', 'InundatedPercentageOfWard', 'MeanInundationDepthM', 'StagnantDurationHours', 'Latitude', 'Longitude']].to_dict(orient="records")

    # 10. Complete Inventory of All Downloadable Data Files
    download_files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            ext = fname.split('.')[-1].upper()
            sz_kb = round(os.path.getsize(fpath) / 1024.0, 1)
            download_files.append({
                "filename": fname,
                "format": ext,
                "size_kb": sz_kb,
                "download_url": f"/data/downloads/{fname}"
            })

    # Consolidate complete master payload
    payload = {
        "metadata": {
            "platform_name": "Bangladesh Climate Risk & Multi-Hazard Intelligence Platform (BCIP)",
            "version": "3.0.0-Production-AllDataConnected",
            "districts_count": len(districts),
            "total_district_features": len(df_m.columns),
            "hdx_datasets_count": len(hdx_manifest),
            "total_catalog_count": len(master_catalog),
            "ai_models_count": len(df_models),
            "feature_importance_count": len(df_fi),
            "cma_wards_count": len(cma_wards),
            "forecast_days": 60,
            "downloadable_files_count": len(download_files),
            "last_updated": "October 2026",
            "citation": "Government of Bangladesh / BMD / BWDB / UN OCHA / WFP / WHO / CEGIS / BCIP Research"
        },
        "districts": districts,
        "cmip6": cmip6_data,
        "ai_models": ai_models_data,
        "future_projections": future_data,
        "forecast_60d": forecast_60d,
        "hdx_manifest": hdx_manifest,
        "master_catalog": master_catalog,
        "live_alerts": live_alerts,
        "satellite_telemetry": satellite_telemetry,
        "live_dispatches": live_dispatches,
        "cma_wards": cma_wards,
        "download_files": download_files
    }

    out_file = os.path.join(STATIC_DATA_DIR, 'platform_data.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"[SUCCESS] Exported Master Connected Payload: {out_file}")
    print(f"          - 64 Districts with all 49 layers + CMIP6 + HDX ({len(df_m.columns)} columns)")
    print(f"          - AI Feature Importances ({len(df_fi)} features ranked)")
    print(f"          - Satellite Telemetry (Himawari, INSAT, Sentinel, 4 radars)")
    print(f"          - {len(master_catalog)} Master datasets & {len(hdx_manifest)} HDX mappings")
    print(f"          - {len(download_files)} Files in /data/downloads/ ready for instant download")
    print("=" * 80)


if __name__ == '__main__':
    build_master_all_data_payload()
