"""
09_future_flood_forecast_models.py
Climate Risk & Urban Flood Intelligence: Future-Oriented Flood Extent Forecasting Pipeline (2030, 2040, 2050)
Incorporates:
  1. Time-series secular trend extrapolation (ETCCDI precipitation + LULC decadal growth).
  2. Monotonically-constrained XGBoost (enforcing physical thermodynamic and hydrological constraints).
  3. Trend-regularized Ridge Regression.
  4. Multi-decadal IPCC scenario simulations (SSP2-4.5 and SSP5-8.5 for 2030, 2040, 2050).
  5. Ward-level exposure, inundation footprint, and economic damage risk projections.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Matplotlib configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets', 'models')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def engineer_features(df):
    """Derives physics-guided composite features for climate trend extrapolation."""
    d = df.copy()
    d['RainfallShockIntensity'] = d['PeakRainfallMm'] / 400.0
    d['UrbanDrainageCongestionIndex'] = (d['ImperviousSurfacePercentage'] / 100.0) * (1.0 / (d['MeanElevationM'] + 1.0))
    d['TidalExposureFactor'] = (1.0 / (d['DistanceToRiverKm'] + 0.5)) * d['RainfallShockIntensity']
    d['NaturalRetentionBuffer'] = (d['VegetationCoveragePercentage'] / 100.0) * (1.0 - d['NaturalDrainageLossIndex'])
    return d

def train_future_forecasting_models():
    print("=" * 80)
    print("PHASE 1: TRAINING PHYSICS-CONSTRAINED FUTURE FORECASTING MODELS")
    print("=" * 80)
    
    raw_df = pd.read_csv(os.path.join(DATA_DIR, 'flood_ml_dataset.csv'))
    df = engineer_features(raw_df)
    
    feature_cols = [
        'PeakRainfallMm', 'DurationDays', 'ImperviousSurfacePercentage', 
        'RainfallShockIntensity', 'UrbanDrainageCongestionIndex', 'TidalExposureFactor',
        'MeanElevationM', 'MaxElevationM', 'MeanSlopeDeg', 'DistanceToRiverKm', 
        'NaturalRetentionBuffer', 'NaturalDrainageLossIndex', 'AreaSqKm', 'EventYear'
    ]
    target = 'InundatedPercentageOfWard'
    
    # Chronological Split: 2004-2022 (Train) vs 2023-2025 (Test)
    train_mask = df['EventKey'] <= 9
    test_mask = df['EventKey'] > 9
    
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, target]
    
    # Monotonic physical constraints:
    # +1: As rainfall, duration, impervious surface, or year increases -> inundation MUST increase
    # -1: As elevation, slope, river distance, or vegetation buffer increases -> inundation MUST decrease
    monotone_dict = {
        'PeakRainfallMm': 1,
        'DurationDays': 1,
        'ImperviousSurfacePercentage': 1,
        'RainfallShockIntensity': 1,
        'UrbanDrainageCongestionIndex': 1,
        'TidalExposureFactor': 1,
        'MeanElevationM': -1,
        'MaxElevationM': -1,
        'MeanSlopeDeg': -1,
        'DistanceToRiverKm': -1,
        'NaturalRetentionBuffer': -1,
        'NaturalDrainageLossIndex': 1,
        'AreaSqKm': 0,
        'EventYear': 1
    }
    constraints = tuple(monotone_dict[f] for f in feature_cols)
    
    # 1. Monotonic XGBoost
    xgb_mono = XGBRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints=constraints,
        random_state=42
    )
    xgb_mono.fit(X_train, y_train)
    
    # 2. Trend-Regularized Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    
    # Out-of-time evaluation
    pred_xgb = xgb_mono.predict(X_test)
    pred_ridge = ridge.predict(X_test)
    pred_blend = 0.6 * pred_xgb + 0.4 * pred_ridge
    
    r2_xgb = r2_score(y_test, pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    mae_xgb = mean_absolute_error(y_test, pred_xgb)
    
    r2_ridge = r2_score(y_test, pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
    mae_ridge = mean_absolute_error(y_test, pred_ridge)
    
    r2_blend = r2_score(y_test, pred_blend)
    rmse_blend = np.sqrt(mean_squared_error(y_test, pred_blend))
    mae_blend = mean_absolute_error(y_test, pred_blend)
    
    print(f" Model Performance on Unseen Future Test Events (2023-2025 Shocks):")
    print(f"  Monotonic XGBoost : R2 = {r2_xgb:.4f} | RMSE = {rmse_xgb:.2f}% | MAE = {mae_xgb:.2f}%")
    print(f"  Trend Ridge       : R2 = {r2_ridge:.4f} | RMSE = {rmse_ridge:.2f}% | MAE = {mae_ridge:.2f}%")
    print(f"  Physics Ensemble  : R2 = {r2_blend:.4f} | RMSE = {rmse_blend:.2f}% | MAE = {mae_blend:.2f}%")
    
    # Save model pipelines
    joblib.dump(xgb_mono, os.path.join(MODELS_DIR, 'future_xgboost_monotonic_pipeline.joblib'))
    joblib.dump(ridge, os.path.join(MODELS_DIR, 'future_trend_ridge_pipeline.joblib'))
    
    metrics_meta = {
        'features': feature_cols,
        'monotone_constraints': monotone_dict,
        'test_r2_blend': round(float(r2_blend), 4),
        'test_rmse_blend': round(float(rmse_blend), 2),
        'test_mae_blend': round(float(mae_blend), 2)
    }
    with open(os.path.join(MODELS_DIR, 'future_forecast_metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_meta, f, indent=2)
        
    return xgb_mono, ridge, feature_cols

def generate_future_scenarios(xgb_model, ridge_model, feature_cols):
    print("\n" + "=" * 80)
    print("PHASE 2: SIMULATING 2030, 2040, AND 2050 DECADAL FLOOD PROJECTIONS")
    print("=" * 80)
    
    dim_area = pd.read_csv(os.path.join(DATA_DIR, 'dim_area.csv'))
    fact_lc = pd.read_csv(os.path.join(DATA_DIR, 'fact_landcover.csv'))
    
    # Extract baseline 2025 landcover for each ward
    lc_2025 = fact_lc[fact_lc['Year'] == 2025].set_index('AreaKey')
    
    # Scenario definitions aligned with IPCC SSP2-4.5 & SSP5-8.5
    scenarios = [
        {
            'HorizonYear': 2030,
            'ScenarioName': '2030 Near-Term (SSP2-4.5 Moderate)',
            'RainfallShockMm': 435.0, # +15% over historical mean monsoon deluge
            'DurationDays': 6,
            'ImperviousGrowthFactor': 1.10, # +10% built-up expansion
            'VegReductionFactor': 0.90,
            'DrainageLossDelta': 0.08,
            'TidalPenaltyKm': 0.1, # Estuarine backflow penalty
            'PopGrowthMultiplier': (1.0 + 0.021) ** (2030 - 2001)
        },
        {
            'HorizonYear': 2040,
            'ScenarioName': '2040 Mid-Term (SSP2-4.5 Accelerated)',
            'RainfallShockMm': 485.0, # +25% extreme rainfall shock
            'DurationDays': 7,
            'ImperviousGrowthFactor': 1.20, # +20% built-up expansion
            'VegReductionFactor': 0.80,
            'DrainageLossDelta': 0.16,
            'TidalPenaltyKm': 0.25,
            'PopGrowthMultiplier': (1.0 + 0.021) ** (2040 - 2001)
        },
        {
            'HorizonYear': 2050,
            'ScenarioName': '2050 Long-Term (SSP5-8.5 High-Emission Shock)',
            'RainfallShockMm': 545.0, # +40% severe tropical convective shock
            'DurationDays': 8,
            'ImperviousGrowthFactor': 1.30, # +30% built-up expansion
            'VegReductionFactor': 0.70,
            'DrainageLossDelta': 0.25,
            'TidalPenaltyKm': 0.45,
            'PopGrowthMultiplier': (1.0 + 0.021) ** (2050 - 2001)
        }
    ]
    
    projection_rows = []
    
    for sc in scenarios:
        h_year = sc['HorizonYear']
        s_name = sc['ScenarioName']
        rain_shock = sc['RainfallShockMm']
        duration = sc['DurationDays']
        
        for _, ward in dim_area.iterrows():
            area_key = int(ward['AreaKey'])
            ward_area = float(ward['AreaSqKm'])
            elev = float(ward['MeanElevationM'])
            dist_river = max(0.1, float(ward['DistanceToRiverKm']) - sc['TidalPenaltyKm'])
            
            # 2025 base landcover
            base_lc = lc_2025.loc[area_key]
            base_imp = float(base_lc['ImperviousSurfacePercentage'])
            base_veg = float(base_lc['VegetationCoveragePercentage'])
            base_drain_loss = float(base_lc['NaturalDrainageLossIndex'])
            
            # Projected LULC
            proj_imp = min(96.0, base_imp * sc['ImperviousGrowthFactor'])
            proj_veg = max(2.0, base_veg * sc['VegReductionFactor'])
            proj_drain_loss = min(1.0, base_drain_loss + sc['DrainageLossDelta'])
            
            # Composite features
            rain_intensity = rain_shock / 400.0
            drain_congestion = (proj_imp / 100.0) * (1.0 / (elev + 1.0))
            tidal_factor = (1.0 / (dist_river + 0.5)) * rain_intensity
            retention_buffer = (proj_veg / 100.0) * (1.0 - proj_drain_loss)
            
            input_vector = pd.DataFrame([{
                'PeakRainfallMm': rain_shock,
                'DurationDays': duration,
                'ImperviousSurfacePercentage': proj_imp,
                'RainfallShockIntensity': rain_intensity,
                'UrbanDrainageCongestionIndex': drain_congestion,
                'TidalExposureFactor': tidal_factor,
                'MeanElevationM': elev,
                'MaxElevationM': float(ward['MaxElevationM']),
                'MeanSlopeDeg': float(ward['MeanSlopeDeg']),
                'DistanceToRiverKm': dist_river,
                'NaturalRetentionBuffer': retention_buffer,
                'NaturalDrainageLossIndex': proj_drain_loss,
                'AreaSqKm': ward_area,
                'EventYear': h_year
            }])[feature_cols]
            
            # Predict using ensemble
            p_xgb = float(xgb_model.predict(input_vector)[0])
            p_ridge = float(ridge_model.predict(input_vector)[0])
            pred_inundated_pct = round(max(5.0, min(98.5, 0.6 * p_xgb + 0.4 * p_ridge)), 2)
            
            inundated_sqkm = round((pred_inundated_pct / 100.0) * ward_area, 3)
            
            # Population and asset exposure
            base_pop_2001 = 55000 * (float(ward['BaselineImperviousRatio']) * 1.5)
            proj_tot_pop = int(base_pop_2001 * sc['PopGrowthMultiplier'])
            density_weight = 1.30 if elev < 6.0 else 0.85
            proj_exp_pop = int(proj_tot_pop * min(0.96, (pred_inundated_pct / 100.0) * density_weight))
            
            val_per_sqkm = 22.0 if 'Commercial' in str(ward['GeomorphicTerrainClass']) else 7.5
            est_damage_usd = round(inundated_sqkm * val_per_sqkm * 1.25, 2)
            
            projection_rows.append({
                'HorizonYear': h_year,
                'ScenarioName': s_name,
                'AreaKey': area_key,
                'WardID': ward['WardID'],
                'WardName': ward['WardName'],
                'Thana': ward['Thana'],
                'AreaSqKm': ward_area,
                'MeanElevationM': elev,
                'DistanceToRiverKm': round(dist_river, 2),
                'GeomorphicTerrainClass': ward['GeomorphicTerrainClass'],
                'ProjectedPeakRainfallMm': rain_shock,
                'ProjectedImperviousPct': round(proj_imp, 2),
                'PredictedInundatedPercentage': pred_inundated_pct,
                'PredictedInundatedAreaSqKm': inundated_sqkm,
                'ProjectedTotalPopulation': proj_tot_pop,
                'ProjectedExposedPopulation': proj_exp_pop,
                'ProjectedDamageUSD_Millions': est_damage_usd
            })
            
    proj_df = pd.DataFrame(projection_rows)
    out_csv = os.path.join(DATA_DIR, 'future_flood_projections_2030_2050.csv')
    proj_df.to_csv(out_csv, index=False)
    print(f" Saved multi-decadal ward projections: {out_csv} ({len(proj_df)} records)")
    
    # -------------------------------------------------------------
    # PUBLICATION FIGURE 1: Future Inundation Trajectories (2001-2050)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: GENERATING FUTURE CLIMATE TRAJECTORY FIGURES")
    print("=" * 80)
    
    # Select sample representative wards
    focus_wards = ['North Agrabad', 'West Bakolia', 'Chandgaon', 'Panchlaish', 'South Pahartali']
    ward_colors = {'North Agrabad': '#d95f02', 'West Bakolia': '#e41a1c', 'Chandgaon': '#377eb8', 'Panchlaish': '#4daf4a', 'South Pahartali': '#984ea3'}
    
    # Historical observations
    hist_df = pd.read_csv(os.path.join(DATA_DIR, 'flood_ml_dataset.csv'))
    hist_agg = hist_df.groupby(['EventYear', 'WardName'])['InundatedPercentageOfWard'].mean().reset_index()
    
    future_agg = proj_df.groupby(['HorizonYear', 'WardName'])['PredictedInundatedPercentage'].mean().reset_index()
    future_agg.rename(columns={'HorizonYear': 'EventYear', 'PredictedInundatedPercentage': 'InundatedPercentageOfWard'}, inplace=True)
    
    fig, ax = plt.subplots(figsize=(13, 6.5))
    
    for w in focus_wards:
        h_sub = hist_agg[hist_agg['WardName'] == w].sort_values(by='EventYear')
        f_sub = future_agg[future_agg['WardName'] == w].sort_values(by='EventYear')
        combined = pd.concat([h_sub, f_sub]).sort_values(by='EventYear')
        
        ax.plot(h_sub['EventYear'], h_sub['InundatedPercentageOfWard'], 'o-', color=ward_colors[w], lw=2.2, label=f"{w} (Historical)")
        ax.plot(combined['EventYear'], combined['InundatedPercentageOfWard'], '--', color=ward_colors[w], lw=2.2, alpha=0.7)
        ax.plot(f_sub['EventYear'], f_sub['InundatedPercentageOfWard'], 's', color=ward_colors[w], markersize=8, label=f"{w} (Projected 2030-2050)")
        
    ax.axvline(2025, color='black', linestyle=':', lw=2, label='Forecast Boundary (2025)')
    ax.set_title('Decadal Flood Inundation Trajectories across Representative Wards (2001–2050)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Timeline Horizon (Year)', fontsize=11)
    ax.set_ylabel('Inundated Extent of Ward (%)', fontsize=11)
    ax.set_xlim(2000, 2052)
    ax.set_ylim(20, 102)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig1_path = os.path.join(ASSETS_DIR, 'future_inundation_trajectories_2001_2050.png')
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # PUBLICATION FIGURE 2: 2030 vs 2050 Ward Risk Hotspots
    # -------------------------------------------------------------
    p30 = proj_df[proj_df['HorizonYear'] == 2030].set_index('WardName')
    p50 = proj_df[proj_df['HorizonYear'] == 2050].set_index('WardName')
    
    top_risk_wards = p50.sort_values(by='PredictedInundatedPercentage', ascending=False).head(15).index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(top_risk_wards))
    height = 0.38
    
    vals_30 = [p30.loc[w, 'PredictedInundatedPercentage'] for w in top_risk_wards]
    vals_50 = [p50.loc[w, 'PredictedInundatedPercentage'] for w in top_risk_wards]
    
    ax.barh(y + height/2, vals_30, height, label='2030 Near-Term (SSP2-4.5: +15% Rain)', color='#38bdf8', edgecolor='black')
    ax.barh(y - height/2, vals_50, height, label='2050 Severe Crisis (SSP5-8.5: +40% Rain)', color='#ef4444', edgecolor='black')
    
    ax.set_yticks(y)
    ax.set_yticklabels(top_risk_wards, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Predicted Ward Inundation Extent (%)', fontsize=11)
    ax.set_title('Top 15 Most Critical Flood Hazard Hotspots: 2030 vs. 2050 Projections', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    for i, (v30, v50) in enumerate(zip(vals_30, vals_50)):
        ax.text(v30 + 1, i + height/2, f"{v30:.1f}%", va='center', fontsize=8.5)
        ax.text(v50 + 1, i - height/2, f"{v50:.1f}%", va='center', fontsize=8.5, fontweight='bold')
        
    ax.set_xlim(0, 110)
    fig2_path = os.path.join(ASSETS_DIR, 'future_ward_risk_hotspots_2030_2050.png')
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # PUBLICATION FIGURE 3: Monotonic Climate Response Curves
    # -------------------------------------------------------------
    # Simulate response curve from 150mm to 650mm extreme rainfall
    test_rains = np.linspace(150, 650, 60)
    sample_lowland = dim_area[dim_area['WardName'] == 'North Agrabad'].iloc[0]
    sample_piedmont = dim_area[dim_area['WardName'] == 'South Pahartali'].iloc[0]
    
    def simulate_curve(ward_meta, imp_pct, yr):
        res = []
        for r in test_rains:
            vec = pd.DataFrame([{
                'PeakRainfallMm': r,
                'DurationDays': 6,
                'ImperviousSurfacePercentage': imp_pct,
                'RainfallShockIntensity': r / 400.0,
                'UrbanDrainageCongestionIndex': (imp_pct/100.0) * (1.0 / (float(ward_meta['MeanElevationM']) + 1.0)),
                'TidalExposureFactor': (1.0 / (float(ward_meta['DistanceToRiverKm']) + 0.5)) * (r / 400.0),
                'MeanElevationM': float(ward_meta['MeanElevationM']),
                'MaxElevationM': float(ward_meta['MaxElevationM']),
                'MeanSlopeDeg': float(ward_meta['MeanSlopeDeg']),
                'DistanceToRiverKm': float(ward_meta['DistanceToRiverKm']),
                'NaturalRetentionBuffer': 0.05,
                'NaturalDrainageLossIndex': 0.20,
                'AreaSqKm': float(ward_meta['AreaSqKm']),
                'EventYear': yr
            }])[feature_cols]
            px = float(xgb_model.predict(vec)[0])
            pr = float(ridge_model.predict(vec)[0])
            res.append(max(0.0, min(100.0, 0.6 * px + 0.4 * pr)))
        return res

    curve_agrabad_2025 = simulate_curve(sample_lowland, 91.0, 2025)
    curve_agrabad_2050 = simulate_curve(sample_lowland, 96.0, 2050)
    curve_pahartali_2025 = simulate_curve(sample_piedmont, 44.0, 2025)
    curve_pahartali_2050 = simulate_curve(sample_piedmont, 58.0, 2050)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(test_rains, curve_agrabad_2025, '-', color='#ef4444', lw=2.5, label='North Agrabad Lowland (2025 Baseline)')
    ax.plot(test_rains, curve_agrabad_2050, '--', color='#b91c1c', lw=2.5, label='North Agrabad Lowland (2050 Growth + Subsidence)')
    ax.plot(test_rains, curve_pahartali_2025, '-', color='#3b82f6', lw=2.5, label='South Pahartali Ridge (2025 Baseline)')
    ax.plot(test_rains, curve_pahartali_2050, '--', color='#1d4ed8', lw=2.5, label='South Pahartali Ridge (2050 Growth)')
    
    ax.axvspan(450, 650, color='red', alpha=0.08, label='Unprecedented Future Climate Range (SSP5-8.5)')
    ax.set_title('Monotonic Physical Response Curves: Inundation Extent vs. Precipitation Shocks', fontsize=13, fontweight='bold')
    ax.set_xlabel('72-Hour Cumulative Extreme Precipitation (mm)', fontsize=11)
    ax.set_ylabel('Projected Ward Inundation Extent (%)', fontsize=11)
    ax.set_xlim(150, 650)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=9.5, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig3_path = os.path.join(ASSETS_DIR, 'climate_response_curves.png')
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig3_path}")
    
    print("\n" + "=" * 80)
    print("PHASE 4: SUMMARY COMPARISON (CITY-WIDE IMPACT)")
    print("=" * 80)
    city_summary = proj_df.groupby('HorizonYear').agg({
        'PredictedInundatedAreaSqKm': 'sum',
        'ProjectedExposedPopulation': 'sum',
        'ProjectedDamageUSD_Millions': 'sum',
        'PredictedInundatedPercentage': 'mean'
    }).reset_index()
    
    print(city_summary.to_string(index=False))
    print("=" * 80)
    return proj_df

if __name__ == '__main__':
    xgb_m, ridge_m, f_cols = train_future_forecasting_models()
    generate_future_scenarios(xgb_m, ridge_m, f_cols)
