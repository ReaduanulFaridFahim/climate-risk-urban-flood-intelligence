"""
08_train_flood_models.py
Climate Risk & Urban Flood Intelligence: Multi-Model Machine Learning Training & Evaluation Pipeline
Models:
  1. Baseline (Dummy Mean Regressor)
  2. Ordinary Least Squares Linear Regression
  3. Ridge Regression (L2 Regularization)
  4. XGBoost Regressor ('agboost' / Gradient Boosted Trees)
  5. AdaBoost Regressor (Adaptive Boosting)
  6. Random Forest Regressor (Ensemble Benchmark)

Strict featurization ordering, 5-fold cross-validation, out-of-time chronological test split (2023-2025),
and publication-grade residual & feature attribution visualizations.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plotting style
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

def mean_absolute_percentage_error_custom(y_true, y_pred):
    """Calculate MAPE handling division by small values."""
    y_true_safe = np.where(np.abs(y_true) < 1e-4, 1e-4, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100.0

def load_and_preprocess_data():
    print("=" * 80)
    print("STEP 1: RELATIONAL INGESTION & DATASET INTEGRATION")
    print("=" * 80)
    
    dim_area = pd.read_csv(os.path.join(DATA_DIR, 'dim_area.csv'))
    dim_event = pd.read_csv(os.path.join(DATA_DIR, 'dim_event.csv'))
    fact_flood = pd.read_csv(os.path.join(DATA_DIR, 'fact_flood_extent.csv'))
    fact_exp = pd.read_csv(os.path.join(DATA_DIR, 'fact_exposure.csv'))
    fact_lc = pd.read_csv(os.path.join(DATA_DIR, 'fact_landcover.csv'))
    
    print(f" Loaded DimArea: {dim_area.shape} (41 municipal wards)")
    print(f" Loaded DimEvent: {dim_event.shape} (12 historic flood/cyclone shocks)")
    print(f" Loaded FactFloodExtent: {fact_flood.shape} (492 observation records)")
    print(f" Loaded FactExposure: {fact_exp.shape} (492 demographic exposure records)")
    print(f" Loaded FactLandCover: {fact_lc.shape} (246 multi-temporal LULC epochs)")
    
    # Event dates & chronological alignment
    dim_event['EventYear'] = pd.to_datetime(dim_event['StartDate']).dt.year
    lulc_years = np.array([2001, 2005, 2010, 2015, 2020, 2025])
    dim_event['MatchedLCYear'] = dim_event['EventYear'].apply(lambda y: lulc_years[np.abs(lulc_years - y).argmin()])
    
    # Star-schema joins to create primary analytical dataset
    df = fact_flood.merge(
        fact_exp[['EventKey', 'AreaKey', 'TotalWardPopulation', 'ExposedPopulation', 
                  'PopulationExposureRatePct', 'ExposedBuiltUpSqKm', 
                  'EstimatedDirectDamageUSD_Millions', 'CompoundRiskIndex']],
        on=['EventKey', 'AreaKey']
    )
    df = df.merge(
        dim_event[['EventKey', 'EventName', 'StartDate', 'EventYear', 'MatchedLCYear', 
                   'EventType', 'DurationDays', 'SeverityClass', 'PeakRainfallMm']],
        on='EventKey'
    )
    df = df.merge(
        dim_area[['AreaKey', 'WardID', 'WardName', 'Thana', 'AreaSqKm', 
                  'MeanElevationM', 'MaxElevationM', 'MeanSlopeDeg', 
                  'BaselineImperviousRatio', 'DistanceToRiverKm', 
                  'GeomorphicTerrainClass', 'Latitude', 'Longitude']],
        on='AreaKey'
    )
    df = df.merge(
        fact_lc[['Year', 'AreaKey', 'BuiltUpSqKm', 'VegetationSqKm', 
                 'WaterBodiesSqKm', 'BareSoilHillsSqKm', 'ImperviousSurfacePercentage', 
                 'VegetationCoveragePercentage', 'NaturalDrainageLossIndex']],
        left_on=['MatchedLCYear', 'AreaKey'],
        right_on=['Year', 'AreaKey']
    )
    
    # Export unified ML dataset
    unified_csv_path = os.path.join(DATA_DIR, 'flood_ml_dataset.csv')
    df.to_csv(unified_csv_path, index=False)
    print(f" Exported unified ML dataset: {unified_csv_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    
    # Data Quality & Missing Value Audit
    print("\n" + "=" * 80)
    print("STEP 2: DATA SANITY CHECKS & STATISTICAL AUDIT")
    print("=" * 80)
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) == 0:
        print(" Data Quality Status: CLEAN (0 missing/null values detected across all columns).")
    else:
        print(" Detected missing values:")
        print(missing_cols)
        
    print(f" Outlier & Range Inspection for Target ['InundatedPercentageOfWard']:")
    target_stats = df['InundatedPercentageOfWard'].describe()
    print(f"  Count: {target_stats['count']:.0f} | Mean: {target_stats['mean']:.2f}% | Std: {target_stats['std']:.2f}%")
    print(f"  Min: {target_stats['min']:.2f}% | 25%: {target_stats['25%']:.2f}% | 50% (Median): {target_stats['50%']:.2f}% | 75%: {target_stats['75%']:.2f}% | Max: {target_stats['max']:.2f}%")
    
    return df

def train_and_evaluate():
    df = load_and_preprocess_data()
    
    # Define Target and Predictors
    target = 'InundatedPercentageOfWard'
    
    numeric_features = [
        'MeanElevationM',
        'MaxElevationM',
        'MeanSlopeDeg',
        'DistanceToRiverKm',
        'PeakRainfallMm',
        'DurationDays',
        'BuiltUpSqKm',
        'VegetationSqKm',
        'WaterBodiesSqKm',
        'ImperviousSurfacePercentage',
        'VegetationCoveragePercentage',
        'NaturalDrainageLossIndex',
        'AreaSqKm'
    ]
    
    categorical_features = [
        'GeomorphicTerrainClass',
        'Thana',
        'EventType',
        'SeverityClass'
    ]
    
    print("\n" + "=" * 80)
    print("STEP 3: CHRONOLOGICAL TEMPORAL SPLIT (ML BEST PRACTICES)")
    print("=" * 80)
    # Split chronologically to avoid temporal group leakage:
    # Training: Events 1-9 (2004-2022, 9 events, 369 rows = 75%)
    # Out-of-time Test: Events 10-12 (2023-2025, 3 events, 123 rows = 25%)
    train_mask = df['EventKey'] <= 9
    test_mask = df['EventKey'] > 9
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]
    
    print(f" Chronological Split:")
    print(f"  Training Set:   Events 1-9  (2004-2022) -> {len(X_train)} samples ({len(X_train)/len(df):.1%})")
    print(f"  Test Set:       Events 10-12 (2023-2025) -> {len(X_test)} samples ({len(X_test)/len(df):.1%})")
    print(f"  Predictor Count: {len(numeric_features)} Numerical + {len(categorical_features)} Categorical Features")
    
    # Preprocessor definition - strict ordering: fit ONLY on training data!
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    models = {
        'Baseline (Mean)': DummyRegressor(strategy='mean'),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'XGBoost (agboost)': XGBRegressor(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=2
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.08,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            n_jobs=2
        )
    }
    
    results = {}
    fitted_pipelines = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING, 5-FOLD CROSS-VALIDATION & EVALUATION")
    print("=" * 80)
    
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # 5-fold cross-validation on Training Data
        t_start = time.time()
        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=False
        )
        t_cv = time.time() - t_start
        
        # Fit on full training set
        t_fit_start = time.time()
        pipe.fit(X_train, y_train)
        fit_time_ms = (time.time() - t_fit_start) * 1000.0
        
        # Predictions
        y_pred_tr = pipe.predict(X_train)
        
        # Inference latency test on test set
        t_inf_start = time.time()
        y_pred_te = pipe.predict(X_test)
        inf_latency_ms = ((time.time() - t_inf_start) / len(X_test)) * 1000.0
        
        # Metrics Calculation
        r2_tr = r2_score(y_train, y_pred_tr)
        rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
        mae_tr = mean_absolute_error(y_train, y_pred_tr)
        mape_tr = mean_absolute_percentage_error_custom(y_train, y_pred_tr)
        
        r2_te = r2_score(y_test, y_pred_te)
        rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))
        mae_te = mean_absolute_error(y_test, y_pred_te)
        mape_te = mean_absolute_percentage_error_custom(y_test, y_pred_te)
        
        cv_r2_mean = np.mean(cv_res['test_r2'])
        cv_r2_std = np.std(cv_res['test_r2'])
        cv_rmse_mean = -np.mean(cv_res['test_neg_root_mean_squared_error'])
        cv_rmse_std = np.std(cv_res['test_neg_root_mean_squared_error'])
        cv_mae_mean = -np.mean(cv_res['test_neg_mean_absolute_error'])
        
        results[name] = {
            'train_r2': round(float(r2_tr), 4),
            'train_rmse': round(float(rmse_tr), 4),
            'train_mae': round(float(mae_tr), 4),
            'train_mape': round(float(mape_tr), 2),
            'cv_r2_mean': round(float(cv_r2_mean), 4),
            'cv_r2_std': round(float(cv_r2_std), 4),
            'cv_rmse_mean': round(float(cv_rmse_mean), 4),
            'cv_rmse_std': round(float(cv_rmse_std), 4),
            'cv_mae_mean': round(float(cv_mae_mean), 4),
            'test_r2': round(float(r2_te), 4),
            'test_rmse': round(float(rmse_te), 4),
            'test_mae': round(float(mae_te), 4),
            'test_mape': round(float(mape_te), 2),
            'fit_time_ms': round(float(fit_time_ms), 2),
            'inf_latency_per_sample_ms': round(float(inf_latency_ms), 4),
            'y_pred_test': y_pred_te,
            'residuals_test': y_test - y_pred_te
        }
        fitted_pipelines[name] = pipe
        
        # Save pipeline model
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_save_path = os.path.join(MODELS_DIR, f"{safe_name}_pipeline.joblib")
        joblib.dump(pipe, model_save_path)
    
    # Print tabular comparison
    print(f"\n{'Model':<20} | {'Train R2':<8} | {'5F-CV R2 (std)':<16} | {'Test R2':<8} | {'Test RMSE':<9} | {'Test MAE':<8} | {'Test MAPE':<9}")
    print("-" * 92)
    for name, m in results.items():
        cv_str = f"{m['cv_r2_mean']:.3f} (+/-{m['cv_r2_std']:.3f})"
        print(f"{name:<20} | {m['train_r2']:8.4f} | {cv_str:<16} | {m['test_r2']:8.4f} | {m['test_rmse']:9.4f} | {m['test_mae']:8.4f} | {m['test_mape']:8.2f}%")
        
    print("\n" + "=" * 80)
    print("STEP 5: PUBLICATION VISUALIZATIONS & RESIDUAL DIAGNOSTICS")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # FIGURE 1: Actual vs. Predicted (3 Key Models)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    key_models = ['Linear Regression', 'XGBoost (agboost)', 'AdaBoost']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    min_val = min(y_test.min(), min(results[m]['y_pred_test'].min() for m in key_models)) - 5
    max_val = max(y_test.max(), max(results[m]['y_pred_test'].max() for m in key_models)) + 5
    
    for ax, m_name, col in zip(axes, key_models, colors):
        y_pred = results[m_name]['y_pred_test']
        r2 = results[m_name]['test_r2']
        rmse = results[m_name]['test_rmse']
        mae = results[m_name]['test_mae']
        
        ax.scatter(y_test, y_pred, alpha=0.75, color=col, edgecolor='black', s=45, label='Observations (Events 10-12)')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Ideal Fit Line')
        
        ax.set_title(f"{m_name}\n$R^2 = {r2:.4f}$ | RMSE = {rmse:.2f}% | MAE = {mae:.2f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Inundated Percentage (%)', fontsize=11)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    axes[0].set_ylabel('Predicted Inundated Percentage (%)', fontsize=11)
    fig.suptitle('Test Set Generalization: Actual vs. Predicted Ward Flood Inundation (2023-2025 Shocks)', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig1_path = os.path.join(ASSETS_DIR, 'actual_vs_predicted.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # FIGURE 2: Residual Analysis (Distributions & Residuals vs Predicted)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    
    for idx, (m_name, col) in enumerate(zip(key_models, colors)):
        residuals = results[m_name]['residuals_test']
        y_pred = results[m_name]['y_pred_test']
        
        # Residual Histogram / KDE
        ax_top = axes[0, idx]
        sns.histplot(residuals, kde=True, color=col, ax=ax_top, bins=15, edgecolor='black', alpha=0.6)
        ax_top.axvline(0, color='red', linestyle='--', lw=2)
        ax_top.axvline(np.mean(residuals), color='darkblue', linestyle=':', lw=1.5, label=f"Mean: {np.mean(residuals):.2f}%")
        ax_top.set_title(f"{m_name}: Residual Distribution", fontsize=11, fontweight='bold')
        ax_top.set_xlabel('Residual (Actual - Predicted) [%]', fontsize=10)
        ax_top.set_ylabel('Ward Count', fontsize=10)
        ax_top.legend(loc='upper right', fontsize=9)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        
        # Residual vs Predicted (Homoscedasticity test)
        ax_bot = axes[1, idx]
        ax_bot.scatter(y_pred, residuals, alpha=0.75, color=col, edgecolor='black', s=45)
        ax_bot.axhline(0, color='red', linestyle='--', lw=2)
        ax_bot.set_title(f"{m_name}: Residuals vs. Fitted Values", fontsize=11, fontweight='bold')
        ax_bot.set_xlabel('Predicted Inundated Percentage (%)', fontsize=10)
        ax_bot.set_ylabel('Residual (Actual - Predicted) [%]', fontsize=10)
        ax_bot.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle('Residual Diagnostic Analysis & Homoscedasticity Assessment (Test Set)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig2_path = os.path.join(ASSETS_DIR, 'residual_analysis.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # FIGURE 3: Feature Importance Comparison
    # -------------------------------------------------------------
    # Extract feature names after One-Hot Encoding
    ohe = fitted_pipelines['Linear Regression'].named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names
    
    # Extract coefficients / importances
    lr_coefs = fitted_pipelines['Linear Regression'].named_steps['regressor'].coef_
    xgb_importances = fitted_pipelines['XGBoost (agboost)'].named_steps['regressor'].feature_importances_
    ada_importances = fitted_pipelines['AdaBoost'].named_steps['regressor'].feature_importances_
    
    feat_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Linear_Regression_Coef': lr_coefs,
        'Linear_Regression_Abs': np.abs(lr_coefs),
        'XGBoost_Importance': xgb_importances,
        'AdaBoost_Importance': ada_importances
    })
    
    # Pick Top 12 features based on XGBoost & Linear Regression
    top_xgb = feat_df.sort_values(by='XGBoost_Importance', ascending=False).head(10)
    top_ada = feat_df.sort_values(by='AdaBoost_Importance', ascending=False).head(10)
    top_lr = feat_df.sort_values(by='Linear_Regression_Abs', ascending=False).head(10)
    
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    
    # Linear Regression Coefficients
    top_lr_sorted = top_lr.sort_values(by='Linear_Regression_Coef', ascending=True)
    lr_bar_cols = ['#d95f02' if c < 0 else '#1b9e77' for c in top_lr_sorted['Linear_Regression_Coef']]
    axes[0].barh(top_lr_sorted['Feature'], top_lr_sorted['Linear_Regression_Coef'], color=lr_bar_cols, edgecolor='black')
    axes[0].axvline(0, color='black', lw=1)
    axes[0].set_title('Linear Regression:\nTop Standardized Coefficients (Beta)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Coefficient Value', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # XGBoost Feature Importance
    top_xgb_sorted = top_xgb.sort_values(by='XGBoost_Importance', ascending=True)
    axes[1].barh(top_xgb_sorted['Feature'], top_xgb_sorted['XGBoost_Importance'], color='#2ca02c', edgecolor='black')
    axes[1].set_title('XGBoost (agboost):\nTop Feature Gain / Relative Importance', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Relative Importance (Gain)', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # AdaBoost Feature Importance
    top_ada_sorted = top_ada.sort_values(by='AdaBoost_Importance', ascending=True)
    axes[2].barh(top_ada_sorted['Feature'], top_ada_sorted['AdaBoost_Importance'], color='#ff7f0e', edgecolor='black')
    axes[2].set_title('AdaBoost:\nTop Feature Gini Importance', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Gini Importance', fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Model Interpretability & Feature Attribution Across Flood Predictors', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(ASSETS_DIR, 'feature_importance.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig3_path}")
    
    # -------------------------------------------------------------
    # FIGURE 4: Benchmark Comparison Bar Charts
    # -------------------------------------------------------------
    model_names_plot = list(results.keys())
    test_r2_vals = [results[m]['test_r2'] for m in model_names_plot]
    cv_r2_vals = [results[m]['cv_r2_mean'] for m in model_names_plot]
    cv_r2_errs = [results[m]['cv_r2_std'] for m in model_names_plot]
    test_rmse_vals = [results[m]['test_rmse'] for m in model_names_plot]
    test_mae_vals = [results[m]['test_mae'] for m in model_names_plot]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(model_names_plot))
    width = 0.35
    
    # Subplot 1: R2 comparison (CV vs Test)
    axes[0].bar(x - width/2, cv_r2_vals, width, yerr=cv_r2_errs, capsize=4, label='5-Fold CV $R^2$ (Train)', color='#4a7bb0', edgecolor='black')
    axes[0].bar(x + width/2, test_r2_vals, width, label='Unseen Test $R^2$ (2023-2025)', color='#2ca02c', edgecolor='black')
    axes[0].set_title('Goodness of Fit ($R^2$ Score - Higher is Better)', fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names_plot, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('$R^2$ Score', fontsize=10)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].axhline(0, color='red', linestyle='--', lw=1)
    axes[0].legend(loc='lower right', fontsize=8.5)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: RMSE comparison
    axes[1].bar(x, test_rmse_vals, width*1.5, color='#d95f02', edgecolor='black')
    axes[1].set_title('Root Mean Squared Error (RMSE % - Lower is Better)', fontsize=11, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names_plot, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('RMSE (% Inundation)', fontsize=10)
    for i, v in enumerate(test_rmse_vals):
        axes[1].text(i, v + 0.3, f"{v:.2f}%", ha='center', fontsize=8.5, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 3: MAE comparison
    axes[2].bar(x, test_mae_vals, width*1.5, color='#7570b3', edgecolor='black')
    axes[2].set_title('Mean Absolute Error (MAE % - Lower is Better)', fontsize=11, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names_plot, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('MAE (% Inundation)', fontsize=10)
    for i, v in enumerate(test_mae_vals):
        axes[2].text(i, v + 0.25, f"{v:.2f}%", ha='center', fontsize=8.5, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Multi-Model Performance Benchmark: Inundation Extent Prediction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4_path = os.path.join(ASSETS_DIR, 'model_comparison_benchmark.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {fig4_path}")
    
    # -------------------------------------------------------------
    # STEP 6: Save Export Summary JSON
    # -------------------------------------------------------------
    # Strip numpy arrays for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {ik: iv for ik, iv in v.items() if ik not in ['y_pred_test', 'residuals_test']}
        
    summary_json_path = os.path.join(MODELS_DIR, 'model_metrics_summary.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n Saved metrics summary JSON: {summary_json_path}")
    
    print("\n" + "=" * 80)
    print("STEP 7: SUMMARY CONCLUSION & PRODUCTION SELECTION")
    print("=" * 80)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_m = results[best_model_name]
    print(f" Optimal Production Model: {best_model_name}")
    print(f"  Test R2: {best_m['test_r2']:.4f} | Test RMSE: {best_m['test_rmse']:.2f}% | Test MAE: {best_m['test_mae']:.2f}% | Test MAPE: {best_m['test_mape']:.2f}%")
    print(f"  Training Time: {best_m['fit_time_ms']:.1f} ms | Inference Latency: {best_m['inf_latency_per_sample_ms']:.3f} ms/sample")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    train_and_evaluate()
