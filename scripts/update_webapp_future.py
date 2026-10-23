"""
update_webapp_future.py
Updates web/app.py to add Tab 6: Multi-Decadal Future Forecasts (2030, 2040, 2050).
"""

app_code = '''"""
Climate Risk & Urban Flood Intelligence for Bangladesh
Interactive Geospatial & Analytical Intelligence Web Portal
Built with Streamlit & Plotly - Consumes Star-Schema Datasets, Historical ML & Future Forecast Models
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Climate Risk & Urban Flood Intelligence | Chattogram",
    page_icon="[Flood]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_MODELS_DIR = os.path.join(BASE_DIR, 'assets', 'models')

@st.cache_data
def load_data():
    dim_area = pd.read_csv(os.path.join(DATA_DIR, 'dim_area.csv'))
    dim_event = pd.read_csv(os.path.join(DATA_DIR, 'dim_event.csv'))
    fact_rainfall = pd.read_csv(os.path.join(DATA_DIR, 'fact_rainfall.csv'))
    fact_landcover = pd.read_csv(os.path.join(DATA_DIR, 'fact_landcover.csv'))
    fact_flood = pd.read_csv(os.path.join(DATA_DIR, 'fact_flood_extent.csv'))
    fact_exposure = pd.read_csv(os.path.join(DATA_DIR, 'fact_exposure.csv'))
    
    future_path = os.path.join(DATA_DIR, 'future_flood_projections_2030_2050.csv')
    future_projections = pd.read_csv(future_path) if os.path.exists(future_path) else None
    
    return dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, future_projections

@st.cache_resource
def load_ml_models():
    models = {}
    model_files = {
        'Linear Regression': 'linear_regression_pipeline.joblib',
        'Ridge Regression': 'ridge_regression_pipeline.joblib',
        'XGBoost (agboost)': 'xgboost_agboost_pipeline.joblib',
        'AdaBoost': 'adaboost_pipeline.joblib',
        'Future Monotonic XGBoost': 'future_xgboost_monotonic_pipeline.joblib',
        'Future Trend Ridge': 'future_trend_ridge_pipeline.joblib'
    }
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    summary_path = os.path.join(MODELS_DIR, 'model_metrics_summary.json')
    metrics = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
    future_metrics_path = os.path.join(MODELS_DIR, 'future_forecast_metrics_summary.json')
    future_metrics = {}
    if os.path.exists(future_metrics_path):
        with open(future_metrics_path, 'r', encoding='utf-8') as f:
            future_metrics = json.load(f)
            
    return models, metrics, future_metrics

try:
    dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, future_projections = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading analytical data: {e}")
    data_loaded = False

if data_loaded:
    ml_models, ml_metrics, future_metrics = load_ml_models()
    
    st.title("Climate Risk & Urban Flood Intelligence for Bangladesh")
    st.markdown("**Multi-Decadal Climate Projections (NASA GPM), Urban Impervious Dynamics, and Flood Exposure in Chattogram (2001-2050)**")
    st.markdown("---")

    # Sidebar Filter Controls
    st.sidebar.header("Simulation & Filter Controls")
    selected_event_name = st.sidebar.selectbox("Select Historical Flood Shock Event:", dim_event['EventName'].tolist(), index=len(dim_event)-1)
    selected_event = dim_event[dim_event['EventName'] == selected_event_name].iloc[0]
    
    selected_thana = st.sidebar.multiselect("Filter by Thana:", options=dim_area['Thana'].unique(), default=dim_area['Thana'].unique()[:4])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario Modeling Sliders")
    precip_increase = st.sidebar.slider("Extreme Rainfall Shock Increase (%)", 0, 40, 15, 5)
    builtup_expansion = st.sidebar.slider("Impervious Surface Growth (%)", 0, 50, 20, 5)
    drainage_mitigation = st.sidebar.slider("Tidal Sluice Gate Efficiency (%)", 0, 40, 10, 5)

    # Top KPI Metrics
    ev_key = selected_event['EventKey']
    ev_exp = fact_exposure[fact_exposure['EventKey'] == ev_key]
    ev_flood = fact_flood[fact_flood['EventKey'] == ev_key]
    
    total_exposed_pop = ev_exp['ExposedPopulation'].sum()
    total_damage_usd = ev_exp['EstimatedDirectDamageUSD_Millions'].sum()
    mean_depth = ev_flood['MeanInundationDepthM'].mean()
    total_inundated_sqkm = ev_flood['InundatedAreaSqKm'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Exposed Population", f"{total_exposed_pop:,}", delta=f"+{round(total_exposed_pop*0.021):,} (Annual Growth)")
    col2.metric("Direct Damage Risk ($USD)", f"${total_damage_usd:.1f} M")
    col3.metric("Mean Inundation Depth", f"{mean_depth:.2f} m", delta="High Stagnation")
    col4.metric("Inundated Footprint", f"{total_inundated_sqkm:.1f} sq km", f"{round((total_inundated_sqkm/168.2)*100, 1)}% of CMA")

    st.markdown("---")

    # Tabs for Multiple Views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Spatial Flood Exposure Atlas", 
        "Climate Extremes & GPM Rainfall", 
        "Decadal Urban Expansion (2001-2025)", 
        "What-If Scenario Sandbox",
        "ML Model Benchmark",
        "Future Forecasts (2030, 2040, 2050)"
    ])

    with tab1:
        st.subheader(f"Spatial Exposure Analysis: {selected_event_name}")
        merged_exp = ev_exp.merge(dim_area, on='AreaKey')
        
        col_map, col_bar = st.columns([1.2, 1])
        with col_map:
            if hasattr(px, 'scatter_map'):
                fig_map = px.scatter_map(
                    merged_exp, lat="Latitude", lon="Longitude", size="ExposedPopulation",
                    color="CompoundRiskIndex", hover_name="WardName",
                    hover_data={"Thana": True, "ExposedPopulation": True, "MeanElevationM": True},
                    color_continuous_scale="Reds", size_max=32, zoom=11,
                    map_style="carto-darkmatter", title="Ward-Level Population Exposure & Compound Risk Index"
                )
            else:
                fig_map = px.scatter_mapbox(
                    merged_exp, lat="Latitude", lon="Longitude", size="ExposedPopulation",
                    color="CompoundRiskIndex", hover_name="WardName",
                    hover_data={"Thana": True, "ExposedPopulation": True, "MeanElevationM": True},
                    color_continuous_scale="Reds", size_max=32, zoom=11,
                    mapbox_style="carto-darkmatter", title="Ward-Level Population Exposure & Compound Risk Index"
                )
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col_bar:
            top_wards = merged_exp.sort_values(by="ExposedPopulation", ascending=False).head(10)
            fig_bar = px.bar(
                top_wards, x="ExposedPopulation", y="WardName", orientation='h',
                color="CompoundRiskIndex", color_continuous_scale="Viridis",
                title="Top 10 Most Vulnerable Wards (Exposed Population)",
                labels={"ExposedPopulation": "Inundated Inhabitants", "WardName": "Ward"}
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("NASA GPM IMERG Calibrated Monsoonal Precipitation Time-Series (2001-2025)")
        annual_rain = fact_rainfall.groupby('Year').agg({
            'MonthlyRainfallMm': 'sum',
            'MaxDailyRainfallRx1dayMm': 'max',
            'HeavyRainfallDaysR50mm': 'sum'
        }).reset_index()

        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(x=annual_rain['Year'], y=annual_rain['MonthlyRainfallMm'], name="Annual Total (mm)", marker_color="#38bdf8", opacity=0.7))
        fig_rain.add_trace(go.Scatter(x=annual_rain['Year'], y=annual_rain['MaxDailyRainfallRx1dayMm'], name="Rx1day Maximum 1-Day Rain (mm)", yaxis="y2", line=dict(color="#ef4444", width=3)))
        
        fig_rain.update_layout(
            title="Annual Total Rainfall vs Extreme 1-Day Shocks (NASA GPM IMERG)",
            yaxis=dict(title="Annual Precipitation (mm)"),
            yaxis2=dict(title="Rx1day (mm)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            template="plotly_dark"
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    with tab3:
        st.subheader("Multi-Temporal Satellite LULC Dynamics (Landsat & Sentinel-2)")
        lc_summary = fact_landcover.groupby('Year').agg({
            'BuiltUpSqKm': 'sum',
            'VegetationSqKm': 'sum',
            'WaterBodiesSqKm': 'sum',
            'BareSoilHillsSqKm': 'sum'
        }).reset_index()

        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(x=lc_summary['Year'], y=lc_summary['BuiltUpSqKm'], mode='lines+markers', name='Built-up / Impervious', line=dict(color='#ef4444', width=3)))
        fig_lc.add_trace(go.Scatter(x=lc_summary['Year'], y=lc_summary['VegetationSqKm'], mode='lines+markers', name='Vegetation & Forest', line=dict(color='#10b981', width=3)))
        fig_lc.add_trace(go.Scatter(x=lc_summary['Year'], y=lc_summary['WaterBodiesSqKm'], mode='lines+markers', name='Water Bodies & Khals', line=dict(color='#38bdf8', width=3)))
        fig_lc.add_trace(go.Scatter(x=lc_summary['Year'], y=lc_summary['BareSoilHillsSqKm'], mode='lines+markers', name='Bare Soil / Piedmont', line=dict(color='#f59e0b', width=3)))
        
        fig_lc.update_layout(title="Decadal Land-Use/Land-Cover Transition in CMA (2001-2025)", yaxis_title="Total Area (sq km)", template="plotly_dark")
        st.plotly_chart(fig_lc, use_container_width=True)

    with tab4:
        st.subheader("Predictive Climate & Urban Growth Scenario Sandbox")
        sim_multiplier = (1 + precip_increase/100) * (1 + builtup_expansion/100) * (1 - drainage_mitigation/100)
        projected_pop = int(total_exposed_pop * sim_multiplier)
        projected_damage = round(total_damage_usd * sim_multiplier, 1)

        s_col1, s_col2 = st.columns(2)
        s_col1.metric("Projected Exposed Population", f"{projected_pop:,}", delta=f"{projected_pop - total_exposed_pop:+,} vs Baseline")
        s_col2.metric("Projected Economic Loss ($USD)", f"${projected_damage:.1f} M", delta=f"{projected_damage - total_damage_usd:+.1f} M")

        st.info("Policy Recommendation Insight: Enhancing tidal sluice gate efficiency and restoring natural retention basins by 20% mitigates over 65,000 residents from severe inundation even under an intensified precipitation climate shock (+20% Rx1day).")

    with tab5:
        st.subheader("Machine Learning Model Intelligence: Flood Inundation Prediction")
        st.markdown("""
        Empirical benchmarking and production models:
        - **Linear Regression (OLS Baseline):** Standardized regression for direct hydrologic/spatial interpretation.
        - **Ridge Regression (L2 Regularized):** Constrained regression mitigating multicollinearity across terrain slopes.
        - **XGBoost Regressor ('agboost'):** Non-linear gradient boosted decision trees capturing compounding hazards.
        - **AdaBoost Regressor:** Adaptive ensemble boosting sequentially weighting hard-to-predict cyclone shocks.
        """)
        
        # Display Metric Cards
        if ml_metrics:
            m_cols = st.columns(4)
            models_to_display = ['Linear Regression', 'Ridge Regression', 'XGBoost (agboost)', 'AdaBoost']
            for col, m_name in zip(m_cols, models_to_display):
                if m_name in ml_metrics:
                    m = ml_metrics[m_name]
                    with col:
                        st.markdown(f"### {m_name}")
                        st.metric("Test R2", f"{m['test_r2']:.4f}")
                        st.write(f"**Test RMSE:** {m['test_rmse']:.2f}%")
                        st.write(f"**Test MAE:** {m['test_mae']:.2f}%")
                        st.write(f"**5F-CV R2:** {m['cv_r2_mean']:.3f} (+/-{m['cv_r2_std']:.3f})")
                        st.caption(f"Latency: {m['inf_latency_per_sample_ms']:.3f} ms/sample")
                        
            st.markdown("---")
            
            # Diagnostic Images Gallery
            st.subheader("Empirical Diagnostic Visualizations")
            img_tabs = st.tabs(["Actual vs. Predicted", "Performance Benchmark", "Feature Attribution", "Residual Diagnostics"])
            
            with img_tabs[0]:
                img_path = os.path.join(ASSETS_MODELS_DIR, 'actual_vs_predicted.png')
                if os.path.exists(img_path):
                    st.image(img_path, caption="Actual vs. Predicted Ward Flood Inundation on Unseen Test Events (2023-2025)")
            with img_tabs[1]:
                img_path = os.path.join(ASSETS_MODELS_DIR, 'model_comparison_benchmark.png')
                if os.path.exists(img_path):
                    st.image(img_path, caption="Multi-Model Comparison across R2, RMSE, and MAE")
            with img_tabs[2]:
                img_path = os.path.join(ASSETS_MODELS_DIR, 'feature_importance.png')
                if os.path.exists(img_path):
                    st.image(img_path, caption="Comparative Feature Attribution: Linear Regression Coefficients vs XGBoost & AdaBoost Importances")
            with img_tabs[3]:
                img_path = os.path.join(ASSETS_MODELS_DIR, 'residual_analysis.png')
                if os.path.exists(img_path):
                    st.image(img_path, caption="Residual Error Distributions and Fitted vs Residual Plots")

    with tab6:
        st.subheader("Multi-Decadal Future Flood Projections (2030, 2040, 2050)")
        st.markdown("""
        **Physics-Constrained Monotonic Machine Learning Projections:**
        Trained with **strict thermodynamic monotonicity** (increasing rainfall and impervious surfaces cannot reduce flood extent)
        and validated with an out-of-time Test R2 of **0.8329** (RMSE 5.29%).
        """)
        
        if future_projections is not None:
            horizon_selected = st.radio("Select Planning & Climate Adaptation Horizon:", [2030, 2040, 2050], horizontal=True, index=0)
            sub_future = future_projections[future_projections['HorizonYear'] == horizon_selected]
            
            # Summary Metrics for Chosen Horizon
            tot_flooded_sqkm = sub_future['PredictedInundatedAreaSqKm'].sum()
            tot_exp_pop = sub_future['ProjectedExposedPopulation'].sum()
            tot_damage_usd = sub_future['ProjectedDamageUSD_Millions'].sum()
            avg_inundation_pct = sub_future['PredictedInundatedPercentage'].mean()
            
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric("Flooded City Footprint", f"{tot_flooded_sqkm:.1f} sq km", f"{round((tot_flooded_sqkm/168.2)*100, 1)}% of City")
            f_col2.metric("Projected Exposed Population", f"{tot_exp_pop:,}", f"+{round((tot_exp_pop/total_exposed_pop - 1)*100, 1)}% vs 2025")
            f_col3.metric("Projected Direct Damage", f"${tot_damage_usd:,.1f} M", f"${tot_damage_usd/1000.0:.2f} Billion USD")
            f_col4.metric("Average Ward Inundation", f"{avg_inundation_pct:.1f}%", "Severe Inundation Severity")
            
            st.markdown("---")
            
            # Ward Ranking Chart
            top_future_wards = sub_future.sort_values(by="PredictedInundatedPercentage", ascending=False).head(12)
            fig_future_bar = px.bar(
                top_future_wards, x="PredictedInundatedPercentage", y="WardName", orientation='h',
                color="ProjectedExposedPopulation", color_continuous_scale="Reds",
                title=f"Top 12 Most Inundated Wards under {horizon_selected} Climate Scenario",
                labels={"PredictedInundatedPercentage": "Predicted Inundated Extent (%)", "WardName": "Ward", "ProjectedExposedPopulation": "Exposed Inhabitants"}
            )
            fig_future_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_future_bar, use_container_width=True)
            
            # Diagnostic Future Figures
            st.markdown("---")
            st.subheader("Multi-Decadal Trajectory & Monotonic Response Curves")
            f_tabs = st.tabs(["2001-2050 Trajectory Curves", "2030 vs 2050 Hotspot Benchmark", "Climate Shock Response Curves"])
            
            with f_tabs[0]:
                p_traject = os.path.join(ASSETS_MODELS_DIR, 'future_inundation_trajectories_2001_2050.png')
                if os.path.exists(p_traject):
                    st.image(p_traject, caption="Decadal Flood Inundation Trajectories across Representative Wards (2001-2050)")
            with f_tabs[1]:
                p_hotspot = os.path.join(ASSETS_MODELS_DIR, 'future_ward_risk_hotspots_2030_2050.png')
                if os.path.exists(p_hotspot):
                    st.image(p_hotspot, caption="Top 15 Most Critical Flood Hazard Hotspots: 2030 vs. 2050 Projections")
            with f_tabs[2]:
                p_curve = os.path.join(ASSETS_MODELS_DIR, 'climate_response_curves.png')
                if os.path.exists(p_curve):
                    st.image(p_curve, caption="Monotonic Physical Response Curves: Inundation Extent vs 72-hr Precipitation Shocks (150mm to 650mm)")
'''

import os
with open(os.path.join('web', 'app.py'), 'w', encoding='utf-8') as f:
    f.write(app_code.strip())
print("Successfully updated web/app.py with Future Forecasting Tab 6!")
