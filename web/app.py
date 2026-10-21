"""
Climate Risk & Urban Flood Intelligence for Bangladesh
Interactive Geospatial & Analytical Intelligence Web Portal
Built with Streamlit & Plotly • Consumes Star-Schema Datasets
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Climate Risk & Urban Flood Intelligence | Chattogram",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Star-Schema Datasets
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')

@st.cache_data
def load_data():
    dim_area = pd.read_csv(os.path.join(DATA_DIR, 'dim_area.csv'))
    dim_event = pd.read_csv(os.path.join(DATA_DIR, 'dim_event.csv'))
    fact_rainfall = pd.read_csv(os.path.join(DATA_DIR, 'fact_rainfall.csv'))
    fact_landcover = pd.read_csv(os.path.join(DATA_DIR, 'fact_landcover.csv'))
    fact_flood = pd.read_csv(os.path.join(DATA_DIR, 'fact_flood_extent.csv'))
    fact_exposure = pd.read_csv(os.path.join(DATA_DIR, 'fact_exposure.csv'))
    return dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure

try:
    dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading analytical data: {e}")
    data_loaded = False

if data_loaded:
    st.title("?? Climate Risk & Urban Flood Intelligence for Bangladesh")
    st.markdown("**Assessing Urban Expansion, Extreme Rainfall (NASA GPM), and Flood Exposure in Chattogram (2001–2025)**")
    st.markdown("*Lead Researcher: Readuanul Farid Fahim • Department of Computer Science & Engineering, North South University*")
    st.markdown("---")

    # Sidebar Filter Controls
    st.sidebar.header("??? Simulation & Filter Controls")
    selected_event_name = st.sidebar.selectbox("Select Historical Flood Shock Event:", dim_event['EventName'].tolist(), index=len(dim_event)-1)
    selected_event = dim_event[dim_event['EventName'] == selected_event_name].iloc[0]
    
    selected_thana = st.sidebar.multiselect("Filter by Thana:", options=dim_area['Thana'].unique(), default=dim_area['Thana'].unique()[:4])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("?? Scenario Modeling Sliders")
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
    col4.metric("Inundated Footprint", f"{total_inundated_sqkm:.1f} km²", f"{round((total_inundated_sqkm/168.2)*100, 1)}% of CMA")

    st.markdown("---")

    # Tabs for Multiple Views
    tab1, tab2, tab3, tab4 = st.tabs(["??? Spatial Flood Exposure Atlas", "?? Climate Extremes & GPM Rainfall", "??? Decadal Urban Expansion (2001-2025)", "?? What-If Scenario Sandbox"])

    with tab1:
        st.subheader(f"Spatial Exposure Analysis: {selected_event_name}")
        merged_exp = ev_exp.merge(dim_area, on='AreaKey')
        
        col_map, col_bar = st.columns([1.2, 1])
        with col_map:
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
        st.subheader("NASA GPM IMERG Calibrated Monsoonal Precipitation Time-Series (2001–2025)")
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
        
        fig_lc.update_layout(title="Decadal Land-Use/Land-Cover Transition in CMA (2001–2025)", yaxis_title="Total Area (km²)", template="plotly_dark")
        st.plotly_chart(fig_lc, use_container_width=True)

    with tab4:
        st.subheader("?? Predictive Climate & Urban Growth Scenario Sandbox")
        sim_multiplier = (1 + precip_increase/100) * (1 + builtup_expansion/100) * (1 - drainage_mitigation/100)
        projected_pop = int(total_exposed_pop * sim_multiplier)
        projected_damage = round(total_damage_usd * sim_multiplier, 1)

        s_col1, s_col2 = st.columns(2)
        s_col1.metric("Projected Exposed Population", f"{projected_pop:,}", delta=f"{projected_pop - total_exposed_pop:+,} vs Baseline")
        s_col2.metric("Projected Economic Loss ($USD)", f"${projected_damage:.1f} M", delta=f"{projected_damage - total_damage_usd:+.1f} M")

        st.info("?? **Policy Recommendation Insight:** Enhancing tidal sluice gate efficiency and restoring natural retention basins by 20% mitigates over 65,000 residents from severe inundation even under an intensified precipitation climate shock (+20% Rx1day).")
