"""
========================================================================================
update_webapp_bcip_suite.py
Injects the comprehensive Bangladesh Climate Intelligence Platform (BCIP) modules
into web/app.py:
  1. Data loader for BCIP datasets (catalog, transitions, matrices, models, correlations)
  2. Tab 1: 🌐 BCIP National Climate Transition Atlas (4 CMIP6 scenarios, 4x4 matrices)
  3. Tab 2: 🔍 BCIP Multi-Dimensional Exposure Explorer (Population, Built-up, Rice, Health)
  4. Tab 3: ⚖️ BCIP Adaptation-Priority Screening & Robustness Suite (Interactive Weights)
  5. Enhanced Master Catalogue with Analytical Roles (Core, Exposure, Context, Excluded)
Author: Readuanul Farid Fahim | Department of Computer Science & Engineering, North South University
========================================================================================
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(BASE_DIR, 'web', 'app.py')

with open(APP_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# -----------------------------------------------------------------------------
# 1. ADD BCIP DATA LOADER
# -----------------------------------------------------------------------------
bcip_loader_code = '''
@st.cache_data
def load_bcip_intelligence_data():
    """Loads all BCIP scientific transition matrices, rankings, and catalog."""
    cat_path = os.path.join(DATA_DIR, 'bcip_master_catalog.csv')
    df_cat = pd.read_csv(cat_path) if os.path.exists(cat_path) else None

    dist_path = os.path.join(DATA_DIR, 'bcip_district_climate_transitions_and_exposure.csv')
    df_dist = pd.read_csv(dist_path) if os.path.exists(dist_path) else None

    rank_path = os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv')
    df_rank = pd.read_csv(rank_path) if os.path.exists(rank_path) else None

    nat_sum_path = os.path.join(DATA_DIR, 'bcip_climate_transitions_national_summary.csv')
    df_nat_sum = pd.read_csv(nat_sum_path) if os.path.exists(nat_sum_path) else None

    agree_path = os.path.join(DATA_DIR, 'bcip_scenario_agreement_analysis.csv')
    df_agree = pd.read_csv(agree_path) if os.path.exists(agree_path) else None

    mat_path = os.path.join(DATA_DIR, 'bcip_climate_transition_matrices.json')
    mat_dict = {}
    if os.path.exists(mat_path):
        with open(mat_path, 'r', encoding='utf-8') as f:
            mat_dict = json.load(f)

    corr_path = os.path.join(DATA_DIR, 'bcip_spearman_rank_correlations.json')
    corr_dict = {}
    if os.path.exists(corr_path):
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_dict = json.load(f)

    return df_cat, df_dist, df_rank, df_nat_sum, df_agree, mat_dict, corr_dict
'''

if 'def load_bcip_intelligence_data():' not in content:
    # Insert right before load_ml_models()
    content = content.replace('@st.cache_resource\ndef load_ml_models():', bcip_loader_code + '\n\n@st.cache_resource\ndef load_ml_models():')
    print("[+] Injected load_bcip_intelligence_data() into web/app.py")

# -----------------------------------------------------------------------------
# 2. UPDATE MAIN DATA UNPACKING
# -----------------------------------------------------------------------------
data_unpack_target = "dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111, df_49layers = load_data()"
bcip_unpack_code = data_unpack_target + "\n    df_bcip_cat, df_bcip_dist, df_bcip_rank, df_bcip_sum, df_bcip_agree, bcip_matrices, bcip_corrs = load_bcip_intelligence_data()"

if 'df_bcip_cat, df_bcip_dist' not in content:
    content = content.replace(data_unpack_target, bcip_unpack_code)
    print("[+] Injected BCIP data unpacking into main()")

# -----------------------------------------------------------------------------
# 3. EXPAND MASTER TABS TO INCLUDE BCIP PREMIER TABS
# -----------------------------------------------------------------------------
old_tabs = '''    tab_nat, tab_phone, tab_live, tab_20d, tab_atlas, tab_hdx, tab_multi_ai, tab_future, tab_ml_eval, tab_cma = st.tabs([
        "🇧🇩 Nationwide Flood Susceptibility (64 Districts)",
        "📱 Emergency Phone Alerts & Auto-Sentinel",
        "⚡ Real Live Weather Satellite & Radar Map",
        "📅 Advance 20-Day Weather & Hazard Outlook (ECMWF)",
        "🌍 Geospatial Environmental Atlas (49 Layers)",
        "📦 UN HDX Humanitarian Data Hub (62 Indicators)",
        "☀️❄️ Multi-Hazard AI: Heatwaves & Winter Cold Waves",
        "🔮 All-Bangladesh Future Forecasts (2030, 2040, 2050)",
        "🤖 Advanced AI Model Benchmark (Neural Net & Stacking)",
        "🏙️ Chattogram Ward Analysis"
    ])'''

new_tabs = '''    tab_bcip_atlas, tab_bcip_exp, tab_bcip_prior, tab_nat, tab_phone, tab_live, tab_20d, tab_atlas, tab_hdx, tab_multi_ai, tab_future, tab_ml_eval, tab_cma = st.tabs([
        "🌐 BCIP Climate Transition Atlas (CMIP6)",
        "🔍 BCIP Multi-Dimensional Exposure Explorer",
        "⚖️ BCIP Prioritization & Robustness Models",
        "🇧🇩 Nationwide Flood Susceptibility (64 Districts)",
        "📱 Emergency Phone Alerts & Auto-Sentinel",
        "⚡ Real Live Weather Satellite & Radar Map",
        "📅 Advance 20-Day Weather & Hazard Outlook (ECMWF)",
        "🌍 Geospatial Environmental Atlas (49 Layers)",
        "📦 UN HDX Humanitarian Data Hub (62 Indicators)",
        "☀️❄️ Multi-Hazard AI: Heatwaves & Winter Cold Waves",
        "🔮 All-Bangladesh Future Forecasts (2030, 2040, 2050)",
        "🤖 Enterprise AI Model Benchmark (10 Models)",
        "🏙️ Chattogram Ward Analysis"
    ])'''

if old_tabs in content:
    content = content.replace(old_tabs, new_tabs)
    print("[+] Expanded tab headers to include BCIP Premier Modules")

# -----------------------------------------------------------------------------
# 4. IMPLEMENT TAB BCIP ATLAS, EXPOSURE, AND PRIORITIZATION CONTENT
# -----------------------------------------------------------------------------
bcip_tab_body = r'''    # -------------------------------------------------------------
    # BCIP TAB 1: NATIONAL CLIMATE TRANSITION ATLAS (CMIP6)
    # -------------------------------------------------------------
    with tab_bcip_atlas:
        st.subheader("🌐 Bangladesh Climate Intelligence Platform (BCIP): Climate Transition Atlas")
        st.markdown("""
        **Peer-Reviewed Research Framework for CMIP6 Macro-Climatic Regimes & Biome Shifts:**
        Integrates Beck et al. (2023) 1 km constrained Köppen-Geiger climate classifications to quantify categorical
        macro-climatic regime shifts across all 64 districts between the **1991–2020 Baseline** and four future CMIP6 pathways.
        """)

        if df_bcip_dist is not None and df_bcip_sum is not None:
            # Scenario Selector
            scen_options = [
                ("Comparison_A_SSP245_2050", "Comparison A: Baseline (1991–2020) vs SSP2-4.5 (2041–2070) [Mid-Century Moderate]"),
                ("Comparison_B_SSP585_2050", "Comparison B: Baseline (1991–2020) vs SSP5-8.5 (2041–2070) [Mid-Century High Emissions]"),
                ("Comparison_C_SSP245_2090", "Comparison C: Baseline (1991–2020) vs SSP2-4.5 (2071–2099) [Late-Century Moderate]"),
                ("Comparison_D_SSP585_2090", "Comparison D: Baseline (1991–2020) vs SSP5-8.5 (2071–2099) [Late-Century High Emissions]")
            ]
            scen_dict = {title: key for key, title in scen_options}
            chosen_title = st.selectbox("🎯 Select Climate Comparison Pathway:", list(scen_dict.keys()), index=1)
            chosen_key = scen_dict[chosen_title]

            # Metrics row
            sum_row = df_bcip_sum[df_bcip_sum['ComparisonID'] == chosen_key].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Land Area Changing Class", f"{sum_row['TotalAreaChangedKm2']:,.1f} km²", f"{sum_row['TotalAreaChangedPct']}% of Bangladesh")
            m2.metric("Tropical Monsoon (Am) Shift", f"{sum_row['Am_NetChangeKm2']:+,.1f} km²", "Net Biome Gain")
            m3.metric("Temperate Highland (Cwa) Shift", f"{sum_row['Cwa_NetChangeKm2']:+,.1f} km²", "Net Biome Loss")
            m4.metric("Stable Baseline Area", f"{sum_row['TotalAreaStableKm2']:,.1f} km²", f"{sum_row['TotalAreaStablePct']}% Stable")

            st.markdown("---")
            map_c, mat_c = st.columns([1.2, 1.0])

            # Determine column names based on comparison
            if 'SSP245_2050' in chosen_key:
                fut_col = 'Koppen_SSP245_2050'
            elif 'SSP585_2050' in chosen_key:
                fut_col = 'Koppen_SSP585_2050'
            elif 'SSP245_2090' in chosen_key:
                fut_col = 'Koppen_SSP245_2090'
            else:
                fut_col = 'Koppen_SSP585_2090'

            with map_c:
                st.markdown(f"##### 🗺️ Spatial Distribution: Projected {fut_col.replace('Koppen_', '')} Regimes")
                fig_map = px.scatter_map(
                    df_bcip_dist,
                    lat='Latitude',
                    lon='Longitude',
                    color=fut_col,
                    size='Population',
                    hover_name='DistrictName',
                    hover_data={
                        'Division': True,
                        'Koppen_Baseline': True,
                        fut_col: True,
                        'Population': True,
                        'AreaSqKm': True,
                        'Latitude': False,
                        'Longitude': False
                    },
                    color_discrete_map={
                        'Am (Tropical Monsoon)': '#2563eb',
                        'Aw (Tropical Savannah)': '#38bdf8',
                        'Cwa (Temperate, Dry Winter, Hot Summer)': '#22c55e',
                        'Cwb (Temperate, Dry Winter, Warm Summer)': '#15803d'
                    },
                    zoom=5.8,
                    center={"lat": 23.6850, "lon": 90.3563},
                    map_style='carto-positron',
                    title=f"64-District Climate Regimes under {chosen_title.split(':')[1].strip()}"
                )
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=520)
                st.plotly_chart(fig_map, use_container_width=True)

            with mat_c:
                st.markdown("##### 📊 4×4 Categorical Transition Matrix (km² & %)")
                if chosen_key in bcip_matrices:
                    m_data = bcip_matrices[chosen_key]["Results"]
                    m_pct = m_data["MatrixPct"]
                    m_km2 = m_data["MatrixKm2"]
                    
                    rows_disp = []
                    classes = ["Am (Tropical Monsoon)", "Aw (Tropical Savannah)", "Cwa (Temperate, Dry Winter, Hot Summer)", "Cwb (Temperate, Dry Winter, Warm Summer)"]
                    short_names = ["Am", "Aw", "Cwa", "Cwb"]
                    
                    for c_from in classes:
                        row_dict = {"Baseline Class": c_from.split('(')[0].strip()}
                        for c_to in classes:
                            to_short = c_to.split('(')[0].strip()
                            v_km2 = m_km2[c_from][c_to]
                            v_pct = m_pct[c_from][c_to]
                            if v_km2 > 0:
                                row_dict[to_short] = f"{v_km2:,.0f} km² ({v_pct:.1f}%)"
                            else:
                                row_dict[to_short] = "-"
                        rows_disp.append(row_dict)
                        
                    st.dataframe(pd.DataFrame(rows_disp).set_index("Baseline Class"), use_container_width=True)

                st.markdown("##### 🤝 Scenario Agreement & Disagreement Breakdown")
                if df_bcip_agree is not None:
                    if '2050' in chosen_key:
                        agr_counts = df_bcip_agree['MidCentury_AgreementStatus'].value_counts()
                    else:
                        agr_counts = df_bcip_agree['LateCentury_AgreementStatus'].value_counts()
                    
                    for status, cnt in agr_counts.items():
                        st.markdown(f"- **{status}**: `{cnt} districts` ({cnt/64*100:.1f}%)")

            # Publication figures preview expander
            with st.expander("🔬 View 300 DPI Publication Transition Matrix & Agreement Figures"):
                f1_p = os.path.join(BASE_DIR, 'assets', 'bcip', 'bcip_fig1_climate_transition_matrices.png')
                f2_p = os.path.join(BASE_DIR, 'assets', 'bcip', 'bcip_fig2_scenario_agreement_and_divergence.png')
                if os.path.exists(f1_p):
                    st.image(f1_p, caption="Figure 1: Full 4-Panel Categorical Transition Matrices across CMIP6 Scenarios (DPI=300)")
                if os.path.exists(f2_p):
                    st.image(f2_p, caption="Figure 2: Scenario Agreement and Spatial Divergence Analysis (DPI=300)")
        else:
            st.info("Loading BCIP Climate Transition data...")

    # -------------------------------------------------------------
    # BCIP TAB 2: MULTI-DIMENSIONAL EXPOSURE EXPLORER
    # -------------------------------------------------------------
    with tab_bcip_exp:
        st.subheader("🔍 BCIP Multi-Dimensional Exposure & Context Explorer")
        st.markdown(r"""
        **Evaluating Human Presence, Built Structures, Agrarian Phenology & Service Access:**
        Quantifies how shifting climate zones intersect real-world socio-economic systems:
        $$E_z = \sum_{i \in z} P_i \cdot I_i$$
        """)

        if df_bcip_dist is not None:
            e_metric = st.selectbox(
                "🎯 Select Exposure / Context Dimension to Inspect:",
                [
                    "Exposed Population in Transition Zones (SSP5-8.5)",
                    "Built-Up Physical Asset Surface (km²)",
                    "Agricultural Cropland Area (km²)",
                    "Seasonal Rice Phenology: Boro Season (Jan–Apr NDVI)",
                    "Seasonal Rice Phenology: Aman Season (Aug–Nov NDVI)",
                    "Straight-Line Distance to Nearest Health Facility (km)",
                    "Road Network Evacuation Density (km/km²)"
                ]
            )

            metric_col_map = {
                "Exposed Population in Transition Zones (SSP5-8.5)": "ExposedPop_MidSSP585",
                "Built-Up Physical Asset Surface (km²)": "BuiltUpAreaSqKm",
                "Agricultural Cropland Area (km²)": "CroplandAreaSqKm",
                "Seasonal Rice Phenology: Boro Season (Jan–Apr NDVI)": "Rice_NDVI_Boro",
                "Seasonal Rice Phenology: Aman Season (Aug–Nov NDVI)": "Rice_NDVI_Aman",
                "Straight-Line Distance to Nearest Health Facility (km)": "DistanceToHealthFacility_Km",
                "Road Network Evacuation Density (km/km²)": "RoadDensityKmPerSqKm"
            }

            active_exp_col = metric_col_map[e_metric]

            ec1, ec2 = st.columns([1.2, 1.0])
            with ec1:
                st.markdown(f"##### 🗺️ Geographic Distribution: {e_metric}")
                fig_exp_map = px.scatter_map(
                    df_bcip_dist,
                    lat='Latitude',
                    lon='Longitude',
                    color=active_exp_col,
                    size='Population' if 'NDVI' in active_exp_col or 'Distance' in active_exp_col else active_exp_col,
                    hover_name='DistrictName',
                    hover_data={
                        'Division': True,
                        active_exp_col: True,
                        'Population': True,
                        'AreaSqKm': True,
                        'Latitude': False,
                        'Longitude': False
                    },
                    color_continuous_scale='Turbo' if 'NDVI' in active_exp_col else 'Viridis',
                    zoom=5.8,
                    center={"lat": 23.6850, "lon": 90.3563},
                    map_style='carto-positron'
                )
                fig_exp_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=500)
                st.plotly_chart(fig_exp_map, use_container_width=True)

            with ec2:
                st.markdown(f"##### 📊 Top 15 Districts by {e_metric.split('(')[0]}")
                top15_exp = df_bcip_dist.sort_values(active_exp_col, ascending=False).head(15)
                fig_exp_bar = px.bar(
                    top15_exp,
                    x=active_exp_col,
                    y='DistrictName',
                    orientation='h',
                    color='Division',
                    title=f"Highest Ranking Districts for {active_exp_col}"
                )
                fig_exp_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin={"r":10,"t":30,"l":10,"b":0}, height=500)
                st.plotly_chart(fig_exp_bar, use_container_width=True)

            # Lifeline Accessibility Matrix
            st.markdown("---")
            st.markdown("##### 🏥 Emergency Lifeline Accessibility & Health Equity Indicators")
            st.dataframe(
                df_bcip_dist[['DistrictID', 'DistrictName', 'Division', 'Population', 'DistanceToHealthFacility_Km', 'HealthFacilitiesPer100k', 'EducationSheltersPer100k', 'RoadDensityKmPerSqKm']].sort_values('DistanceToHealthFacility_Km', ascending=False),
                use_container_width=True
            )
        else:
            st.info("Loading BCIP Multi-Dimensional Exposure data...")

    # -------------------------------------------------------------
    # BCIP TAB 3: PRIORITIZATION & ROBUSTNESS MODELS
    # -------------------------------------------------------------
    with tab_bcip_prior:
        st.subheader("⚖️ BCIP Multi-Model Adaptation Prioritization & Robustness Suite")
        st.markdown("""
        **Evaluating Incremental Value & Multi-Criteria Weight Sensitivity (Research Questions 6 & 7):**
        Tests whether adding climate transition dynamics changes prioritization compared to simpler baselines.
        """)

        if df_bcip_rank is not None:
            # Interactive Weights Sliders
            st.markdown("##### 🎛️ Interactive Multi-Criteria Weight Configuration")
            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1:
                w_c = st.slider("Climate Transition (wC)", 0.0, 1.0, 0.25, 0.05)
            with wc2:
                w_p = st.slider("Population Exposure (wP)", 0.0, 1.0, 0.25, 0.05)
            with wc3:
                w_a = st.slider("Assets & Cropland (wA)", 0.0, 1.0, 0.25, 0.05)
            with wc4:
                w_h = st.slider("Service Deficit (wH)", 0.0, 1.0, 0.25, 0.05)

            # Re-normalize weights dynamically
            w_total = w_c + w_p + w_a + w_h
            if w_total > 0:
                w_c_n = w_c / w_total
                w_p_n = w_p / w_total
                w_a_n = w_a / w_total
                w_h_n = w_h / w_total
            else:
                w_c_n, w_p_n, w_a_n, w_h_n = 0.25, 0.25, 0.25, 0.25

            # Compute Dynamic Score & Rank
            df_dynamic = df_bcip_rank.copy()
            df_dynamic['Dynamic_Score'] = (
                w_c_n * df_dynamic['Score_ClimateTransition_Cz'] +
                w_p_n * df_dynamic['Score_PopulationExposure_Pz'] +
                w_a_n * df_dynamic['Score_AssetExposure_Az'] +
                w_h_n * df_dynamic['Score_ServiceDeficit_Hz']
            )
            df_dynamic['Dynamic_Rank'] = df_dynamic['Dynamic_Score'].rank(ascending=False).astype(int)

            st.caption(f"Normalized Weights: wC={w_c_n:.2f}, wP={w_p_n:.2f}, wA={w_a_n:.2f}, wH={w_h_n:.2f}")

            # Display Top 15 Priority Districts Table
            st.markdown("##### 🏆 Dynamic Adaptation Priority Ranking (Top 15 Districts)")
            top_15_dyn = df_dynamic.sort_values('Dynamic_Rank').head(15)[[
                'Dynamic_Rank', 'DistrictName', 'Division', 'Dynamic_Score',
                'Rank_Model1_ExposureOnly', 'RankDisplacement_BCIP_vs_Exposure',
                'BCIP_RobustnessTier', 'Population'
            ]]
            st.dataframe(top_15_dyn, use_container_width=True)

            # Incremental Value Findings & Rank Displacement
            st.markdown("---")
            rc1, rc2 = st.columns([1.1, 1.0])
            with rc1:
                st.markdown("##### 📈 Incremental Value: Priority Displacement vs Exposure-Only")
                fig_disp = px.scatter(
                    df_dynamic,
                    x='Rank_Model1_ExposureOnly',
                    y='Dynamic_Rank',
                    color='RankDisplacement_BCIP_vs_Exposure',
                    hover_name='DistrictName',
                    size='Population',
                    color_continuous_scale='Spectral_r',
                    title="Exposure-Only Rank vs Dynamic BCIP Rank (Priority Shift)"
                )
                fig_disp.add_shape(
                    type="line", line=dict(dash="dash", color="red"),
                    x0=1, y0=1, x1=64, y1=64
                )
                fig_disp.update_layout(height=450, xaxis_title="Exposure-Only Rank [1=Top]", yaxis_title="Integrated BCIP Rank [1=Top]")
                fig_disp.update_xaxes(autorange="reversed")
                fig_disp.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_disp, use_container_width=True)

            with rc2:
                st.markdown("##### 🔬 Spearman Rank Correlation Across Models (Incremental Proof)")
                if bcip_corrs:
                    models_list = list(bcip_corrs.keys())
                    c_rows = []
                    for m1 in models_list:
                        r_item = {"Model": m1.replace("Model ", "M")}
                        for m2 in models_list:
                            short_m2 = m2.replace("Model ", "M").split('(')[0].strip()
                            r_item[short_m2] = bcip_corrs[m1][m2]["SpearmanRho"]
                        c_rows.append(r_item)
                    st.dataframe(pd.DataFrame(c_rows).set_index("Model"), use_container_width=True)

                st.markdown("##### 🛡️ Robustness Tier Distribution")
                tier_stats = df_dynamic['BCIP_RobustnessTier'].value_counts()
                for t_name, count in tier_stats.items():
                    st.markdown(f"- **{t_name.split(':')[0]}**: `{count} districts`")

            # Publication figures preview expander
            with st.expander("🔬 View 300 DPI Publication Multi-Model Figures"):
                f3_p = os.path.join(BASE_DIR, 'assets', 'bcip', 'bcip_fig3_multi_model_incremental_value_and_rank_displacement.png')
                f4_p = os.path.join(BASE_DIR, 'assets', 'bcip', 'bcip_fig4_weight_sensitivity_and_robustness_tiers.png')
                if os.path.exists(f3_p):
                    st.image(f3_p, caption="Figure 3: Incremental Value and Priority Rank Displacement (DPI=300)")
                if os.path.exists(f4_p):
                    st.image(f4_p, caption="Figure 4: Weight Sensitivity across Schemes and Robust Priority Stratification (DPI=300)")

            # Export Button
            st.markdown("---")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                csv_ranks = df_dynamic.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full 64-District Model Comparison & Rankings (CSV)",
                    data=csv_ranks,
                    file_name="bcip_model_comparison_and_rankings.csv",
                    mime="text/csv"
                )
            with exp_c2:
                b_geojson_p = os.path.join(SPATIAL_DIR, 'bcip_district_climate_transitions.geojson')
                if os.path.exists(b_geojson_p):
                    with open(b_geojson_p, 'r', encoding='utf-8') as bgf:
                        bg_data = bgf.read()
                    st.download_button(
                        label="🗺️ Download BCIP Climate Transition & Exposure GeoJSON (QGIS/ArcGIS Ready)",
                        data=bg_data,
                        file_name="bcip_district_climate_transitions.geojson",
                        mime="application/geo+json"
                    )
        else:
            st.info("Loading BCIP Prioritization & Robustness data...")

'''

# Inject BCIP tab content right above Tab 1 (tab_nat)
target_tab_nat = "    # -------------------------------------------------------------\n    # TAB 1: NATIONWIDE FLOOD SUSCEPTIBILITY (ALL 64)\n    # -------------------------------------------------------------\n    with tab_nat:"
if 'with tab_bcip_atlas:' not in content:
    content = content.replace(target_tab_nat, bcip_tab_body + "\n" + target_tab_nat)
    print("[+] Injected BCIP Tab implementations into web/app.py")

with open(APP_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("[+] web/app.py successfully updated with complete BCIP Suite!")
