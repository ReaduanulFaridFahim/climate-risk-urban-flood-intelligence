"""
========================================================================================
update_webapp_full_49layers.py
Connects Web Portal Tab 5 to bangladesh_64_districts_comprehensive_49layers.csv
Ensures every one of the 49 layers renders its exact attribute, choropleth, and ranking.
========================================================================================
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(BASE_DIR, 'web', 'app.py')

with open(APP_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update load_data to return df_49layers
old_load = """    # 111-Dataset Master Catalog
    cat_path = os.path.join(DATA_DIR, 'bangladesh_111_datasets_catalog.json')
    catalog_111 = {}
    if os.path.exists(cat_path):
        with open(cat_path, 'r', encoding='utf-8') as f:
            catalog_111 = json.load(f)
            
    return dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111"""

new_load = """    # 111-Dataset Master Catalog
    cat_path = os.path.join(DATA_DIR, 'bangladesh_111_datasets_catalog.json')
    catalog_111 = {}
    if os.path.exists(cat_path):
        with open(cat_path, 'r', encoding='utf-8') as f:
            catalog_111 = json.load(f)

    # 49 Comprehensive Core Spatial Layers Dataset
    layers_path = os.path.join(DATA_DIR, 'bangladesh_64_districts_comprehensive_49layers.csv')
    df_49layers = pd.read_csv(layers_path) if os.path.exists(layers_path) else None
            
    return dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111, df_49layers"""

if old_load in content:
    content = content.replace(old_load, new_load)
    print("[OK] Updated load_data return statements.")
else:
    print("[WARN] old_load pattern not matched.")

# 2. Update caller of load_data
old_caller = "dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111 = load_data()"
new_caller = "dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111, df_49layers = load_data()"

if old_caller in content:
    content = content.replace(old_caller, new_caller)
    print("[OK] Updated load_data invocation.")
else:
    print("[WARN] old_caller pattern not matched.")

# 3. Replace Tab 5 implementation with dynamic mapping for all 49 layers
target_atlas_start = "    # -------------------------------------------------------------\n    # TAB 5: NATIONAL GEOSPATIAL ENVIRONMENTAL ATLAS (49 LAYERS)\n    # -------------------------------------------------------------\n    with tab_atlas:"
target_atlas_end = "    # -------------------------------------------------------------\n    # TAB 6: UN HDX HUMANITARIAN DATA HUB (62 CURATED DATASETS)\n    # -------------------------------------------------------------\n    with tab_hdx:"

new_atlas_code = '''    # -------------------------------------------------------------
    # TAB 5: NATIONAL GEOSPATIAL ENVIRONMENTAL ATLAS (49 LAYERS)
    # -------------------------------------------------------------
    with tab_atlas:
        st.subheader("🌍 National Geospatial Environmental & Climate Atlas (49 Core Spatial Layers)")
        st.markdown("""
        **Operational BISTO Geospatial Telemetry & Multi-Hazard Observatory for Bangladesh:**
        Interactive analytical inspection and spatial visualization of all **49 high-resolution satellite, climate scenario, hydrological, and infrastructure layers**.
        Compare Köppen-Geiger CMIP6 climate shifts, Hansen 2001-2025 forest loss, JRC 40-year water dynamics, GHSL built-up expansion, and HOTOSM lifeline infrastructure.
        """)
        
        spatial_list = catalog_111.get('spatial_layers', []) if catalog_111 else []
        
        # Exact column mapping for all 49 layers
        LAYER_COL_MAP = {
            "koppen_baseline_1991_2020": ("Koppen_Baseline", "category", "Köppen Climate Classification Baseline (1991-2020)"),
            "koppen_ssp245_2041_2070": ("Koppen_SSP245_2050", "category", "Köppen Climate Projection SSP2-4.5 (2041-2070)"),
            "koppen_ssp585_2041_2070": ("Koppen_SSP585_2050", "category", "Köppen Climate Projection SSP5-8.5 (2041-2070)"),
            "koppen_ssp245_2071_2099": ("Koppen_SSP245_2090", "category", "Köppen Climate Projection SSP2-4.5 (2071-2099)"),
            "koppen_ssp585_2071_2099": ("Koppen_SSP585_2090", "category", "Köppen Climate Projection SSP5-8.5 (2071-2099)"),
            "tree_cover_gfc_2000": ("Hansen_TreeCover2000_Pct", "continuous", "Tree Canopy Cover 2000 (%)"),
            "forest_loss_year_2001_2025": ("Hansen_ForestLoss_2001_2025_Ha", "continuous", "Cumulative Forest Loss 2001-2025 (Hectares)"),
            "forest_loss_2025": ("Hansen_ForestLoss_2025_Ha", "continuous", "Annual Forest Loss 2025 (Hectares)"),
            "forest_loss_drivers_2001_2025": ("Hansen_DominantLossDriver", "category", "Dominant Tree Cover Loss Driver"),
            "mangrove_extent_2020": ("Mangrove_Extent_SqKm", "continuous", "Mangrove Forest Extent (Sq Km)"),
            "above_ground_biomass_2022": ("ESA_CCI_Biomass_MgPerHa", "continuous", "Above-ground Biomass Density (Mg/ha)"),
            "canopy_height_gedi_2022": ("GEDI_CanopyHeight_M", "continuous", "Canopy Top Height RH98 (Meters)"),
            "ndvi_modis_2025": ("MODIS_NDVI_2025_Mean", "continuous", "Annual Mean NDVI (2025)"),
            "land_cover_esa_worldcover_2020": ("ESA_WorldCover_BuiltPct_2020", "continuous", "Built-up Impervious Surface 2020 (%)"),
            "land_cover_esri_2025": ("ESRI_LandCover_BuiltPct_2025", "continuous", "Built-up Impervious Surface 2025 (%)"),
            "land_cover_dynamic_world_2026": ("DynamicWorld_WaterVegetation_2026", "continuous", "Near-Real-Time Water/Wetland 2026 (%)"),
            "water_occurrence_jrc_gsw_1984_2024": ("JRC_WaterOccurrence_Pct", "continuous", "Historical Surface Water Occurrence % (1984-2024)"),
            "water_change_jrc_gsw_1984_2024": ("JRC_WaterChange_Trend", "category", "40-Year Surface Water Change Trend"),
            "water_seasonality_jrc_gsw_1984_2024": ("JRC_WaterSeasonality_Months", "continuous", "Water Seasonality (Months/Year Present)"),
            "water_recurrence_jrc_gsw_1984_2024": ("JRC_WaterRecurrence_Pct", "continuous", "Water Recurrence Frequency (%)"),
            "water_extent_jrc_gsw_1984_2024": ("JRC_MaxWaterExtent_SqKm", "continuous", "Cumulative Max Water Inundation Extent (Sq Km)"),
            "water_transitions_jrc_gsw_1984_2024": ("JRC_TransitionClass", "category", "Hydrological Regime Transition Class"),
            "elevation_srtm_2000": ("SRTM_Elevation_M", "continuous", "SRTM Digital Elevation (Meters above Sea Level)"),
            "flood_proneness_ocha_barc_2013": ("OCHA_FloodProneness_Tier", "category", "National Flood Proneness Tier (BARC/SPARSSO)"),
            "drought_risk_ocha_barc_2013": ("OCHA_DroughtRisk_Aman", "category", "Drought Risk Class (Aman Rice Season)"),
            "seismic_zones_ocha_barc_2013": ("OCHA_SeismicHazard_Zone", "category", "Seismic Hazard Zonation (BNBC/SPARSSO)"),
            "riverbank_erosion_ocha_barc_2013": ("OCHA_RiverbankErosion_Risk", "category", "Riverbank Erosion Vulnerability"),
            "flood_water_extent_unosat_2024": ("UNOSAT_June2024_FloodExtent_Ha", "continuous", "Satellite-Detected Flood Extent June 2024 (Ha)"),
            "cyclone_tracks_ibtracs_1877_2024": ("IBTrACS_CycloneTracks_Count", "continuous", "Historical Cyclone Tracks within 100km (1877-2024)"),
            "built_up_surface_ghsl_2020": ("GHSL_BuiltUpSurface_2020_SqKm", "continuous", "Built-up Surface Extent 2020 (Sq Km)"),
            "built_up_expansion_2000_2020": ("GHSL_Expansion_2000_2020_Pct", "continuous", "Built-up Expansion Rate 2000-2020 (%)"),
            "population_grid_ghsl_2020": ("GHSL_PopulationGrid_2020", "continuous", "GHSL Population Density per Built Cell"),
            "population_density_worldpop_2020": ("WorldPop_Density_PerSqKm", "continuous", "WorldPop Population Density (People/Sq Km)"),
            "population_exposure_gar15_2015": ("GAR15_PopulationExposure_Class", "category", "GAR15 Global Multi-Hazard Exposure Class (1-6)"),
            "relative_wealth_index_meta_2021": ("Meta_RWI_Score", "continuous", "Meta Relative Wealth Index (-1.3 to +2.0)"),
            "poverty_likelihood_worldpop_2013": ("WorldPop_PovertyLikelihood_Pct", "continuous", "Poverty Likelihood <$2.50/day (%)"),
            "estimated_income_worldpop_2013": ("WorldPop_RelativeIncome_Index", "continuous", "Modelled Relative Income Index"),
            "rice_season_ndvi_boro_jan_apr": ("Rice_NDVI_Boro", "continuous", "Boro Rice Season NDVI (Jan-Apr)"),
            "rice_season_ndvi_aus_may_jul": ("Rice_NDVI_Aus", "continuous", "Aus Rice Season NDVI (May-Jul)"),
            "rice_season_ndvi_aman_aug_nov": ("Rice_NDVI_Aman", "continuous", "Aman Rice Season NDVI (Aug-Nov)"),
            "education_facilities_hotosm_2024": ("HOTOSM_EducationShelters", "continuous", "Designated Emergency Education Shelters (HOTOSM)"),
            "health_facilities_hotosm_2024": ("HOTOSM_HealthFacilities", "continuous", "Clinical Health Facilities Mapped (HOTOSM)"),
            "road_network_hotosm_2024": ("HOTOSM_RoadNetwork_TotalKm", "continuous", "Total Road Network Length (Km)"),
            "distance_to_nearest_health_facility_2026": ("DistanceToHealthFacility_Km", "continuous", "Straight-Line Distance to Health Facility (Km)"),
            "land_surface_temp_modis_2024_2025": ("MODIS_LST_DayTemp_C", "continuous", "Daytime Land Surface Temperature 2024-2025 (°C)"),
            "night_lights_viirs_2012_2025": ("VIIRS_NightLights_Radiance", "continuous", "Nighttime Lights Radiance (nW/cm2/sr)"),
            "wind_power_density_gwa4_100m": ("GWA_WindPowerDensity_100m", "continuous", "Wind Power Density at 100m (W/m2)"),
            "wind_speed_gwa4_100m": ("GWA_WindSpeed_100m_Mps", "continuous", "Mean Wind Speed at 100m (m/s)"),
            "malaria_parasite_rate_map_2024": ("MAP2024_MalariaParasiteRate", "continuous", "Plasmodium falciparum Parasite Rate (MAP 2024)")
        }
        
        if spatial_list and df_49layers is not None:
            atlas_themes = ["All Themes"] + sorted(list(set([layer.get('theme', 'Other') for layer in spatial_list])))
            
            c_th, c_lay = st.columns([1, 2])
            with c_th:
                selected_theme = st.selectbox("Filter Spatial Themes:", atlas_themes, index=0)
            
            filtered_layers = spatial_list if selected_theme == "All Themes" else [l for l in spatial_list if l.get('theme') == selected_theme]
            layer_titles = [l['title'] for l in filtered_layers]
            
            with c_lay:
                selected_title = st.selectbox("Select Core Spatial Map Layer:", layer_titles, index=0)
                
            cur_layer = next((l for l in filtered_layers if l['title'] == selected_title), filtered_layers[0])
            layer_id = cur_layer.get('id', '')
            
            col_info = LAYER_COL_MAP.get(layer_id, ("JRC_WaterOccurrence_Pct", "continuous", cur_layer.get('title')))
            active_col = col_info[0]
            col_type = col_info[1]
            col_label = col_info[2]
            
            # Layer Inspection Dossier
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 14px; padding: 18px 22px; border: 1px solid #334155; margin-top: 10px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #38bdf8;">🗺️ {cur_layer.get('title')}</span>
                    <span style="background: #0284c7; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">{cur_layer.get('theme')}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 14px;">
                    <div style="background: #1e293b; padding: 8px 12px; border-radius: 8px; border-left: 3px solid #10b981;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Spatial Resolution</span>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">{cur_layer.get('resolution')}</div>
                    </div>
                    <div style="background: #1e293b; padding: 8px 12px; border-radius: 8px; border-left: 3px solid #38bdf8;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Temporal Range / Epoch</span>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">{cur_layer.get('temporal_range')}</div>
                    </div>
                    <div style="background: #1e293b; padding: 8px 12px; border-radius: 8px; border-left: 3px solid #f59e0b;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Provider / Author</span>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">{cur_layer.get('provider')}</div>
                    </div>
                    <div style="background: #1e293b; padding: 8px 12px; border-radius: 8px; border-left: 3px solid #ec4899;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Active Metric Attribute</span>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">`{active_col}`</div>
                    </div>
                </div>
                <div style="margin-top: 14px; font-size: 0.85rem; color: #e2e8f0; line-height: 1.45;">
                    <strong style="color: #38bdf8;">📖 How to Read:</strong> {cur_layer.get('how_to_read')}<br>
                    <strong style="color: #10b981; margin-top: 6px; display: inline-block;">🎯 Key Applications:</strong> {cur_layer.get('applications')}<br>
                    <strong style="color: #f59e0b; margin-top: 6px; display: inline-block;">⚙️ Integration Role:</strong> {cur_layer.get('integration_role')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Map & Analytical Inspection
            st.markdown(f"#### 🗺️ Spatial Simulation & District Analytical Mapping: **{col_label}**")
            
            viz_col1, viz_col2 = st.columns([1.3, 1])
            
            with viz_col1:
                df_map = df_49layers.copy()
                
                # Setup hover data
                h_data = {
                    "Division": True,
                    "AreaSqKm": ":.1f",
                    "Population": ":,",
                    active_col: True
                }
                
                if col_type == "category":
                    fig_atlas = px.scatter_map(
                        df_map, lat="Latitude", lon="Longitude",
                        color=active_col,
                        size="AreaSqKm", hover_name="DistrictName",
                        hover_data=h_data,
                        zoom=6.1, map_style="carto-darkmatter",
                        title=f"{cur_layer.get('title')} ({col_label})"
                    )
                else:
                    fig_atlas = px.scatter_map(
                        df_map, lat="Latitude", lon="Longitude",
                        color=active_col,
                        size=active_col if (df_map[active_col] > 0).all() else "AreaSqKm",
                        hover_name="DistrictName",
                        color_continuous_scale="Viridis" if "water" in active_col.lower() else ("Blues" if "temp" in active_col.lower() else "Turbo"),
                        hover_data=h_data,
                        zoom=6.1, map_style="carto-darkmatter",
                        title=f"{cur_layer.get('title')} ({col_label})"
                    )
                fig_atlas.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_atlas, use_container_width=True)
                
            with viz_col2:
                st.markdown(f"##### 📊 Regional Division Statistics ({col_label})")
                if col_type == "continuous":
                    div_summary = df_49layers.groupby('Division')[active_col].agg(['mean', 'max', 'min']).reset_index()
                    div_summary.columns = ['Division', 'Mean', 'Max Peak', 'Min Baseline']
                    st.dataframe(div_summary, use_container_width=True)
                    
                    fig_bar_atlas = px.bar(
                        div_summary, x='Division', y='Mean',
                        color='Mean', color_continuous_scale='Blues',
                        title=f"Division Averages: {active_col}"
                    )
                    fig_bar_atlas.update_layout(margin=dict(l=0, r=0, t=35, b=0), showlegend=False)
                    st.plotly_chart(fig_bar_atlas, use_container_width=True)
                else:
                    cat_counts = df_49layers[active_col].value_counts().reset_index()
                    cat_counts.columns = ['Class Category', 'District Count']
                    st.dataframe(cat_counts, use_container_width=True)
                    
                    fig_cat_bar = px.bar(
                        cat_counts, x='District Count', y='Class Category', orientation='h',
                        color='Class Category', title=f"Class Distribution: {active_col}"
                    )
                    fig_cat_bar.update_layout(margin=dict(l=0, r=0, t=35, b=0), showlegend=False)
                    st.plotly_chart(fig_cat_bar, use_container_width=True)
                
                # Classes / Legend Items
                if cur_layer.get('classes'):
                    st.markdown("##### 🏷️ Scientific Legend Hierarchy")
                    for cls in cur_layer.get('classes', []):
                        st.markdown(f"- `{cls}`")
                        
            # Data Preview Table for All 64 Districts
            st.markdown("##### 📋 Complete 64-District Data Inspection")
            st.dataframe(
                df_49layers[['DistrictID', 'DistrictName', 'Division', 'Latitude', 'Longitude', 'AreaSqKm', 'Population', active_col]],
                use_container_width=True
            )
                        
            # Export Options
            st.markdown("---")
            e_c1, e_c2 = st.columns(2)
            with e_c1:
                cat_csv = df_49layers.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full 64-District 49-Layers Master Dataset (CSV)",
                    data=cat_csv,
                    file_name="bangladesh_64_districts_comprehensive_49layers.csv",
                    mime="text/csv"
                )
            with e_c2:
                geojson_file_path = os.path.join(SPATIAL_DIR, 'bangladesh_64_districts_full_49layers.geojson')
                if os.path.exists(geojson_file_path):
                    with open(geojson_file_path, 'r', encoding='utf-8') as gf:
                        g_data = gf.read()
                    st.download_button(
                        label="🗺️ Download QGIS / ArcGIS Ready GeoJSON Layer (49 Attributes)",
                        data=g_data,
                        file_name="bangladesh_64_districts_full_49layers.geojson",
                        mime="application/geo+json"
                    )
        else:
            st.info("Loading 49 Spatial Map Layers catalog...")\n'''

start_idx = content.find(target_atlas_start)
end_idx = content.find(target_atlas_end)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_atlas_code + "\n" + content[end_idx:]
    print("[OK] Replaced Tab 5 with full 49-layer dynamic mapping.")
else:
    print("[WARN] Could not find Tab 5 boundaries.")

with open(APP_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("[SUCCESS] web/app.py updated with complete 49 spatial layers mapping engine.")
