"""
========================================================================================
UPDATE WEBAPP WITH 111-DATASET CATALOG:
- Tab 5: National Geospatial Environmental Atlas (49 Spatial Layers)
- Tab 6: UN HDX Humanitarian Data Hub (62 Curated Datasets)
========================================================================================
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(BASE_DIR, 'web', 'app.py')

with open(APP_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update load_data to load catalog_111
old_load = """    # Automated Live Dispatches
    disp_path = os.path.join(DATA_DIR, 'bangladesh_automated_live_dispatches.csv')
    df_dispatches = pd.read_csv(disp_path) if os.path.exists(disp_path) else None
            
    return dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches"""

new_load = """    # Automated Live Dispatches
    disp_path = os.path.join(DATA_DIR, 'bangladesh_automated_live_dispatches.csv')
    df_dispatches = pd.read_csv(disp_path) if os.path.exists(disp_path) else None

    # 111-Dataset Master Catalog
    cat_path = os.path.join(DATA_DIR, 'bangladesh_111_datasets_catalog.json')
    catalog_111 = {}
    if os.path.exists(cat_path):
        with open(cat_path, 'r', encoding='utf-8') as f:
            catalog_111 = json.load(f)
            
    return dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111"""

if old_load in content:
    content = content.replace(old_load, new_load)
    print("[OK] Updated load_data function.")
else:
    print("[WARN] old_load pattern not matched.")

# 2. Update load_data invocation
old_call = "dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches = load_data()"
new_call = "dim_area, dim_event, fact_rainfall, fact_landcover, fact_flood, fact_exposure, nat_dist, nat_pred, future_projections_64, multi_df, alerts_data, sat_data, fc_20day, df_dispatches, catalog_111 = load_data()"

if old_call in content:
    content = content.replace(old_call, new_call)
    print("[OK] Updated load_data invocation.")
else:
    print("[WARN] old_call pattern not matched.")

# 3. Update tab definitions
old_tabs = """    # Master Multi-Hazard Tab System (8 Feature Tabs)
    tab_nat, tab_phone, tab_live, tab_20d, tab_multi_ai, tab_future, tab_ml_eval, tab_cma = st.tabs([
        "🇧🇩 Nationwide Flood Susceptibility (64 Districts)",
        "📱 Emergency Phone Alerts & Auto-Sentinel",
        "⚡ Real Live Weather Satellite & Radar Map",
        "📅 Advance 20-Day Weather & Hazard Outlook (ECMWF)",
        "☀️❄️ Multi-Hazard AI: Heatwaves & Winter Cold Waves",
        "🔮 All-Bangladesh Future Forecasts (2030, 2040, 2050)",
        "🤖 Advanced AI Model Benchmark (Neural Net & Stacking)",
        "🏙️ Chattogram Ward Analysis"
    ])"""

new_tabs = """    # Master Multi-Hazard Tab System (10 Comprehensive Feature Tabs with 111 Datasets)
    tab_nat, tab_phone, tab_live, tab_20d, tab_atlas, tab_hdx, tab_multi_ai, tab_future, tab_ml_eval, tab_cma = st.tabs([
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
    ])"""

if old_tabs in content:
    content = content.replace(old_tabs, new_tabs)
    print("[OK] Updated tab definitions to 10 tabs.")
else:
    print("[WARN] old_tabs pattern not matched.")

# 4. Insert Tab 5 and Tab 6 implementation before Tab 5 (now Tab 7) Multi-Hazard AI
target_multi = """    # -------------------------------------------------------------
    # TAB 5: MULTI-HAZARD AI: HEATWAVES & WINTER COLD WAVES
    # -------------------------------------------------------------
    with tab_multi_ai:"""

new_tabs_impl = '''    # -------------------------------------------------------------
    # TAB 5: NATIONAL GEOSPATIAL ENVIRONMENTAL ATLAS (49 LAYERS)
    # -------------------------------------------------------------
    with tab_atlas:
        st.subheader("🌍 National Geospatial Environmental & Climate Atlas (4Core Spatial Layers)")
        st.markdown("""
        **BISTO Spatial Telemetry & Multi-Hazard Geospatial Observatory:**
        Interactive analytical inspection of **49 high-resolution satellite, climate scenario, hydrological, and infrastructure layers** covering Bangladesh.
        Compare Köppen-Geiger CMIP6 climate shifts, Hansen 2001-2025 forest loss, JRC 40-year water dynamics, GHSL built-up expansion, and HOTOSM lifeline infrastructure.
        """)
        
        spatial_list = catalog_111.get('spatial_layers', []) if catalog_111 else []
        
        if spatial_list:
            atlas_themes = ["All Themes"] + sorted(list(set([layer.get('theme', 'Other') for layer in spatial_list])))
            
            c_th, c_lay = st.columns([1, 2])
            with c_th:
                selected_theme = st.selectbox("Filter Spatial Themes:", atlas_themes, index=0)
            
            filtered_layers = spatial_list if selected_theme == "All Themes" else [l for l in spatial_list if l.get('theme') == selected_theme]
            layer_titles = [l['title'] for l in filtered_layers]
            
            with c_lay:
                selected_title = st.selectbox("Select Core Spatial Map Layer:", layer_titles, index=0)
                
            cur_layer = next((l for l in filtered_layers if l['title'] == selected_title), filtered_layers[0])
            
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
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">License</span>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">{cur_layer.get('license')}</div>
                    </div>
                </div>
                <div style="margin-top: 14px; font-size: 0.85rem; color: #e2e8f0; line-height: 1.45;">
                    <strong style="color: #38bdf8;">📖 How to Read:</strong> {cur_layer.get('how_to_read')}<br>
                    <strong style="color: #10b981; margin-top: 6px; display: inline-block;">🎯 Key Applications:</strong> {cur_layer.get('applications')}<br>
                    <strong style="color: #f59e0b; margin-top: 6px; display: inline-block;">⚙️ Integration Role:</strong> {cur_layer.get('integration_role')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Layer Visualizer
            st.markdown("#### 🗺️ Spatial Simulation & District Analytical Mapping")
            
            viz_col1, viz_col2 = st.columns([1.3, 1])
            
            with viz_col1:
                # Dynamic Map rendering based on layer category
                df_map = nat_pred.copy() if nat_pred is not None else pd.DataFrame()
                
                if 'Koppen' in cur_layer.get('id', '') or 'koppen' in cur_layer.get('id', ''):
                    # Köppen climate classification display
                    fig_atlas = px.scatter_map(
                        df_map, lat="Latitude", lon="Longitude",
                        color="KoppenClimateClass" if "KoppenClimateClass" in df_map.columns else "Division",
                        size="AreaSqKm", hover_name="DistrictName",
                        hover_data={"Division": True, "KoppenClimateClass": True, "MeanElevationM": True},
                        zoom=6.1, map_style="carto-darkmatter",
                        title=f"{cur_layer.get('title')} - Climate Zone Distribution"
                    )
                elif 'water' in cur_layer.get('id', '').lower():
                    # JRC Water occurrence / recurrence
                    fig_atlas = px.scatter_map(
                        df_map, lat="Latitude", lon="Longitude",
                        color="JRC_WaterOccurrencePct" if "JRC_WaterOccurrencePct" in df_map.columns else "PredictedInundatedPercentage",
                        size="PredictedInundatedPercentage", hover_name="DistrictName",
                        color_continuous_scale="Viridis",
                        hover_data={"Division": True, "DistanceToMajorRiverKm": True, "MeanElevationM": True},
                        zoom=6.1, map_style="carto-darkmatter",
                        title=f"{cur_layer.get('title')} - 40-Year Surface Water Occurrence"
                    )
                elif 'health' in cur_layer.get('id', '').lower() or 'road' in cur_layer.get('id', '').lower():
                    # Infrastructure facilities
                    color_prop = "HOTOSM_HealthFacilityCount" if "health" in cur_layer.get('id', '').lower() else "HOTOSM_RoadDensityKmPerSqKm"
                    fig_atlas = px.scatter_map(
                        df_map, lat="Latitude", lon="Longitude",
                        color=color_prop if color_prop in df_map.columns else "BaselineImperviousRatio",
                        size="Population", hover_name="DistrictName",
                        color_continuous_scale="Plasma",
                        hover_data={"Division": True, "Population": ":,"},
                        zoom=6.1, map_style="carto-darkmatter",
                        title=f"{cur_layer.get('title')} - District Density Profile"
                    )
                else:
                    # General Multi-hazard mapping
                    fig_atlas = px.scatter_map(
                        df_map, lat="Latitude", lon="Longitude",
                        color="Meta_RWI_WealthIndex" if "Meta_RWI_WealthIndex" in df_map.columns else "PredictedInundatedPercentage",
                        size="AreaSqKm", hover_name="DistrictName",
                        color_continuous_scale="Turbo",
                        hover_data={"Division": True, "GHSL_BuiltUpExpansionPct": True},
                        zoom=6.1, map_style="carto-darkmatter",
                        title=f"{cur_layer.get('title')} - Nationwide Zonal Representation"
                    )
                fig_atlas.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_atlas, use_container_width=True)
                
            with viz_col2:
                st.markdown("##### 📊 Regional Division Zonal Aggregation")
                if nat_pred is not None:
                    # Aggregate summary
                    agg_cols = ['Division']
                    metric_col = 'JRC_WaterOccurrencePct' if 'water' in cur_layer.get('id', '').lower() else ('Meta_RWI_WealthIndex' if 'wealth' in cur_layer.get('id', '').lower() or 'poverty' in cur_layer.get('id', '').lower() else 'PredictedInundatedPercentage')
                    if metric_col in nat_pred.columns:
                        div_summary = nat_pred.groupby('Division')[metric_col].agg(['mean', 'max', 'min']).reset_index()
                        div_summary.columns = ['Division', 'Mean Index', 'Peak Hotspot', 'Low Baseline']
                        st.dataframe(div_summary, use_container_width=True)
                        
                        fig_bar_atlas = px.bar(
                            div_summary, x='Division', y='Mean Index',
                            color='Mean Index', color_continuous_scale='Blues',
                            title=f"Division-Level {metric_col} Averages"
                        )
                        fig_bar_atlas.update_layout(margin=dict(l=0, r=0, t=35, b=0), showlegend=False)
                        st.plotly_chart(fig_bar_atlas, use_container_width=True)
                
                # Classes / Legend Items
                if cur_layer.get('classes'):
                    st.markdown("##### 🏷️ Layer Classification Classes")
                    for cls in cur_layer.get('classes', []):
                        st.markdown(f"- `{cls}`")
                        
            # Export Layer Metadata
            st.markdown("---")
            cat_csv = pd.DataFrame(spatial_list).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete 49-Spatial-Layer Registry (CSV)",
                data=cat_csv,
                file_name="bangladesh_49_core_spatial_layers_registry.csv",
                mime="text/csv"
            )
        else:
            st.info("Loading 49 Spatial Map Layers catalog...")

    # -------------------------------------------------------------
    # TAB 6: UN HDX HUMANITARIAN DATA HUB (62 CURATED DATASETS)
    # -------------------------------------------------------------
    with tab_hdx:
        st.subheader("📦 UN Humanitarian Data Exchange (HDX) Bangladesh Intelligence Hub")
        st.markdown("""
        **Authoritative External UN & International Agency Open Data Portal (62 Curated Datasets):**
        Curated by BISTO from the **Humanitarian Data Exchange (HDX)** for Bangladesh across food security, health, refugee displacement, poverty, and emergency hazard zonation.
        Each dataset includes direct access links to official UN repositories on [data.humdata.org](https://data.humdata.org).
        """)
        
        hdx_list = catalog_111.get('hdx_datasets', []) if catalog_111 else []
        
        if hdx_list:
            # Cluster metrics row
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            h_col1.metric("Curated UN Datasets", "62", "Humanitarian Data Exchange (HDX)")
            h_col2.metric("Thematic Clusters", "12 Clusters", "Food, Health, Disasters, WASH")
            h_col3.metric("Key Global Agencies", "15+ UN Agencies", "WFP, FAO, WHO, IOM, UNHCR, BBS")
            h_col4.metric("Temporal Coverage", "1900 - 2029", "Historical to Predictive VIEWS")
            
            st.markdown("---")
            
            # Interactive Filter & Search
            f_col1, f_col2 = st.columns([1, 2])
            
            clusters = ["ALL CLUSTERS (62)"] + sorted(list(set([d.get('cluster', 'Other') for d in hdx_list])))
            with f_col1:
                selected_cluster = st.selectbox("Filter by Humanitarian Cluster:", clusters, index=0)
                
            with f_col2:
                search_query = st.text_input("🔍 Search 62 Datasets by Name, Provider, or Keyword:", placeholder="e.g. food security, cholera, flood extent, displacement, census, WHO...")
                
            # Filter logic
            filtered_hdx = hdx_list
            if selected_cluster != "ALL CLUSTERS (62)":
                filtered_hdx = [d for d in filtered_hdx if d.get('cluster') == selected_cluster]
                
            if search_query:
                q = search_query.lower()
                filtered_hdx = [
                    d for d in filtered_hdx
                    if q in d.get('title', '').lower() or q in d.get('source', '').lower() or q in d.get('description', '').lower() or q in d.get('use_case', '').lower()
                ]
                
            st.markdown(f"**Showing {len(filtered_hdx)} of 62 Curated Datasets**")
            
            # Responsive Card Grid (2-column layout)
            for i in range(0, len(filtered_hdx), 2):
                col_a, col_b = st.columns(2)
                
                # Card A
                with col_a:
                    dA = filtered_hdx[i]
                    cluster_icon = "🌾" if "Agri" in dA.get('cluster','') else ("⚠️" if "Disaster" in dA.get('cluster','') else ("🏥" if "Health" in dA.get('cluster','') else ("👥" if "Pop" in dA.get('cluster','') else ("💧" if "Water" in dA.get('cluster','') or "WASH" in dA.get('cluster','') else "🗂️"))))
                    st.markdown(f"""
                    <div style="background:#1e293b; border-radius:12px; padding:16px; border:1px solid #334155; margin-bottom:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
                        <div>
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:0.75rem; font-weight:700; color:#38bdf8; text-transform:uppercase;">{cluster_icon} {dA.get('cluster')}</span>
                                <span style="background:#0f172a; color:#94a3b8; font-size:0.7rem; padding:2px 8px; border-radius:8px;">{dA.get('format')}</span>
                            </div>
                            <div style="font-size:0.95rem; font-weight:700; color:#ffffff; margin-top:6px;">{dA.get('title')}</div>
                            <div style="font-size:0.78rem; color:#94a3b8; margin-top:2px;">🏛️ {dA.get('source')} | 📅 {dA.get('coverage')}</div>
                            <div style="font-size:0.8rem; color:#cbd5e1; margin-top:8px; line-height:1.35;">{dA.get('description')}</div>
                            <div style="font-size:0.75rem; color:#10b981; margin-top:6px;">💡 <strong>Use Case:</strong> {dA.get('use_case')}</div>
                        </div>
                        <div style="margin-top:12px; border-top:1px solid #334155; padding-top:8px; display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.7rem; color:#64748b;">⚖️ {dA.get('license')}</span>
                            <a href="{dA.get('hdx_url')}" target="_blank" style="background:#0284c7; color:#ffffff; text-decoration:none; padding:4px 12px; border-radius:6px; font-size:0.75rem; font-weight:700;">Open on HDX ↗</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Card B (if exists)
                if i + 1 < len(filtered_hdx):
                    with col_b:
                        dB = filtered_hdx[i+1]
                        cluster_icon_b = "🌾" if "Agri" in dB.get('cluster','') else ("⚠️" if "Disaster" in dB.get('cluster','') else ("🏥" if "Health" in dB.get('cluster','') else ("👥" if "Pop" in dB.get('cluster','') else ("💧" if "Water" in dB.get('cluster','') or "WASH" in dB.get('cluster','') else "🗂️"))))
                        st.markdown(f"""
                        <div style="background:#1e293b; border-radius:12px; padding:16px; border:1px solid #334155; margin-bottom:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
                            <div>
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-size:0.75rem; font-weight:700; color:#38bdf8; text-transform:uppercase;">{cluster_icon_b} {dB.get('cluster')}</span>
                                    <span style="background:#0f172a; color:#94a3b8; font-size:0.7rem; padding:2px 8px; border-radius:8px;">{dB.get('format')}</span>
                                </div>
                                <div style="font-size:0.95rem; font-weight:700; color:#ffffff; margin-top:6px;">{dB.get('title')}</div>
                                <div style="font-size:0.78rem; color:#94a3b8; margin-top:2px;">🏛️ {dB.get('source')} | 📅 {dB.get('coverage')}</div>
                                <div style="font-size:0.8rem; color:#cbd5e1; margin-top:8px; line-height:1.35;">{dB.get('description')}</div>
                                <div style="font-size:0.75rem; color:#10b981; margin-top:6px;">💡 <strong>Use Case:</strong> {dB.get('use_case')}</div>
                            </div>
                            <div style="margin-top:12px; border-top:1px solid #334155; padding-top:8px; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:0.7rem; color:#64748b;">⚖️ {dB.get('license')}</span>
                                <a href="{dB.get('hdx_url')}" target="_blank" style="background:#0284c7; color:#ffffff; text-decoration:none; padding:4px 12px; border-radius:6px; font-size:0.75rem; font-weight:700;">Open on HDX ↗</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Export Master HDX Catalog
            st.markdown("---")
            hdx_csv = pd.DataFrame(hdx_list).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete 62-Dataset Curated HDX Catalog (CSV)",
                data=hdx_csv,
                file_name="bangladesh_62_curated_hdx_catalog.csv",
                mime="text/csv"
            )
        else:
            st.info("Loading 62 Curated UN HDX datasets...")

    # -------------------------------------------------------------
    # TAB 7: MULTI-HAZARD AI: HEATWAVES & WINTER COLD WAVES
    # -------------------------------------------------------------
    with tab_multi_ai:'''

if target_multi in content:
    content = content.replace(target_multi, new_tabs_impl)
    print("[OK] Inserted Tab 5 & Tab 6 implementation.")
else:
    print("[WARN] target_multi pattern not matched.")

with open(APP_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("[SUCCESS] web/app.py updated with 111-dataset catalog integration.")
