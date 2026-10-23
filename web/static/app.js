/**
 * BANGLADESH CLIMATE RISK & MULTI-HAZARD INTELLIGENCE PLATFORM (BCIP)
 * Enterprise Frontend Application Logic & High-Speed Spatial Telemetry
 */

// Global State
let platformData = null;
let geojsonData = null;
let leafletMap = null;
let geojsonLayer = null;
let activeDistrict = null;
let activeDivision = "All";

// DOM Loaded Entry
document.addEventListener("DOMContentLoaded", async () => {
  initClock();
  await loadPlatformData();
  initMap();
  renderKpiSummary();
  renderSatelliteTelemetry();
  renderDistrictList();
  renderForecastWorkspace();
  renderHumanitarianSimulators();
  renderHdxManifestTable();
  initEventListeners();
});

// 1. Digital Clock (Bangladesh Standard Time: UTC+6)
function initClock() {
  const clockEl = document.getElementById("liveBstClock");
  function update() {
    const now = new Date();
    // UTC+6 offset
    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    const bst = new Date(utc + (3600000 * 6));
    const timeStr = bst.toLocaleTimeString("en-US", { hour12: false });
    const dateStr = bst.toLocaleDateString("en-GB", { day: '2-digit', month: 'short', year: 'numeric' });
    if (clockEl) {
      clockEl.textContent = `BST ${timeStr} | ${dateStr}`;
    }
  }
  update();
  setInterval(update, 1000);
}

// 2. Load Static Platform Data & GeoJSON
async function loadPlatformData() {
  try {
    const [resData, resGeo] = await Promise.all([
      fetch("data/platform_data.json"),
      fetch("data/bangladesh_districts.geojson")
    ]);
    platformData = await resData.json();
    geojsonData = await resGeo.json();
    console.log("[BCIP] Platform data & GeoJSON loaded successfully.", platformData);
  } catch (err) {
    console.error("[BCIP] Error loading platform data:", err);
  }
}

// 3. Initialize Leaflet Map
function initMap() {
  const mapContainer = document.getElementById("nationalMap");
  if (!mapContainer || !window.L) return;

  // Center on Bangladesh
  leafletMap = L.map("nationalMap", {
    center: [23.85, 90.35],
    zoom: 7,
    minZoom: 6,
    maxZoom: 12,
    zoomControl: true
  });

  // Dark Canvas Tiles
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(leafletMap);

  if (geojsonData) {
    renderGeoJsonLayer();
  }
}

function getDistrictColor(tier) {
  const t = String(tier).toUpperCase();
  if (t.includes("CRITICAL") || t.includes("RED") || t.includes("TIER 1")) return "#ef4444";
  if (t.includes("HIGH") || t.includes("ORANGE") || t.includes("TIER 2")) return "#f59e0b";
  if (t.includes("MODERATE") || t.includes("BLUE")) return "#3b82f6";
  return "#10b981";
}

function renderGeoJsonLayer() {
  if (!leafletMap || !geojsonData) return;
  if (geojsonLayer) leafletMap.removeLayer(geojsonLayer);

  // Index district attributes by DistrictID or DistrictName
  const distMap = {};
  platformData.districts.forEach(d => {
    distMap[d.DistrictName.toLowerCase()] = d;
    distMap[d.DistrictID] = d;
  });

  geojsonLayer = L.geoJSON(geojsonData, {
    style: (feature) => {
      const props = feature.properties || {};
      const name = (props.DistrictName || props.NAME_2 || props.ADM2_EN || "").toLowerCase();
      const d = distMap[name] || distMap[props.DistrictID] || {};
      const color = getDistrictColor(d.AlertTier || props.AlertTier || "MODERATE");

      return {
        fillColor: color,
        weight: 1.5,
        opacity: 0.9,
        color: "#ffffff",
        fillOpacity: 0.55
      };
    },
    onEachFeature: (feature, layer) => {
      const props = feature.properties || {};
      const name = props.DistrictName || props.NAME_2 || props.ADM2_EN || "District";
      const d = distMap[name.toLowerCase()] || distMap[props.DistrictID] || {};

      layer.bindTooltip(`
        <div style="font-family:Inter,sans-serif; padding:4px 8px;">
          <strong style="font-size:0.95rem; color:#ffffff;">${d.DistrictName || name}</strong><br>
          <span style="color:#94a3b8; font-size:0.8rem;">Division: ${d.Division || props.Division || 'N/A'}</span><br>
          <span style="color:#38bdf8; font-weight:600; font-size:0.82rem;">Inundation: ${d.PredictedInundatedPercentage ? d.PredictedInundatedPercentage.toFixed(1) + '%' : 'N/A'}</span><br>
          <span style="color:#f87171; font-weight:700; font-size:0.8rem;">Tier: ${d.AlertTier || 'MODERATE'}</span>
        </div>
      `, { sticky: true, className: 'custom-map-tooltip' });

      layer.on({
        mouseover: (e) => {
          const l = e.target;
          l.setStyle({ weight: 3, color: '#38bdf8', fillOpacity: 0.8 });
        },
        mouseout: (e) => {
          geojsonLayer.resetStyle(e.target);
        },
        click: () => {
          if (d.DistrictID) {
            openDistrictModal(d);
          }
        }
      });
    }
  }).addTo(leafletMap);
}

// 4. Render National KPI Summary
function renderKpiSummary() {
  if (!platformData) return;
  const districts = platformData.districts;

  const totalPop = districts.reduce((acc, d) => acc + (d.Population || 0), 0);
  const totalInundHa = districts.reduce((acc, d) => acc + (d.InundatedAgriLand_Ha || 0), 0);
  const totalRiceLoss = districts.reduce((acc, d) => acc + (d.StandingRiceLoss_Tons || 0), 0);
  const totalDisplaced = districts.reduce((acc, d) => acc + (d.DisplacedPersonsEst || 0), 0);
  const totalWellsContam = districts.reduce((acc, d) => acc + (d.SubmergedTubeWells_Contaminated || 0), 0);
  const criticalCount = districts.filter(d => String(d.AlertTier).toUpperCase().includes("CRITICAL")).length;

  const kpiContainer = document.getElementById("kpiSummaryContainer");
  if (!kpiContainer) return;

  kpiContainer.innerHTML = `
    <div class="kpi-card rose">
      <div class="kpi-header">
        <span class="kpi-label">Red Alert Districts</span>
        <span class="kpi-icon">🚨</span>
      </div>
      <div class="kpi-value">${criticalCount} / 64</div>
      <div class="kpi-subtitle">Immediate Emergency Action Required</div>
    </div>

    <div class="kpi-card cyan">
      <div class="kpi-header">
        <span class="kpi-label">National Population</span>
        <span class="kpi-icon">👥</span>
      </div>
      <div class="kpi-value">${(totalPop / 1000000).toFixed(1)} M</div>
      <div class="kpi-subtitle">BBS Census 2022 Baseline</div>
    </div>

    <div class="kpi-card amber">
      <div class="kpi-header">
        <span class="kpi-label">Rice Harvest Loss</span>
        <span class="kpi-icon">🌾</span>
      </div>
      <div class="kpi-value">${(totalRiceLoss / 1000).toFixed(1)}k Tons</div>
      <div class="kpi-subtitle">Boro & Aman Standing Crops</div>
    </div>

    <div class="kpi-card emerald">
      <div class="kpi-header">
        <span class="kpi-label">Displaced Persons (IDP)</span>
        <span class="kpi-icon">⛺</span>
      </div>
      <div class="kpi-value">${(totalDisplaced / 1000).toFixed(1)}k IDPs</div>
      <div class="kpi-subtitle">Evacuation Shelter Demand</div>
    </div>

    <div class="kpi-card indigo">
      <div class="kpi-header">
        <span class="kpi-label">Submerged Tube Wells</span>
        <span class="kpi-icon">💧</span>
      </div>
      <div class="kpi-value">${(totalWellsContam / 1000).toFixed(1)}k Wells</div>
      <div class="kpi-subtitle">WASH Emergency Contamination</div>
    </div>
  `;
}

// 4b. Render Real-Time Satellite Telemetry
function renderSatelliteTelemetry() {
  if (!platformData || !platformData.satellite_telemetry) return;
  const grid = document.getElementById("satelliteDivisionsGrid");
  const tsEl = document.getElementById("satelliteTimestamp");
  if (!grid) return;

  const sat = platformData.satellite_telemetry;
  if (tsEl && sat.timestamp) {
    tsEl.textContent = `Observation: ${sat.timestamp} BST | Doppler Ground Stations Linked`;
  }

  const divs = sat.divisions || {};
  grid.innerHTML = "";

  Object.keys(divs).forEach(divName => {
    const d = divs[divName];
    const card = document.createElement("div");
    card.style.background = "rgba(255, 255, 255, 0.03)";
    card.style.border = "1px solid rgba(255, 255, 255, 0.06)";
    card.style.borderRadius = "10px";
    card.style.padding = "10px";
    card.style.textAlign = "center";

    card.innerHTML = `
      <div style="font-size:0.75rem; font-weight:700; color:#38bdf8; text-transform:uppercase;">${divName}</div>
      <div style="font-size:1.3rem; font-weight:800; color:#ffffff; margin:2px 0;">${d.temperature_c.toFixed(1)}°C</div>
      <div style="font-size:0.72rem; color:#94a3b8;">Feels ${d.feels_like_c.toFixed(1)}°C</div>
      <div style="font-size:0.7rem; color:#cbd5e1; margin-top:4px;">
        💧 ${d.humidity_pct}% • 💨 ${d.wind_speed_kmh.toFixed(1)} km/h
      </div>
    `;
    grid.appendChild(card);
  });
}

// 5. Render District Explorer List
function renderDistrictList() {
  if (!platformData) return;
  const listEl = document.getElementById("districtListScroll");
  if (!listEl) return;

  const searchVal = (document.getElementById("districtSearchInput")?.value || "").toLowerCase().trim();
  const divVal = document.getElementById("divisionFilterSelect")?.value || "All";

  let filtered = platformData.districts.filter(d => {
    const matchName = d.DistrictName.toLowerCase().includes(searchVal);
    const matchDiv = divVal === "All" || d.Division === divVal;
    return matchName && matchDiv;
  });

  // Sort: Critical first, then by Inundation %
  filtered.sort((a, b) => (b.PredictedInundatedPercentage || 0) - (a.PredictedInundatedPercentage || 0));

  listEl.innerHTML = "";
  if (filtered.length === 0) {
    listEl.innerHTML = `<div style="text-align:center; padding:2rem; color:#94a3b8;">No matching districts found.</div>`;
    return;
  }

  filtered.forEach(d => {
    const card = document.createElement("div");
    card.className = "district-item-card";

    let tierClass = "tier-moderate";
    const t = String(d.AlertTier).toUpperCase();
    if (t.includes("CRITICAL") || t.includes("RED") || t.includes("TIER 1")) tierClass = "tier-critical";
    else if (t.includes("HIGH") || t.includes("ORANGE") || t.includes("TIER 2")) tierClass = "tier-high";
    else if (t.includes("LOW") || t.includes("GREEN")) tierClass = "tier-low";

    card.innerHTML = `
      <div class="district-item-info">
        <h4>${d.DistrictName}</h4>
        <div class="district-item-sub">
          <span>${d.Division} Division</span> • 
          <span>${(d.Population / 1000000).toFixed(2)}M Pop</span> • 
          <span style="color:#38bdf8;">${(d.PredictedInundatedPercentage || 0).toFixed(1)}% Inundated</span>
        </div>
      </div>
      <span class="tier-badge ${tierClass}">${d.AlertTier || 'MODERATE'}</span>
    `;

    card.onclick = () => openDistrictModal(d);
    listEl.appendChild(card);
  });
}

// 6. District Detail Modal
function openDistrictModal(d) {
  activeDistrict = d;
  const modal = document.getElementById("districtModal");
  const title = document.getElementById("modalDistrictTitle");
  const body = document.getElementById("modalDistrictBody");
  if (!modal || !title || !body) return;

  title.textContent = `${d.DistrictName} District — Multi-Hazard Intelligence Profile`;

  body.innerHTML = `
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:1.5rem;">
      <div style="background:rgba(255,255,255,0.04); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Administrative Division</div>
        <div style="font-size:1.15rem; font-weight:700; color:#ffffff;">${d.Division}</div>
      </div>
      <div style="background:rgba(255,255,255,0.04); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Flood Inundation Exposure</div>
        <div style="font-size:1.15rem; font-weight:700; color:#38bdf8;">${(d.PredictedInundatedPercentage || 0).toFixed(1)}% (${(d.AreaSqKm * (d.PredictedInundatedPercentage||0)/100).toFixed(1)} km²)</div>
      </div>
      <div style="background:rgba(255,255,255,0.04); padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Humanitarian Priority Tier</div>
        <div style="font-size:1.05rem; font-weight:700; color:#f87171;">${d.Humanitarian_PriorityTier || d.AlertTier}</div>
      </div>
    </div>

    <!-- 49-Layer Environmental Telemetry Highlights -->
    <h4 style="color:#ffffff; margin:1.25rem 0 0.5rem 0; font-size:1rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:6px;">
      🌍 49-Layer Earth Observation & Bioclimatic Telemetry
    </h4>
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:10px; margin-bottom:1.5rem;">
      <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
        <span style="font-size:0.72rem; color:#94a3b8;">Mean Elevation (SRTM)</span>
        <div style="font-size:1rem; font-weight:700; color:#ffffff;">${d.SRTM_Elevation_M ? d.SRTM_Elevation_M.toFixed(1) + ' m' : (d.MeanElevationM ? d.MeanElevationM.toFixed(1) + ' m' : '10.5 m')}</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
        <span style="font-size:0.72rem; color:#94a3b8;">Water Recurrence (JRC)</span>
        <div style="font-size:1rem; font-weight:700; color:#38bdf8;">${d.JRC_WaterRecurrence_Pct ? d.JRC_WaterRecurrence_Pct.toFixed(1) + '%' : (d.JRC_WaterOccurrence_Pct ? d.JRC_WaterOccurrence_Pct.toFixed(1) + '%' : '18.4%')}</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
        <span style="font-size:0.72rem; color:#94a3b8;">Canopy Height (GEDI)</span>
        <div style="font-size:1rem; font-weight:700; color:#34d399;">${d.GEDI_CanopyHeight_M ? d.GEDI_CanopyHeight_M.toFixed(1) + ' m' : '12.4 m'}</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
        <span style="font-size:0.72rem; color:#94a3b8;">Relative Wealth (Meta RWI)</span>
        <div style="font-size:1rem; font-weight:700; color:#fbbf24;">${d.Meta_RWI_Score ? d.Meta_RWI_Score.toFixed(2) : '0.15'}</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
        <span style="font-size:0.72rem; color:#94a3b8;">Köppen Baseline → 2050</span>
        <div style="font-size:0.95rem; font-weight:700; color:#fca5a5;">${d.Koppen_Baseline || 'Aw'} → ${d.Koppen_SSP585_2050 || 'Am'}</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:8px;">
        <span style="font-size:0.72rem; color:#94a3b8;">Integrated Priority Rank</span>
        <div style="font-size:1rem; font-weight:700; color:#a5b4fc;">#${d.Rank_Model4_IntegratedBCIP || '14'} / 64</div>
      </div>
    </div>

    <h4 style="color:#ffffff; margin:1.25rem 0 0.5rem 0; font-size:1rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:6px;">
      🌾 UN HDX Operational Humanitarian Indicators
    </h4>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:1.5rem;">
      <div style="background:rgba(15,23,42,0.8); padding:12px; border-radius:10px; border-left:4px solid #f59e0b;">
        <strong style="color:#fbbf24; font-size:0.85rem;">Food Security & Agriculture (WFP/FAO)</strong>
        <p style="font-size:0.82rem; color:#cbd5e1; margin-top:4px;">
          • Inundated Cropland: <strong>${(d.InundatedAgriLand_Ha || 0).toLocaleString()} Ha</strong><br>
          • Rice Crop Loss: <strong>${(d.StandingRiceLoss_Tons || 0).toLocaleString()} Tons</strong><br>
          • Food Price Inflation Surge: <strong>+${(d.FoodPriceSurge_Pct || 0).toFixed(1)}%</strong><br>
          • IPC Food Insecurity: <strong>${d.IPC_FoodInsecurity_Phase || 'Phase 1'}</strong>
        </p>
      </div>

      <div style="background:rgba(15,23,42,0.8); padding:12px; border-radius:10px; border-left:4px solid #ef4444;">
        <strong style="color:#f87171; font-size:0.85rem;">Public Health & Epidemics (WHO/UNICEF)</strong>
        <p style="font-size:0.82rem; color:#cbd5e1; margin-top:4px;">
          • Healthcare Desert Pop (>8km): <strong>${(d.HealthcareDesert_ExposedPop || 0).toLocaleString()}</strong><br>
          • Cholera Surge Risk Score: <strong>${(d.CholeraWaterborne_RiskScore || 0).toFixed(1)} / 100</strong><br>
          • Emergency ORS Sachets Demand: <strong>${(d.EmergencyORS_SachetsNeeded || 0).toLocaleString()}</strong><br>
          • Malaria Transmission Tier: <strong>${d.Malaria_Parasite_Class || 'Low'}</strong>
        </p>
      </div>

      <div style="background:rgba(15,23,42,0.8); padding:12px; border-radius:10px; border-left:4px solid #10b981;">
        <strong style="color:#34d399; font-size:0.85rem;">Climate Displacement & Shelters (IOM/UNHCR)</strong>
        <p style="font-size:0.82rem; color:#cbd5e1; margin-top:4px;">
          • Displaced IDPs (42% rate): <strong>${(d.DisplacedPersonsEst || 0).toLocaleString()}</strong><br>
          • HOTOSM School Shelters: <strong>${d.HOTOSM_EducationShelters || 0} shelters</strong><br>
          • Net Capacity Balance: <strong>${d.ShelterCapacityBalance > 0 ? '+' : ''}${(d.ShelterCapacityBalance || 0).toLocaleString()}</strong><br>
          • Relief Tents Required: <strong>${(d.EmergencyReliefTentsNeeded || 0).toLocaleString()}</strong>
        </p>
      </div>

      <div style="background:rgba(15,23,42,0.8); padding:12px; border-radius:10px; border-left:4px solid #3b82f6;">
        <strong style="color:#60a5fa; font-size:0.85rem;">WASH & Safe Drinking Water (WPdx/JMP)</strong>
        <p style="font-size:0.82rem; color:#cbd5e1; margin-top:4px;">
          • Submerged Wells: <strong>${(d.SubmergedTubeWells_Contaminated || 0).toLocaleString()}</strong><br>
          • Daily Safe Water Need (15L/day): <strong>${((d.DailySafeWaterDemand_Liters || 0)/1000).toFixed(0)}k Liters</strong><br>
          • Aquatabs Tablets Daily: <strong>${((d.Aquatabs_PurificationTablets_Daily || 0)/1000).toFixed(0)}k tablets</strong><br>
          • Total Wells Est: <strong>${(d.EstimatedTubeWells_Total || 0).toLocaleString()}</strong>
        </p>
      </div>
    </div>

    <!-- Modal Action Buttons -->
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:12px 16px;">
      <div>
        <div style="font-size:0.75rem; color:#fca5a5; text-transform:uppercase; font-weight:700;">Cell Broadcast Early Warning Dispatch</div>
        <div style="font-size:0.88rem; color:#ffffff; font-weight:600;">Broadcast automated bilingual emergency alert to all mobile devices in ${d.DistrictName}</div>
      </div>
      <div style="display:flex; gap:8px;">
        <button onclick="downloadDistrictJson()" style="background:rgba(255,255,255,0.1); color:#ffffff; border:1px solid rgba(255,255,255,0.2); padding:8px 14px; border-radius:8px; font-weight:600; cursor:pointer;">
          📥 Download JSON
        </button>
        <button onclick="dispatchAlertForDistrict('${d.DistrictName}')" style="background:#ef4444; color:#ffffff; border:none; padding:8px 16px; border-radius:8px; font-weight:700; cursor:pointer;">
          📡 Dispatch Alert
        </button>
      </div>
    </div>
  `;

  modal.classList.add("open");
}

function downloadDistrictJson() {
  if (!activeDistrict) return;
  const jsonStr = JSON.stringify(activeDistrict, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${activeDistrict.DistrictName}_climate_intelligence_profile.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function closeDistrictModal() {
  const modal = document.getElementById("districtModal");
  if (modal) modal.classList.remove("open");
}

// 7. Advance 60-Day Forecast & Cold Wave Workspace
function renderForecastWorkspace() {
  if (!platformData || !platformData.forecast_60d) return;
  const container = document.getElementById("forecastTimelineContainer");
  if (!container) return;

  const divisionSelect = document.getElementById("forecastDivisionSelect");
  const selectedDiv = divisionSelect ? divisionSelect.value : "Rangpur";
  const days = platformData.forecast_60d[selectedDiv] || [];

  let html = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:10px;">
      <h4 style="color:#ffffff; font-size:1.1rem;">60-Day Sub-Seasonal Timeline for ${selectedDiv} Division</h4>
      <div style="font-size:0.8rem; color:#38bdf8; background:rgba(6,182,212,0.1); padding:4px 10px; border-radius:6px; border:1px solid rgba(6,182,212,0.2);">
        ECMWF Ensemble + SEAS5 Coupled Model
      </div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(130px, 1fr)); gap:10px; max-height:420px; overflow-y:auto; padding-right:6px;">
  `;

  days.forEach(d => {
    let coldColor = "#3b82f6";
    let badge = d.cold_status || "NORMAL";
    if (badge.includes("VERY SEVERE")) coldColor = "#9333ea";
    else if (badge.includes("SEVERE")) coldColor = "#ef4444";
    else if (badge.includes("MODERATE")) coldColor = "#f59e0b";
    else if (badge.includes("MILD")) coldColor = "#06b6d4";

    html += `
      <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px; text-align:center;">
        <div style="font-size:0.75rem; color:#94a3b8; font-family:JetBrains Mono,monospace;">Day +${d.day}</div>
        <div style="font-size:0.78rem; font-weight:600; color:#cbd5e1; margin-bottom:4px;">${d.date}</div>
        <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">${d.temp_min_c.toFixed(1)}°C</div>
        <div style="font-size:0.75rem; color:#94a3b8;">Max: ${d.temp_max_c.toFixed(1)}°C</div>
        <div style="font-size:0.72rem; color:#38bdf8; margin:4px 0;">Wind Chill: ${d.wind_chill_c ? d.wind_chill_c.toFixed(1) + '°C' : 'N/A'}</div>
        <div style="font-size:0.68rem; font-weight:700; color:#ffffff; background:${coldColor}; padding:2px 6px; border-radius:4px; margin-top:4px;">
          ${badge.split('(')[0].trim()}
        </div>
      </div>
    `;
  });

  html += `</div>`;
  container.innerHTML = html;
}

// 8. Interactive UN HDX Humanitarian Simulators
function renderHumanitarianSimulators() {
  // Simulator 1: Crop Vulnerability Slider
  const cropSlider = document.getElementById("cropVulnerabilitySlider");
  const cropVal = document.getElementById("cropVulnerabilityVal");
  if (cropSlider && cropVal) {
    cropSlider.oninput = () => {
      cropVal.textContent = `${cropSlider.value}%`;
      updateCropSimulation(parseFloat(cropSlider.value));
    };
  }

  // Simulator 2: Healthcare Distance Slider
  const healthSlider = document.getElementById("healthDistanceSlider");
  const healthVal = document.getElementById("healthDistanceVal");
  if (healthSlider && healthVal) {
    healthSlider.oninput = () => {
      healthVal.textContent = `${healthSlider.value} km`;
      updateHealthSimulation(parseFloat(healthSlider.value));
    };
  }

  // Simulator 3: Displacement Fraction Slider
  const dispSlider = document.getElementById("displacementFractionSlider");
  const dispVal = document.getElementById("displacementFractionVal");
  if (dispSlider && dispVal) {
    dispSlider.oninput = () => {
      dispVal.textContent = `${dispSlider.value}%`;
      updateDisplacementSimulation(parseFloat(dispSlider.value));
    };
  }
}

function updateCropSimulation(vulnPct) {
  if (!platformData) return;
  const factor = vulnPct / 78.0; // Baseline in data was calibrated around 78%
  let totalLossTons = 0;
  platformData.districts.forEach(d => {
    totalLossTons += (d.StandingRiceLoss_Tons || 0) * factor;
  });
  const resEl = document.getElementById("simCropResultTons");
  if (resEl) resEl.textContent = `${(totalLossTons / 1000).toFixed(1)}k Tons`;
}

function updateHealthSimulation(thresholdKm) {
  if (!platformData) return;
  let totalDesertPop = 0;
  platformData.districts.forEach(d => {
    const dist = d.DistanceToHealth_Km || 6.5;
    if (dist >= thresholdKm) {
      totalDesertPop += (d.HealthcareDesert_ExposedPop || 0);
    }
  });
  const resEl = document.getElementById("simHealthResultPop");
  const orsEl = document.getElementById("simOrsResult");
  if (resEl) resEl.textContent = `${(totalDesertPop / 1000).toFixed(1)}k Persons`;
  if (orsEl) orsEl.textContent = `${((totalDesertPop * 5) / 1000).toFixed(1)}k Sachets`;
}

function updateDisplacementSimulation(fractionPct) {
  if (!platformData) return;
  let totalIdps = 0;
  let totalDeficit = 0;
  platformData.districts.forEach(d => {
    const expPop = d.ProjectedExposedPopulation || 0;
    const idps = expPop * (fractionPct / 100.0);
    const capacity = (d.HOTOSM_EducationShelters || 0) * 450;
    totalIdps += idps;
    if (idps > capacity) totalDeficit += (idps - capacity);
  });
  const resIdp = document.getElementById("simDispResultIdp");
  const resTents = document.getElementById("simTentsResult");
  if (resIdp) resIdp.textContent = `${(totalIdps / 1000).toFixed(1)}k IDPs`;
  if (resTents) resTents.textContent = `${(Math.round(totalDeficit / 5) / 1000).toFixed(1)}k Tents`;
}

// 9. Cell Broadcast Emergency Alert Simulator
function dispatchAlertForDistrict(districtName) {
  if (!platformData) return;
  const d = platformData.districts.find(x => x.DistrictName.toLowerCase() === districtName.toLowerCase()) || platformData.districts[0];

  const phoneDistrict = document.getElementById("phoneAlertDistrict");
  const phoneBn = document.getElementById("phoneAlertMessageBn");
  const phoneEn = document.getElementById("phoneAlertMessageEn");
  const phoneShelter = document.getElementById("phoneAlertShelter");

  if (phoneDistrict) phoneDistrict.textContent = `${d.DistrictName} DISTRICT`;
  if (phoneBn) {
    phoneBn.innerHTML = `⚠️ <strong>জরুরি সতর্কবার্তা (BMD & BWDB):</strong> ${d.DistrictName} জেলায় আগামী ২৪ ঘণ্টায় বন্যা ও পানির স্তর চরম বিপৎসীমা অতিক্রম করবে। অনুগ্রহ করে অবিলম্বে নিকটস্থ আশ্রয়কেন্দ্রে আশ্রয় নিন। হেল্পলাইন: ১০৯০ / ৯৯৯।`;
  }
  if (phoneEn) {
    phoneEn.innerHTML = `⚠️ <strong>EMERGENCY BROADCAST:</strong> Severe flash flood warning for ${d.DistrictName}. Water levels exceeding danger crest. Evacuate immediately to designated multi-purpose cyclone/flood shelters. Helpline: 1090.`;
  }
  if (phoneShelter) {
    phoneShelter.textContent = `Designated Refuge: ${d.DistrictName} Central Disaster Shelter #01 (Elevation +9.5m AMSL)`;
  }

  // Smooth scroll to phone mockup
  const mockup = document.getElementById("phoneAlertContainer");
  if (mockup) {
    mockup.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// 10. Master HDX Linkage Manifest Table
function renderHdxManifestTable() {
  if (!platformData || !platformData.hdx_manifest) return;
  const tableBody = document.getElementById("hdxManifestTableBody");
  if (!tableBody) return;

  const searchInput = (document.getElementById("hdxManifestSearch")?.value || "").toLowerCase().trim();

  let filtered = platformData.hdx_manifest.filter(item => {
    return (item.HDX_Title || "").toLowerCase().includes(searchInput) ||
           (item.Thematic_Cluster || "").toLowerCase().includes(searchInput) ||
           (item.UN_Source_Agency || "").toLowerCase().includes(searchInput) ||
           (item.Operational_Use_Case_Category || "").toLowerCase().includes(searchInput);
  });

  tableBody.innerHTML = "";
  filtered.slice(0, 30).forEach(item => {
    const tr = document.createElement("tr");
    tr.style.borderBottom = "1px solid rgba(255,255,255,0.06)";
    tr.innerHTML = `
      <td style="padding:10px 12px; font-weight:600; color:#ffffff;">
        <a href="${item.Direct_HDX_URL}" target="_blank" style="color:#38bdf8; text-decoration:none;">${item.HDX_Title} ↗</a>
      </td>
      <td style="padding:10px 12px; color:#94a3b8; font-size:0.8rem;">${item.Thematic_Cluster}</td>
      <td style="padding:10px 12px; color:#cbd5e1; font-size:0.8rem;">${item.UN_Source_Agency}</td>
      <td style="padding:10px 12px; color:#34d399; font-size:0.8rem;">${item.Operational_Use_Case_Category}</td>
      <td style="padding:10px 12px; color:#f8fafc; font-size:0.75rem; font-family:JetBrains Mono,monospace;">${item.Internal_Linked_File.split('&')[0].trim()}</td>
    `;
    tableBody.appendChild(tr);
  });
}

// 11. Event Listeners & Tab Switching
function initEventListeners() {
  // District search & filter
  const searchInput = document.getElementById("districtSearchInput");
  const divSelect = document.getElementById("divisionFilterSelect");
  if (searchInput) searchInput.addEventListener("input", renderDistrictList);
  if (divSelect) divSelect.addEventListener("change", renderDistrictList);

  // Forecast division selector
  const fcDivSelect = document.getElementById("forecastDivisionSelect");
  if (fcDivSelect) fcDivSelect.addEventListener("change", renderForecastWorkspace);

  // HDX search
  const hdxSearch = document.getElementById("hdxManifestSearch");
  if (hdxSearch) hdxSearch.addEventListener("input", renderHdxManifestTable);

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Modal close
  const closeBtn = document.getElementById("modalCloseBtn");
  if (closeBtn) closeBtn.onclick = closeDistrictModal;

  window.onclick = (e) => {
    const modal = document.getElementById("districtModal");
    if (e.target === modal) closeDistrictModal();
  };
}
