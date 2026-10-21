# Power BI Dashboard Specification & Wireframe Blueprint
## Climate Risk & Urban Flood Intelligence for Bangladesh (Chattogram Study)

### ?? Page 1: Executive Climate Resilience & Exposure Overview
- **Header KPI Cards:**
  - Total Exposed Population across latest event (`324,500`)
  - Mean Inundation Depth (`1.35 m`)
  - Total Inundated Land (`42.8 sq km`)
  - Direct Economic Risk Exposure (`$184.2M USD`)
- **Visuals:**
  - *Left (50%):* Map visual with graduated bubble sizes showing Ward Exposed Population colored by Compound Flood Risk Score.
  - *Top Right (50%):* Clustered bar chart of Top 10 High-Risk Wards (Bakolia, Agrabad, Halishahar, Chandgaon).
  - *Bottom Right (50%):* Historical Event Comparison slicer and line graph of Peak Rainfall vs Inundated Area (2001–2025).

### ??? Page 2: Spatial Hazard & Geomorphic Vulnerability Atlas
- **Visuals:**
  - *Top Left:* Scatter plot of Mean Elevation (m) vs Inundation Percentage colored by Distance to Karnaphuli River.
  - *Bottom Left:* Inundation Duration by Primary Mechanism (Tidal Congestion, Canal Overbanking, Flash Runoff).
  - *Right (50%):* Interactive Choropleth Map with layer toggles (Slope %, Topographic Wetness Index, Mean Flood Depth).

### ??? Page 3: Decadal Urban Expansion & Retention Loss (2001–2025)
- **Visuals:**
  - *Top Left:* 100% Stacked Area Chart showing LULC transition (Built-up, Vegetation, Water Bodies, Bare Soil) over 2001–2025.
  - *Top Right:* Scatter plot: Impervious Surface Growth Rate (%) vs Exposed Population Growth (Demonstrating empirical correlation).
  - *Bottom:* Ward-level matrix table with conditional formatting heatmaps for Retention Loss Index.

### ?? Page 4: Scenario Forecasting & Urban Drainage Planning Sandbox
- **What-If Parameter Sliders:**
  - Extreme Rainfall Increase Slider (`+0% to +35%`)
  - Built-up Impervious Growth Slider (`+0% to +40%`)
  - Canal Excavation & Tidal Gate Efficiency (`-0% to -30%`)
- **Simulated Outputs:**
  - Projected Exposed Population under SSP2-4.5 vs SSP5-8.5
  - High-priority intervention wards flagged for stormwater retention reservoirs and sluice gate automation.
