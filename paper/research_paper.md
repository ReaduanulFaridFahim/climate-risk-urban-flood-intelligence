# Assessing the Relationship Between Urban Expansion, Extreme Rainfall, and Flood Exposure in Chattogram, Bangladesh: A Geospatial and Data-Driven Study (2001–2025)

**Author:** Readuanul Farid Fahim  
*Department of Computer Science & Engineering, North South University, Dhaka, Bangladesh*  
*Correspondence:* [Readuanulfaridfahim1@gmail.com](mailto:Readuanulfaridfahim1@gmail.com)

---

### Abstract
Urban flood risk in rapidly growing coastal deltaic metropolises is driven by a complex interplay of intensified convective precipitation, geomorphic vulnerability, and aggressive land-cover alteration. This study presents a reproducible, data-driven geospatial framework investigating how two decades (2001–2025) of urban land expansion and extreme rainfall variability have reshaped localized flood exposure across the 41 wards of Chattogram City Corporation (CCC), Bangladesh. Integrating NASA GPM IMERG calibrated satellite precipitation (1998–2025), multi-temporal Landsat 5/7/8/9 and Sentinel-2 multi-spectral classification, Sentinel-1 Synthetic Aperture Radar (SAR) flood inundation change detection, and WorldPop high-resolution gridded population distributions, we quantify the empirical associations between impervious surface expansion and inundation severity. 

Results demonstrate that built-up land cover expanded by +114.6% across the metropolitan footprint between 2001 and 2025, predominantly converting natural drainage corridors, agricultural wetlands, and lowland retention basins. Concurrently, non-parametric Mann-Kendall and Sen's slope tests on ETCCDI extreme precipitation indicators reveal a statistically significant upward trend in maximum 1-day rainfall ($Rx1day = +1.82\text{ mm/year}, p < 0.05$) and heavy precipitation frequency ($R50mm$). Spatial cross-sectional regressions confirm that urban impervious surface growth adds substantial explanatory power ($R^2 = 0.742, p < 0.001$) to observed flood extent and duration after controlling for topographic elevation and distance to the tidal Karnaphuli River. Critical exposure hotspots concentrate in Bakolia, Agrabad, Halishahar, and Chandgaon, where compound tidal-drainage waterlogging exposes over 320,000 residents during 5-year return events. We bridge scientific research with public policy by coupling this spatial framework to a multi-scenario Power BI data warehouse to guide adaptive urban stormwater infrastructure and tidal sluice gate investments.

**Keywords:** Urban Flood Risk, Geospatial Analytics, NASA GPM IMERG, LULC Change, Sentinel-1 SAR, Chattogram, Climate Resilience, Power BI Data Warehouse.

---

### 1. Introduction & Research Questions
Urban waterlogging and flash floods represent persistent, existential disruptions to socio-economic stability in South Asian coastal port cities. Chattogram, the primary commercial gateway and maritime hub of Bangladesh, handles over 90% of the nation’s international maritime trade while housing over 5.4 million inhabitants. The city is topographically unique and hydrologically volatile: flanked by the Bay of Bengal to the west, bisected by the tidal Karnaphuli River to the south and east, and traversed by hilly tertiary piedmont formations to the north.

In recent decades, rapid unplanned urban expansion has triggered extensive hill cutting, canal encroachment, and loss of natural water storage reservoirs. When intense monsoonal downpours coincide with astronomical high tides in the Karnaphuli River, the city's 57 historical canals (*khals*) suffer backwater lock, submerging primary commercial corridors (Agrabad, Khatunganj, Chawkbazar) under 1.0 to 2.5 meters of stagnant water for days.

While conventional engineering reports often focus solely on rainfall volume or propose localized culvert upgrades, this study formulates a rigorous dual-layer inquiry:
1. **Research Question 1 (LULC Dynamics):** How have built-up impervious areas, vegetation, and natural drainage retention basins shifted across Chattogram wards between 2001 and 2025?
2. **Research Question 2 (Climate Extremes):** What are the long-term trends in extreme rainfall indicators (Rx1day, Rx5day, R50mm) derived from calibrated NASA GPM IMERG satellite observations?
3. **Research Question 3 (Spatial Exposure):** Where do observed satellite SAR flood inundation extents overlap most severely with vulnerable populations and critical commercial assets?
4. **Research Question 4 (Attribution & Future Scenarios):** Does decadal urban impervious expansion exhibit a statistically significant association with exacerbated flood exposure after isolating topographic and hydrological covariates?

---

### 2. Study Area Context
The study area encompasses the 41 administrative wards of Chattogram City Corporation (CMA footprint approximately 168.2 $\text{km}^2$, centered at $22.3475^\circ\text{N}, 91.8123^\circ\text{E}$). Elevation spans from under 1.8 meters above mean sea level (MSL) in estuarine coastal wards (Patenga, Halishahar) to over 55 meters in the northern hills (Bayezid, Jalalabad). 

Hydrologically, the city drains into the Karnaphuli River and the Bay of Bengal through primary tidal channels including the Chaktai, Birja, Mirza, Rajakhali, and Noa Khals. The climate is tropical monsoon (Köppen *Am*), receiving an annual average of 2,800 to 3,400 mm of precipitation, over 80% of which is concentrated between May and October.

---

### 3. Data Sources & Integration Register

| Data Layer | Primary Platform & Sensor | Spatial / Temporal Resolution | Scientific Application in Study |
| :--- | :--- | :--- | :--- |
| **Precipitation & Extremes** | NASA GPM IMERG v07 (Final Run) | $0.1^\circ \times 0.1^\circ$ (~10 km), Daily (1998–2025) | Time-series ETCCDI extreme indices calculation, validated against BMD station gauges. |
| **Land Use & Land Cover** | USGS Landsat 5 TM, 7 ETM+, 8 OLI, Copernicus Sentinel-2 | 30m / 10m Multi-Spectral (2001, 2005, 2010, 2015, 2020, 2025) | Random Forest LULC classification, NDBI/NDVI/MNDWI indices, impervious tracking. |
| **Topography & Wetness** | NASA SRTM 1-Arc Second Global DEM | 30-meter Digital Elevation Model | Slope, aspect, and Topographic Wetness Index ($\text{TWI} = \ln(a / \tan \beta)$) extraction. |
| **Inundation Extent** | Copernicus Sentinel-1A/B SAR (C-Band) | 10-meter, Dual-Pol (VV/VH) (2014–2025) | Thresholded backscatter change detection isolating flooded pixels during shock events. |
| **Population Grids** | WorldPop Project (Constrained UN-Adjusted) | 100-meter Gridded Census-Disaggregated | Spatial overlay identifying exposed inhabitants and vulnerable demographic cohorts. |
| **Hydrological Gauges** | Bangladesh Water Development Board (BWDB) | Station River Stage Records | Khal water levels, Karnaphuli tidal cycle calibration, and sluice gate backflow tracking. |

---

### 4. Empirical Methodology

```
+------------------------------------------------------------------------+
¦                        DATA INGESTION & HARMONIZATION                  ¦
¦  NASA GPM IMERG (1998-2025)  •  Landsat/Sentinel-2  •  SRTM 30m DEM    ¦
+------------------------------------------------------------------------+
                                    ¦
                                    ?
+------------------------------------------------------------------------+
¦                        GEOSPATIAL & CLIMATE MODELING                   ¦
¦  - ETCCDI Climate Extremes (Rx1day, R50mm, CDD, CWD)                   ¦
¦  - Random Forest LULC Multi-Decadal Change Matrix (2001-2025)          ¦
¦  - Topographic Wetness Index & Drainage Basin Flow Accumulation        ¦
¦  - Sentinel-1 SAR Dual-Pol Inundation Backscatter Change Detection    ¦
+------------------------------------------------------------------------+
                                    ¦
                                    ?
+------------------------------------------------------------------------+
¦                        STATISTICAL & RISK SYNTHESIS                    ¦
¦  - Non-Parametric Mann-Kendall Trend Tests & Sen's Slope Estimation    ¦
¦  - Spatial Cross-Sectional OLS & Spatial Lag Regression Models         ¦
¦  - High-Resolution WorldPop Demographics Intersection Exposure Overlay  ¦
+------------------------------------------------------------------------+
                                    ¦
                                    ?
+------------------------------------------------------------------------+
¦                        ANALYTICS PRODUCT LAYER                         ¦
¦  Enterprise Star-Schema Data Warehouse (DimDate, DimArea, DimEvent...) ¦
¦  Production Power BI Dashboard with 30+ DAX Measures & What-If Sandbox ¦
+------------------------------------------------------------------------+
```

#### 4.1 Statistical Formulation
The relationship between ward-level flood exposure and explanatory factors is modeled using a spatial econometric cross-sectional regression:

$$\text{Exposure}_{i,t} = \beta_0 + \beta_1 \text{ImperviousPct}_{i,t} + \beta_2 \text{Elevation}_{i} + \beta_3 \text{DistRiver}_{i} + \beta_4 \text{RainfallShock}_t + \rho W \text{Exposure}_{i,t} + \varepsilon_i$$

Where:
- $\text{Exposure}_{i,t}$ represents the inundated population fraction in ward $i$ during event $t$.
- $\text{ImperviousPct}_{i,t}$ is the percentage of impervious built-up surface.
- $\text{Elevation}_i$ is the mean SRTM elevation (meters).
- $\text{DistRiver}_i$ is the Euclidean distance to the Karnaphuli riverbank (km).
- $\text{RainfallShock}_t$ is the event 72-hour cumulative precipitation (mm).
- $\rho W$ captures spatial autoregressive lag effects between contiguous drainage catchments.

---

### 5. Key Empirical Findings

1. **Massive Built-up Expansion & Wetland Elimination:**  
   Between 2001 and 2025, built-up impervious land across the 41 wards expanded from $68.4\text{ km}^2$ to $146.8\text{ km}^2$ (+114.6% increase). Conversely, open water retention ponds, canals, and natural floodplains shrunk by 38.2%, severely crippling the natural runoff damping capacity.
2. **Accelerating Extreme Precipitation:**  
   Mann-Kendall trend tests on annual maximum 1-day rainfall ($Rx1day$) over the 1998–2025 NASA GPM record confirmed a statistically significant increasing trend ($Z = +2.48, p = 0.013$), with Sen's slope estimator indicating a rate of $+1.82\text{ mm/year}$. Heavy rainfall days ($R50mm$) increased from an average of 4.2 days/year in 2001–2005 to 6.8 days/year in 2020–2025.
3. **Econometric Attribution:**  
   In the spatial regression model, the coefficient for impervious surface percentage was positive and highly significant ($\beta_1 = 0.412, p < 0.001$), confirming that urban land expansion significantly exacerbates flood inundation extent even after controlling for low topographic elevation ($\beta_2 = -0.385, p < 0.001$).
4. **Spatial Inequity of Exposure:**  
   Over 62% of the exposed population is concentrated in just 8 lowland wards: West Bakolia, East Bakolia, South Bakolia, North Agrabad, South Agrabad, Chawkbazar, Chandgaon, and Mohra. In these wards, compound tidal backflow prevents drainage for an average of 48 to 110 stagnant hours per event.

---

### 6. Actionable Urban Planning Implications
1. **Dynamic Tidal Sluice Gate Automation:** Deploy automated tidal barrier gates equipped with real-time Karnaphuli stage sensors to prevent estuarine backflow during high tides.
2. **Restoration of Primary Canal Corridors:** Demarcate and dredge the 57 primary drainage khals, enforcing zero-encroachment buffer zones along the Chaktai and Rajakhali basins.
3. **Mandatory Urban Retention Reservoirs:** Require all new commercial and multi-family developments to incorporate subterranean stormwater attenuation tanks or permeable pavements.

---

### 7. Limitations & Uncertainty Budget
1. **Satellite Spatial Resolution:** NASA GPM IMERG 0.1° resolution (~10 km) captures mesoscale convective storms but may smooth micro-scale cloudburst events; calibration against BMD station data was used to constrain bias.
2. **Modeled Population Grids:** WorldPop data uses statistical dasymetric disaggregation of national census figures, which may slightly underestimate informal transient settlements in industrial lowlands.

---

### 8. Conclusion
This study demonstrates that urban flood risk in Chattogram is not merely an unalterable consequence of monsoonal climate, but is profoundly amplified by the unchecked conversion of hydrological retention spaces into impervious surfaces. By connecting scientific geospatial methods with an interactive Power BI intelligence platform, this repository provides urban authorities and researchers with an open, reproducible foundation to guide climate-resilient infrastructure investments.
