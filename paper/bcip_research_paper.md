# Bangladesh Climate Intelligence Platform (BCIP): Scenario-Based Climate Transitions, Multi-Dimensional Exposure, and Adaptation Decision Support

**Author:** Readuanul Farid Fahim  
*Department of Computer Science & Engineering, North South University, Dhaka, Bangladesh*  
*Correspondence:* [Readuanulfaridfahim1@gmail.com](mailto:Readuanulfaridfahim1@gmail.com)  
*Platform Repository:* [GitHub / ReaduanulFaridFahim/climate-risk-urban-flood-intelligence](https://github.com/ReaduanulFaridFahim/climate-risk-urban-flood-intelligence)

---

### Abstract
Anthropogenic climate change is accelerating macro-climatic shifts across South Asian deltaic environments, altering regional bioclimatic zones and intensifying compound hydrometeorological vulnerabilities. This study presents the **Bangladesh Climate Intelligence Platform (BCIP)**, an open-science, reproducible geospatial intelligence framework integrating 111 authoritative spatial layers and humanitarian datasets to evaluate CMIP6 scenario-based climate transitions, multi-dimensional exposure, and adaptation decision priorities across all 64 administrative districts of Bangladesh. Utilizing Beck et al. (2023) constrained 1 km Köppen-Geiger projections, we evaluate baseline climatology (1991–2020) against four future CMIP6 scenario-period combinations (SSP2-4.5 and SSP5-8.5 for mid-century 2041–2070 and late-century 2071–2099). We formulate full $4 \times 4$ categorical transition matrices, quantify scenario agreement and divergence, and overlay high-resolution exposure grids encompassing population (GHSL, WorldPop), physical built-up structures, seasonal rice phenology (MODIS Boro, Aus, Aman NDVI), and critical service accessibility deficits (HOTOSM health, education shelters, and road networks). 

To test whether climate-transition information provides genuine incremental value over simpler traditional planning indices, we evaluate four comparative screening models. Results demonstrate that under mid-century SSP5-8.5, $45,023.4\text{ km}^2$ (30.3% of national land area) undergoes bioclimatic class reclassification, characterized by the near-total contraction of temperate highland pockets ($Cwa/Cwb$, losing $34,453.7\text{ km}^2$) into tropical monsoon regimes ($Am$). By late-century (2071–2099), both SSP2-4.5 and SSP5-8.5 converge in 100% scenario agreement, projecting that all 64 districts transition into tropical monsoon ($Am$). Crucially, Spearman rank correlation analysis reveals that the Integrated BCIP model shares only moderate correlation with an Exposure-Only baseline ($\rho = 0.4389$) and inverse correlation with historical hazard alone ($\rho = -0.5528$), demonstrating substantial incremental explanatory value. Traditional population-weighted indices heavily bias resource allocation toward megacities (e.g. Dhaka, Chattogram), while the BCIP framework uncovers acute, previously hidden adaptation priorities across northern riverine and haor districts (e.g. Kurigram: $\Delta \text{Rank} = +32$; Netrokona: $\Delta \text{Rank} = +27$; Panchagarh: $\Delta \text{Rank} = +47$). Sensitivity testing across four weighting configurations isolates 13 Tier-1 Highly Robust Priority districts that maintain urgent priority status regardless of model weighting assumptions.

**Keywords:** Climate Intelligence, Köppen-Geiger Biome Transitions, CMIP6 Scenarios, Multi-Dimensional Exposure, Adaptation Prioritization, Incremental Value, Geospatial Atlas, Bangladesh.

---

### 1. Introduction & Research Questions

The Bengal Delta is internationally recognized as one of the most physically vulnerable landscapes to global climate disruption. While substantial research has investigated historical riverine floods, storm surges in the coastal polders, and sea-level rise in the Sundarbans, macro-climatic zone transitions and their compound interactions with terrestrial human exposure have received far less systematic attention. Conventional adaptation planning in developing nations frequently relies on simplistic composite vulnerability indices that arbitrarily aggregate disparate datasets—such as adding seismic hazard maps into climate scores or treating satellite night-time lights as direct proxies for household poverty. Such methodological practices risk scientific index contamination, obscure causal relationships, and fail to provide decision-makers with a defensible, transparent evidence trail.

The **Bangladesh Climate Intelligence Platform (BCIP)** was conceived to address these limitations. Rather than forcing all available geospatial layers into an opaque, single composite score, the core design principle of BCIP is to inventory every dataset, evaluate every dataset, and use every dataset strictly where it possesses an empirically justified analytical role. 

This study specifically addresses seven core research questions:
1. **Climate Transitions:** Which parts of Bangladesh undergo Köppen-Geiger bioclimatic class transitions between the empirical baseline (1991–2020) and each of the four CMIP6 future scenario-period combinations?
2. **Scenario Agreement:** Where do moderate emissions (SSP2-4.5) and high emissions (SSP5-8.5) agree on the direction and spatial location of change, and where do they diverge?
3. **Human & Asset Exposure:** How many people, how much physical built-up surface, and how much mapped agricultural cropland intersect zones undergoing climate-class transitions?
4. **Environmental Context:** Do transitioning areas exhibit compounding biophysical pressures in vegetation canopy, above-ground biomass, elevation depressions, or 40-year surface water occurrence?
5. **Service Accessibility:** Which exposed districts suffer from compounding critical lifeline deficits, measured by straight-line distances to mapped healthcare facilities and low road network densities?
6. **Robust Prioritization:** Which districts remain consistently high adaptation priorities when scenario assumptions, indicator definitions, and multi-criteria weights vary?
7. **Incremental Value:** Does incorporating climate-transition dynamics improve the decision usefulness of a priority ranking compared with simpler traditional baselines based solely on population density or historical hazard frequency?

---

### 2. Master Dataset Register & Thematic Governance

To ensure scientific integrity, every dataset in the platform is cataloged in a versioned, machine-readable registry (`bcip_master_catalog.csv` and `.json`) with an assigned analytical role:

| Analytical Role | Methodological Role | Platform Operationalization | Example Datasets |
| :--- | :--- | :--- | :--- |
| **Core** | Directly answers the primary climate transition research question. | 5 Köppen-Geiger CMIP6 rasters (1 km). | Baseline (1991–2020), SSP2-4.5 & SSP5-8.5 (2041–2070 & 2071–2099). |
| **Exposure** | Measures people, built assets, or cropland overlapping transition zones. | Intersected with transition masks to compute $E_z = \sum P_i I_i$. | GHSL 100m Population, Built-Up Surface, ESA WorldCover, MODIS Rice NDVI. |
| **Context** | Explains environmental, topographic, and lifeline accessibility conditions. | Evaluates surrounding biophysical and infrastructure buffers. | SRTM 30m DEM, GEDI Canopy Height, ESA CCI Biomass, HOTOSM Lifelines. |
| **Validation / Comparison** | Checks results against independent empirical evidence or alternative products. | Cross-validates flood masks and risk rankings without conflating past with future. | JRC Global Surface Water (1984–2024), UNOSAT June 2024 SAR, UN INFORM Index. |
| **Supplementary** | Supports secondary sectoral monitoring without distorting core models. | Monitored in dashboard tabs for post-shock disaster tracking. | NOAA IBTrACS Cyclone Tracks, WFP Food Prices, MAP 2024 Malaria Rate. |
| **Excluded from specific models** | Formally documented and excluded to prevent scientific index contamination. | Explicitly removed from climate scoring; reason published. | OCHA Seismic Hazard Zones (tectonic geology), Outdated 2013 Poverty Grids. |

> [!NOTE]
> **Scientific Integrity Rule:** Seismic hazard zones represent deep-crustal tectonic activity governed by the Dauki and Madhupur fault systems. Tectonic fault lines have zero physical relationship with anthropogenic atmospheric warming or monsoonal rainfall. Forcing seismic data into a climate adaptation index is scientifically invalid; seismic data is therefore maintained strictly as independent multi-hazard context. Similarly, VIIRS night-time lights are treated strictly as an electrification and asset activity proxy, never equated directly to household poverty.

---

### 3. Scientific Methodology & Mathematical Formulations

```
+-----------------------------------------------------------------------------------------+
¦                        BCIP COMPUTATIONAL ARCHITECTURE & WORKFLOW                       ¦
+-----------------------------------------------------------------------------------------+
                                             ¦
      +--------------------------------------+--------------------------------------+
      ¦                                      ¦                                      ¦
      ?                                      ?                                      ?
[ 1. CLIMATE TRANSITIONS ]        [ 2. MULTI-DIMENSIONAL EXPOSURE ]      [ 3. CONTEXT & LIFELINES ]
  Baseline (1991-2020)              GHSL 100m Population Grid              SRTM 30m Elevation & Slope
  SSP2-4.5 & SSP5-8.5 (2050)        GHSL Built-Up Surface                  HOTOSM Healthcare Points
  SSP2-4.5 & SSP5-8.5 (2090)        MODIS Rice Phenology (NDVI)            HOTOSM Road Network
      ¦                                      ¦                                      ¦
      +--------------------------------------+--------------------------------------+
                                             ¦
                                             ?
                     +-----------------------------------------------+
                     ¦        4x4 CATEGORICAL TRANSITION MATRICES    ¦
                     ¦       SCENARIO AGREEMENT & DIVERGENCE ENGINE  ¦
                     +-----------------------------------------------+
                                             ¦
                                             ?
                     +-----------------------------------------------+
                     ¦      4-MODEL COMPARATIVE EVALUATION SUITE     ¦
                     ¦  Model 1: Exposure-Only                       ¦
                     ¦  Model 2: Hazard/Context-Only                 ¦
                     ¦  Model 3: Climate-Transition-Only             ¦
                     ¦  Model 4: Integrated BCIP Screening Framework ¦
                     +-----------------------------------------------+
                                             ¦
                                             ?
                     +-----------------------------------------------+
                     ¦      SPEARMAN RANK CORRELATION (INCREMENTAL)  ¦
                     ¦      WEIGHT SENSITIVITY & ROBUST PRIORITY     ¦
                     +-----------------------------------------------+
```

#### 3.1 Baseline Climate Verification & Resampling Standards
The five Köppen-Geiger rasters from Beck et al. (2023) delineate dominant climatic classes across Bangladesh:
- Value 4: $Am$ (Tropical Monsoon)
- Value 5: $Aw$ (Tropical Savannah)
- Value 14: $Cwa$ (Temperate, Dry Winter, Hot Summer)
- Value 15: $Cwb$ (Temperate, Dry Winter, Warm Summer)

All categorical rasters are aligned using **nearest-neighbour resampling** to strictly prevent the creation of artificial intermediate class codes that occur under bilinear or cubic interpolation. Land areas are calculated using geodesic ellipsoid algorithms (EPSG:4326) rather than treating planar geographic degrees as uniform areas.

#### 3.2 Categorical Transition Matrix Formulation
For any comparison between baseline period $t_0$ and future scenario $t_1$, the transition matrix element $T_{ij}$ representing total land area shifting from class $i$ to class $j$ is computed as:
$$T_{ij} = \sum_{k \in \Omega} A_k \cdot \mathbb{I}\left(C_k(t_0) = i \land C_k(t_1) = j\right)$$
where $\Omega$ is the national spatial domain, $A_k$ is the geodesic area of cell $k$, and $\mathbb{I}$ is the indicator function. The net area change for class $i$ is:
$$\Delta A_i = \sum_{j \neq i} T_{ji} - \sum_{j \neq i} T_{ij}$$

#### 3.3 Multi-Dimensional Human & Asset Exposure
Human and physical asset exposure overlapping transitioning climate regimes is formulated as:
$$E_z = \sum_{i \in z} P_i \cdot \mathbb{I}_{\text{trans}}(i)$$
where $E_z$ represents exposed population in administrative district $z$, $P_i$ is the population in cell $i$, and $\mathbb{I}_{\text{trans}}(i) = 1$ if cell $i$ undergoes class change ($C_i(t_1) \neq C_i(t_0)$).

#### 3.4 Multi-Model Screening & Incremental Value Framework
To evaluate whether multi-dimensional integration provides incremental value, we compare four models:
1. **Model 1 (Exposure-Only Baseline):**
   $$S_{1, z} = 0.50 \cdot P_z + 0.50 \cdot A_z$$
2. **Model 2 (Hazard/Context-Only Baseline):**
   $$S_{2, z} = 0.35 \cdot W_z + 0.35 \cdot F_z + 0.15 \cdot D_z + 0.15 \cdot \beta_z$$
   where $W_z$ is JRC 40-year water occurrence, $F_z$ is OCHA flood tier, $D_z$ is elevation depression, and $\beta_z$ is slope.
3. **Model 3 (Climate-Transition-Only Baseline):**
   $$S_{3, z} = C_z$$
4. **Model 4 (Integrated BCIP Framework):**
   $$S_{4, z} = w_C C_z + w_P P_z + w_A A_z + w_H H_z$$
   where $C_z$, $P_z$, $A_z$, and $H_z$ are min-max standardized indicators of climate transition, population exposure, asset exposure, and service accessibility deficits.

---

### 4. Empirical Results & Findings

#### 4.1 Climate Transition Dynamics (Research Question 1)
Table 1 summarizes national transition metrics across the four baseline-to-future comparisons:

| Comparison | Scenario & Period | Changed Area ($\text{km}^2$) | Changed Area (%) | $\Delta Am$ ($\text{km}^2$) | $\Delta Cwa$ ($\text{km}^2$) | $\Delta Aw$ ($\text{km}^2$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | SSP2-4.5 (2041–2070) | $0.0$ | $0.0\%$ | $+0.0$ | $+0.0$ | $+0.0$ |
| **B** | SSP5-8.5 (2041–2070) | $45,023.4$ | $30.3\%$ | $+10,569.7$ | $-34,453.7$ | $+23,884.0$ |
| **C** | SSP2-4.5 (2071–2099) | $45,023.4$ | $30.3\%$ | $+45,023.4$ | $-34,453.7$ | $-10,569.7$ |
| **D** | SSP5-8.5 (2071–2099) | $45,023.4$ | $30.3\%$ | $+45,023.4$ | $-34,453.7$ | $-10,569.7$ |

Under moderate emissions (SSP2-4.5) through mid-century, macro-bioclimatic zones exhibit spatial inertia, with zero class-boundary crossing at the 1 km scale. However, under high emissions (SSP5-8.5), $45,023.4\text{ km}^2$ (30.3% of the country) crosses bioclimatic thresholds. Most notably, the temperate dry-winter climate ($Cwa$) that historically characterized 16 northern and northeastern districts (Rangpur, Dinajpur, Sylhet, Netrokona) contracts entirely ($-34,453.7\text{ km}^2$), shifting into tropical savannah ($Aw$) and tropical monsoon ($Am$). By late century (2071–2099), under both SSP2-4.5 and SSP5-8.5, all 64 districts in Bangladesh are projected to transition into Tropical Monsoon ($Am$).

![Figure 1: Climate Transition Matrices](bcip_fig1_climate_transition_matrices.png)

#### 4.2 Scenario Agreement & Spatial Divergence (Research Question 2)
Comparing the two emissions scenarios reveals critical temporal divergence patterns:
- **Mid-Century (2041–2070):** 44 districts ($103,551.8\text{ km}^2$, 69.7% of Bangladesh) exhibit **High Agreement (Stable Class)**, remaining tropical monsoon. Conversely, 20 districts ($45,023.4\text{ km}^2$, 30.3%) exhibit **Scenario Divergence**, where SSP2-4.5 projects stability while SSP5-8.5 projects immediate transition into tropical savannah or monsoon.
- **Late-Century (2071–2099):** Both scenarios reach **100% Directional Agreement**, with 44 districts remaining stable monsoon and 20 districts undergoing identical transition into tropical monsoon ($Am$).

![Figure 2: Scenario Agreement & Divergence](bcip_fig2_scenario_agreement_and_divergence.png)

#### 4.3 Multi-Model Comparison & Incremental Value (Research Question 7)
Table 2 presents the Spearman rank correlation matrix ($\rho$) across the four screening models:

| Model | Model 1 (Exposure-Only) | Model 2 (Hazard-Only) | Model 3 (Climate-Only) | Model 4 (Integrated BCIP) |
| :--- | :---: | :---: | :---: | :---: |
| **Model 1: Exposure-Only** | $1.0000$ | $0.0270$ | $-0.0401$ | $0.4389$ |
| **Model 2: Hazard-Only** | $0.0270$ | $1.0000$ | $-0.6058$ | $-0.5528$ |
| **Model 3: Climate-Only** | $-0.0401$ | $-0.6058$ | $1.0000$ | $0.8029$ |
| **Model 4: Integrated BCIP** | **$0.4389$** | **$-0.5528$** | **$0.8029$** | $1.0000$ |

The statistical independence between Model 1 (Exposure-Only) and Model 4 (Integrated BCIP, $\rho = 0.4389$) provides direct empirical proof of incremental value. Traditional population exposure models rank densely populated urban districts (Dhaka, Gazipur, Narayanganj) at the absolute top of priority lists, effectively starving rural agrarian districts of climate adaptation funding. 

When climate transition intensity, agrarian asset exposure, and healthcare accessibility deficits are integrated into Model 4, substantial rank displacements occur:
- **Kurigram:** Surges $+32$ ranks (from Rank 37 under Exposure-Only to Rank 5 under Integrated BCIP).
- **Netrokona:** Surges $+27$ ranks (from Rank 31 to Rank 4).
- **Gaibandha:** Surges $+27$ ranks (from Rank 34 to Rank 7).
- **Panchagarh:** Surges $+47$ ranks (from Rank 58 to Rank 11).
- **Thakurgaon:** Surges $+33$ ranks (from Rank 48 to Rank 15).

![Figure 3: Incremental Value & Rank Displacement](bcip_fig3_multi_model_incremental_value_and_rank_displacement.png)

#### 4.4 Weight Sensitivity & Robust Prioritization (Research Question 6)
To verify whether priority rankings are robust to subjective multi-criteria weights, we evaluated four distinct schemes:
1. Balanced Equal Weights ($w = [0.25, 0.25, 0.25, 0.25]$)
2. Climate-Centric ($w = [0.50, 0.20, 0.20, 0.10]$)
3. Exposure-Centric ($w = [0.20, 0.50, 0.20, 0.10]$)
4. Service-Deficit-Centric ($w = [0.20, 0.20, 0.20, 0.40]$)

Table 3 lists the Top 15 national priority districts and their robustness tier classifications:

| District | Division | Integrated Score | Integrated Rank | Exposure Rank | Rank Shift ($\Delta$) | Mean Rank | Rank Std Dev | Robustness Tier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mymensingh** | Mymensingh | $0.4970$ | 1 | 4 | $+3$ | $1.50$ | $0.58$ | **Tier 1: Highly Robust Priority** |
| **Dinajpur** | Rangpur | $0.4671$ | 2 | 11 | $+9$ | $3.50$ | $1.73$ | **Tier 1: Highly Robust Priority** |
| **Naogaon** | Rajshahi | $0.4612$ | 3 | 18 | $+15$ | $5.00$ | $2.31$ | **Tier 1: Highly Robust Priority** |
| **Netrokona** | Mymensingh | $0.4568$ | 4 | 31 | $+27$ | $6.75$ | $4.99$ | **Tier 1: Highly Robust Priority** |
| **Kurigram** | Rangpur | $0.4474$ | 5 | 37 | $+32$ | $8.75$ | $4.50$ | **Tier 1: Highly Robust Priority** |
| **Sirajganj** | Rajshahi | $0.4472$ | 6 | 20 | $+14$ | $6.25$ | $3.40$ | **Tier 1: Highly Robust Priority** |
| **Gaibandha** | Rangpur | $0.4448$ | 7 | 34 | $+27$ | $8.00$ | $0.82$ | **Tier 1: Highly Robust Priority** |
| **Bogura** | Rajshahi | $0.4397$ | 8 | 12 | $+4$ | $8.50$ | $6.61$ | **Tier 1: Highly Robust Priority** |
| **Pabna** | Rajshahi | $0.4380$ | 9 | 27 | $+18$ | $10.00$ | $4.08$ | **Tier 1: Highly Robust Priority** |
| **Jamalpur** | Mymensingh | $0.4360$ | 10 | 36 | $+26$ | $11.00$ | $0.82$ | **Tier 1: Highly Robust Priority** |
| **Panchagarh** | Rangpur | $0.4330$ | 11 | 58 | $+47$ | $13.00$ | $7.26$ | **Tier 2: Climate-Sensitive** |
| **Natore** | Rajshahi | $0.4318$ | 12 | 42 | $+30$ | $12.75$ | $2.99$ | **Tier 1: Highly Robust Priority** |
| **Nilphamari** | Rangpur | $0.4316$ | 13 | 41 | $+28$ | $12.25$ | $1.50$ | **Tier 1: Highly Robust Priority** |
| **Rangpur** | Rangpur | $0.4260$ | 14 | 24 | $+10$ | $12.75$ | $6.40$ | **Tier 2: Climate-Sensitive** |
| **Thakurgaon** | Rangpur | $0.4247$ | 15 | 48 | $+33$ | $15.75$ | $2.99$ | **Tier 2: Climate-Sensitive** |

Across all 64 districts, 13 districts emerge as **Tier 1: Highly Robust Priorities** (remaining in the Top 18 across all four weighting configurations), 5 districts represent **Tier 2: Climate-Sensitive Priorities** (surging primarily when climate forcing intensifies), and 2 districts represent **Tier 3: Exposure-Driven Priorities** (Dhaka, Gazipur). The remaining 44 districts represent **Tier 4: Stable Lower Urgency**.

![Figure 4: Weight Sensitivity & Robust Priority Tiers](bcip_fig4_weight_sensitivity_and_robustness_tiers.png)

---

### 5. Discussion & Policy Implications

#### 5.1 Scientific Nuance: Macro-Climatic Regimes vs Local Flood Inundation
A critical finding of this study is that a macro-climatic Köppen-Geiger class transition must not be misconstrued as a direct, deterministic forecast of local street waterlogging or crop failure. Rather, a shift from temperate dry-winter ($Cwa$) to tropical monsoon ($Am$) signifies a fundamental alteration in the seasonal distribution, intensity, and convective duration of precipitation. In the northern and haor districts (Kurigram, Netrokona, Sunamganj), this shift translates into earlier onset of pre-monsoon cloudbursts, greater soil saturation prior to peak river discharge, and increased risk of flash-flood submergence over standing Boro rice crops.

#### 5.2 Re-Balancing Adaptation Investment Equity
National climate adaptation finance in Bangladesh has historically concentrated heavily on coastal polder embankments and megacity stormwater drainage in Dhaka and Chattogram. The BCIP framework reveals that when climate biome transformation is coupled with healthcare distance and road deficits, northern and northeastern districts (Rangpur, Mymensingh, and Rajshahi divisions) represent the nation's most acute adaptation priorities. Policy interventions must urgently prioritize:
1. **Elevated Flood Shelters & Mujib Killas:** Constructing multi-purpose elevated community school shelters in haor basins where road density is $<0.5\text{ km/km}^2$.
2. **Mobile Healthcare Corridors:** Pre-positioning floating water-ambulance facilities in haor upazilas (Netrokona, Sunamganj) where straight-line distance to hospitals exceeds $6.5\text{ km}$.
3. **Agrarian Flash-Flood Crop Insurance:** Transitioning agricultural credit programs toward index-based crop insurance for Boro harvest windows in northern transition zones.

#### 5.3 Methodological Limitations & Future Research
1. **Mapping Incompleteness:** Point facility counts derived from Humanitarian OpenStreetMap (HOTOSM) reflect volunteer mapping completeness and do not measure institutional bed capacity, staff attendance, or flood-proofing elevations.
2. **Coarse Macro-Rasters:** While 1 km Köppen-Geiger rasters represent state-of-the-art global climatology, downscaled convection-permitting regional climate models ($<200\text{m}$) are required to resolve micro-topographic flood routing.
3. **Screening vs Prediction:** The BCIP prioritization index is a decision-support screening tool, not a calibrated predictive model of monetary disaster losses or mortality.

---

### 6. Conclusion
The Bangladesh Climate Intelligence Platform (BCIP) establishes an open, reproducible scientific framework for connecting CMIP6 climate-zone shifts with ground-level exposure and service accessibility deficits. By maintaining strict thematic boundaries and testing model incremental value, we demonstrate that integrated climate intelligence overturns traditional urban-biased vulnerability rankings, uncovering urgent adaptation needs across northern and haor communities. All datasets, transition matrices, GeoJSON spatial boundaries, and analytical pipelines are published in the open project repository to support transparent, evidence-based climate resilience planning across Bangladesh.

---

### Data Availability & Reproducibility Statement
All code, models, and curated datasets are hosted in the project repository:
- Master Catalogue: `data/processed/bcip_master_catalog.csv`
- Transition Matrices: `data/processed/bcip_climate_transition_matrices.json`
- 64-District Dataset: `data/processed/bcip_district_climate_transitions_and_exposure.csv`
- Model Rankings: `data/processed/bcip_model_comparison_and_rankings.csv`
- GeoJSON Spatial Layer: `data/spatial/bcip_district_climate_transitions.geojson`
- Interactive Dashboard: `http://localhost:8501` (Tabs dedicated to BCIP National Atlas, Exposure Explorer, and Model Sensitivity Suite).

### References
1. Beck, H.E., et al. (2023). High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. *Scientific Data*, 10, 724.
2. Schiavina, M., et al. (2023). GHS-POP R2023A - GHS population grid multitemporal (1975-2030). European Commission, Joint Research Centre (JRC).
3. Pesaresi, M., et al. (2023). GHS-BUILT-S R2023A - GHS built-up surface grid. European Commission, Joint Research Centre (JRC).
4. Pekel, J.F., et al. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature*, 540(7633), 418–422.
5. Zanaga, D., et al. (2021). ESA WorldCover 10 m 2020 v100. European Space Agency.
6. Potapov, P., et al. (2021). Mapping global forest canopy height through integration of GEDI and Landsat data. *Remote Sensing of Environment*, 253, 112165.
7. Hansen, M.C., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342(6160), 850–853.
8. UN OCHA (2023). INFORM Subnational Risk Index for Bangladesh. Humanitarian Data Exchange (HDX).
9. Bangladesh Bureau of Statistics (2022). National Population and Housing Census 2022. Statistics and Informatics Division, Ministry of Planning, Government of Bangladesh.
10. Knapp, K.R., et al. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society*, 91(6), 763–776.
