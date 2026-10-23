# 📚 BCIP Data Dictionary & Schema Specification

This document details the column definitions, data types, units, and scientific interpretations for all primary datasets in the platform.

---

## 1. Master Dataset Register (`bcip_master_catalog.csv`)

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `DatasetID` | String | Unique identifier (e.g. `BCIP-CLI-001`). |
| `Title` | String | Formal descriptive title of the dataset. |
| `Theme` | String | BISTO / HDX thematic domain. |
| `Provider` | String | Authoritative publishing institution. |
| `SourceURL` | String | Direct canonical URL. |
| `License` | String | Legal open-access license (e.g. CC BY 4.0, Public Domain). |
| `AnalyticalRole` | Enum | Analytical role: `Core`, `Exposure`, `Context`, `Validation / Comparison`, `Supplementary`, `Excluded from a specific model`. |
| `InclusionExclusionDecision` | String | Formal inclusion/exclusion decision. |
| `Justification` | String | Scientific rationale for role assignment. |
| `FileChecksum` | String | 16-character SHA-256 deterministic verification checksum. |

---

## 2. 64-District Climate Transitions & Exposure (`bcip_district_climate_transitions_and_exposure.csv`)

| Field Name | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `DistrictID` | Integer | ID | Official BBS administrative district identifier (1 to 64). |
| `DistrictName` | String | Name | Official statutory English district name. |
| `Division` | String | Name | Administrative division (8 divisions). |
| `Latitude` | Float | Deg N | Centroid latitude (WGS 84). |
| `Longitude` | Float | Deg E | Centroid longitude (WGS 84). |
| `AreaSqKm` | Float | $\text{km}^2$ | Official geographic land area. |
| `Population` | Integer | Persons | Enumerated population (BBS Census 2022 ground truth). |
| `Koppen_Baseline` | String | Class | Dominant Köppen-Geiger class (1991–2020 baseline). |
| `Koppen_SSP585_2050` | String | Class | Projected class under SSP5-8.5 (2041–2070). |
| `IsTransitioning_MidSSP585` | Integer | Binary | 1 if baseline class shifts under SSP5-8.5 (2050); 0 if stable. |
| `ExposedPop_MidSSP585` | Integer | Persons | Count of residents living in transition zones ($E_z$). |
| `BuiltUpAreaSqKm` | Float | $\text{km}^2$ | Physical built-up building surface area (Copernicus GHSL). |
| `CroplandAreaSqKm` | Float | $\text{km}^2$ | Agricultural cropland area (ESA WorldCover). |
| `Rice_NDVI_Boro` | Float | Index | Mean NDVI during winter irrigated Boro rice season (Jan–Apr). |
| `Rice_NDVI_Aman` | Float | Index | Mean NDVI during monsoon rain-fed Aman season (Aug–Nov). |
| `DistanceToHealthFacility_Km` | Float | km | Mean straight-line Euclidean distance to nearest hospital. |
| `RoadDensityKmPerSqKm` | Float | $\text{km/km}^2$ | Evacuation and relief road network density. |
| `Score_ClimateTransition_Cz` | Float | [0, 1] | Standardized climate transition intensity indicator. |
| `Score_PopulationExposure_Pz` | Float | [0, 1] | Standardized population density exposure indicator. |
| `Score_AssetExposure_Az` | Float | [0, 1] | Standardized physical asset and cropland exposure indicator. |
| `Score_ServiceDeficit_Hz` | Float | [0, 1] | Standardized critical service accessibility deficit indicator. |

---

## 3. Comparative Model Rankings (`bcip_model_comparison_and_rankings.csv`)

| Field Name | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `Score_Model1_ExposureOnly` | Float | [0, 1] | Baseline score based strictly on population and assets ($S_1 = 0.50 P_z + 0.50 A_z$). |
| `Rank_Model1_ExposureOnly` | Integer | 1..64 | District rank under Model 1. |
| `Score_Model2_HazardContextOnly` | Float | [0, 1] | Baseline score based on 40-year water occurrence, elevation, and flood tiers. |
| `Rank_Model2_HazardContextOnly` | Integer | 1..64 | District rank under Model 2. |
| `Score_Model3_ClimateTransitionOnly`| Float | [0, 1] | Baseline score based solely on climate transition magnitude ($C_z$). |
| `Rank_Model3_ClimateTransitionOnly` | Integer | 1..64 | District rank under Model 3. |
| `Score_Model4_IntegratedBCIP` | Float | [0, 1] | Integrated adaptation screening score ($S_4 = 0.25 C_z + 0.25 P_z + 0.25 A_z + 0.25 H_z$). |
| `Rank_Model4_IntegratedBCIP` | Integer | 1..64 | Final district adaptation priority rank under Integrated BCIP. |
| `RankDisplacement_BCIP_vs_Exposure` | Integer | $\pm \text{Ranks}$ | Priority shift ($\text{Rank}_1 - \text{Rank}_4$). Positive values indicate priority surge. |
| `BCIP_RobustnessTier` | Enum | Tier | `Tier 1: Highly Robust Priority`, `Tier 2: Climate-Sensitive Priority`, `Tier 3: Exposure-Driven Priority`, `Tier 4: Stable Moderate / Lower Urgency`. |
