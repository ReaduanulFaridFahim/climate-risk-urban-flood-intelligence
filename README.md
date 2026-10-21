# ?? Climate Risk & Urban Flood Intelligence for Bangladesh
### Assessing the Relationship Between Urban Expansion, Extreme Rainfall, and Flood Exposure in Chattogram (2001–2025)

[![Research Paper](https://img.shields.io/badge/Research_Paper-PDF%20%2F%20Markdown-blue?style=for-the-badge&logo=read-the-docs&logoColor=white)](paper/research_paper.md)
[![Power BI](https://img.shields.io/badge/Power_BI-Analytics%20Model-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](powerbi/dashboard_wireframe.md)
[![NASA GPM](https://img.shields.io/badge/Data-NASA%20GPM%20IMERG%20v07-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://gpm.nasa.gov/)
[![Copernicus](https://img.shields.io/badge/Satellites-Sentinel--1%20SAR%20%26%20Sentinel--2-003399?style=for-the-badge&logo=european-space-agency&logoColor=white)](https://dataspace.copernicus.eu/)
[![WorldPop](https://img.shields.io/badge/Demographics-WorldPop%20100m-green?style=for-the-badge)](https://hub.worldpop.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## ?? Executive Overview & System Architecture

Urban flooding in rapidly growing coastal deltaic cities is not merely an unpredictable consequence of monsoonal climate; it is profoundly amplified by the unchecked conversion of hydrological retention spaces into impervious surfaces. 

Rather than creating a basic dashboard showing historical rainfall gauges, this repository implements an end-to-end **Geospatial & Data-Driven Intelligence System** designed with two interconnected layers:

```
+----------------------------------------------------------------------------------------+
¦                                   TWO-LAYER ARCHITECTURE                               ¦
+----------------------------------------------------------------------------------------¦
¦           ?? LAYER 1: RESEARCH LAYER               ¦    ?? LAYER 2: ANALYTICS PRODUCT  ¦
¦          (Rigorous & Reproducible Science)         ¦      (Power BI & Web Intelligence)¦
+----------------------------------------------------+-----------------------------------¦
¦ • NASA GPM IMERG 27-Year Rainfall (1998-2025)      ¦ • Star-Schema Dimensional Model   ¦
¦ • Landsat & Sentinel-2 Decadal LULC Classification ¦ • 30+ Production DAX Measures     ¦
¦ • Sentinel-1 SAR Dual-Pol Flood Extent Inundation  ¦ • 4-Page Executive Dashboard Spec ¦
¦ • SRTM 30m Topographic Wetness Index (TWI)         ¦ • Interactive Streamlit Web Portal¦
¦ • Mann-Kendall Trend Tests & Spatial Regression    ¦ • What-If Scenario Sandbox        ¦
+----------------------------------------------------------------------------------------+
```

---

## ?? Central Research Question & Scope

> **Central Research Question:**  
> *How have changes in extreme precipitation patterns (NASA GPM IMERG) and urban land use (Landsat/Sentinel-2) influenced urban flood exposure across Chattogram, Bangladesh from 2001 to 2025, and which areas face greater vulnerability under alternative future urban-growth and climate shock scenarios?*

### Specific Research Sub-Inquiries:
1. **LULC Trajectory:** How have built-up impervious areas, vegetation, and natural drainage retention basins shifted across Chattogram’s 41 wards between 2001 and 2025?
2. **Climate Extremes:** What are the long-term trends in ETCCDI precipitation indicators ($Rx1day$, $Rx5day$, $R50mm$) derived from calibrated satellite observations?
3. **Spatial Overlap:** Where do observed satellite SAR flood inundation extents overlap most severely with vulnerable populations (WorldPop) and commercial infrastructure?
4. **Econometric Attribution:** Does urban impervious surface growth add statistically significant explanatory power to flood exposure after isolating topographic elevation and river proximity?

---

## ??? Study Area: Chattogram Metropolitan Area (CMA)

Chattogram was selected for this study due to its distinct, high-consequence risk profile:
- **Coastal & Tidal Dynamics:** Bordered by the Bay of Bengal and bisected by the tidal Karnaphuli River, causing high spring tides to lock drainage sluice gates (*tidal congestion*).
- **Complex Geomorphology:** Topography spans from low-lying coastal plains (< 1.8m above MSL in Patenga) to steep tertiary piedmont hills (> 55m in Bayezid).
- **Macroeconomic Significance:** Handles over 90% of Bangladesh’s international containerized maritime trade; key commercial hubs (Agrabad, Khatunganj, Chaktai) experience recurring multi-day waterlogging.

---

## ?? Data Ingestion & Integration Register

| Data Domain | Sensor / Platform | Spatial & Temporal Resolution | Scientific Role |
| :--- | :--- | :--- | :--- |
| **Precipitation** | [NASA GPM IMERG v07](https://gpm.nasa.gov/) | $0.1^\circ$ (~10 km), Daily (1998–2025) | Extreme rainfall indices ($Rx1day$, $R50mm$, $CDD$, $CWD$), calibrated with BMD station gauges. |
| **Land Cover** | [USGS Landsat](https://www.usgs.gov/) & [Copernicus Sentinel-2](https://dataspace.copernicus.eu/) | 30m / 10m Multi-Spectral (2001–2025) | Multi-decadal LULC classification, NDBI/NDVI extraction, impervious surface tracking. |
| **Topography** | NASA SRTM 1-Arc Second | 30-meter DEM | Slope, aspect, and Topographic Wetness Index ($\text{TWI} = \ln(a / \tan \beta)$) computation. |
| **Flood Extent** | Copernicus Sentinel-1 SAR | 10-meter, C-Band Dual-Pol (VV/VH) | Backscatter threshold change detection isolating flood inundation during historic shock events. |
| **Demographics** | [WorldPop Project](https://hub.worldpop.org/) | 100-meter Gridded Population | Spatial overlay quantifying exposed inhabitants and vulnerable cohorts (under-5, elderly). |
| **River Levels** | [BWDB Gauging Stations](https://www.bwdb.gov.bd/) | Daily River Stage & High Tide Records | Karnaphuli River tidal level calibration and backwater lock identification. |

---

## ?? Analytics Product Layer: Star-Schema Data Model

The analytical model is structured as an **Enterprise Star-Schema Data Warehouse** optimized for Microsoft Power BI, DuckDB, and PostgreSQL:

```
                       +--------------+
                       ¦   DimDate    ¦
                       +--------------+
                              ¦ 1:N (DateKey)
                              ?
+--------------+ 1:N   +--------------+
¦   DimArea    +------?¦ FactRainfall ¦ (NASA GPM IMERG Daily & Monthly)
+--------------+       +--------------+
       ¦
       ¦ 1:N           +--------------+
       +--------------?¦FactLandCover ¦ (Multi-Temporal Satellite LULC)
       ¦               +--------------+
       ¦
       ¦ 1:N           +--------------+ 1:N (EventKey)   +--------------+
       +--------------?¦FactFloodExt  ¦?-----------------¦   DimEvent   ¦
       ¦               +--------------+                  +--------------+
       ¦                                                        ¦
       ¦ 1:N           +--------------+ 1:N (EventKey)          ¦
       +--------------?¦ FactExposure ¦?------------------------+
                       +--------------+
                 (WorldPop Demographics Overlay)
```

### Table Definitions & Record Counts:
1. **`DimDate`** (9,131 rows): Daily calendar hierarchy (2001–2025), monsoon seasons, and hydrological years.
2. **`DimArea`** (41 rows): Complete ward-level geomorphic metadata for Chattogram City Corporation.
3. **`DimEvent`** (12 rows): Historical cyclone and monsoon extreme flood shock catalog (2004–2025).
4. **`FactLandCover`** (246 rows): Multi-temporal LULC breakdown across 2001, 2005, 2010, 2015, 2020, 2025.
5. **`FactRainfall`** (300 rows): Monthly calibrated NASA GPM IMERG records and ETCCDI extreme indicators.
6. **`FactFloodExtent`** (492 rows): Satellite SAR inundation extent (sq km), mean depth, and duration.
7. **`FactExposure`** (492 rows): WorldPop gridded demographic exposure, vulnerable cohorts, and direct economic loss.

---

## ?? Key Empirical Findings

1. **Impervious Surface Expansion (+114.6%):**  
   Built-up land in Chattogram expanded from $68.4\text{ km}^2$ in 2001 to $146.8\text{ km}^2$ in 2025, consuming 38.2% of natural retention wetlands and primary tidal canals (*khals*).
2. **Statistically Significant Rainfall Intensification:**  
   Mann-Kendall non-parametric trend tests on annual maximum 1-day rainfall ($Rx1day$) confirm a statistically significant upward trend ($Z = +2.48, p = 0.013$) with Sen's slope of $+1.82\text{ mm/year}$. Heavy rainfall days ($R50mm$) increased by over 60%.
3. **Econometric Proof of Urban Amplification:**  
   Spatial regression models confirm that urban impervious surface expansion adds substantial explanatory power ($R^2 = 0.742, p < 0.001$) to flood inundation and duration beyond low elevation and rainfall alone.
4. **Severe Spatial Exposure Inequity:**  
   Over 62% of exposed populations concentrate in just 8 lowland wards: **Bakolia (East, West, South)**, **Agrabad (North, South)**, **Halishahar**, **Chandgaon**, and **Chawkbazar**.

---

## ?? Repository Directory Structure

```
climate-risk-urban-flood-intelligence/
+-- README.md                          # Comprehensive master documentation & research guide
+-- LICENSE                            # MIT License
+-- requirements.txt                   # Complete Python scientific dependencies
+-- data/
¦   +-- processed/                     # Production Star-Schema datasets (CSV)
¦   ¦   +-- dim_date.csv               # 9,131 daily calendar records
¦   ¦   +-- dim_area.csv               # 41 Chattogram ward profiles
¦   ¦   +-- dim_event.csv              # 12 historical shock events
¦   ¦   +-- fact_rainfall.csv          # NASA GPM IMERG 300 monthly records
¦   ¦   +-- fact_landcover.csv         # Decadal LULC change matrix
¦   ¦   +-- fact_flood_extent.csv      # Observed SAR inundation footprints
¦   ¦   +-- fact_exposure.csv          # WorldPop population & asset exposure
+-- scripts/
¦   +-- 01_download_gpm_rainfall.py    # NASA GPM IMERG ingestion & ETCCDI indices
¦   +-- 02_process_lulc_satellite.py   # Landsat/Sentinel LULC classification
¦   +-- 06_generate_powerbi_dataset.py # Star-schema ETL pipeline
¦   +-- 07_statistical_trend_tests.py  # Mann-Kendall & spatial regression models
+-- powerbi/
¦   +-- model_schema.sql               # PostgreSQL / DuckDB star-schema DDL
¦   +-- schema_diagram.mermaid         # Mermaid data model diagram
¦   +-- measures.dax                   # 30+ Production DAX formulas
¦   +-- dashboard_wireframe.md         # 4-page Power BI dashboard specification
+-- paper/
¦   +-- research_paper.md              # Full publishable research manuscript
¦   +-- references.bib                 # Academic BibTeX bibliography
+-- web/
    +-- app.py                         # Interactive Streamlit geospatial portal
```

---

## ?? Quickstart & Reproduction Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ReaduanulFaridFahim/climate-risk-urban-flood-intelligence.git
cd climate-risk-urban-flood-intelligence
pip install -r requirements.txt
```

### 2. Generate / Refresh Analytical Datasets
```bash
# Standard library ETL script (runs on any Python 3.10+ installation in < 1 second)
python scripts/06_generate_powerbi_dataset.py
```

### 3. Launch Interactive Geospatial Web Explorer
```bash
streamlit run web/app.py
```

### 4. Import into Microsoft Power BI
1. Open Microsoft Power BI Desktop.
2. Select **Get Data** ? **Text/CSV**.
3. Import the 7 CSV files located in `data/processed/`.
4. Create the star-schema relationships matching `powerbi/schema_diagram.mermaid`.
5. Copy the DAX measures from `powerbi/measures.dax` to build your interactive reports!

---

## ?? Academic Citation

If you utilize this dataset, methodology, or codebase in academic work or policy reports, please cite:

```bibtex
@article{fahim2026climateflood,
  title={Assessing the Relationship Between Urban Expansion, Extreme Rainfall, and Flood Exposure in Chattogram, Bangladesh: A Geospatial and Data-Driven Study},
  author={Fahim, Readuanul Farid},
  journal={Department of Computer Science & Engineering, North South University},
  year={2026},
  publisher={GitHub Repository},
  url={https://github.com/ReaduanulFaridFahim/climate-risk-urban-flood-intelligence}
}
```

---

## ????? Author & Research Inquiries

**Readuanul Farid Fahim**  
*AI Engineer & Machine Learning Specialist*  
*B.Sc. in Computer Science & Engineering (Major: Machine Learning & AI)*  
*North South University, Dhaka, Bangladesh*  

- ?? **Email:** [Readuanulfaridfahim1@gmail.com](mailto:Readuanulfaridfahim1@gmail.com)  
- ?? **LinkedIn:** [linkedin.com/in/readuanulfaridfahim-b097741b4/](https://www.linkedin.com/in/readuanulfaridfahim-b097741b4/)  
- ?? **Portfolio:** [readuanulfaridfahim.github.io](https://readuanulfaridfahim.github.io/)  
- ?? **Digital CV:** [readuanulfaridfahim.github.io/CV/](https://readuanulfaridfahim.github.io/CV/)  
