# 🛡️ BCIP Production Operations & Engineering Runbook

**Platform:** Bangladesh Climate Intelligence Platform (BCIP)  
**Maintainer:** Readuanul Farid Fahim | Department of Computer Science & Engineering, North South University  
**Contact:** Readuanulfaridfahim1@gmail.com  
**Version:** 1.0.0-PROD  

---

## 1. System Architecture Overview

BCIP operates as a decoupled, microservices-ready intelligence platform:

```
[ External Satellite & UN Open Data ]
  • Beck et al. (2023) CMIP6 Köppen-Geiger (1 km)
  • Copernicus GHSL (100m Population & Built Surface)
  • NASA SRTM (30m Elevation) & LP DAAC MODIS
  • UN HDX 62 Humanitarian Series & OCHA Boundaries
                   │
                   ▼
[ Data Engineering & Processing Engine (Python 3.12) ]
  • scripts/17_build_bcip_master_catalog.py
  • scripts/18_climate_transition_and_scenario_agreement_engine.py
  • scripts/19_adaptation_priority_and_robustness_models.py
  • scripts/20_generate_bcip_publication_figures.py
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ Analytical Storage ]  [ Spatial Storage ]
  • CSV / JSON Data       • RFC 7946 GeoJSON
  • Serialized Models     • EPSG:4326 Boundaries
         │                   │
         └─────────┬─────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│        High-Performance REST API Service (FastAPI)      │
│  Port: 8000  •  Workers: 4  •  Docs: /docs             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│        Interactive National Web Atlas (Streamlit)       │
│  Port: 8501  •  Session State  •  Plotly Express Maps   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Deployment Instructions

### A. Local Virtual Environment Deployment (Development / Staging)
```bash
# 1. Activate environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux / macOS

# 2. Run complete data pipelines (15 seconds)
python scripts/run_all_bcip_pipelines.py

# 3. Launch REST API Service (background or dedicated terminal)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# 4. Launch Streamlit National Web Atlas
streamlit run web/app.py --server.port 8501 --server.headless true
```

### B. Containerized Production Deployment (Docker Compose)
```bash
# 1. Build and launch all services in detached mode
docker compose up -d --build

# 2. Verify container status and health
docker compose ps

# 3. Inspect application logs
docker compose logs -f bcip-api
docker compose logs -f bcip-web
```

---

## 3. Automated Health Checks & Telemetry

### API Healthcheck Endpoint
* **URL:** `GET http://localhost:8000/health`
* **Expected Response:** `200 OK`
```json
{
  "status": "Healthy & Operational",
  "cached_datasets": ["bcip_master_catalog.csv", "bcip_model_comparison_and_rankings.csv"],
  "districts_count": 64,
  "datasets_inventoried": 111
}
```

---

## 4. Automated Testing & Continuous Integration

Run the 19-test automated test suite:
```bash
pytest tests/ -v --durations=10
```

### Acceptance Test Gates:
1. `test_data_integrity.py`: Verifies that all 111 datasets and 64 districts are intact with valid SHA-256 checksums and geographic bounding boxes.
2. `test_climate_transitions.py`: Verifies mathematical conservation of total land area across all $4 \times 4$ transition matrices ($\sum T_{ij} = A_{\text{total}}$) and net class balances.
3. `test_models.py`: Verifies score bounds $[0, 1]$, rank permutations, Spearman rank correlation stability, and incremental value proof ($\rho < 0.70$).
4. `test_api.py`: Validates all 10 FastAPI endpoints, Pydantic schemas, and GeoJSON streaming.

---

## 5. Incident Response & Troubleshooting

| Symptom | Probable Cause | Action |
| :--- | :--- | :--- |
| `503 Service Unavailable` on `/health` | Missing processed CSV or GeoJSON assets. | Run `python scripts/run_all_bcip_pipelines.py` to regenerate all analytical files. |
| Streamlit `NameError: SPATIAL_DIR` | Path constant missing in top declaration. | Ensure `SPATIAL_DIR` and `ASSETS_DIR` are declared in `web/app.py`. |
| Port 8000 or 8501 conflict | Existing process binding the port. | Run `netstat -ano \| findstr :8000` and kill the lingering PID. |
