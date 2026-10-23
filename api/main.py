"""
========================================================================================
api/main.py
BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - PRODUCTION REST API
Enterprise FastAPI service serving dataset metadata, CMIP6 transition matrices,
multi-dimensional exposure metrics, dynamic prioritization, and GeoJSON spatial feeds.
========================================================================================
"""

import os
import json
import logging
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.requests import Request
from starlette.responses import Response
from pydantic import BaseModel, Field
import pandas as pd

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("bcip-api")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPATIAL_DIR = os.path.join(BASE_DIR, 'data', 'spatial')

# Initialize FastAPI App
app = FastAPI(
    title="Bangladesh Climate Intelligence Platform (BCIP) API",
    description=(
        "Production-grade RESTful API serving peer-reviewed CMIP6 bioclimatic transition matrices, "
        "multi-dimensional exposure overlays, and dynamic adaptation decision-support scores across all 64 districts."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# METRICS & LATENCY TELEMETRY MIDDLEWARE
# -----------------------------------------------------------------------------
_api_metrics: Dict[str, Any] = {
    "start_time": time.time(),
    "total_requests": 0,
    "total_errors": 0,
    "total_latency_ms": 0.0,
    "endpoint_hits": {}
}


@app.middleware("http")
async def track_metrics_and_latency(request: Request, call_next):
    start = time.time()
    endpoint = request.url.path
    _api_metrics["total_requests"] += 1
    _api_metrics["endpoint_hits"][endpoint] = _api_metrics["endpoint_hits"].get(endpoint, 0) + 1
    
    try:
        response: Response = await call_next(request)
        if response.status_code >= 400:
            _api_metrics["total_errors"] += 1
    except Exception as exc:
        _api_metrics["total_errors"] += 1
        raise exc
    finally:
        latency_ms = (time.time() - start) * 1000.0
        _api_metrics["total_latency_ms"] += latency_ms

    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    return response

# -----------------------------------------------------------------------------
# DATA CACHE & LOADERS
# -----------------------------------------------------------------------------
_data_cache: Dict[str, Any] = {}


def get_cached_dataset(filename: str) -> pd.DataFrame:
    if filename not in _data_cache:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Dataset {filename} not found on server.")
        _data_cache[filename] = pd.read_csv(path)
    return _data_cache[filename]


def get_cached_json(filename: str) -> dict:
    if filename not in _data_cache:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"JSON asset {filename} not found.")
        with open(path, 'r', encoding='utf-8') as f:
            _data_cache[filename] = json.load(f)
    return _data_cache[filename]


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------
class PrioritizationRequest(BaseModel):
    w_C: float = Field(0.25, ge=0.0, le=1.0, description="Weight for Climate Transition Intensity (wC)")
    w_P: float = Field(0.25, ge=0.0, le=1.0, description="Weight for Population Density Exposure (wP)")
    w_A: float = Field(0.25, ge=0.0, le=1.0, description="Weight for Built Asset & Cropland Exposure (wA)")
    w_H: float = Field(0.25, ge=0.0, le=1.0, description="Weight for Service Accessibility Deficit (wH)")
    top_n: Optional[int] = Field(15, ge=1, le=64, description="Number of top districts to return")


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", tags=["Platform Info"])
def root(request: Request):
    accept = request.headers.get("accept", "")
    index_path = os.path.join(STATIC_WEB_DIR, 'index.html')
    if "text/html" in accept and os.path.exists(index_path):
        return FileResponse(index_path)

    return {
        "platform": "Bangladesh Climate Intelligence Platform (BCIP)",
        "version": "1.0.0",
        "documentation": "/docs",
        "portal_ui": "/portal",
        "streamlit_lab": "http://localhost:8501/",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "catalogue": "/api/v1/catalogue",
            "transitions": "/api/v1/transitions",
            "districts": "/api/v1/districts",
            "biomes": "/api/v1/biomes/summary",
            "prioritize": "/api/v1/prioritize",
            "correlations": "/api/v1/correlations",
            "export_csv": "/api/v1/export/csv",
            "spatial_geojson": "/api/v1/spatial/geojson",
            "streamlit": "/streamlit"
        }
    }


@app.get("/streamlit", tags=["Platform Info"])
def redirect_to_streamlit():
    """Redirects to the live Streamlit interactive research suite on port 8501."""
    return RedirectResponse(url="http://localhost:8501/")


@app.get("/health", tags=["System Health"])
def health_check():
    required_files = [
        os.path.join(DATA_DIR, 'bcip_master_catalog.csv'),
        os.path.join(DATA_DIR, 'bcip_climate_transitions_national_summary.csv'),
        os.path.join(DATA_DIR, 'bcip_district_climate_transitions_and_exposure.csv'),
        os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv'),
        os.path.join(SPATIAL_DIR, 'bcip_district_climate_transitions.geojson')
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        return JSONResponse(status_code=503, content={"status": "Degraded", "missing_assets": missing})
    return {
        "status": "Healthy & Operational",
        "cached_datasets": list(_data_cache.keys()),
        "districts_count": 64,
        "datasets_inventoried": 111
    }


@app.get("/api/v1/catalogue", tags=["Master Catalog"])
def get_catalogue(
    theme: Optional[str] = Query(None, description="Filter by theme domain"),
    role: Optional[str] = Query(None, description="Filter by analytical role (Core, Exposure, Context, etc.)"),
    search: Optional[str] = Query(None, description="Search keyword in title or description")
):
    df_cat = get_cached_dataset('bcip_master_catalog.csv')
    filtered = df_cat.copy()
    if theme:
        filtered = filtered[filtered['Theme'].str.contains(theme, case=False, na=False)]
    if role:
        filtered = filtered[filtered['AnalyticalRole'].str.contains(role, case=False, na=False)]
    if search:
        filtered = filtered[
            filtered['Title'].str.contains(search, case=False, na=False) |
            filtered['Justification'].str.contains(search, case=False, na=False)
        ]
    return {
        "total_catalog_count": len(df_cat),
        "filtered_count": len(filtered),
        "datasets": filtered.fillna("").to_dict(orient="records")
    }


@app.get("/api/v1/transitions", tags=["Climate Transitions"])
def get_transitions():
    df_sum = get_cached_dataset('bcip_climate_transitions_national_summary.csv')
    matrices = get_cached_json('bcip_climate_transition_matrices.json')
    df_agree = get_cached_dataset('bcip_scenario_agreement_analysis.csv')
    return {
        "national_summary": df_sum.fillna("").to_dict(orient="records"),
        "transition_matrices": matrices,
        "mid_century_agreement": df_agree['MidCentury_AgreementStatus'].value_counts().to_dict()
    }


@app.get("/api/v1/districts", tags=["Districts & Exposure"])
def get_districts(
    division: Optional[str] = Query(None, description="Filter by administrative division"),
    tier: Optional[str] = Query(None, description="Filter by BCIP robustness tier")
):
    df_dist = get_cached_dataset('bcip_model_comparison_and_rankings.csv')
    filtered = df_dist.copy()
    if division:
        filtered = filtered[filtered['Division'].str.lower() == division.lower()]
    if tier:
        filtered = filtered[filtered['BCIP_RobustnessTier'].str.contains(tier, case=False, na=False)]
    return {
        "count": len(filtered),
        "districts": filtered.fillna("").to_dict(orient="records")
    }


@app.get("/api/v1/districts/{district_id}", tags=["Districts & Exposure"])
def get_district_by_id(district_id: int):
    df_dist = get_cached_dataset('bcip_model_comparison_and_rankings.csv')
    matched = df_dist[df_dist['DistrictID'] == district_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"DistrictID {district_id} not found.")
    return matched.fillna("").iloc[0].to_dict()


@app.post("/api/v1/prioritize", tags=["Adaptation Prioritization Engine"])
def compute_dynamic_prioritization(req: PrioritizationRequest):
    df_rank = get_cached_dataset('bcip_model_comparison_and_rankings.csv').copy()
    
    # Normalize weights
    total_w = req.w_C + req.w_P + req.w_A + req.w_H
    if total_w > 0:
        wc_n = req.w_C / total_w
        wp_n = req.w_P / total_w
        wa_n = req.w_A / total_w
        wh_n = req.w_H / total_w
    else:
        wc_n, wp_n, wa_n, wh_n = 0.25, 0.25, 0.25, 0.25

    # Compute custom dynamic score
    df_rank['Dynamic_Score'] = (
        wc_n * df_rank['Score_ClimateTransition_Cz'] +
        wp_n * df_rank['Score_PopulationExposure_Pz'] +
        wa_n * df_rank['Score_AssetExposure_Az'] +
        wh_n * df_rank['Score_ServiceDeficit_Hz']
    )
    df_rank['Dynamic_Rank'] = df_rank['Dynamic_Score'].rank(ascending=False).astype(int)
    df_rank['RankDisplacement_vs_Exposure'] = df_rank['Rank_Model1_ExposureOnly'] - df_rank['Dynamic_Rank']

    top_districts = df_rank.sort_values('Dynamic_Rank').head(req.top_n)[[
        'Dynamic_Rank', 'DistrictID', 'DistrictName', 'Division', 'Dynamic_Score',
        'Rank_Model1_ExposureOnly', 'RankDisplacement_vs_Exposure', 'BCIP_RobustnessTier', 'Population'
    ]]

    return {
        "normalized_weights": {"wC": round(wc_n, 4), "wP": round(wp_n, 4), "wA": round(wa_n, 4), "wH": round(wh_n, 4)},
        "top_priority_districts": top_districts.fillna("").to_dict(orient="records")
    }


@app.get("/api/v1/correlations", tags=["Research Models"])
def get_correlations():
    return get_cached_json('bcip_spearman_rank_correlations.json')


@app.get("/metrics", tags=["System Observability"])
def get_metrics(format: Optional[str] = Query("json", description="Output format: 'json' or 'prometheus'")):
    uptime_sec = time.time() - _api_metrics["start_time"]
    reqs = _api_metrics["total_requests"]
    avg_latency = (_api_metrics["total_latency_ms"] / reqs) if reqs > 0 else 0.0

    if format == "prometheus":
        lines = [
            "# HELP bcip_api_requests_total Total HTTP requests received",
            "# TYPE bcip_api_requests_total counter",
            f"bcip_api_requests_total {_api_metrics['total_requests']}",
            "# HELP bcip_api_errors_total Total HTTP 4xx/5xx errors returned",
            "# TYPE bcip_api_errors_total counter",
            f"bcip_api_errors_total {_api_metrics['total_errors']}",
            "# HELP bcip_api_uptime_seconds Process uptime in seconds",
            "# TYPE bcip_api_uptime_seconds gauge",
            f"bcip_api_uptime_seconds {uptime_sec:.2f}",
            "# HELP bcip_api_latency_ms_avg Average request latency in milliseconds",
            "# TYPE bcip_api_latency_ms_avg gauge",
            f"bcip_api_latency_ms_avg {avg_latency:.2f}"
        ]
        for ep, hits in _api_metrics["endpoint_hits"].items():
            ep_clean = ep.replace('"', '\\"')
            lines.append(f'bcip_endpoint_requests_total{{path="{ep_clean}"}} {hits}')
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_sec, 2),
        "total_requests": _api_metrics["total_requests"],
        "total_errors": _api_metrics["total_errors"],
        "average_latency_ms": round(avg_latency, 2),
        "endpoint_hits": _api_metrics["endpoint_hits"]
    }


@app.get("/api/v1/biomes/summary", tags=["Regional Macro-Biomes"])
def get_biomes_summary():
    df_b = get_cached_dataset('bcip_regional_biomes_summary.csv')
    return {
        "biomes_count": len(df_b),
        "biomes": df_b.fillna("").to_dict(orient="records")
    }


@app.get("/api/v1/export/csv", tags=["Data Export Feeds"])
def export_dataset_csv(dataset: str = Query(..., description="Dataset key: 'catalog', 'transitions', 'rankings', 'biomes', 'agreement'")):
    mapping = {
        "catalog": "bcip_master_catalog.csv",
        "transitions": "bcip_district_climate_transitions_and_exposure.csv",
        "rankings": "bcip_model_comparison_and_rankings.csv",
        "biomes": "bcip_regional_biomes_summary.csv",
        "agreement": "bcip_scenario_agreement_analysis.csv"
    }
    if dataset not in mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Permitted options: {list(mapping.keys())}"
        )
    filename = mapping[dataset]
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    return FileResponse(file_path, media_type="text/csv", filename=filename)


@app.get("/api/v1/spatial/geojson", tags=["Geospatial Data Feeds"])
def get_geojson_layer(
    bbox: Optional[str] = Query(None, description="Optional bounding box 'min_lon,min_lat,max_lon,max_lat' to filter features")
):
    p = os.path.join(SPATIAL_DIR, 'bcip_district_climate_transitions.geojson')
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="GeoJSON layer not generated.")

    if not bbox:
        return FileResponse(p, media_type="application/geo+json", filename="bcip_district_climate_transitions.geojson")

    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError()
        min_lon, min_lat, max_lon, max_lat = parts
    except Exception:
        raise HTTPException(status_code=400, detail="bbox must be in format 'min_lon,min_lat,max_lon,max_lat' with float values.")

    with open(p, 'r', encoding='utf-8') as f:
        geo = json.load(f)

    filtered_features = []
    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        lon = props.get("Longitude")
        lat = props.get("Latitude")
        if lon is not None and lat is not None:
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                filtered_features.append(feat)
        else:
            filtered_features.append(feat)

    return {
        "type": "FeatureCollection",
        "bbox_filter": [min_lon, min_lat, max_lon, max_lat],
        "features_count": len(filtered_features),
        "features": filtered_features
    }


# -----------------------------------------------------------------------------
# STATIC WEB PORTAL & DATA DOWNLOADS MOUNT
# -----------------------------------------------------------------------------
if os.path.exists(STATIC_WEB_DIR):
    app.mount("/portal", StaticFiles(directory=STATIC_WEB_DIR, html=True), name="portal_site")
    app.mount("/static", StaticFiles(directory=STATIC_WEB_DIR), name="static_assets")

if os.path.exists(DATA_DIR):
    app.mount("/data/downloads", StaticFiles(directory=DATA_DIR), name="data_downloads")


@app.get("/api/v1/data/inventory", tags=["Data Export Feeds"])
def list_available_data_files():
    """Returns a full inventory of all processed research data files available for direct download."""
    files = []
    if os.path.exists(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.isfile(fpath):
                ext = fname.split('.')[-1].upper()
                files.append({
                    "filename": fname,
                    "format": ext,
                    "size_bytes": os.path.getsize(fpath),
                    "size_kb": round(os.path.getsize(fpath) / 1024.0, 1),
                    "download_url": f"/data/downloads/{fname}"
                })
    return {
        "status": "success",
        "total_files": len(files),
        "files": files
    }


