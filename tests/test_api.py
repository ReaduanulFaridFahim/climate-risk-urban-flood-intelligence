"""
End-to-End Test Suite for FastAPI Production Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["platform"] == "Bangladesh Climate Intelligence Platform (BCIP)"
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy & Operational"
    assert data["districts_count"] == 64
    assert data["datasets_inventoried"] == 111


def test_catalogue_endpoint():
    response = client.get("/api/v1/catalogue")
    assert response.status_code == 200
    data = response.json()
    assert data["total_catalog_count"] == 111
    assert len(data["datasets"]) == 111


def test_catalogue_filter_by_role():
    response = client.get("/api/v1/catalogue?role=Core")
    assert response.status_code == 200
    data = response.json()
    assert data["filtered_count"] == 5
    for item in data["datasets"]:
        assert item["AnalyticalRole"] == "Core"


def test_transitions_endpoint():
    response = client.get("/api/v1/transitions")
    assert response.status_code == 200
    data = response.json()
    assert "national_summary" in data
    assert "transition_matrices" in data
    assert len(data["national_summary"]) == 2


def test_districts_endpoint():
    response = client.get("/api/v1/districts")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 64
    assert len(data["districts"]) == 64


def test_district_by_id_endpoint():
    response = client.get("/api/v1/districts/61")  # Mymensingh
    assert response.status_code == 200
    data = response.json()
    assert data["DistrictName"] == "Mymensingh"
    assert "Score_Model4_IntegratedBCIP" in data


def test_prioritization_post_endpoint():
    payload = {
        "w_C": 0.40,
        "w_P": 0.20,
        "w_A": 0.20,
        "w_H": 0.20,
        "top_n": 10
    }
    response = client.post("/api/v1/prioritize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "normalized_weights" in data
    assert len(data["top_priority_districts"]) == 10
    # Top 1 district must have dynamic rank 1
    assert data["top_priority_districts"][0]["Dynamic_Rank"] == 1


def test_correlations_endpoint():
    response = client.get("/api/v1/correlations")
    assert response.status_code == 200
    data = response.json()
    assert "Model 4 (Integrated BCIP)" in data


def test_spatial_geojson_endpoint():
    response = client.get("/api/v1/spatial/geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 64


import os

@pytest.mark.skipif(not os.path.exists("web/static"), reason="Static web portal unmounted")
def test_web_portal_route():
    response = client.get("/portal/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "BANGLADESH CLIMATE INTELLIGENCE" in response.text


@pytest.mark.skipif(not os.path.exists("web/static"), reason="Static web portal unmounted")
def test_root_browser_html_request():
    response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Bangladesh Climate Risk & Flood Intelligence Platform" in response.text


@pytest.mark.skipif(not os.path.exists("web/static"), reason="Static web portal unmounted")
def test_all_multipage_routes():
    pages = [
        "index.html",
        "geospatial.html",
        "ai-models.html",
        "weather-forecast.html",
        "humanitarian.html",
        "emergency-alert.html",
        "chattogram.html",
        "data-catalog.html",
        "analytics-lab.html"
    ]
    for page in pages:
        res = client.get(f"/portal/{page}")
        assert res.status_code == 200, f"Page {page} returned {res.status_code}"
        assert "text/html" in res.headers.get("content-type", "")
        assert "BANGLADESH CLIMATE INTELLIGENCE" in res.text






