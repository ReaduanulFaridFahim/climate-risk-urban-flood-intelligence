"""
========================================================================================
tests/test_boundary_conditions.py
Comprehensive QA Test Suite: Boundary Conditions, Edge Cases, Telemetry, and Validations
========================================================================================
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_latency_header_present():
    """Verify latency middleware attaches X-Process-Time-Ms header."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    latency = float(response.headers["X-Process-Time-Ms"])
    assert latency >= 0.0


def test_metrics_json_and_prometheus():
    """Verify telemetry endpoint in both JSON and Prometheus formats."""
    # Trigger requests
    client.get("/")
    client.get("/health")

    # JSON format
    resp_json = client.get("/metrics")
    assert resp_json.status_code == 200
    data = resp_json.json()
    assert data["status"] == "healthy"
    assert data["total_requests"] >= 2
    assert "uptime_seconds" in data
    assert "average_latency_ms" in data

    # Prometheus format
    resp_prom = client.get("/metrics?format=prometheus")
    assert resp_prom.status_code == 200
    assert "text/plain" in resp_prom.headers["content-type"]
    content = resp_prom.text
    assert "bcip_api_requests_total" in content
    assert "bcip_api_uptime_seconds" in content
    assert "bcip_endpoint_requests_total" in content


def test_prioritize_zero_weights_fallback():
    """Edge Case: All zero weights should gracefully fall back to equal weights without ZeroDivisionError."""
    payload = {
        "w_C": 0.0,
        "w_P": 0.0,
        "w_A": 0.0,
        "w_H": 0.0,
        "top_n": 5
    }
    response = client.post("/api/v1/prioritize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["normalized_weights"]["wC"] == 0.25
    assert data["normalized_weights"]["wP"] == 0.25
    assert len(data["top_priority_districts"]) == 5


def test_prioritize_single_dimension():
    """Edge Case: Weight concentrated in 100% climate transition."""
    payload = {
        "w_C": 1.0,
        "w_P": 0.0,
        "w_A": 0.0,
        "w_H": 0.0,
        "top_n": 10
    }
    response = client.post("/api/v1/prioritize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["normalized_weights"]["wC"] == 1.0
    assert data["normalized_weights"]["wP"] == 0.0
    assert len(data["top_priority_districts"]) == 10


def test_prioritize_negative_weight_validation_error():
    """Boundary Case: Negative weight must trigger HTTP 422 Unprocessable Entity."""
    payload = {
        "w_C": -0.5,
        "w_P": 0.5,
        "w_A": 0.5,
        "w_H": 0.5,
        "top_n": 5
    }
    response = client.post("/api/v1/prioritize", json=payload)
    assert response.status_code == 422


def test_prioritize_excess_weight_validation_error():
    """Boundary Case: Weight exceeding 1.0 must trigger HTTP 422."""
    payload = {
        "w_C": 1.5,
        "w_P": 0.0,
        "w_A": 0.0,
        "w_H": 0.0,
        "top_n": 5
    }
    response = client.post("/api/v1/prioritize", json=payload)
    assert response.status_code == 422


def test_district_out_of_range_404():
    """Boundary Case: District ID not found must return HTTP 404."""
    response = client.get("/api/v1/districts/9999")
    assert response.status_code == 404
    assert "DistrictID 9999 not found" in response.json()["detail"]


def test_export_csv_valid():
    """Verify CSV streaming export for registered datasets."""
    response = client.get("/api/v1/export/csv?dataset=catalog")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "DatasetID" in response.text


def test_export_csv_invalid_dataset():
    """Boundary Case: Invalid dataset key must return HTTP 400 Bad Request."""
    response = client.get("/api/v1/export/csv?dataset=non_existent_dataset")
    assert response.status_code == 400
    assert "Invalid dataset" in response.json()["detail"]


def test_spatial_geojson_bbox_valid():
    """Verify GeoJSON bounding box filtering for central Bangladesh."""
    # Bounding box around central Bangladesh: lon 90.0 to 91.0, lat 23.0 to 24.5
    response = client.get("/api/v1/spatial/geojson?bbox=90.0,23.0,91.0,24.5")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features_count" in data
    assert data["features_count"] > 0
    assert data["features_count"] <= 64


def test_spatial_geojson_bbox_invalid():
    """Boundary Case: Invalid bbox string format must return HTTP 400."""
    response = client.get("/api/v1/spatial/geojson?bbox=invalid,bbox")
    assert response.status_code == 400
