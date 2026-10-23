# 📡 BCIP REST API Specification (OpenAPI 3.0)

**Base URL:** `http://localhost:8000`  
**Interactive Swagger UI:** [`http://localhost:8000/docs`](http://localhost:8000/docs)  
**ReDoc Documentation:** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)  

---

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Platform metadata, version, and endpoint sitemap. |
| `GET` | `/health` | Cluster and dataset asset health status. |
| `GET` | `/api/v1/catalogue` | Query the 111-dataset master register with filtering by theme, role, and search keyword. |
| `GET` | `/api/v1/transitions` | National transition summary and $4 \times 4$ categorical transition matrices. |
| `GET` | `/api/v1/districts` | Complete 64-district dataset with transition flags, exposure metrics, and service deficits. |
| `GET` | `/api/v1/districts/{district_id}` | Detailed climate adaptation profile for a specific administrative district. |
| `POST` | `/api/v1/prioritize` | Dynamic adaptation prioritization engine accepting custom multi-criteria weights. |
| `GET` | `/api/v1/correlations` | Spearman rank correlation matrix demonstrating incremental value. |
| `GET` | `/api/v1/spatial/geojson` | Streams RFC 7946 GeoJSON boundary polygon layer for desktop QGIS / ArcGIS. |

---

## Endpoint Details & Examples

### 1. Dynamic Adaptation Prioritization
* **Endpoint:** `POST /api/v1/prioritize`
* **Request Body (JSON):**
```json
{
  "w_C": 0.40,
  "w_P": 0.20,
  "w_A": 0.20,
  "w_H": 0.20,
  "top_n": 5
}
```
* **Sample Response (JSON):**
```json
{
  "normalized_weights": {
    "wC": 0.4,
    "wP": 0.2,
    "wA": 0.2,
    "wH": 0.2
  },
  "top_priority_districts": [
    {
      "Dynamic_Rank": 1,
      "DistrictID": 61,
      "DistrictName": "Mymensingh",
      "Division": "Mymensingh",
      "Dynamic_Score": 0.497,
      "Rank_Model1_ExposureOnly": 4,
      "RankDisplacement_vs_Exposure": 3,
      "BCIP_RobustnessTier": "Tier 1: Highly Robust Priority (Top 18 in All Schemes)",
      "Population": 5312000
    }
  ]
}
```

### 2. Querying Master Catalog
* **Endpoint:** `GET /api/v1/catalogue?role=Core`
* **Sample Curl:**
```bash
curl -X GET "http://localhost:8000/api/v1/catalogue?role=Core" -H "accept: application/json"
```

### 3. Streaming Desktop GeoJSON
* **Endpoint:** `GET /api/v1/spatial/geojson`
* **Sample Curl:**
```bash
curl -o bangladesh_bcip_atlas.geojson "http://localhost:8000/api/v1/spatial/geojson"
```
