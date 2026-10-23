"""
Unit and Integration Tests for BCIP Data Assets & Geospatial Integrity
"""

import os
import json
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPATIAL_DIR = os.path.join(BASE_DIR, 'data', 'spatial')


def test_master_catalog_completeness():
    csv_path = os.path.join(DATA_DIR, 'bcip_master_catalog.csv')
    json_path = os.path.join(DATA_DIR, 'bcip_master_catalog.json')

    assert os.path.exists(csv_path), "bcip_master_catalog.csv does not exist"
    assert os.path.exists(json_path), "bcip_master_catalog.json does not exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 111, f"Expected 111 datasets, found {len(df)}"

    required_cols = [
        'DatasetID', 'Title', 'Theme', 'Provider', 'SourceURL', 'License',
        'AnalyticalRole', 'InclusionExclusionDecision', 'Justification', 'FileChecksum'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required catalog column: {col}"
        assert df[col].isnull().sum() == 0, f"Null values found in catalog column: {col}"

    # Verify checksums are non-empty 16-char hashes
    assert df['FileChecksum'].str.len().min() == 16


def test_64_districts_dataset_integrity():
    dist_csv = os.path.join(DATA_DIR, 'bcip_district_climate_transitions_and_exposure.csv')
    assert os.path.exists(dist_csv), "bcip_district_climate_transitions_and_exposure.csv missing"

    df = pd.read_csv(dist_csv)
    assert len(df) == 64, f"Expected 64 districts, found {len(df)}"

    # Check vital geographic and demographic boundaries
    assert df['Latitude'].between(20.5, 26.7).all(), "District latitudes out of Bangladesh bounds"
    assert df['Longitude'].between(88.0, 92.8).all(), "District longitudes out of Bangladesh bounds"
    assert (df['Population'] > 50000).all(), "Unrealistic low population detected"
    assert (df['AreaSqKm'] > 100).all(), "Unrealistic low district area detected"

    # Total area check (Bangladesh ~147,570 km²)
    total_area = df['AreaSqKm'].sum()
    assert 140000 <= total_area <= 155000, f"Unexpected national total area: {total_area} km²"


def test_spatial_geojson_specification():
    geojson_path = os.path.join(SPATIAL_DIR, 'bcip_district_climate_transitions.geojson')
    assert os.path.exists(geojson_path), "GeoJSON file missing"

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert data.get('type') == 'FeatureCollection', "Invalid GeoJSON root type"
    features = data.get('features', [])
    assert len(features) == 64, f"Expected 64 features in GeoJSON, found {len(features)}"

    # Validate first feature structure
    f0 = features[0]
    assert 'geometry' in f0
    assert f0['geometry']['type'] == 'Polygon'
    coords = f0['geometry']['coordinates'][0]
    assert len(coords) >= 4, "Polygon must have at least 4 coordinate vertices"
    assert coords[0] == coords[-1], "Polygon must be closed"


def test_60day_forecast_and_cold_wave_integrity():
    fc_json = os.path.join(DATA_DIR, 'bangladesh_60day_weather_forecast.json')
    fc_csv = os.path.join(DATA_DIR, 'bangladesh_60day_weather_forecast.csv')
    fc_20_json = os.path.join(DATA_DIR, 'bangladesh_20day_weather_forecast.json')

    assert os.path.exists(fc_json), "bangladesh_60day_weather_forecast.json missing"
    assert os.path.exists(fc_csv), "bangladesh_60day_weather_forecast.csv missing"
    assert os.path.exists(fc_20_json), "bangladesh_20day_weather_forecast.json missing"

    with open(fc_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    expected_divs = {'Dhaka', 'Chattogram', 'Sylhet', 'Rajshahi', 'Khulna', 'Barishal', 'Rangpur', 'Mymensingh'}
    assert set(data.keys()) == expected_divs, f"Division mismatch: {set(data.keys())}"

    for div_name, days in data.items():
        assert len(days) == 60, f"{div_name} does not have exactly 60 days of forecasts (found {len(days)})"

        for d in days:
            assert 'temp_max_c' in d and 'temp_min_c' in d
            assert 'wind_chill_c' in d, "Missing perceived wind chill"
            assert 'fog_duration_hrs' in d, "Missing fog duration"
            assert 'cold_status' in d, "Missing BMD cold status"
            assert 'boro_seedbed_risk' in d, "Missing Boro seedbed agricultural advisory"
            assert 'cold_wave_risk_score' in d, "Missing cold wave risk score"

            t_min = d['temp_min_c']
            c_status = d['cold_status']

            # Validate official BMD Cold Wave thresholds
            if t_min < 4.0:
                assert "VERY SEVERE" in c_status, f"Invalid BMD tier for {t_min}°C: {c_status}"
            elif t_min < 6.0:
                assert "SEVERE COLD WAVE" in c_status, f"Invalid BMD tier for {t_min}°C: {c_status}"
            elif t_min < 8.0:
                assert "MODERATE COLD WAVE" in c_status, f"Invalid BMD tier for {t_min}°C: {c_status}"
            elif t_min <= 10.0:
                assert "MILD COLD WAVE" in c_status, f"Invalid BMD tier for {t_min}°C: {c_status}"

    df = pd.read_csv(fc_csv)
    assert len(df) == 480, f"Expected 480 rows in 60-day CSV, found {len(df)}"
    assert 'wind_chill_c' in df.columns
    assert 'fog_duration_hrs' in df.columns


def test_hdx_humanitarian_usecases_data_integrity():
    csv_path = os.path.join(DATA_DIR, 'bangladesh_hdx_humanitarian_usecases_district_summary.csv')
    assert os.path.exists(csv_path), "bangladesh_hdx_humanitarian_usecases_district_summary.csv missing"

    df = pd.read_csv(csv_path)
    assert len(df) == 64, f"Expected 64 districts in HDX summary, found {len(df)}"

    required_indicators = [
        # Administrative & Exposure
        'DistrictID', 'DistrictName', 'Division', 'Population', 'AreaSqKm',
        'PredictedInundatedPercentage', 'ProjectedExposedPopulation',
        # Use Case 1: Food Security & Crop Shock
        'InundatedAgriLand_Ha', 'StandingRiceLoss_Tons', 'FoodPriceSurge_Pct',
        'IPC_FoodInsecurity_Phase', 'WFP_CopingStrategyIndex_CSI',
        # Use Case 2: Public Health & Epidemic Deserts
        'DistanceToHealth_Km', 'HealthcareDesert_ExposedPop', 'CholeraWaterborne_RiskScore',
        'EmergencyORS_SachetsNeeded', 'Malaria_Parasite_Class',
        # Use Case 3: Displacement & Shelters
        'DisplacedPersonsEst', 'HOTOSM_EducationShelters', 'ShelterCapacityBalance',
        'ShelterSufficiencyTier', 'EmergencyReliefTentsNeeded',
        # Use Case 4: WASH & Water Well Contamination
        'EstimatedTubeWells_Total', 'SubmergedTubeWells_Contaminated',
        'DailySafeWaterDemand_Liters', 'Aquatabs_PurificationTablets_Daily',
        # Use Case 5: INFORM Benchmark
        'UN_INFORM_RiskScore', 'AI_MultiHazardRiskScore', 'ConcordanceDelta',
        'Humanitarian_PriorityTier'
    ]

    for col in required_indicators:
        assert col in df.columns, f"Missing required HDX indicator: {col}"
        assert df[col].isnull().sum() == 0, f"Null values found in HDX indicator: {col}"

    # Demographic & Humanitarian Boundary checks
    assert (df['StandingRiceLoss_Tons'] >= 0).all()
    assert (df['DailySafeWaterDemand_Liters'] >= 0).all()
    assert (df['CholeraWaterborne_RiskScore'].between(0, 100)).all()
    assert (df['UN_INFORM_RiskScore'].between(1.0, 10.0)).all()
    assert (df['AI_MultiHazardRiskScore'].between(1.0, 10.0)).all()


def test_hdx_dataset_linkage_manifest_completeness():
    json_path = os.path.join(DATA_DIR, 'hdx_dataset_linkage_manifest.json')
    csv_path = os.path.join(DATA_DIR, 'hdx_dataset_linkage_manifest.csv')

    assert os.path.exists(json_path), "hdx_dataset_linkage_manifest.json missing"
    assert os.path.exists(csv_path), "hdx_dataset_linkage_manifest.csv missing"

    with open(json_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    assert len(manifest) == 62, f"Expected 62 linked HDX datasets, found {len(manifest)}"

    categories = set()
    for item in manifest:
        assert 'HDX_Dataset_ID' in item and 'HDX_Title' in item
        assert 'Direct_HDX_URL' in item and item['Direct_HDX_URL'].startswith('https://data.humdata.org/dataset/')
        assert 'Internal_Linked_File' in item and len(item['Internal_Linked_File']) > 0
        assert 'Linked_Features_Columns' in item and len(item['Linked_Features_Columns']) > 0
        assert 'Operational_Formula' in item and len(item['Operational_Formula']) > 5
        categories.add(item.get('Operational_Use_Case_Category'))

    # Confirm key operational categories are present
    assert len(categories) >= 5, f"Expected at least 5 operational categories, found {len(categories)}"



