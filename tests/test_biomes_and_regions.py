"""
========================================================================================
tests/test_biomes_and_regions.py
QA Test Suite for Regional Ecological Macro-Biomes and District Stratification
========================================================================================
"""

import os
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def test_biomes_endpoint():
    """Verify GET /api/v1/biomes/summary returns 5 macro-biomes."""
    response = client.get("/api/v1/biomes/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["biomes_count"] == 5
    biome_names = [b["EcologicalBiome"] for b in data["biomes"]]
    expected_biomes = [
        "Northwest Barind Arid Zone",
        "Chattogram Hill Tracts & Ridge Zone",
        "Active Brahmaputra-Meghna Central Floodplain",
        "Haor Basin Depression Zone",
        "Coastal Polders & Mangrove Estuary"
    ]
    for eb in expected_biomes:
        assert eb in biome_names


def test_biomes_district_allocation_integrity():
    """Verify that district counts across the 5 biomes sum precisely to 64."""
    biomes_path = os.path.join(DATA_DIR, 'bcip_regional_biomes_summary.csv')
    assert os.path.exists(biomes_path)
    df_biomes = pd.read_csv(biomes_path)
    assert len(df_biomes) == 5
    assert df_biomes['DistrictCount'].sum() == 64


def test_biomes_demographic_and_spatial_reconciliation():
    """Verify that biome populations and areas reconcile with district-level totals."""
    df_biomes = pd.read_csv(os.path.join(DATA_DIR, 'bcip_regional_biomes_summary.csv'))
    df_dist = pd.read_csv(os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv'))

    total_pop_biomes = df_biomes['TotalPopulation'].sum()
    total_pop_dist = df_dist['Population'].sum()
    assert abs(total_pop_biomes - total_pop_dist) == 0

    total_area_biomes = df_biomes['TotalAreaSqKm'].sum()
    total_area_dist = df_dist['AreaSqKm'].sum()
    assert abs(total_area_biomes - total_area_dist) < 1.0  # float rounding tolerance


def test_biomes_policy_strategies_populated():
    """Verify that every macro-biome has non-empty recommended adaptation strategies."""
    df_biomes = pd.read_csv(os.path.join(DATA_DIR, 'bcip_regional_biomes_summary.csv'))
    for _, row in df_biomes.iterrows():
        assert len(str(row['RecommendedAdaptationStrategy'])) > 10
        assert row['DistrictCount'] > 0
        assert row['TotalPopulation'] > 0
