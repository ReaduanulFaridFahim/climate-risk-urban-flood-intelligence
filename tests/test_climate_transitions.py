"""
Mathematical Validation of CMIP6 Climate Transition Matrices & Scenario Agreement
"""

import os
import json
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def test_transition_matrices_area_conservation():
    mat_path = os.path.join(DATA_DIR, 'bcip_climate_transition_matrices.json')
    assert os.path.exists(mat_path), "Missing transition matrices JSON"

    with open(mat_path, 'r', encoding='utf-8') as f:
        matrices = json.load(f)

    expected_comparisons = [
        "Comparison_A_SSP245_2050",
        "Comparison_B_SSP585_2050"
    ]

    for comp_id in expected_comparisons:
        assert comp_id in matrices, f"Missing comparison {comp_id} in matrices JSON"
        res = matrices[comp_id]["Results"]
        total_nat_area = res["TotalNationalAreaKm2"]
        m_km2 = res["MatrixKm2"]

        # Sum of all matrix cells must equal total national area
        sum_cells = sum(m_km2[c_from][c_to] for c_from in m_km2 for c_to in m_km2[c_from])
        assert abs(sum_cells - total_nat_area) < 1.0, f"Area not conserved in {comp_id}: sum={sum_cells}, expected={total_nat_area}"

        # Sum of changed area + stable area must equal total area
        assert abs((res["TotalAreaChangedKm2"] + res["TotalAreaStableKm2"]) - total_nat_area) < 1.0

        # Percentages must sum to 100%
        m_pct = res["MatrixPct"]
        sum_pct = sum(m_pct[c_from][c_to] for c_from in m_pct for c_to in m_pct[c_from])
        assert abs(sum_pct - 100.0) < 0.5, f"Percentages do not sum to 100% in {comp_id}: {sum_pct}%"


def test_class_net_balance_conservation():
    mat_path = os.path.join(DATA_DIR, 'bcip_climate_transition_matrices.json')
    with open(mat_path, 'r', encoding='utf-8') as f:
        matrices = json.load(f)

    for comp_id, c_data in matrices.items():
        stats = c_data["Results"]["ClassStats"]
        net_changes = [v["NetChangeKm2"] for v in stats.values()]
        # Net balance across all classes must sum to zero
        assert abs(sum(net_changes)) < 1.0, f"Net balance does not sum to zero in {comp_id}: sum={sum(net_changes)}"


def test_scenario_agreement_integrity():
    agree_path = os.path.join(DATA_DIR, 'bcip_scenario_agreement_analysis.csv')
    assert os.path.exists(agree_path), "Missing scenario agreement dataset"

    df_agree = pd.read_csv(agree_path)
    assert len(df_agree) == 64, f"Expected 64 districts in agreement table, found {len(df_agree)}"

    # Valid status values
    valid_statuses = [
        "High Agreement (Stable Class)",
        "High Agreement (Identical Transition)",
        "Scenario Divergence (One Transitions, One Stable)",
        "Scenario Divergence (Different Transition Classes)"
    ]

    assert df_agree['MidCentury_AgreementStatus'].isin(valid_statuses).all()
