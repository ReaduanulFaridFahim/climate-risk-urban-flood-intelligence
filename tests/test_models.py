"""
Verification of Multi-Model Scoring, Incremental Value & Weight Sensitivity
"""

import os
import json
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def test_model_scores_and_ranks_validity():
    rank_path = os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv')
    assert os.path.exists(rank_path), "Missing model comparison rankings dataset"

    df = pd.read_csv(rank_path)
    assert len(df) == 64

    score_cols = [
        'Score_Model1_ExposureOnly',
        'Score_Model2_HazardContextOnly',
        'Score_Model3_ClimateTransitionOnly',
        'Score_Model4_IntegratedBCIP'
    ]
    for sc in score_cols:
        assert sc in df.columns, f"Missing score column: {sc}"
        assert df[sc].between(0.0, 1.0).all(), f"Scores in {sc} not bounded in [0, 1]"

    # Continuous models must produce strict 1..64 permutations
    for rc in ['Rank_Model1_ExposureOnly', 'Rank_Model2_HazardContextOnly', 'Rank_Model4_IntegratedBCIP']:
        assert rc in df.columns, f"Missing rank column: {rc}"
        assert set(df[rc].values) == set(range(1, 65)), f"Ranks in {rc} are not a valid 1..64 permutation"

    # Categorical/tied Model 3 ranks must fall within valid range [1, 64]
    assert df['Rank_Model3_ClimateTransitionOnly'].between(1, 64).all()


def test_spearman_rank_correlations_incremental_value():
    corr_path = os.path.join(DATA_DIR, 'bcip_spearman_rank_correlations.json')
    assert os.path.exists(corr_path), "Missing Spearman correlations JSON"

    with open(corr_path, 'r', encoding='utf-8') as f:
        corrs = json.load(f)

    models = [
        "Model 1 (Exposure-Only)",
        "Model 2 (Hazard-Only)",
        "Model 3 (Climate-Only)",
        "Model 4 (Integrated BCIP)"
    ]

    for m in models:
        assert m in corrs, f"Model {m} missing from correlation matrix"
        # Diagonal must be 1.0
        assert abs(corrs[m][m]["SpearmanRho"] - 1.0) < 1e-4

    # Incremental value assertion: Model 4 must not be trivially identical to Model 1
    rho_1_4 = corrs["Model 1 (Exposure-Only)"]["Model 4 (Integrated BCIP)"]["SpearmanRho"]
    assert rho_1_4 < 0.70, f"Model 4 is too collinear with Model 1 (rho={rho_1_4}); lacks incremental value"


def test_tier1_robust_districts():
    rank_path = os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv')
    df = pd.read_csv(rank_path)

    tier1_districts = df[df['BCIP_RobustnessTier'].str.contains('Tier 1')]
    assert len(tier1_districts) >= 10, f"Expected at least 10 Tier-1 districts, found {len(tier1_districts)}"

    # All Tier 1 districts must maintain high priority (max rank <= 20) across all weight schemes
    scheme_cols = ['Rank_Weight_Balanced_Equal', 'Rank_Weight_Climate_Heavy', 'Rank_Weight_Exposure_Heavy', 'Rank_Weight_Service_Deficit']
    for _, row in tier1_districts.iterrows():
        for col in scheme_cols:
            assert row[col] <= 20, f"District {row['DistrictName']} exceeded rank 20 in {col}"
