"""
========================================================================================
modules/deltasim.py
PILLAR 4: DELTASIM - NATIONAL CLIMATE POLICY & BUDGET SIMULATION SANDBOX
========================================================================================
Interactive simulation sandbox where policymakers allocate national adaptation capital
($500M budget) across four strategic interventions to simulate 2030, 2040, and 2050
national resilience outcomes and calculate a composite National Resilience Score.
========================================================================================
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def _load_future_data():
    path = os.path.join(DATA_DIR, 'bangladesh_future_projections_2030_2050_all64.csv')
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def run_policy_simulation(
    budget_millions_usd: float = 500.0,
    polder_pct: float = 35.0,
    shelter_pct: float = 25.0,
    afforestation_pct: float = 20.0,
    farmer_subsidy_pct: float = 20.0
) -> dict:
    """
    Executes multi-decadal simulation based on capital allocations.
    """
    total_alloc = polder_pct + shelter_pct + afforestation_pct + farmer_subsidy_pct
    if abs(total_alloc - 100.0) > 0.1:
        # Normalize
        polder_pct = (polder_pct / total_alloc) * 100.0
        shelter_pct = (shelter_pct / total_alloc) * 100.0
        afforestation_pct = (afforestation_pct / total_alloc) * 100.0
        farmer_subsidy_pct = (farmer_subsidy_pct / total_alloc) * 100.0

    polder_budget = (polder_pct / 100.0) * budget_millions_usd
    shelter_budget = (shelter_pct / 100.0) * budget_millions_usd
    afforest_budget = (afforestation_pct / 100.0) * budget_millions_usd
    subsidy_budget = (farmer_subsidy_pct / 100.0) * budget_millions_usd

    # Unmitigated Baseline (2050 Long-Term Severe):
    # Mean Inundation: 61.19%, Flooded: 91,228 sq km, Exposed: 105.6 Million, Loss: $693.2 Billion
    unmitigated_flooded_km2 = 91227.9
    unmitigated_exposed_pop = 105600000
    unmitigated_damage_billion = 693.2

    # Mitigation Effectiveness Curves
    # Polders directly reduce physical flooded footprint
    flood_reduction_km2 = min(22000.0, polder_budget * 75.0 + afforest_budget * 35.0)
    mitigated_flooded_km2 = round(unmitigated_flooded_km2 - flood_reduction_km2, 1)

    # Shelters + Subsidies directly protect lives and prevent catastrophic displacement
    lives_protected = int(min(28000000, shelter_budget * 85000 + subsidy_budget * 45000))
    residual_exposed_pop = max(0, unmitigated_exposed_pop - lives_protected)

    # Economic Damage Avoided ($Billion)
    damage_avoided_billion = round(min(210.0, (flood_reduction_km2 / unmitigated_flooded_km2) * 450.0 + (subsidy_budget / 10.0)), 1)
    residual_damage_billion = round(max(0.0, unmitigated_damage_billion - damage_avoided_billion), 1)

    # Composite National Resilience Score (0 to 100)
    # Balanced allocations score higher than extreme single-bucket bets
    diversity_penalty = max(0.0, (max(polder_pct, shelter_pct, afforestation_pct, farmer_subsidy_pct) - 50.0) * 0.4)
    resilience_score = float(np.clip(
        (damage_avoided_billion / 210.0) * 40.0 +
        (lives_protected / 28000000) * 40.0 +
        (flood_reduction_km2 / 22000.0) * 20.0 -
        diversity_penalty,
        15.0, 98.5
    ))

    # Evaluation Grade
    if resilience_score >= 85.0:
        evaluation = "VISIONARY CLIMATE STRATEGY 🏆 (Optimal Cross-Sector Balance)"
        recommendation = "Exceptional balance: Protects agricultural base while fortifying physical coastal defenses."
    elif resilience_score >= 70.0:
        evaluation = "EFFECTIVE ADAPTATION PLAN 🛡️ (Strong Multi-Sector Protection)"
        recommendation = "Solid deployment. Consider slightly increasing shelter capacity in haor wetlands."
    elif resilience_score >= 50.0:
        evaluation = "MODERATE MITIGATION ⚠️ (Vulnerable Pockets Remain)"
        recommendation = "Uneven capital spread leaves either farmers or coastal polders exposed to extreme shocks."
    else:
        evaluation = "DEFICIENT RESILIENCE 🚨 (Critical Under-Investment)"
        recommendation = "Budget allocation insufficient to prevent widespread 2050 decadal displacement."

    return {
        'total_budget_usd_millions': budget_millions_usd,
        'allocation': {
            'coastal_polders_millions_usd': round(polder_budget, 1),
            'cyclone_shelters_millions_usd': round(shelter_budget, 1),
            'mangrove_afforestation_millions_usd': round(afforest_budget, 1),
            'farmer_direct_subsidies_millions_usd': round(subsidy_budget, 1)
        },
        'unmitigated_2050': {
            'flooded_area_sqkm': unmitigated_flooded_km2,
            'exposed_population': unmitigated_exposed_pop,
            'projected_damage_usd_billions': unmitigated_damage_billion
        },
        'mitigated_2050': {
            'flooded_area_sqkm': mitigated_flooded_km2,
            'flood_area_prevented_sqkm': round(flood_reduction_km2, 1),
            'lives_protected': lives_protected,
            'residual_exposed_population': residual_exposed_pop,
            'damage_avoided_usd_billions': damage_avoided_billion,
            'residual_damage_usd_billions': residual_damage_billion
        },
        'national_resilience_score': round(resilience_score, 1),
        'evaluation': evaluation,
        'recommendation': recommendation
    }
