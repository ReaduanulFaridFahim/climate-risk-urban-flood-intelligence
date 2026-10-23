"""
========================================================================================
20_generate_bcip_publication_figures.py
BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - PUBLICATION FIGURES GENERATOR (300 DPI)
Generates 4 high-resolution diagnostic and scientific research figures for the publication:
  - Fig 1: 4-Panel Climate Transition Matrices Heatmap
  - Fig 2: Mid-Century vs Late-Century Scenario Agreement & Divergence Analysis
  - Fig 3: Incremental Value: Rank Displacement & Spearman Correlation Suite
  - Fig 4: Weight Sensitivity & Robust Priority Tier Stratification
Author: Readuanul Farid Fahim | Department of Computer Science & Engineering, North South University
========================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets', 'bcip')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Publication Typography & Style Setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9


def generate_all_publication_figures():
    print("=" * 90)
    print("BCIP STEP 4: GENERATING 300 DPI PUBLICATION FIGURES")
    print("=" * 90)

    # Load required data
    with open(os.path.join(DATA_DIR, 'bcip_climate_transition_matrices.json'), 'r', encoding='utf-8') as f:
        matrices_data = json.load(f)

    with open(os.path.join(DATA_DIR, 'bcip_spearman_rank_correlations.json'), 'r', encoding='utf-8') as f:
        corr_data = json.load(f)

    df_rankings = pd.read_csv(os.path.join(DATA_DIR, 'bcip_model_comparison_and_rankings.csv'))
    df_agree = pd.read_csv(os.path.join(DATA_DIR, 'bcip_scenario_agreement_analysis.csv'))

    classes_short = ["Am", "Aw", "Cwa", "Cwb"]
    classes_full = [
        "Am (Tropical Monsoon)",
        "Aw (Tropical Savannah)",
        "Cwa (Temperate, Dry Winter, Hot Summer)",
        "Cwb (Temperate, Dry Winter, Warm Summer)"
    ]

    # ---------------------------------------------------------------------------------
    # FIGURE 1: 2-PANEL MID-CENTURY CLIMATE TRANSITION MATRICES HEATMAP
    # ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        "Figure 1: Baseline-to-Future Köppen-Geiger Climate Biome Transition Matrices\n"
        "Quantifying Land Area Reclassification (km² and % of Bangladesh) under CMIP6 Mid-Century Scenarios",
        fontsize=14, fontweight='bold', y=0.98
    )

    comparisons_keys = [
        ("Comparison_A_SSP245_2050", "A: Baseline (1991–2020) vs SSP2-4.5 (2041–2070)", axes[0]),
        ("Comparison_B_SSP585_2050", "B: Baseline (1991–2020) vs SSP5-8.5 (2041–2070)", axes[1])
    ]

    for comp_key, comp_title, ax in comparisons_keys:
        m_pct = matrices_data[comp_key]["Results"]["MatrixPct"]
        m_km2 = matrices_data[comp_key]["Results"]["MatrixKm2"]
        
        # Build 4x4 array
        arr_pct = np.zeros((4, 4))
        annot_arr = np.empty((4, 4), dtype=object)
        
        for i, c_from in enumerate(classes_full):
            for j, c_to in enumerate(classes_full):
                val_pct = m_pct[c_from][c_to]
                val_km2 = m_km2[c_from][c_to]
                arr_pct[i, j] = val_pct
                if val_km2 > 0:
                    annot_arr[i, j] = f"{val_km2:,.0f} km²\n({val_pct:.1f}%)"
                else:
                    annot_arr[i, j] = "0 km²\n(0.0%)"

        sns.heatmap(
            arr_pct, annot=annot_arr, fmt="", cmap="Blues",
            xticklabels=classes_short, yticklabels=classes_short,
            cbar_kws={'label': 'Percentage of Total Area (%)'},
            ax=ax, vmin=0, vmax=70, linewidths=0.5, linecolor='gray'
        )
        ax.set_title(comp_title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Future Projected Class", fontsize=10, fontweight='bold')
        ax.set_ylabel("Baseline Class (1991–2020)", fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    f1_path = os.path.join(ASSETS_DIR, 'bcip_fig1_climate_transition_matrices.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 1: {f1_path}")

    # ---------------------------------------------------------------------------------
    # FIGURE 2: SCENARIO AGREEMENT & SPATIAL DIVERGENCE ANALYSIS (MID-CENTURY)
    # ---------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Figure 2: CMIP6 Scenario Agreement & Divergence Analysis (SSP2-4.5 vs SSP5-8.5)\n"
        "Evaluating Directional Concordance and Regional Vulnerability Across 64 Districts",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Mid-Century Breakdown by Status
    mid_summary = df_agree.groupby('MidCentury_AgreementStatus').agg({
        'AreaSqKm': 'sum',
        'DistrictID': 'count'
    }).reset_index()
    mid_summary['AreaPct'] = (mid_summary['AreaSqKm'] / df_agree['AreaSqKm'].sum()) * 100.0

    colors_mid = ['#10b981', '#f59e0b', '#3b82f6'][:len(mid_summary)]
    bars1 = ax1.barh(mid_summary['MidCentury_AgreementStatus'], mid_summary['AreaPct'], color=colors_mid, edgecolor='black', alpha=0.85)
    ax1.set_title("Mid-Century (2041–2070): Agreement Status Distribution", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Percentage of National Land Area (%)", fontsize=10)
    ax1.set_xlim(0, 100)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontweight='bold', fontsize=9)

    # Mid-Century Breakdown by Division
    div_summary = df_agree.groupby(['Division', 'MidCentury_AgreementStatus'])['DistrictID'].count().unstack().fillna(0)
    div_summary.plot(kind='bar', stacked=True, ax=ax2, colormap='Spectral', edgecolor='black', alpha=0.85)
    ax2.set_title("Scenario Concordance by Administrative Division", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Number of Districts", fontsize=10)
    ax2.set_xlabel("Division", fontsize=10)
    ax2.legend(title="Agreement Status", fontsize=8, loc='upper left')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    f2_path = os.path.join(ASSETS_DIR, 'bcip_fig2_scenario_agreement_and_divergence.png')
    plt.savefig(f2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 2: {f2_path}")

    # ---------------------------------------------------------------------------------
    # FIGURE 3: INCREMENTAL VALUE & RANK DISPLACEMENT ANALYSIS
    # ---------------------------------------------------------------------------------
    fig, (ax_corr, ax_scat) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Figure 3: Incremental Value of Climate-Transition Integration vs Baseline Rankings\n"
        "Demonstrating Statistical Independence & Priority Surges in Northern/Riverine Districts",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Heatmap of Spearman Correlations
    corr_models = list(corr_data.keys())
    c_mat = np.zeros((len(corr_models), len(corr_models)))
    for i, m1 in enumerate(corr_models):
        for j, m2 in enumerate(corr_models):
            c_mat[i, j] = corr_data[m1][m2]["SpearmanRho"]

    short_names = ["1. Exposure-Only", "2. Hazard-Only", "3. Climate-Only", "4. Integrated BCIP"]
    sns.heatmap(
        c_mat, annot=True, fmt=".3f", cmap="coolwarm",
        xticklabels=short_names, yticklabels=short_names,
        ax=ax_corr, vmin=-0.8, vmax=1.0, cbar_kws={'label': 'Spearman Rank Correlation (rho)'}
    )
    ax_corr.set_title("Spearman Rank Correlation Across Screening Models", fontsize=11, fontweight='bold')

    # Scatter: Model 1 (Exposure-Only) Rank vs Model 4 (Integrated BCIP) Rank
    # Highlights rank displacement
    scatter = ax_scat.scatter(
        df_rankings['Rank_Model1_ExposureOnly'],
        df_rankings['Rank_Model4_IntegratedBCIP'],
        c=df_rankings['RankDisplacement_BCIP_vs_Exposure'],
        cmap='Spectral_r', s=80, edgecolor='black', alpha=0.85
    )
    cbar = plt.colorbar(scatter, ax=ax_scat)
    cbar.set_label("Rank Displacement (Delta Rank = Model 1 - Model 4)", fontsize=9)

    # 1:1 Identity Line
    ax_scat.plot([1, 64], [1, 64], 'r--', lw=1.5, label='Equal Priority Line (Delta = 0)')
    ax_scat.set_title("District Priority Shift: Exposure-Only vs Integrated BCIP", fontsize=11, fontweight='bold')
    ax_scat.set_xlabel("Model 1 Rank (Exposure-Only Baseline) [1 = Highest Priority]", fontsize=10)
    ax_scat.set_ylabel("Model 4 Rank (Integrated BCIP Framework) [1 = Highest Priority]", fontsize=10)
    ax_scat.set_xlim(0, 65)
    ax_scat.set_ylim(0, 65)
    ax_scat.invert_xaxis()
    ax_scat.invert_yaxis()
    ax_scat.legend(loc='lower right')

    # Annotate key surging districts
    annot_districts = ['Kurigram', 'Netrokona', 'Gaibandha', 'Panchagarh', 'Mymensingh', 'Dinajpur']
    for d_name in annot_districts:
        r_match = df_rankings[df_rankings['DistrictName'] == d_name]
        if not r_match.empty:
            x_val = r_match['Rank_Model1_ExposureOnly'].values[0]
            y_val = r_match['Rank_Model4_IntegratedBCIP'].values[0]
            ax_scat.annotate(
                d_name, (x_val, y_val),
                xytext=(x_val - 2, y_val - 2),
                fontsize=8, fontweight='bold', color='#1e3a8a',
                arrowprops=dict(arrowstyle="->", color='#1e3a8a', lw=1)
            )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    f3_path = os.path.join(ASSETS_DIR, 'bcip_fig3_multi_model_incremental_value_and_rank_displacement.png')
    plt.savefig(f3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 3: {f3_path}")

    # ---------------------------------------------------------------------------------
    # FIGURE 4: WEIGHT SENSITIVITY & ROBUST PRIORITY TIERS
    # ---------------------------------------------------------------------------------
    fig, (ax_box, ax_tier) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Figure 4: Weight Sensitivity & Robust Priority Tier Stratification\n"
        "Testing Priority Stability Across 4 Weighting Configurations across 64 Districts",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Boxplot of rank distribution for Top 15 districts across the 4 weighting schemes
    top_15_names = df_rankings.sort_values('Rank_Model4_IntegratedBCIP').head(15)['DistrictName'].tolist()
    scheme_cols = ['Rank_Weight_Balanced_Equal', 'Rank_Weight_Climate_Heavy', 'Rank_Weight_Exposure_Heavy', 'Rank_Weight_Service_Deficit']
    
    top_15_ranks = []
    for d_name in top_15_names:
        row = df_rankings[df_rankings['DistrictName'] == d_name]
        vals = row[scheme_cols].values[0]
        for v in vals:
            top_15_ranks.append({'DistrictName': d_name, 'Rank': v})
            
    df_box = pd.DataFrame(top_15_ranks)
    sns.boxplot(
        x='Rank', y='DistrictName', data=df_box, ax=ax_box,
        palette='Blues_r', orient='h', boxprops=dict(alpha=0.8)
    )
    ax_box.set_title("Rank Range Across 4 Weight Schemes (Top 15 Districts)", fontsize=11, fontweight='bold')
    ax_box.set_xlabel("District Rank [1 = Highest Priority]", fontsize=10)
    ax_box.set_ylabel("District Name", fontsize=10)

    # Robustness Tier Distribution
    tier_counts = df_rankings['BCIP_RobustnessTier'].value_counts()
    tier_labels = [
        "Tier 1: Highly Robust Priority\n(Top 18 in All Schemes)",
        "Tier 2: Climate-Sensitive\n(Surges under Climate Shift)",
        "Tier 3: Exposure-Driven\n(Dense Population Center)",
        "Tier 4: Stable Moderate\n/ Lower Urgency"
    ]
    tier_palette = ['#dc2626', '#f59e0b', '#2563eb', '#64748b']
    
    # Match order
    ordered_counts = [
        tier_counts.get("Tier 1: Highly Robust Priority (Top 18 in All Schemes)", 0),
        tier_counts.get("Tier 2: Climate-Sensitive Priority (Surges under Climate Shift)", 0),
        tier_counts.get("Tier 3: Exposure-Driven Priority (Dense Population Center)", 0),
        tier_counts.get("Tier 4: Stable Moderate / Lower Urgency", 0)
    ]

    bars_t = ax_tier.bar(tier_labels, ordered_counts, color=tier_palette, edgecolor='black', alpha=0.85)
    ax_tier.set_title("National Stratification into Robust Priority Tiers", fontsize=11, fontweight='bold')
    ax_tier.set_ylabel("Number of Districts", fontsize=10)
    ax_tier.set_ylim(0, 50)
    for bar in bars_t:
        h = bar.get_height()
        ax_tier.text(bar.get_x() + bar.get_width()/2, h + 1, f"{int(h)} ({h/64*100:.1f}%)", ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    f4_path = os.path.join(ASSETS_DIR, 'bcip_fig4_weight_sensitivity_and_robustness_tiers.png')
    plt.savefig(f4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 4: {f4_path}")

    print("=" * 90)
    print("ALL 4 PUBLICATION FIGURES GENERATED AT 300 DPI.")
    print("=" * 90)


if __name__ == '__main__':
    generate_all_publication_figures()
