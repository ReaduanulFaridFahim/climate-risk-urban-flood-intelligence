"""
========================================================================================
run_all_bcip_pipelines.py
BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - END-TO-END PIPELINE RUNNER
Executes the complete research and data engineering pipeline from scratch:
  Step 1: Build 111-Dataset Master Register & Catalog (scripts/17_build_bcip_master_catalog.py)
  Step 2: Compute 4x4 Climate Transitions, Matrices & Exposure (scripts/18_climate_transition_and_scenario_agreement_engine.py)
  Step 3: Multi-Model Screening, Incremental Value & Robustness (scripts/19_adaptation_priority_and_robustness_models.py)
  Step 4: Generate 300 DPI Publication Diagnostic Figures (scripts/20_generate_bcip_publication_figures.py)
Author: Readuanul Farid Fahim | Department of Computer Science & Engineering, North South University
========================================================================================
"""

import os
import sys
import subprocess
import time


def run_step(step_num: int, title: str, script_path: str):
    print("\n" + "=" * 90)
    print(f"STEP {step_num}: {title.upper()}")
    print(f"Executing: {script_path}")
    print("=" * 90)
    
    t0 = time.time()
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    dt = time.time() - t0
    
    print(res.stdout)
    if res.returncode != 0:
        print(f"[!] ERROR in {script_path}:")
        print(res.stderr)
        sys.exit(res.returncode)
        
    print(f"[+] Completed Step {step_num} successfully in {dt:.2f} seconds.")


def main():
    print("*" * 90)
    print("BANGLADESH CLIMATE INTELLIGENCE PLATFORM (BCIP) - FULL REPRODUCIBILITY SUITE")
    print("Scenario-Based Climate Transitions, Multi-Dimensional Exposure, and Adaptation Support")
    print("*" * 90)
    
    t_start = time.time()
    
    run_step(1, "Master Dataset Register & Catalog Ingestion", "scripts/17_build_bcip_master_catalog.py")
    run_step(2, "Climate Transition Matrices & Scenario Agreement", "scripts/18_climate_transition_and_scenario_agreement_engine.py")
    run_step(3, "Multi-Model Prioritization & Robustness Evaluation", "scripts/19_adaptation_priority_and_robustness_models.py")
    run_step(4, "Publication Figures Generation (300 DPI)", "scripts/20_generate_bcip_publication_figures.py")
    run_step(5, "Regional Ecological Biomes & Policy Profiling", "scripts/21_generate_biome_and_upazila_profiles.py")
    run_step(6, "UN HDX Humanitarian Integration & Dispatches", "scripts/22_generate_hdx_humanitarian_usecases.py")
    if os.path.exists("scripts/23_build_website_payload.py"):
        run_step(7, "Master Web Payload & Multi-Dataset Harmonization", "scripts/23_build_website_payload.py")
    
    total_time = time.time() - t_start
    print("\n" + "*" * 90)
    print(f"[SUCCESS] ALL BCIP PIPELINES EXECUTED AND REPRODUCED IN {total_time:.2f} SECONDS.")
    print("All datasets, transition matrices, rankings, and publication figures are ready.")
    print("Launch interactive platform: streamlit run web/app.py or uvicorn api.main:app")
    print("*" * 90)


if __name__ == '__main__':
    main()
